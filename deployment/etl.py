"""
Session CSV → transition tuples for batch training.

Reads completed session CSVs, validates data, computes rewards matching
env.step(), and produces (s, a, r, s', done, mask) tuples ready for
agent.store_transition().
"""

import csv
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

from deployment.config import (
    CLEARANCE_BONUS,
    DISCOUNT_LEVELS,
    EPISODE_LENGTH,
    HOLDING_COST_PER_STEP,
    N_ACTIONS,
    STEP_HOURS,
    WASTE_PENALTY_MULTIPLIER,
    ProductionConfig,
)
from deployment.state import StateConstructor

# Type alias for a transition tuple
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]


class SessionETL:
    """Converts session CSV files into RL transition tuples.

    Parameters
    ----------
    config : ProductionConfig
        Paths configuration.
    """

    def __init__(self, config: ProductionConfig):
        self.config = config

    def load_session_csv(self, csv_path: str) -> List[dict]:
        """Read and validate a session CSV.

        Returns list of row dicts with proper types. Validates:
          - Progressive discount constraint (discount_idx never decreases)
          - Required columns present
          - Numeric type conversions
        """
        rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {
                    "session_id": row["session_id"],
                    "sku_name": row["sku_name"],
                    "step": int(row["step"]),
                    "timestamp": row["timestamp"],
                    "discount_pct": float(row["discount_pct"]),
                    "discount_idx": int(row["discount_idx"]),
                    "units_sold": int(row["units_sold"]),
                    "inventory_before": int(row["inventory_before"]),
                    "inventory_after": int(row["inventory_after"]),
                    "revenue": float(row["revenue"]),
                    "day_of_week": int(row["day_of_week"]),
                    "time_block": int(row["time_block"]),
                }
                rows.append(parsed)

        # Validate progressive constraint
        for i in range(1, len(rows)):
            if rows[i]["discount_idx"] < rows[i - 1]["discount_idx"]:
                raise ValueError(
                    f"Progressive constraint violated at step {rows[i]['step']} "
                    f"in {csv_path}: discount_idx went from "
                    f"{rows[i-1]['discount_idx']} to {rows[i]['discount_idx']}"
                )

        return rows

    def session_to_transitions(
        self,
        rows: List[dict],
        base_price: float,
        cost_per_unit: float,
        product_features: np.ndarray,
    ) -> List[Transition]:
        """Convert a session's rows into (s, a, r, s', done, mask) tuples.

        Reward formula exactly matches env.step():
          - Step reward: revenue - holding_cost
          - Terminal waste: -cost_per_unit * WASTE_PENALTY_MULTIPLIER * remaining
          - Clearance bonus: +CLEARANCE_BONUS * initial_inventory (if cleared early)
        """
        if len(rows) < 2:
            return []

        initial_inventory = rows[0]["inventory_before"]
        state_ctor = StateConstructor(product_features, initial_inventory)

        transitions = []
        recent_sales = []
        total_sold = 0

        for t in range(len(rows)):
            row = rows[t]

            # Skip steps where inventory was already 0
            if row["inventory_before"] == 0:
                recent_sales.append(0)
                continue

            # Build s_t (state BEFORE this step's action)
            s_t = state_ctor.build_state(
                step_count=row["step"],
                inventory_remaining=row["inventory_before"],
                current_discount_idx=row["discount_idx"],
                time_block=row["time_block"],
                day_of_week=row["day_of_week"],
                recent_sales=recent_sales,
                total_sold=total_sold,
            )

            # Update running trackers (sales from this step)
            recent_sales.append(row["units_sold"])
            total_sold += row["units_sold"]

            # --- Compute reward (matches env.step()) ---
            price = base_price * (1.0 - DISCOUNT_LEVELS[row["discount_idx"]])
            revenue = price * row["units_sold"]
            holding = (
                HOLDING_COST_PER_STEP
                * price
                * row["inventory_after"]
                * (STEP_HOURS / 4.0)
            )
            reward = revenue - holding

            # Check termination
            inventory_cleared = row["inventory_after"] == 0
            window_expired = (row["step"] + 1) >= EPISODE_LENGTH
            done = inventory_cleared or window_expired

            # Terminal rewards
            if done:
                if row["inventory_after"] > 0:
                    waste_cost = (
                        cost_per_unit
                        * WASTE_PENALTY_MULTIPLIER
                        * row["inventory_after"]
                    )
                    reward -= waste_cost
                if inventory_cleared and not window_expired:
                    bonus = CLEARANCE_BONUS * initial_inventory
                    reward += bonus

            # Build s_{t+1}
            next_step = row["step"] + 1
            next_time_block = (row["time_block"] + 1) % (24 // STEP_HOURS)
            next_day_of_week = row["day_of_week"]
            # Handle midnight crossing
            if next_time_block < row["time_block"]:
                next_day_of_week = (next_day_of_week + 1) % 7

            s_t1 = state_ctor.build_state(
                step_count=next_step,
                inventory_remaining=row["inventory_after"],
                current_discount_idx=row["discount_idx"],
                time_block=next_time_block,
                day_of_week=next_day_of_week,
                recent_sales=recent_sales,
                total_sold=total_sold,
            )

            # Action mask for s_{t+1}
            if done:
                next_mask = np.ones(N_ACTIONS, dtype=bool)
            else:
                next_mask = np.zeros(N_ACTIONS, dtype=bool)
                next_mask[row["discount_idx"]:] = True

            transitions.append((
                s_t,
                row["discount_idx"],
                reward,
                s_t1,
                done,
                next_mask,
            ))

            if done:
                break

        return transitions

    def load_sessions_for_date_range(
        self,
        start_date: str,
        end_date: str,
        sku_name: str,
    ) -> List[str]:
        """Find all session CSV paths for a SKU within a date range.

        Parameters
        ----------
        start_date, end_date : str
            ISO date strings (YYYY-MM-DD), inclusive.
        sku_name : str
            SKU name to filter for.

        Returns
        -------
        list of str
            Sorted paths to matching session CSVs.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        paths = []
        current = start
        while current <= end:
            date_str = current.isoformat()
            csv_path = self.config.session_csv_path(date_str, sku_name)
            if os.path.exists(csv_path):
                paths.append(csv_path)
            current += timedelta(days=1)

        return sorted(paths)

    def etl_sessions(
        self,
        csv_paths: List[str],
        base_price: float,
        cost_per_unit: float,
        product_features: np.ndarray,
    ) -> List[Transition]:
        """Run full ETL pipeline on multiple session CSVs.

        Returns all transitions concatenated.
        """
        all_transitions = []
        for path in csv_paths:
            try:
                rows = self.load_session_csv(path)
                transitions = self.session_to_transitions(
                    rows, base_price, cost_per_unit, product_features
                )
                all_transitions.extend(transitions)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping %s: %s", path, e
                )
        return all_transitions
