"""
State vector construction for production inference and ETL.

Replicates env._get_obs() + AugmentedProductEnv feature concatenation,
but takes explicit parameters instead of reading simulator internals.
"""

import numpy as np

from deployment.config import (
    EPISODE_LENGTH,
    MARKDOWN_WINDOW_HOURS,
    N_ACTIONS,
    N_TIME_BLOCKS,
    STEP_HOURS,
)


class StateConstructor:
    """Builds the 14-dim state vector from session data.

    Exactly replicates:
      - env._get_obs() (features 0-9)
      - AugmentedProductEnv (features 10-13)

    Parameters
    ----------
    product_features : np.ndarray[4]
        Pre-computed [price_norm, cost_frac_norm, inventory_norm, pack_size_norm].
    initial_inventory : int
        Inventory at session start (actual_initial_inventory).
    """

    def __init__(self, product_features: np.ndarray, initial_inventory: int):
        self.product_features = np.asarray(product_features, dtype=np.float32)
        self.initial_inventory = max(initial_inventory, 1)

    def build_state(
        self,
        step_count: int,
        inventory_remaining: int,
        current_discount_idx: int,
        time_block: int,
        day_of_week: int,
        recent_sales: list,
        total_sold: int,
    ) -> np.ndarray:
        """Build a 14-dim state vector.

        Parameters
        ----------
        step_count : int
            Steps completed so far (0 at session start).
        inventory_remaining : int
            Current unsold inventory.
        current_discount_idx : int
            Active discount index (0-10).
        time_block : int
            Hour-of-day // step_hours (0-11 for 2h steps).
        day_of_week : int
            0=Monday ... 6=Sunday.
        recent_sales : list[int]
            Per-step sales history for velocity (last 3 used).
        total_sold : int
            Cumulative units sold this session.

        Returns
        -------
        np.float32[14]
        """
        # [0] hours_remaining (normalized)
        hours_remaining = (EPISODE_LENGTH - step_count) * STEP_HOURS
        hours_remaining_norm = hours_remaining / MARKDOWN_WINDOW_HOURS

        # [1] inventory_remaining (normalized)
        inv_norm = inventory_remaining / self.initial_inventory

        # [2] current_discount_idx (normalized)
        discount_norm = current_discount_idx / max(N_ACTIONS - 1, 1)

        # [3-4] time-of-day cyclical encoding
        tod_angle = 2.0 * np.pi * time_block / N_TIME_BLOCKS
        tod_sin = (np.sin(tod_angle) + 1.0) / 2.0
        tod_cos = (np.cos(tod_angle) + 1.0) / 2.0

        # [5-6] day-of-week cyclical encoding
        dow_angle = 2.0 * np.pi * day_of_week / 7.0
        dow_sin = (np.sin(dow_angle) + 1.0) / 2.0
        dow_cos = (np.cos(dow_angle) + 1.0) / 2.0

        # [7] recent_velocity (normalized)
        velocity = float(np.mean(recent_sales[-3:])) if recent_sales else 0.0
        max_velocity = self.initial_inventory * 0.5
        velocity_norm = min(velocity / max(max_velocity, 1.0), 1.0)

        # [8] sell_through_rate
        if step_count > 0:
            ideal_rate = self.initial_inventory / EPISODE_LENGTH
            actual_rate = total_sold / step_count
            sell_through_rate = min(actual_rate / max(ideal_rate, 1e-3), 1.0)
        else:
            sell_through_rate = 0.0

        # [9] projected_clearance
        if inventory_remaining <= 0 or step_count >= EPISODE_LENGTH:
            projected_clearance = 1.0
        else:
            remaining_steps = EPISODE_LENGTH - step_count
            projected_sales = velocity * remaining_steps
            projected_clearance = min(
                projected_sales / max(inventory_remaining, 1), 1.0
            )

        # [0-9] base observation
        base_obs = np.array([
            hours_remaining_norm,
            inv_norm,
            discount_norm,
            tod_sin,
            tod_cos,
            dow_sin,
            dow_cos,
            velocity_norm,
            sell_through_rate,
            projected_clearance,
        ], dtype=np.float32)

        base_obs = np.clip(base_obs, 0.0, 1.0)

        # [10-13] product features
        return np.concatenate([base_obs, self.product_features])
