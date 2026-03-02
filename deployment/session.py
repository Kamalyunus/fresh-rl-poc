"""
Daytime session tracking: manages active markdown sessions and logs steps to CSV.

The caller's code (pricing service, scheduler) drives the loop:
  1. SessionManager.start_session() — creates session + CSV file
  2. SessionManager.get_current_state() — returns 14-dim state for inference
  3. (caller gets discount from PricingAgent)
  4. (caller waits 2h, observes sales)
  5. SessionManager.record_step() — updates internal state + appends CSV row
"""

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from deployment.config import (
    DISCOUNT_LEVELS,
    EPISODE_LENGTH,
    N_ACTIONS,
    SESSION_CSV_COLUMNS,
    STEP_HOURS,
    ProductionConfig,
)
from deployment.state import StateConstructor


@dataclass
class ActiveSession:
    """Per-session state tracking."""

    session_id: str
    sku_name: str
    category: str
    base_price: float
    cost_per_unit: float
    initial_inventory: int
    start_time: datetime

    # Mutable state
    step_count: int = 0
    current_discount_idx: int = 0
    inventory_remaining: int = 0
    total_sold: int = 0
    total_revenue: float = 0.0
    recent_sales: list = field(default_factory=list)

    # Components
    state_constructor: Optional[StateConstructor] = field(default=None, repr=False)
    csv_path: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.inventory_remaining == 0 or self.step_count >= EPISODE_LENGTH

    @property
    def time_block(self) -> int:
        """Current time block based on start_time + steps elapsed."""
        elapsed_hours = self.step_count * STEP_HOURS
        current_hour = self.start_time.hour + elapsed_hours
        return (current_hour // STEP_HOURS) % (24 // STEP_HOURS)

    @property
    def day_of_week(self) -> int:
        """Current day of week (0=Mon, 6=Sun), accounting for midnight crossings."""
        elapsed_hours = self.step_count * STEP_HOURS
        current_hour = self.start_time.hour + elapsed_hours
        day_offset = current_hour // 24
        return (self.start_time.weekday() + day_offset) % 7

    def action_masks(self) -> np.ndarray:
        """Return boolean mask: True for valid actions (>= current discount idx)."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[self.current_discount_idx:] = True
        return mask


class SessionManager:
    """Manages active markdown sessions during daytime operation.

    Parameters
    ----------
    config : ProductionConfig
        Paths and settings.
    """

    def __init__(self, config: ProductionConfig):
        self.config = config
        self._sessions: Dict[str, ActiveSession] = {}

    def start_session(
        self,
        session_id: str,
        sku_name: str,
        category: str,
        initial_inventory: int,
        base_price: float,
        cost_per_unit: float,
        product_features: np.ndarray,
        start_time: datetime,
    ) -> ActiveSession:
        """Start a new markdown session.

        Creates the ActiveSession, initializes the StateConstructor,
        and creates the CSV file with headers.
        """
        state_ctor = StateConstructor(product_features, initial_inventory)

        date_str = start_time.strftime("%Y-%m-%d")
        csv_path = self.config.session_csv_path(date_str, sku_name)

        session = ActiveSession(
            session_id=session_id,
            sku_name=sku_name,
            category=category,
            base_price=base_price,
            cost_per_unit=cost_per_unit,
            initial_inventory=initial_inventory,
            start_time=start_time,
            inventory_remaining=initial_inventory,
            state_constructor=state_ctor,
            csv_path=csv_path,
        )

        # Write CSV header
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SESSION_CSV_COLUMNS)

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ActiveSession:
        """Get an active session by ID."""
        if session_id not in self._sessions:
            raise KeyError(f"No active session '{session_id}'")
        return self._sessions[session_id]

    def get_current_state(self, session_id: str) -> np.ndarray:
        """Build the 14-dim state vector for the current session state."""
        s = self.get_session(session_id)
        return s.state_constructor.build_state(
            step_count=s.step_count,
            inventory_remaining=s.inventory_remaining,
            current_discount_idx=s.current_discount_idx,
            time_block=s.time_block,
            day_of_week=s.day_of_week,
            recent_sales=s.recent_sales,
            total_sold=s.total_sold,
        )

    def record_step(
        self,
        session_id: str,
        discount_idx: int,
        units_sold: int,
        inventory_after: int,
        revenue: float,
        timestamp: datetime,
    ) -> None:
        """Record an observed step outcome and append to CSV.

        Call this after the 2h window elapses and sales are observed.
        """
        s = self.get_session(session_id)

        if s.is_terminal:
            raise ValueError(f"Session '{session_id}' is already terminal")

        # Enforce progressive constraint
        discount_idx = max(discount_idx, s.current_discount_idx)
        discount_pct = float(DISCOUNT_LEVELS[discount_idx] * 100)

        # Write CSV row
        row = [
            s.session_id,
            s.sku_name,
            s.step_count,
            timestamp.isoformat(),
            f"{discount_pct:.0f}",
            discount_idx,
            units_sold,
            s.inventory_remaining,  # inventory_before
            inventory_after,
            f"{revenue:.2f}",
            s.day_of_week,
            s.time_block,
        ]
        with open(s.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Update internal state
        s.recent_sales.append(units_sold)
        s.total_sold += units_sold
        s.total_revenue += revenue
        s.current_discount_idx = discount_idx
        s.inventory_remaining = inventory_after
        s.step_count += 1

    def end_session(self, session_id: str) -> ActiveSession:
        """Remove a session from active tracking and return it."""
        return self._sessions.pop(session_id)

    @property
    def active_session_ids(self) -> list:
        return list(self._sessions.keys())

    def resume_from_csv(
        self,
        csv_path: str,
        sku_name: str,
        category: str,
        base_price: float,
        cost_per_unit: float,
        product_features: np.ndarray,
        start_time: datetime,
    ) -> ActiveSession:
        """Recover a session by replaying logged CSV steps.

        Used for crash recovery: reads the CSV, reconstructs internal state,
        and re-registers the session as active.
        """
        rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"Empty session CSV: {csv_path}")

        session_id = rows[0]["session_id"]
        initial_inventory = int(rows[0]["inventory_before"])

        state_ctor = StateConstructor(product_features, initial_inventory)

        session = ActiveSession(
            session_id=session_id,
            sku_name=sku_name,
            category=category,
            base_price=base_price,
            cost_per_unit=cost_per_unit,
            initial_inventory=initial_inventory,
            start_time=start_time,
            state_constructor=state_ctor,
            csv_path=csv_path,
        )

        # Replay steps to reconstruct state
        for row in rows:
            units_sold = int(row["units_sold"])
            session.recent_sales.append(units_sold)
            session.total_sold += units_sold
            session.total_revenue += float(row["revenue"])
            session.current_discount_idx = int(row["discount_idx"])
            session.inventory_remaining = int(row["inventory_after"])
            session.step_count += 1

        self._sessions[session_id] = session
        return session
