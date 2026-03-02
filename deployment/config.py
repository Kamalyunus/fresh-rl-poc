"""
Production configuration: constants, paths, and defaults matching POC v2.1.

All shared constants live here so other deployment modules import from one place.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Environment constants (must match POC) ────────────────────────────────

STEP_HOURS = 2
N_ACTIONS = 11
EPISODE_LENGTH = 12  # 24h / 2h
MARKDOWN_WINDOW_HOURS = 24
STATE_DIM = 14  # 10 base + 4 product features
N_TIME_BLOCKS = 24 // STEP_HOURS  # 12

DISCOUNT_LEVELS = np.array(
    [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
)

# ── DQN defaults (v2.1) ──────────────────────────────────────────────────

HIDDEN_DIM = 128
LR = 5e-4
GAMMA = 0.97
N_STEP = 5
BUFFER_SIZE = 50_000
BATCH_SIZE = 32
USE_PER = True
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-5
EPSILON_START = 0.10   # production: warm-started, lower than POC
EPSILON_END = 0.02
EPSILON_DECAY = 0.999
TAU = 0.005
TAU_START = 0.005
TAU_END = 0.005
TAU_WARMUP_STEPS = 0
HOLD_ACTION_PROB = 0.5

# ── Reward constants (must match env.step()) ──────────────────────────────

WASTE_PENALTY_MULTIPLIER = 3.0
HOLDING_COST_PER_STEP = 0.02
CLEARANCE_BONUS = 1.0

# ── Batch training defaults ──────────────────────────────────────────────

TRAINING_STEPS_PER_SESSION = 50
LOOKBACK_DAYS = 30
WARMUP_STEPS = 1000
TRANSFER_EPSILON = 0.30   # epsilon when cold-starting from pooled model
ELEVATED_PER_PRIORITY = 5.0

# ── Session CSV columns ──────────────────────────────────────────────────

SESSION_CSV_COLUMNS = [
    "session_id", "sku_name", "step", "timestamp",
    "discount_pct", "discount_idx", "units_sold",
    "inventory_before", "inventory_after", "revenue",
    "day_of_week", "time_block",
]


@dataclass
class ProductionConfig:
    """Constructs all deployment paths from a single base directory.

    Usage:
        config = ProductionConfig(base_dir="deployment")
        config.ensure_dirs()
    """

    base_dir: str = "deployment"

    # Derived paths (set in __post_init__)
    data_dir: str = field(init=False)
    sessions_dir: str = field(init=False)
    models_dir: str = field(init=False)
    pooled_models_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    product_master_path: str = field(init=False)

    def __post_init__(self):
        self.data_dir = os.path.join(self.base_dir, "data")
        self.sessions_dir = os.path.join(self.data_dir, "sessions")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.pooled_models_dir = os.path.join(self.models_dir, "_pooled")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.product_master_path = os.path.join(self.data_dir, "product_master.csv")

    def ensure_dirs(self):
        """Create all required directories."""
        for d in [self.sessions_dir, self.models_dir, self.pooled_models_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

    def session_dir_for_date(self, date_str: str) -> str:
        """Return path to sessions/{date}/, creating it if needed."""
        d = os.path.join(self.sessions_dir, date_str)
        os.makedirs(d, exist_ok=True)
        return d

    def session_csv_path(self, date_str: str, sku_name: str) -> str:
        """Return path: data/sessions/{date}/S-{date}-{sku}.csv"""
        return os.path.join(
            self.session_dir_for_date(date_str),
            f"S-{date_str}-{sku_name}.csv",
        )

    def model_dir_for_sku(self, sku_name: str) -> str:
        """Return path: models/{sku_name}/, creating it if needed."""
        d = os.path.join(self.models_dir, sku_name)
        os.makedirs(d, exist_ok=True)
        return d

    def checkpoint_path(self, sku_name: str) -> str:
        """Return path: models/{sku_name}/agent.pt"""
        return os.path.join(self.model_dir_for_sku(sku_name), "agent.pt")

    def prev_checkpoint_path(self, sku_name: str) -> str:
        """Return path: models/{sku_name}/agent_prev.pt"""
        return os.path.join(self.model_dir_for_sku(sku_name), "agent_prev.pt")

    def pooled_model_path(self, category: str, variant: str = "plain") -> str:
        """Return path: models/_pooled/{category}/pooled_{cat}_{variant}_2h.pt"""
        cat_dir = os.path.join(self.pooled_models_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        return os.path.join(cat_dir, f"pooled_{category}_{variant}_2h.pt")

    def batch_log_path(self, date_str: str) -> str:
        """Return path: logs/batch_train_{date}.log"""
        return os.path.join(self.logs_dir, f"batch_train_{date_str}.log")

    def batch_report_path(self, date_str: str) -> str:
        """Return path: logs/batch_report_{date}.json"""
        return os.path.join(self.logs_dir, f"batch_report_{date_str}.json")
