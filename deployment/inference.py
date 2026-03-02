"""
Pricing agent wrapper for production inference.

Loads a DQN checkpoint with 3-tier fallback:
  1. Per-SKU checkpoint (best, from batch training)
  2. Pooled category model (zero-shot generalization)
  3. BackloadedProgressive baseline (rule-based fallback)

All inference errors are caught and fall back to the baseline.
"""

import logging
import os
from typing import Optional

import numpy as np

from deployment.config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    DISCOUNT_LEVELS,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    HIDDEN_DIM,
    HOLD_ACTION_PROB,
    LR,
    N_ACTIONS,
    N_STEP,
    PER_ALPHA,
    PER_BETA_END,
    PER_BETA_START,
    PER_EPSILON,
    STATE_DIM,
    TRANSFER_EPSILON,
    USE_PER,
    ProductionConfig,
)

logger = logging.getLogger(__name__)


class PricingAgent:
    """Production pricing agent with model loading and baseline fallback.

    Parameters
    ----------
    sku_name : str
        Product SKU name.
    category : str
        Product category (for pooled model fallback).
    base_price : float
        Regular shelf price.
    cost_per_unit : float
        Cost per unit (for reward shaping scale).
    initial_inventory : int
        Session starting inventory (for reward shaping scale).
    config : ProductionConfig
        Paths configuration.
    """

    def __init__(
        self,
        sku_name: str,
        category: str,
        base_price: float,
        cost_per_unit: float,
        initial_inventory: int,
        config: ProductionConfig,
    ):
        self.sku_name = sku_name
        self.category = category
        self.base_price = base_price
        self.cost_per_unit = cost_per_unit
        self.initial_inventory = initial_inventory
        self.config = config

        self._agent = None
        self._baseline = None
        self._model_source = None

    def _create_agent(self, epsilon: float = EPSILON_START):
        """Create a fresh DQNAgent with production defaults."""
        from fresh_rl.dqn_agent import DQNAgent

        agent = DQNAgent(
            state_dim=STATE_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=HIDDEN_DIM,
            lr=LR,
            gamma=GAMMA,
            epsilon_start=epsilon,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            reward_shaping=True,
            waste_cost_scale=self.cost_per_unit * 3.0 * self.initial_inventory,
            use_per=USE_PER,
            per_alpha=PER_ALPHA,
            per_beta_start=PER_BETA_START,
            per_beta_end=PER_BETA_END,
            per_epsilon=PER_EPSILON,
            n_step=N_STEP,
            hold_action_prob=HOLD_ACTION_PROB,
        )
        return agent

    def load_model(self) -> str:
        """Load model with 3-tier fallback. Returns source description.

        Fallback order:
          1. Per-SKU checkpoint: models/{sku}/agent.pt
          2. Pooled category model: models/_pooled/{cat}/pooled_{cat}_plain_2h.pt
          3. BackloadedProgressive baseline (no model)
        """
        # Tier 1: per-SKU checkpoint
        sku_path = self.config.checkpoint_path(self.sku_name)
        if os.path.exists(sku_path):
            try:
                agent = self._create_agent()
                agent.load(sku_path)
                self._agent = agent
                self._model_source = f"per-sku:{sku_path}"
                logger.info("Loaded per-SKU model for %s", self.sku_name)
                return self._model_source
            except Exception as e:
                logger.warning("Failed to load per-SKU model for %s: %s", self.sku_name, e)

        # Tier 2: pooled category model
        pooled_path = self.config.pooled_model_path(self.category, "plain")
        if os.path.exists(pooled_path):
            try:
                agent = self._create_agent(epsilon=TRANSFER_EPSILON)
                agent.load_pretrained(pooled_path)
                self._agent = agent
                self._model_source = f"pooled:{pooled_path}"
                logger.info("Loaded pooled model for %s (category=%s)", self.sku_name, self.category)
                return self._model_source
            except Exception as e:
                logger.warning("Failed to load pooled model for %s: %s", self.sku_name, e)

        # Tier 3: baseline fallback
        from fresh_rl.baselines import BackloadedProgressive

        self._baseline = BackloadedProgressive(n_actions=N_ACTIONS)
        self._agent = None
        self._model_source = "baseline:BackloadedProgressive"
        logger.info("Using baseline fallback for %s", self.sku_name)
        return self._model_source

    def get_agent(self):
        """Return the underlying DQNAgent (or None if using baseline)."""
        return self._agent

    def get_discount(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
    ) -> dict:
        """Get discount recommendation.

        Returns
        -------
        dict with keys:
            discount_idx : int (0-10)
            discount_pct : float (0.20-0.70)
            price : float (discounted price)
            source : str (model source description)
            q_values : list[float] or None
        """
        q_values = None
        source = self._model_source or "unloaded"

        if self._agent is not None:
            try:
                # Get Q-values for logging
                import torch
                with torch.no_grad():
                    state_t = torch.FloatTensor(np.array(state).reshape(1, -1))
                    q_vals = self._agent.q_network(state_t).numpy()[0].copy()
                q_values = q_vals.tolist()

                discount_idx = self._agent.select_action(state, action_mask=action_mask)
            except Exception as e:
                logger.error("Inference error for %s, falling back to baseline: %s", self.sku_name, e)
                discount_idx = self._get_baseline_action(state, action_mask)
                source = f"fallback:baseline (error: {e})"
                q_values = None
        else:
            discount_idx = self._get_baseline_action(state, action_mask)

        discount_pct = float(DISCOUNT_LEVELS[discount_idx])
        price = self.base_price * (1.0 - discount_pct)

        return {
            "discount_idx": discount_idx,
            "discount_pct": discount_pct,
            "price": round(price, 2),
            "source": source,
            "q_values": q_values,
        }

    def _get_baseline_action(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        """Get action from BackloadedProgressive baseline."""
        if self._baseline is None:
            from fresh_rl.baselines import BackloadedProgressive
            self._baseline = BackloadedProgressive(n_actions=N_ACTIONS)

        # Baseline uses 10-dim obs (first 10 features)
        action = self._baseline.select_action(state[:10])
        # Enforce mask
        valid = np.where(action_mask)[0]
        if action not in valid:
            action = int(valid[0]) if len(valid) > 0 else 0
        return action
