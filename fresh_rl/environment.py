"""
Gymnasium environment for markdown channel intraday progressive discounting.

Models a perishable product on an ecommerce markdown landing page:
- Items enter 1-2 days before expiry with fixed inventory (no replenishment)
- RL agent controls progressive deep discounts at configurable intervals (2h or 4h)
- Discounts can only go deeper (never revert to a shallower level)

State: [hours_remaining, inventory_remaining, current_discount_idx,
        tod_sin, tod_cos, dow_sin, dow_cos,
        recent_velocity, sell_through_rate,
        projected_clearance]  (10-dim, all normalized to [0,1])
Action: discrete discount levels (6 levels for 4h, 11 levels for 2h)
Reward: revenue - waste_penalty - holding_cost + clearance_bonus
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Discount levels by step_hours configuration
DISCOUNT_LEVELS_BY_STEP = {
    4: np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70]),
    2: np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]),
}

# Intraday demand multipliers by step_hours configuration
INTRADAY_PATTERN_BY_STEP = {
    4: {0: 0.3, 1: 0.5, 2: 0.9, 3: 1.2, 4: 1.5, 5: 0.8},
    2: {0: 0.2, 1: 0.4, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0,
        6: 1.1, 7: 1.3, 8: 1.5, 9: 1.5, 10: 1.0, 11: 0.5},
}


class MarkdownChannelEnv(gym.Env):
    """
    Gymnasium environment for markdown channel progressive discounting.

    The agent decides what discount to apply each interval (2h or 4h),
    constrained so discounts can only go deeper over time. The episode
    ends when the markdown window expires or all inventory is sold.
    """

    metadata = {"render_modes": ["human"]}

    # Day-of-week demand multipliers (Mon=0 ... Sun=6)
    DOW_PATTERN = [0.8, 0.85, 0.9, 1.0, 1.1, 1.3, 1.2]

    def __init__(
        self,
        markdown_window_hours: int = 24,
        initial_inventory: int = 20,
        base_price: float = 5.0,
        base_markdown_demand: float = 5.0,
        price_elasticity: float = 3.0,
        cost_per_unit: float = 2.0,
        waste_penalty_multiplier: float = 3.0,
        holding_cost_per_step: float = 0.02,
        clearance_bonus: float = 1.0,
        inventory_noise_std: float = 2.0,
        step_hours: int = 4,
        seed: int = None,
    ):
        super().__init__()

        self.step_hours = step_hours
        self.n_time_blocks = 24 // step_hours
        self.markdown_window_hours = markdown_window_hours
        self.episode_length = markdown_window_hours // step_hours
        self.initial_inventory = initial_inventory
        self.base_price = base_price
        self.base_markdown_demand = base_markdown_demand
        self.price_elasticity = price_elasticity
        self.cost_per_unit = cost_per_unit
        self.waste_penalty_multiplier = waste_penalty_multiplier
        self.holding_cost_per_step = holding_cost_per_step
        self.clearance_bonus = clearance_bonus
        self.inventory_noise_std = inventory_noise_std

        # Discount levels and intraday pattern based on step_hours
        self.DISCOUNT_LEVELS = DISCOUNT_LEVELS_BY_STEP[step_hours]
        self.INTRADAY_PATTERN = INTRADAY_PATTERN_BY_STEP[step_hours]

        # Action space: discrete discount levels (6 for 4h, 11 for 2h)
        self.action_space = spaces.Discrete(len(self.DISCOUNT_LEVELS))

        # Observation space: 10-dim, all normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(10, dtype=np.float32),
            high=np.ones(10, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self.inventory_remaining = 0
        self.current_discount_idx = 0
        self.step_count = 0
        self.day_of_week = 0
        self.time_of_day = 0  # time block index (0 to n_time_blocks-1)
        self.recent_sales = []
        self.actual_initial_inventory = 0

        # Episode metrics
        self.total_revenue = 0.0
        self.total_waste = 0
        self.total_sold = 0
        self.total_holding_cost = 0.0

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def _projected_clearance(self):
        """Ratio of projected remaining sales to remaining inventory based on observed velocity."""
        if self.inventory_remaining <= 0 or self.step_count >= self.episode_length:
            return 1.0
        velocity = np.mean(self.recent_sales[-3:]) if self.recent_sales else 0.0
        remaining_steps = self.episode_length - self.step_count
        projected_sales = velocity * remaining_steps
        return min(projected_sales / max(self.inventory_remaining, 1), 1.0)

    def _get_obs(self):
        velocity = np.mean(self.recent_sales[-3:]) if self.recent_sales else 0.0
        max_velocity = self.actual_initial_inventory * 0.5

        # Cyclical encoding for time_of_day
        tod_angle = 2.0 * np.pi * self.time_of_day / self.n_time_blocks
        tod_sin = (np.sin(tod_angle) + 1.0) / 2.0
        tod_cos = (np.cos(tod_angle) + 1.0) / 2.0

        # Cyclical encoding for day_of_week
        dow_angle = 2.0 * np.pi * self.day_of_week / 7.0
        dow_sin = (np.sin(dow_angle) + 1.0) / 2.0
        dow_cos = (np.cos(dow_angle) + 1.0) / 2.0

        # Sell-through rate: actual vs ideal pace to clear inventory
        if self.step_count > 0:
            ideal_rate = self.actual_initial_inventory / self.episode_length
            actual_rate = self.total_sold / self.step_count
            sell_through_rate = min(actual_rate / max(ideal_rate, 1e-3), 1.0)
        else:
            sell_through_rate = 0.0

        obs = np.array([
            self._hours_remaining() / self.markdown_window_hours,              # [0]
            self.inventory_remaining / max(self.actual_initial_inventory, 1),   # [1]
            self.current_discount_idx / max(len(self.DISCOUNT_LEVELS) - 1, 1), # [2]
            tod_sin,                                                           # [3]
            tod_cos,                                                           # [4]
            dow_sin,                                                           # [5]
            dow_cos,                                                           # [6]
            min(velocity / max(max_velocity, 1), 1.0),                         # [7]
            sell_through_rate,                                                 # [8]
            self._projected_clearance(),                                       # [9]
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _hours_remaining(self):
        steps_remaining = self.episode_length - self.step_count
        return steps_remaining * self.step_hours

    def action_masks(self):
        """Return boolean mask: True for valid actions (>= current discount idx)."""
        mask = np.zeros(len(self.DISCOUNT_LEVELS), dtype=bool)
        mask[self.current_discount_idx:] = True
        return mask

    def _demand_model(self, price):
        """
        Compute stochastic demand for the current step.

        demand = Poisson(base_markdown_demand * price_effect * intraday_effect * dow_effect)
        """
        # Price effect: exponential response to discount depth
        price_effect = np.exp(-self.price_elasticity * (price / self.base_price - 1.0))

        # Intraday effect: traffic varies by time of day
        intraday_effect = self.INTRADAY_PATTERN[self.time_of_day]

        # Day-of-week effect
        dow_effect = self.DOW_PATTERN[self.day_of_week]

        expected = self.base_markdown_demand * price_effect * intraday_effect * dow_effect
        expected *= self.step_hours / 4.0  # scale demand for step duration
        actual = self.np_random.poisson(max(0.1, expected))
        return int(actual)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize starting day/time
        self.day_of_week = int(self.np_random.integers(0, 7))
        self.time_of_day = int(self.np_random.integers(0, self.n_time_blocks))

        # Noisy initial inventory
        noise = self.np_random.normal(0, self.inventory_noise_std)
        self.actual_initial_inventory = max(1, int(self.initial_inventory + noise))
        self.inventory_remaining = self.actual_initial_inventory

        self.current_discount_idx = 0
        self.step_count = 0
        self.recent_sales = []

        # Reset metrics
        self.total_revenue = 0.0
        self.total_waste = 0
        self.total_sold = 0
        self.total_holding_cost = 0.0

        return self._get_obs(), {}

    def step(self, action):
        # Enforce progressive constraint: can only go deeper
        action = max(action, self.current_discount_idx)
        self.current_discount_idx = action

        discount = self.DISCOUNT_LEVELS[action]
        price = self.base_price * (1.0 - discount)

        reward = 0.0
        units_sold = 0

        # Generate demand and sell
        if self.inventory_remaining > 0:
            demand = self._demand_model(price)
            units_sold = min(demand, self.inventory_remaining)
            self.inventory_remaining -= units_sold

            revenue = price * units_sold
            reward += revenue
            self.total_revenue += revenue
            self.total_sold += units_sold

        # Holding cost for remaining inventory (scaled by step duration)
        holding = self.holding_cost_per_step * price * self.inventory_remaining * (self.step_hours / 4.0)
        reward -= holding
        self.total_holding_cost += holding

        # Track sales
        self.recent_sales.append(units_sold)

        # Advance time
        self.step_count += 1
        self.time_of_day = (self.time_of_day + 1) % self.n_time_blocks

        # Check termination
        inventory_cleared = self.inventory_remaining == 0
        window_expired = self.step_count >= self.episode_length
        terminated = inventory_cleared or window_expired

        # Terminal rewards/penalties
        if terminated:
            if self.inventory_remaining > 0:
                # Waste penalty for unsold inventory
                waste_cost = (self.cost_per_unit * self.waste_penalty_multiplier
                              * self.inventory_remaining)
                reward -= waste_cost
                self.total_waste = self.inventory_remaining

            if inventory_cleared and not window_expired:
                # Clearance bonus: sold everything before deadline
                bonus = self.clearance_bonus * self.actual_initial_inventory
                reward += bonus

        truncated = False

        info = {
            "units_sold": units_sold,
            "discount": discount,
            "price": price,
            "revenue": price * units_sold,
            "total_inventory": self.inventory_remaining,
            "step": self.step_count,
            "hours_remaining": self._hours_remaining(),
        }

        if terminated:
            clearance_rate = self.total_sold / max(self.actual_initial_inventory, 1)
            info["episode_stats"] = {
                "total_revenue": self.total_revenue,
                "total_waste": self.total_waste,
                "total_sold": self.total_sold,
                "initial_inventory": self.actual_initial_inventory,
                "waste_rate": self.total_waste / max(self.actual_initial_inventory, 1),
                "clearance_rate": clearance_rate,
                "total_holding_cost": self.total_holding_cost,
                "cleared": int(inventory_cleared and not window_expired),
            }

        return self._get_obs(), reward, terminated, truncated, info


class MarkdownProductEnv(MarkdownChannelEnv):
    """
    Extended environment with configurable product profiles for the
    markdown channel. Each product represents a different perishability
    category with realistic markdown-specific parameters.

    Profiles are resolved from the product catalog (110 SKUs across
    7 categories + legacy). See fresh_rl.product_catalog for details.
    """

    def __init__(self, product_name: str = "salad_mix", **kwargs):
        from fresh_rl.product_catalog import get_profile
        profile = get_profile(product_name)
        profile.update(kwargs)  # allow overrides
        super().__init__(**profile)
        self.product_name = product_name
