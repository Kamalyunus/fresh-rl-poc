"""
Baseline (rule-based) policies for markdown channel progressive discounting.

These serve as benchmarks to compare against the RL agent:
1. ImmediateDeepDiscount: Always picks the deepest discount (70%)
2. LinearProgressive: Evenly steps 20%→70% over the markdown window
3. BackloadedProgressive: 20% first half, ramp 30%→70% in second half
4. DemandResponsive: Adjusts discount based on velocity and urgency
5. FixedMarkdown: Stays at a fixed discount level (instantiated at 20% and 40%)
6. RandomPolicy: Random from valid actions
"""

import numpy as np


class BasePolicy:
    """Base class for all policies."""

    def __init__(self, name: str):
        self.name = name

    def select_action(self, obs: np.ndarray, env=None) -> int:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class ImmediateDeepDiscount(BasePolicy):
    """Always picks the deepest discount (last action)."""

    def __init__(self, n_actions: int = 6):
        super().__init__("Immediate Deep 70%")
        self.n_actions = n_actions

    def select_action(self, obs, env=None):
        if env is not None:
            return env.action_space.n - 1
        return self.n_actions - 1


class LinearProgressive(BasePolicy):
    """
    Evenly steps from 20% to 70% over the markdown window.

    Maps elapsed fraction of episode to a discount index.
    Enforces progressive constraint via max with current_discount_idx.
    """

    def __init__(self, n_actions: int = 6):
        super().__init__("Linear Progressive")
        self.n_actions = n_actions

    def select_action(self, obs, env=None):
        n = env.action_space.n if env is not None else self.n_actions

        if env is None:
            hours_frac = obs[0]
            elapsed_frac = 1.0 - hours_frac
        else:
            elapsed_frac = env.step_count / max(env.episode_length, 1)

        action = int(elapsed_frac * n)
        action = min(action, n - 1)

        if env is not None:
            action = max(action, env.current_discount_idx)
        return action


class BackloadedProgressive(BasePolicy):
    """
    20% for the first half of the window, then ramp to deepest in the second half.

    This mimics a strategy of starting conservatively and getting aggressive
    only when time is running out.
    """

    def __init__(self, n_actions: int = 6):
        super().__init__("Backloaded Progressive")
        self.n_actions = n_actions

    def select_action(self, obs, env=None):
        n = env.action_space.n if env is not None else self.n_actions

        if env is None:
            hours_frac = obs[0]
            elapsed_frac = 1.0 - hours_frac
        else:
            elapsed_frac = env.step_count / max(env.episode_length, 1)

        if elapsed_frac < 0.5:
            action = 0  # 20%
        else:
            # Map second half [0.5, 1.0] → actions [1, n-1]
            second_half_frac = (elapsed_frac - 0.5) / 0.5
            action = 1 + int(second_half_frac * (n - 1))
            action = min(action, n - 1)

        if env is not None:
            action = max(action, env.current_discount_idx)
        return action


class DemandResponsive(BasePolicy):
    """
    Adjusts discount based on recent sales velocity and urgency.

    - Increases discount by 1 level when velocity drops below threshold
    - Jumps 2 levels when time < 30% remaining and inventory > 30%
    - Otherwise holds current discount
    """

    def __init__(self, n_actions: int = 6):
        super().__init__("Demand Responsive")
        self.n_actions = n_actions

    def select_action(self, obs, env=None):
        n = env.action_space.n if env is not None else self.n_actions

        if env is None:
            hours_frac = obs[0]
            inv_frac = obs[1]
            velocity_frac = obs[7]
            current_idx = int(obs[2] * (n - 1))
        else:
            hours_frac = (env.episode_length - env.step_count) / max(env.episode_length, 1)
            inv_frac = env.inventory_remaining / max(env.actual_initial_inventory, 1)
            velocity = np.mean(env.recent_sales[-3:]) if env.recent_sales else 0.0
            velocity_frac = velocity / max(env.actual_initial_inventory * 0.5, 1)
            current_idx = env.current_discount_idx

        action = current_idx

        # Emergency: little time left with lots of inventory
        if hours_frac < 0.3 and inv_frac > 0.3:
            action = min(current_idx + 2, n - 1)
        # Slow sales: bump discount
        elif velocity_frac < 0.3:
            action = min(current_idx + 1, n - 1)

        if env is not None:
            action = max(action, env.current_discount_idx)
        return action


class FixedMarkdown(BasePolicy):
    """Stays at a fixed discount level throughout the episode."""

    def __init__(self, discount_idx: int = 0, name: str = None, n_actions: int = 6):
        self.discount_idx = discount_idx
        # Generate label dynamically based on n_actions
        if name is not None:
            label = name
        else:
            from fresh_rl.environment import DISCOUNT_LEVELS_BY_STEP
            # Try to find matching config; default to percentage from 6-level
            levels_6 = DISCOUNT_LEVELS_BY_STEP[4]
            if discount_idx < len(levels_6):
                discount_pct = int(levels_6[discount_idx] * 100)
            else:
                discount_pct = 20 + discount_idx * 5
            label = f"Fixed {discount_pct}%"
        super().__init__(label)

    def select_action(self, obs, env=None):
        action = self.discount_idx
        if env is not None:
            action = max(action, env.current_discount_idx)
        return action


class RandomPolicy(BasePolicy):
    """Random discount from valid actions (>= current discount index)."""

    def __init__(self, n_actions: int = 6, seed: int = None):
        super().__init__("Random")
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs, env=None):
        if env is not None:
            valid = np.where(env.action_masks())[0]
            return int(self.rng.choice(valid))
        else:
            return int(self.rng.integers(0, self.n_actions))


def get_all_baselines(n_actions=6, seed=None):
    """Return a list of all baseline policies (7 total)."""
    # Find the index closest to 40% discount
    from fresh_rl.environment import DISCOUNT_LEVELS_BY_STEP
    # Pick levels from the matching config
    for step_h, levels in DISCOUNT_LEVELS_BY_STEP.items():
        if len(levels) == n_actions:
            idx_40 = int(np.argmin(np.abs(levels - 0.40)))
            break
    else:
        idx_40 = 2  # fallback

    return [
        ImmediateDeepDiscount(n_actions=n_actions),
        LinearProgressive(n_actions=n_actions),
        BackloadedProgressive(n_actions=n_actions),
        DemandResponsive(n_actions=n_actions),
        FixedMarkdown(discount_idx=0, name="Fixed 20%", n_actions=n_actions),
        FixedMarkdown(discount_idx=idx_40, name="Fixed 40%", n_actions=n_actions),
        RandomPolicy(n_actions=n_actions, seed=seed),
    ]
