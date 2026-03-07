"""
Pooled category environment: one Gym env wrapping all SKUs in a category.

The agent sees a 14-dim state = 10-dim base env obs + 4-dim product features,
allowing a single model to learn pricing across all ~22 SKUs in a category.

Product features are observable attributes (price, cost fraction, inventory,
pack size) — no simulator leakage (elasticity, base_demand excluded).

Usage:
    env = PooledCategoryEnv("meats", step_hours=2, seed=42)
    obs, info = env.reset(options={"product": "salmon_fillet"})
    # obs is 14-dim: [base_10_obs..., price_norm, cost_frac_norm, inv_norm, pack_norm]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from fresh_rl.environment import MarkdownProductEnv
from fresh_rl.product_catalog import (
    get_product_names,
    get_product_features,
    generate_catalog,
    CATEGORIES,
)


class AugmentedProductEnv(gym.Wrapper):
    """Wraps MarkdownProductEnv to append 4 product features (10→14 dim).

    Used for pooled→per-SKU transfer learning: the per-SKU agent sees the same
    14-dim state as the pooled model, with product features held constant.
    """

    def __init__(self, env, product_name, inventory_mult=1.0):
        super().__init__(env)
        self._features = get_product_features(product_name, inventory_mult)
        base_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=np.zeros(base_dim + 4, dtype=np.float32),
            high=np.ones(base_dim + 4, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.concatenate([obs, self._features]), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return np.concatenate([obs, self._features]), r, term, trunc, info

    def action_masks(self):
        return self.env.action_masks()

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped env (for baselines)."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


class PooledCategoryEnv(gym.Env):
    """Gymnasium env that pools all SKUs in a category behind a single interface.

    Each reset() can switch the active product. Observations are augmented with
    4 normalized product features appended to the base 10-dim state, yielding
    a 14-dim observation space.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        category: str,
        step_hours: int = 4,
        seed: int = 42,
        demand_mult: float = 1.0,
        inventory_mult: float = 1.0,
    ):
        super().__init__()
        self.category = category
        self.step_hours = step_hours
        self.seed = seed
        self.demand_mult = demand_mult
        self.inventory_mult = inventory_mult

        self.products = get_product_names(category)
        if not self.products:
            raise ValueError(f"No products found for category '{category}'")

        catalog = generate_catalog()

        # Create one MarkdownProductEnv per SKU
        self._envs = {}
        for name in self.products:
            profile = catalog[name]
            env_kwargs = dict(product_name=name, step_hours=step_hours, seed=seed)
            if demand_mult != 1.0:
                env_kwargs["base_markdown_demand"] = round(
                    profile.get("base_markdown_demand", 5.0) * demand_mult, 1
                )
            if inventory_mult != 1.0:
                env_kwargs["initial_inventory"] = int(
                    profile.get("initial_inventory", 20) * inventory_mult
                )
            self._envs[name] = MarkdownProductEnv(**env_kwargs)

        # Pre-compute product features (static per SKU)
        self._features = {
            name: get_product_features(name, inventory_mult=inventory_mult)
            for name in self.products
        }

        # Active product (set on reset)
        self._active_name = self.products[0]
        self._active_env = self._envs[self._active_name]

        # 14-dim observation: 10 base + 4 product features
        sample_env = self._active_env
        base_dim = sample_env.observation_space.shape[0]
        self._base_dim = base_dim
        self._feat_dim = 4
        total_dim = base_dim + self._feat_dim

        self.observation_space = spaces.Box(
            low=np.zeros(total_dim, dtype=np.float32),
            high=np.ones(total_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = sample_env.action_space

    @property
    def active_product(self) -> str:
        return self._active_name

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Append product features to base observation."""
        return np.concatenate([obs, self._features[self._active_name]])

    def reset(self, seed=None, options=None):
        # Switch active product if specified
        if options and "product" in options:
            name = options["product"]
            if name not in self._envs:
                raise ValueError(
                    f"Unknown product '{name}' for category '{self.category}'. "
                    f"Available: {self.products}"
                )
            self._active_name = name
            self._active_env = self._envs[name]

        obs, info = self._active_env.reset(seed=seed)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._active_env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def action_masks(self):
        return self._active_env.action_masks()

    def __getattr__(self, name):
        """Delegate attribute access to the active inner env.

        This lets baselines access env.step_count, env.inventory_remaining, etc.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._active_env, name)


def pooled_prefill(env, agent, episodes_per_product, products, seed=42,
                   collect_per_product=False):
    """Pre-fill replay buffer with baseline policy transitions through a PooledCategoryEnv.

    Runs baseline policies (from DEFAULT_BASELINE_MIX) across all products,
    producing 14-dim transitions (product features appended automatically by env).

    Parameters
    ----------
    env : PooledCategoryEnv
        The pooled environment.
    agent : DQNAgent
        Agent whose replay buffer will be filled (uses store_transition for shaping).
    episodes_per_product : int
        Number of prefill episodes per product.
    products : list of str
        Product names to prefill.
    seed : int
        Random seed.
    collect_per_product : bool
        When True, also collect raw transitions per product (for TL prefill later).

    Returns
    -------
    int or (int, dict)
        Total transitions added. If collect_per_product=True, also returns
        dict mapping product name -> list of (obs, action, reward, next_obs, done, mask).
    """
    from fresh_rl.baselines import (
        LinearProgressive,
        BackloadedProgressive,
        DemandResponsive,
        FixedMarkdown,
    )
    from fresh_rl.historical_data import DEFAULT_BASELINE_MIX

    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n

    # Build baseline policies and weights
    from fresh_rl.historical_data import get_baseline_by_name
    names = list(DEFAULT_BASELINE_MIX.keys())
    weights = np.array([DEFAULT_BASELINE_MIX[n] for n in names])
    weights /= weights.sum()
    baselines = [get_baseline_by_name(n, n_actions=n_actions, seed=seed) for n in names]

    # Temporarily elevate PER priority for historical data
    has_per = hasattr(agent.replay_buffer, "max_priority")
    if has_per:
        old_max = agent.replay_buffer.max_priority
        agent.replay_buffer.max_priority = 5.0

    total_transitions = 0
    if collect_per_product:
        from collections import defaultdict
        transitions_by_product = defaultdict(list)

    for product in products:
        for ep in range(episodes_per_product):
            # Pick a baseline for this episode
            idx = rng.choice(len(baselines), p=weights)
            policy = baselines[idx]

            # Deterministic seed: seed + ep per product (matches training seed pattern)
            obs, _ = env.reset(
                seed=seed + ep,
                options={"product": product},
            )
            done = False

            while not done:
                # Baselines need access to inner env attributes
                action = policy.select_action(obs[:env._base_dim], env=env)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                next_mask = (
                    env.action_masks() if not done
                    else np.ones(n_actions, dtype=bool)
                )

                agent.store_transition(obs, action, reward, next_obs, done, next_mask)
                total_transitions += 1

                if collect_per_product:
                    transitions_by_product[product].append(
                        (obs.copy(), action, reward, next_obs.copy(), done, next_mask.copy())
                    )

                obs = next_obs

    if has_per:
        agent.replay_buffer.max_priority = old_max

    if collect_per_product:
        return total_transitions, dict(transitions_by_product)
    return total_transitions
