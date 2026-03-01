"""
Generate synthetic historical markdown pricing data from baseline policies.

Simulates what a retailer's historical pricing logs would look like by running
baseline policies through the environment and collecting transitions. Used to
pre-fill replay buffers for warm-starting DQN training.
"""

import numpy as np
from fresh_rl.environment import MarkdownProductEnv
from fresh_rl.baselines import (
    LinearProgressive,
    BackloadedProgressive,
    DemandResponsive,
    FixedMarkdown,
)


# Default baseline mix simulating realistic retailer behavior
DEFAULT_BASELINE_MIX = {
    "linear_progressive": 0.30,
    "backloaded_progressive": 0.25,
    "demand_responsive": 0.25,
    "fixed_20": 0.10,
    "fixed_40": 0.10,
}


def get_baseline_by_name(name: str, n_actions: int = 6, seed: int = None):
    """Map a string name to an instantiated baseline policy."""
    from fresh_rl.environment import DISCOUNT_LEVELS_BY_STEP

    # Find idx closest to 40% for the current action count
    for step_h, levels in DISCOUNT_LEVELS_BY_STEP.items():
        if len(levels) == n_actions:
            idx_40 = int(np.argmin(np.abs(levels - 0.40)))
            break
    else:
        idx_40 = 2

    mapping = {
        "linear_progressive": lambda: LinearProgressive(n_actions=n_actions),
        "backloaded_progressive": lambda: BackloadedProgressive(n_actions=n_actions),
        "demand_responsive": lambda: DemandResponsive(n_actions=n_actions),
        "fixed_20": lambda: FixedMarkdown(discount_idx=0, name="Fixed 20%", n_actions=n_actions),
        "fixed_40": lambda: FixedMarkdown(discount_idx=idx_40, name="Fixed 40%", n_actions=n_actions),
    }

    if name not in mapping:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(mapping.keys())}")
    return mapping[name]()


class HistoricalDataGenerator:
    """
    Generates synthetic historical transitions by running baseline policies
    through the markdown environment.

    Parameters
    ----------
    product : str
        Product name for MarkdownProductEnv.
    step_hours : int
        Step hours for the environment.
    baseline_mix : dict or None
        Mapping of baseline name -> weight. Defaults to DEFAULT_BASELINE_MIX.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        product: str = "salad_mix",
        step_hours: int = 4,
        baseline_mix: dict = None,
        seed: int = None,
    ):
        self.product = product
        self.step_hours = step_hours
        self.baseline_mix = baseline_mix or DEFAULT_BASELINE_MIX
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, n_episodes: int):
        """
        Run baseline policies through the environment and collect transitions.

        Returns list of (state, action, reward, next_state, done, next_action_mask) tuples.
        """
        env = MarkdownProductEnv(
            product_name=self.product,
            step_hours=self.step_hours,
            seed=self.seed,
        )
        n_actions = env.action_space.n

        # Build baseline instances and weights
        names = list(self.baseline_mix.keys())
        weights = np.array([self.baseline_mix[n] for n in names])
        weights /= weights.sum()
        baselines = [get_baseline_by_name(n, n_actions=n_actions, seed=self.seed) for n in names]

        transitions = []

        for ep in range(n_episodes):
            # Pick a baseline for this episode
            idx = self.rng.choice(len(baselines), p=weights)
            policy = baselines[idx]

            obs, _ = env.reset()
            done = False

            while not done:
                action = policy.select_action(obs, env=env)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                next_action_mask = (
                    env.action_masks() if not done
                    else np.ones(n_actions, dtype=bool)
                )

                transitions.append((
                    obs.copy(), action, reward,
                    next_obs.copy(), float(done),
                    next_action_mask.copy(),
                ))
                obs = next_obs

        return transitions

    def fill_buffer(self, buffer, n_episodes: int, initial_priority: float = 5.0, agent=None):
        """
        Generate historical data and push into a replay buffer.

        If an agent is provided, transitions are pushed via agent.store_transition()
        so that reward shaping is applied consistently. Otherwise falls back to
        buffer.push() with raw rewards.

        For PrioritizedReplayBuffer, temporarily sets max_priority to
        initial_priority so historical data gets high initial sampling weight.

        Parameters
        ----------
        buffer : ReplayBuffer or PrioritizedReplayBuffer
            The buffer to fill.
        n_episodes : int
            Number of historical episodes to generate.
        initial_priority : float
            Priority for historical transitions (PER only).
        agent : DQNAgent or None
            If provided, use agent.store_transition() to apply reward shaping.

        Returns
        -------
        int
            Number of transitions added.
        """
        transitions = self.generate(n_episodes)

        # If PER buffer, temporarily elevate max_priority
        has_per = hasattr(buffer, 'max_priority')
        if has_per:
            old_max = buffer.max_priority
            buffer.max_priority = initial_priority

        for state, action, reward, next_state, done, next_action_mask in transitions:
            if agent is not None:
                agent.store_transition(state, action, reward, next_state, done, next_action_mask)
            else:
                buffer.push(state, action, reward, next_state, done, next_action_mask)

        if has_per:
            buffer.max_priority = old_max

        return len(transitions)
