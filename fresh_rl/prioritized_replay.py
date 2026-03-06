"""
Prioritized Experience Replay buffer using SumTree for O(log n) sampling.

Implements the proportional prioritization variant from Schaul et al. (2015)
with importance-sampling weight correction and stratified sampling.

References:
    Schaul et al., "Prioritized Experience Replay", ICLR 2016
"""

import numpy as np
import torch
from fresh_rl.sumtree import SumTree


class PrioritizedReplayBuffer:
    """
    Replay buffer with proportional prioritization.

    Transitions are sampled with probability proportional to priority^alpha.
    Importance-sampling weights correct for the non-uniform sampling bias,
    with beta annealing from beta_start to beta_end over training.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    n_actions : int
        Number of discrete actions (for action mask sizing).
    alpha : float
        Prioritization exponent. 0 = uniform, 1 = fully prioritized.
    beta_start : float
        Initial IS correction exponent (partial correction early on).
    beta_end : float
        Final IS correction exponent (full correction = 1.0).
    beta_anneal_steps : int or None
        Steps over which beta anneals. If None, defaults to capacity.
    epsilon_per : float
        Small constant added to TD errors to prevent zero-priority transitions.
    """

    def __init__(
        self,
        capacity: int = 10000,
        n_actions: int = 6,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = None,
        epsilon_per: float = 1e-5,
    ):
        self.capacity = capacity
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps if beta_anneal_steps else capacity
        self.epsilon_per = epsilon_per

        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self._sample_step = 0

    def push(self, state, action, reward, next_state, done, next_action_mask=None):
        """Store a transition with max priority (ensures it's sampled at least once)."""
        if next_action_mask is None:
            next_action_mask = np.ones(self.n_actions, dtype=bool)

        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
            np.array(next_action_mask, dtype=bool),
        )

        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int):
        """
        Sample a batch using stratified proportional sampling.

        Returns
        -------
        states, actions, rewards, next_states, dones, next_action_masks,
        indices (tree indices for priority update), is_weights (IS correction)
        """
        indices = []
        priorities = []
        batch = []

        total = self.tree.total()
        if total == 0:
            # Fallback: shouldn't happen with epsilon, but be safe
            total = 1e-8

        # Stratified sampling: divide [0, total) into batch_size segments
        segment = total / batch_size

        # Anneal beta
        beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start)
            * self._sample_step / max(self.beta_anneal_steps, 1),
        )
        self._sample_step += 1

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = np.random.uniform(low, high)
            tree_idx, priority, data = self.tree.get(cumsum)

            # Guard against sampling empty slots
            if data is None:
                cumsum = np.random.uniform(0, total)
                tree_idx, priority, data = self.tree.get(cumsum)
            if data is None:
                continue

            indices.append(tree_idx)
            priorities.append(priority)
            batch.append(data)

        if len(batch) == 0:
            return None

        # Build arrays
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        next_action_masks = np.array([b[5] for b in batch], dtype=bool)

        indices = np.array(indices)
        priorities = np.array(priorities, dtype=np.float64)

        # Importance-sampling weights
        N = self.tree.size
        probs = priorities / total
        probs = np.clip(probs, 1e-10, None)  # numerical safety
        is_weights = (N * probs) ** (-beta)
        is_weights /= is_weights.max()  # normalize so max weight = 1

        return (
            states, actions, rewards, next_states, dones, next_action_masks,
            indices, is_weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        td_errors = np.asarray(td_errors, dtype=np.float64)
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.epsilon_per) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, abs(td_err) + self.epsilon_per)

    def _get_all_transitions(self):
        """Return list of all stored transitions."""
        return [self.tree.data[i] for i in range(self.tree.size)]

    def _get_all_priorities(self):
        """Return list of leaf priorities for all stored transitions."""
        return [
            self.tree.tree[i + self.tree.capacity - 1]
            for i in range(self.tree.size)
        ]

    def save(self, path):
        """Save buffer contents and priorities to disk."""
        transitions = self._get_all_transitions()
        data = {
            "transitions": [
                (s.tolist(), int(a), float(r), s2.tolist(), float(d), m.tolist())
                for s, a, r, s2, d, m in transitions
            ],
            "priorities": self._get_all_priorities(),
            "max_priority": self.max_priority,
            "sample_step": self._sample_step,
        }
        torch.save(data, path)

    def load(self, path):
        """Load buffer contents and priorities from disk."""
        data = torch.load(path, weights_only=False)
        for (s, a, r, s2, d, m), priority in zip(
            data["transitions"], data["priorities"]
        ):
            self.push(
                np.array(s, dtype=np.float32), a, r,
                np.array(s2, dtype=np.float32), d,
                np.array(m, dtype=bool),
            )
            # Fix priority (push uses max_priority, we want the saved priority)
            idx = (self.tree.write_idx - 1) % self.tree.capacity
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)
        self.max_priority = data.get("max_priority", 1.0)
        self._sample_step = data.get("sample_step", 0)

    def __len__(self):
        return self.tree.size
