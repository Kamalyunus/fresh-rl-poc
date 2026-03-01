"""
Deep Q-Network (DQN) agent with action masking for progressive discounting.

This is a lightweight, educational implementation suitable for the POC.
No PyTorch/TensorFlow dependency — just NumPy neural networks.

Features:
- 2-layer MLP Q-network with ReLU activations
- Experience replay buffer with action mask support
- Epsilon-greedy exploration over valid actions only
- Target network with periodic sync
- Optional potential-based reward shaping (urgency-aware)
"""

import numpy as np
from collections import deque
import pickle


class ReplayBuffer:
    """Fixed-size circular replay buffer with action mask support."""

    def __init__(self, capacity: int = 10000, n_actions: int = 6):
        self.buffer = deque(maxlen=capacity)
        self.n_actions = n_actions

    def push(self, state, action, reward, next_state, done, next_action_mask=None):
        if next_action_mask is None:
            next_action_mask = np.ones(self.n_actions, dtype=bool)
        self.buffer.append((state, action, reward, next_state, done, next_action_mask))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        next_action_masks = np.array([b[5] for b in batch], dtype=bool)
        return states, actions, rewards, next_states, dones, next_action_masks

    def __len__(self):
        return len(self.buffer)


class NumpyMLP:
    """
    Simple 2-layer MLP implemented in pure NumPy.

    Architecture: input -> hidden1 (ReLU) -> hidden2 (ReLU) -> output
    Uses He initialization and Adam optimizer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 1e-3):
        self.lr = lr

        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)

        # Adam optimizer state
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x):
        """Forward pass returning Q-values and cached activations for backprop."""
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        q_values = a2 @ self.W3 + self.b3
        cache = (x, z1, a1, z2, a2)
        return q_values, cache

    def predict(self, x):
        """Forward pass without caching (for inference)."""
        q, _ = self.forward(x)
        return q

    def backward(self, cache, q_values, targets, actions, is_weights=None):
        """
        Backpropagation for DQN loss.
        Only updates Q-values for taken actions (masked MSE loss).

        When is_weights is provided (PER), each sample's gradient is scaled
        by its importance-sampling weight to correct for non-uniform sampling.
        """
        x, z1, a1, z2, a2 = cache
        batch_size = x.shape[0]

        # Compute gradient of loss w.r.t. output (only for taken actions)
        dq = np.zeros_like(q_values)
        for i in range(batch_size):
            weight = is_weights[i] if is_weights is not None else 1.0
            dq[i, actions[i]] = 2.0 * weight * (q_values[i, actions[i]] - targets[i]) / batch_size

        # Layer 3
        dW3 = a2.T @ dq
        db3 = dq.sum(axis=0)
        da2 = dq @ self.W3.T

        # ReLU 2
        da2 = da2 * (z2 > 0)

        # Layer 2
        dW2 = a1.T @ da2
        db2 = da2.sum(axis=0)
        da1 = da2 @ self.W2.T

        # ReLU 1
        da1 = da1 * (z1 > 0)

        # Layer 1
        dW1 = x.T @ da1
        db1 = da1.sum(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]

        # Gradient clipping
        for i in range(len(grads)):
            np.clip(grads[i], -1.0, 1.0, out=grads[i])

        # Adam update
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * grad ** 2
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            v_hat = self.v[i] / (1 - 0.999 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Sync references (since we modified in place)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.params

    def copy_from(self, other):
        """Copy weights from another network (for target network sync)."""
        for i in range(len(self.params)):
            np.copyto(self.params[i], other.params[i])

    def soft_update_from(self, other, tau: float):
        """Polyak averaging: target = tau * online + (1 - tau) * target."""
        for i in range(len(self.params)):
            self.params[i] *= (1.0 - tau)
            self.params[i] += tau * other.params[i]


class DQNAgent:
    """
    Deep Q-Network agent with action masking for progressive discounting.

    Features:
    - Epsilon-greedy exploration over valid actions only
    - Experience replay with action masks
    - Target network
    - Optional potential-based reward shaping (urgency-aware)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 6,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        reward_shaping: bool = False,
        waste_cost_scale: float = None,
        seed: int = None,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = None,
        per_epsilon: float = 1e-5,
        double_dqn: bool = True,
        soft_target_tau: float = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.reward_shaping = reward_shaping
        self.waste_cost_scale = waste_cost_scale
        self.use_per = use_per
        self.double_dqn = double_dqn
        self.soft_target_tau = soft_target_tau

        if seed is not None:
            np.random.seed(seed)

        # Networks
        self.q_network = NumpyMLP(state_dim, hidden_dim, n_actions, lr=lr)
        self.target_network = NumpyMLP(state_dim, hidden_dim, n_actions, lr=lr)
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        if use_per:
            from fresh_rl.prioritized_replay import PrioritizedReplayBuffer
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                n_actions=n_actions,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=per_beta_end,
                beta_anneal_steps=per_beta_anneal_steps,
                epsilon_per=per_epsilon,
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, n_actions=n_actions)

        # Training stats
        self.train_step = 0
        self.losses = []
        self.episode_rewards = []

    def select_action(self, state, action_mask=None, env=None):
        """Epsilon-greedy action selection over valid actions only."""
        if action_mask is None:
            action_mask = np.ones(self.n_actions, dtype=bool)

        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            valid_actions = np.arange(self.n_actions)

        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))
        else:
            state = np.array(state).reshape(1, -1)
            q_values = self.q_network.predict(state)[0].copy()
            # Mask invalid actions with -inf
            q_values[~action_mask] = -np.inf
            return int(np.argmax(q_values))

    def _shape_reward(self, state, action, reward, next_state):
        """
        Potential-based reward shaping using cost-aware quadratic urgency.

        Phi(s) = -inventory_frac * waste_cost_scale * (1 + time_pressure^2)

        where:
            inventory_frac = normalized inventory remaining (state[1])
            time_pressure = 1 - hours_remaining_norm (state[0])
            waste_cost_scale = cost_per_unit * waste_penalty_multiplier * initial_inventory

        The scale is derived from the environment's actual waste penalty so
        it generalizes across SKUs without per-product tuning. Quadratic
        time pressure makes the last steps matter much more than early ones.

        Preserves optimal policy (Ng et al., 1999).
        """
        if not self.reward_shaping:
            return reward

        scale = self.waste_cost_scale if self.waste_cost_scale is not None else 5.0

        def potential(s):
            hours_remaining_norm = s[0]  # already normalized
            inventory_norm = s[1]        # already normalized
            time_pressure = 1.0 - hours_remaining_norm
            return -inventory_norm * scale * (1.0 + time_pressure ** 2)

        phi_s = potential(state)
        phi_s_next = potential(next_state)
        shaped = reward + self.gamma * phi_s_next - phi_s
        return shaped

    def store_transition(self, state, action, reward, next_state, done, next_action_mask=None):
        """Store a transition in replay buffer (with optional reward shaping)."""
        shaped_reward = self._shape_reward(state, action, reward, next_state)
        self.replay_buffer.push(state, action, shaped_reward, next_state, done, next_action_mask)

    def train_step_fn(self):
        """Perform one training step (sample batch + gradient update)."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample from replay buffer
        sample = self.replay_buffer.sample(self.batch_size)
        if sample is None:
            return None

        if self.use_per:
            states, actions, rewards, next_states, dones, next_action_masks, \
                per_indices, is_weights = sample
        else:
            states, actions, rewards, next_states, dones, next_action_masks = sample
            per_indices = None
            is_weights = None

        # Compute target Q-values with action masking
        if self.double_dqn:
            # Double DQN: online network selects action, target network evaluates
            online_next_q = self.q_network.predict(next_states)
            masked_online = online_next_q.copy()
            masked_online[~next_action_masks] = -np.inf
            best_next_actions = np.argmax(masked_online, axis=1)

            target_next_q = self.target_network.predict(next_states)
            max_next_q = target_next_q[np.arange(len(best_next_actions)), best_next_actions]
        else:
            # Vanilla DQN: target network selects and evaluates
            next_q = self.target_network.predict(next_states)
            masked_next_q = next_q.copy()
            masked_next_q[~next_action_masks] = -np.inf
            max_next_q = np.max(masked_next_q, axis=1)

        # Guard against all-masked edge case (terminal states)
        max_next_q = np.where(np.isinf(max_next_q), 0.0, max_next_q)

        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Forward pass
        q_values, cache = self.q_network.forward(states)

        # Compute per-sample TD errors
        predicted = np.array([q_values[i, actions[i]] for i in range(len(actions))])
        td_errors = predicted - targets
        loss = np.mean((td_errors) ** 2)
        self.losses.append(loss)

        # Backward pass (with IS weights for PER)
        self.q_network.backward(cache, q_values, targets, actions, is_weights=is_weights)

        # Update priorities in PER buffer
        if self.use_per and per_indices is not None:
            self.replay_buffer.update_priorities(per_indices, td_errors)

        # Update target network
        self.train_step += 1
        if self.soft_target_tau is not None:
            # Soft update (Polyak averaging) every step
            self.target_network.soft_update_from(self.q_network, self.soft_target_tau)
        elif self.train_step % self.target_update_freq == 0:
            # Hard copy periodically
            self.target_network.copy_from(self.q_network)

        return loss

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save agent to file."""
        data = {
            "q_params": [p.copy() for p in self.q_network.params],
            "target_params": [p.copy() for p in self.target_network.params],
            "epsilon": self.epsilon,
            "train_step": self.train_step,
            "losses": self.losses[-1000:],  # last 1000
            "episode_rewards": self.episode_rewards,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load agent from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        for i, p in enumerate(data["q_params"]):
            np.copyto(self.q_network.params[i], p)
        for i, p in enumerate(data["target_params"]):
            np.copyto(self.target_network.params[i], p)
        self.epsilon = data["epsilon"]
        self.train_step = data["train_step"]
        self.losses = data.get("losses", [])
        self.episode_rewards = data.get("episode_rewards", [])
