"""
Deep Q-Network (DQN) agent with action masking for progressive discounting.

PyTorch-based Double DQN with soft target updates.

Features:
- 2-layer MLP Q-network (PyTorch nn.Module)
- Experience replay buffer with action mask support
- Epsilon-greedy exploration over valid actions only
- Double DQN with Polyak-averaged target network
- Optional potential-based reward shaping (urgency-aware)
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque


class NStepAccumulator:
    """Accumulates n transitions and computes n-step returns before pushing to buffer.

    Sits between store_transition() and the replay buffer. Collects n transitions,
    computes G_n = r_0 + gamma*r_1 + ... + gamma^(n-1)*r_{n-1}, then pushes
    (s_0, a_0, G_n, s_n, done_n, mask_n) to the underlying buffer.

    At episode end (done=True), flushes remaining transitions with shorter
    lookbacks — all have done=True so the bootstrap term is 0.
    """

    def __init__(self, n_step: int, gamma: float, buffer):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = buffer
        self._deque = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done, next_action_mask):
        self._deque.append((state, action, reward, next_state, done, next_action_mask))
        if done:
            self._flush()
        elif len(self._deque) == self.n_step:
            self._push_nstep()

    def _push_nstep(self):
        """Compute n-step return from full deque and push oldest transition."""
        G = 0.0
        for i, (_, _, r, _, _, _) in enumerate(self._deque):
            G += (self.gamma ** i) * r
        s_0, a_0 = self._deque[0][0], self._deque[0][1]
        _, _, _, s_n, done_n, mask_n = self._deque[-1]
        self.buffer.push(s_0, a_0, G, s_n, done_n, mask_n)
        self._deque.popleft()

    def _flush(self):
        """Flush remaining transitions at episode end."""
        while self._deque:
            G = 0.0
            for i, (_, _, r, _, _, _) in enumerate(self._deque):
                G += (self.gamma ** i) * r
            s_0, a_0 = self._deque[0][0], self._deque[0][1]
            _, _, _, s_n, done_n, mask_n = self._deque[-1]
            self.buffer.push(s_0, a_0, G, s_n, done_n, mask_n)
            self._deque.popleft()


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


class DQNAgent:
    """
    Double DQN agent with action masking for progressive discounting.

    Features:
    - Epsilon-greedy exploration over valid actions only
    - Experience replay with action masks
    - Double DQN with Polyak-averaged (soft) target updates
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
        reward_shaping: bool = False,
        waste_cost_scale: float = None,
        seed: int = None,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = None,
        per_epsilon: float = 1e-5,
        n_step: int = 1,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_step = n_step
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.reward_shaping = reward_shaping
        self.waste_cost_scale = waste_cost_scale
        self.use_per = use_per
        self.tau = 0.005

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device("cpu")

        # Networks
        self.q_network = self._build_network(state_dim, hidden_dim, n_actions)
        self.target_network = self._build_network(state_dim, hidden_dim, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

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

        # N-step accumulator (wraps buffer when n_step > 1)
        if n_step > 1:
            self._nstep = NStepAccumulator(n_step, gamma, self.replay_buffer)

        # Training stats
        self.train_step = 0
        self.losses = []
        self.episode_rewards = []

    @staticmethod
    def _build_network(state_dim, hidden_dim, n_actions):
        """Build a 2-layer MLP with Kaiming initialization."""
        net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        return net

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
            with torch.no_grad():
                state_t = torch.FloatTensor(np.array(state).reshape(1, -1))
                q_values = self.q_network(state_t).numpy()[0].copy()
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
        if self.n_step > 1:
            self._nstep.push(state, action, shaped_reward, next_state, done, next_action_mask)
        else:
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

        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)
        next_masks_t = torch.BoolTensor(next_action_masks)

        # Current Q-values for taken actions
        q_values = self.q_network(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN targets: online network selects, target evaluates
        with torch.no_grad():
            online_next_q = self.q_network(next_states_t)
            # Mask invalid actions
            masked_online = online_next_q.clone()
            masked_online[~next_masks_t] = -float("inf")
            best_next_actions = masked_online.argmax(dim=1)

            target_next_q = self.target_network(next_states_t)
            max_next_q = target_next_q.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            # Guard against all-masked edge case (terminal states)
            all_masked = ~next_masks_t.any(dim=1)
            max_next_q[all_masked] = 0.0

            targets = rewards_t + (self.gamma ** self.n_step) * max_next_q * (1 - dones_t)

        # TD errors (for PER priority updates)
        td_errors = (current_q - targets).detach().numpy()

        # Compute loss
        if is_weights is not None:
            is_weights_t = torch.FloatTensor(is_weights)
            loss = (is_weights_t * (current_q - targets) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(current_q, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)

        # Update priorities in PER buffer
        if self.use_per and per_indices is not None:
            self.replay_buffer.update_priorities(per_indices, td_errors)

        # Soft target update (Polyak averaging) every step
        self.train_step += 1
        with torch.no_grad():
            for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return loss_val

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save agent to file."""
        data = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step": self.train_step,
            "losses": self.losses[-2000:],
            "episode_rewards": self.episode_rewards,
        }
        torch.save(data, path)

    def load(self, path):
        """Load agent from file."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(data["q_network"])
        self.target_network.load_state_dict(data["target_network"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon = data["epsilon"]
        self.train_step = data["train_step"]
        self.losses = data.get("losses", [])
        self.episode_rewards = data.get("episode_rewards", [])

    def load_pretrained(self, path):
        """Load only network weights from a pre-trained agent (no optimizer/epsilon/history)."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(data["q_network"])
        self.target_network.load_state_dict(data["q_network"])  # sync target to q_network
