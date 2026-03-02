# Technical Architecture — Fresh RL POC

Detailed technical documentation of the reinforcement learning system for perishable product markdown pricing. Covers the full architecture, algorithms, data structures, and training pipeline.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [MDP Formulation](#mdp-formulation)
3. [Episode Lifecycle](#episode-lifecycle)
4. [Neural Network (PyTorch MLP)](#neural-network-pytorch-mlp)
5. [Deep Q-Network (DQN)](#deep-q-network-dqn)
6. [Double DQN](#double-dqn)
7. [Target Network and Soft Updates](#target-network-and-soft-updates)
8. [Experience Replay](#experience-replay)
9. [Prioritized Experience Replay (PER)](#prioritized-experience-replay-per)
10. [SumTree Data Structure](#sumtree-data-structure)
11. [N-Step Returns](#n-step-returns)
12. [Exploration Bias Correction](#exploration-bias-correction)
13. [Action Masking (Progressive Constraint)](#action-masking-progressive-constraint)
14. [Reward Shaping](#reward-shaping)
15. [Historical Data Pre-filling](#historical-data-pre-filling)
16. [Demand Model](#demand-model)
17. [Baseline Policies](#baseline-policies)
18. [Training Pipeline](#training-pipeline)
19. [Portfolio Runner](#portfolio-runner)
20. [Product Catalog](#product-catalog)
21. [Pooled Category Training](#pooled-category-training)

---

## System Architecture

```
+----------------------------------------------------------------------+
|                          Portfolio Runner                              |
|  (scripts/run_portfolio.py — parallel training across 150 SKUs)       |
+------+-------+-------+-------+-------+-------+-------+-------+------+
       |       |       |       |       |       |       |       |
       v       v       v       v       v       v       v       v
+----------------------------------------------------------------------+
|                     Training Loop (scripts/train.py)                  |
|                                                                      |
|  +---------------------+      +----------------------------------+   |
|  | Historical Pre-fill |----->| Warmup Phase (gradient steps on  |   |
|  | (baseline rollouts) |      | buffered data, no env interaction)|  |
|  +---------------------+      +----------------------------------+   |
|                                        |                             |
|                                        v                             |
|  +-------------------------------------------------------------------+
|  |                    Online Training Loop                           |
|  |                                                                   |
|  |  +-----------+     +-----------+     +------------------+        |
|  |  |    Env    |---->|   Agent   |---->| Replay Buffer    |        |
|  |  | (Gym MDP) |<----|  (DQN)    |<----| (Uniform or PER) |        |
|  |  +-----------+     +-----------+     +------------------+        |
|  |       |                  |                    |                   |
|  |       |            +-----+------+     +-------+------+           |
|  |       |            | Online Net |     | Target Net   |           |
|  |       |            | (Q-values) |     | (TD targets) |           |
|  |       |            +-----+------+     +-------+------+           |
|  |       |                  |                    |                   |
|  |       |                  +---- soft update ---+                  |
|  |       |                  |    (tau=0.005)                        |
|  |       |                  |                                       |
|  |       +---> Reward Shaping (optional, potential-based)           |
|  +-------------------------------------------------------------------+
|                                                                      |
|  +-------------------------------------------------------------------+
|  |                    Evaluation Phase                               |
|  |  DQN (plain) vs DQN (shaped) vs 7 Baselines                     |
|  |  -> greedy rollouts, aggregate metrics, comparison tables        |
|  +-------------------------------------------------------------------+
+----------------------------------------------------------------------+
```

### Component Dependency Graph

```
product_catalog.py -----> environment.py -----> dqn_agent.py
        |                      |                     |
        |                      |                +----+----+
        |                      |                |         |
        |                      v                v         v
        |                 baselines.py    sumtree.py  prioritized_replay.py
        |                      |
        v                      v
   pooled_env.py -------> historical_data.py
```

### Data Flow Per Training Step

```
Environment                Agent                     Replay Buffer
    |                        |                            |
    |--- obs, mask --------->|                            |
    |                        |--- epsilon-greedy -------->|
    |<--- action ------------|   (masked Q-values)       |
    |                        |                            |
    |--- next_obs, reward -->|                            |
    |    done, info          |                            |
    |                        |--- store_transition() --->|
    |                        |    (with reward shaping)   |
    |                        |                            |
    |                        |<--- sample batch ---------|
    |                        |    (with IS weights if PER)|
    |                        |                            |
    |                        |--- compute TD targets ---->|
    |                        |    (Double DQN)            |
    |                        |                            |
    |                        |--- loss.backward() ------->|
    |                        |    + optimizer.step()      |
    |                        |                            |
    |                        |--- update priorities ----->|
    |                        |    (PER only)              |
    |                        |                            |
    |                        |--- soft update target ---->|
    |                        |    (tau=0.005)             |
```

---

## MDP Formulation

The markdown pricing problem is modeled as a finite-horizon Markov Decision Process.

### State Space (10-dimensional, normalized to [0,1])

| Index | Feature | Formula / Normalization | Signal |
|-------|---------|------------------------|--------|
| 0 | `hours_remaining` | `/ markdown_window_hours` | Time pressure — how much time left to sell |
| 1 | `inventory_remaining` | `/ actual_initial_inventory` | Stock level — what fraction remains unsold |
| 2 | `current_discount_idx` | `/ (n_actions - 1)` | Current pricing state — where in the discount ladder |
| 3 | `tod_sin` | `(sin(2π·tod/n_blocks) + 1) / 2` | Cyclical time-of-day (sin component) |
| 4 | `tod_cos` | `(cos(2π·tod/n_blocks) + 1) / 2` | Cyclical time-of-day (cos component) |
| 5 | `dow_sin` | `(sin(2π·dow/7) + 1) / 2` | Cyclical day-of-week (sin component) |
| 6 | `dow_cos` | `(cos(2π·dow/7) + 1) / 2` | Cyclical day-of-week (cos component) |
| 7 | `recent_velocity` | `/ (initial_inventory * 0.5)` | Sales momentum — rolling mean of last 3 steps |
| 8 | `sell_through_rate` | `(total_sold/step_count) / (initial_inv/episode_len)` | Actual vs ideal sell-through pace (1.0 = on track) |
| 9 | `projected_clearance` | `(recent_velocity * remaining_steps) / inventory_remaining` | Whether current sales pace can clear remaining inventory before deadline |

**Cyclical encoding** uses sin/cos pairs so the network learns that 11pm neighbors midnight and Sunday neighbors Monday — linear floats put these maximally apart. **Sell-through rate** gives the agent a direct "am I ahead or behind pace?" signal without requiring the network to learn the division between inventory and time features. **Projected clearance** extrapolates from the observed recent sales velocity (rolling mean of last 3 steps) to estimate whether the current pace will clear remaining inventory before the deadline — a value near 1.0 means the agent can hold the current discount, while low values signal urgency to go deeper. Uses only observable data (past sales, time, inventory) with no access to demand model internals.

All features are clipped to [0, 1] after normalization. This keeps the neural network's input distribution stable across products with vastly different absolute scales.

**Pooled mode (14-dim)**: In pooled category training, 4 additional product features are appended: `price_norm`, `cost_frac_norm`, `inventory_norm`, `pack_size_norm`. These are static per product (constant within an episode) and enable a single model to learn product-conditional strategies. See [Pooled Category Training](#pooled-category-training).

### Action Space

| Mode | Actions | Levels | Granularity |
|------|---------|--------|-------------|
| **2h** (recommended) | 11 | 20%, 25%, 30%, ..., 65%, 70% | 5pp steps |
| **4h** | 6 | 20%, 30%, 40%, 50%, 60%, 70% | 10pp steps |

The action is a discount index. Due to the **progressive constraint**, the agent can only select actions with index >= the current discount index. This is enforced via action masking (see [Action Masking](#action-masking-progressive-constraint)).

### Reward Function

At each step:

```
R_step = revenue - holding_cost
       = (price * units_sold) - (0.02 * price * remaining_inventory * step_hours/4)
```

At the terminal step, additional components:

```
R_terminal = R_step - waste_penalty + clearance_bonus
```

where:
- **Waste penalty**: `cost_per_unit * 3.0 * unsold_inventory` — applied when inventory remains at deadline
- **Clearance bonus**: `1.0 * initial_inventory` — awarded only if all inventory was sold before the window expired

| Component | Formula | Purpose |
|-----------|---------|---------|
| Revenue | `price * qty_sold` | Drives profitable sales |
| Holding cost | `0.02 * price * remaining_inv * (step_h/4)` | Penalizes holding unsold stock |
| Waste penalty | `cost * 3.0 * unsold_at_deadline` | Heavy penalty for leftover waste |
| Clearance bonus | `1.0 * initial_inventory` | Rewards complete sell-through |

The waste penalty multiplier of 3.0x cost reflects real-world disposal costs, brand damage, and sustainability penalties. The holding cost is small (2% of current price per 4h step) but creates a continuous signal favoring quicker sales.

### Transition Dynamics

The state transitions stochastically — given the same state and action, different demand realizations lead to different next states. The demand model (Poisson with price/time/DOW modulation) introduces the stochasticity. See [Demand Model](#demand-model) for details.

### Discount Factor (gamma = 0.97)

`gamma = 0.97` means the agent values a reward 1 step in the future at 97% of its current value. For a 24h product with 4h steps (6 steps total), the effective horizon discount is `0.97^6 = 0.83` — the agent still cares substantially about the terminal waste penalty even from the first step. For 2h mode (12 steps), `0.97^12 = 0.69`.

---

## Episode Lifecycle

A complete walkthrough of what happens during one training episode.

### Phase 1: Reset

```
env.reset()
  |
  +-- Randomize day_of_week (0-6) and time_of_day (0 to n_time_blocks-1)
  +-- Sample initial inventory: base +/- N(0, noise_std=2.0), min=1
  +-- Reset discount index to 0 (20%)
  +-- Reset step counter, sales history, and metrics
  +-- Return initial observation (10-dim normalized vector)
```

The randomized start time and day create natural variation — the agent must learn policies that work across Monday mornings (low traffic) and Saturday evenings (peak traffic).

### Phase 2: Step Loop

For each time step within the markdown window:

```
Step t:
  |
  +-- Agent observes state s_t = [hours_rem, inv_rem, disc_idx, tod_sin, tod_cos, dow_sin, dow_cos, velocity, sell_through, proj_clearance]
  |
  +-- Agent gets action mask: [False, False, True, True, True, True]
  |     (e.g., if current_discount_idx=2, actions 0,1 are masked)
  |
  +-- Agent selects action a_t via epsilon-greedy:
  |     With prob epsilon: random from valid actions
  |     With prob 1-epsilon: argmax Q(s_t, a) over valid actions
  |
  +-- Environment enforces progressive constraint: a_t = max(a_t, current_idx)
  |
  +-- Environment computes demand:
  |     discount = DISCOUNT_LEVELS[a_t]          (e.g., 0.40)
  |     price = base_price * (1 - discount)       (e.g., $5.00 * 0.60 = $3.00)
  |     demand = Poisson(base_demand * price_effect * intraday * dow)
  |     units_sold = min(demand, inventory_remaining)
  |
  +-- Environment computes step reward:
  |     revenue = price * units_sold
  |     holding = 0.02 * price * remaining_inv * (step_hours/4)
  |     reward = revenue - holding
  |
  +-- If terminal (inventory=0 or window expired):
  |     reward -= cost * 3.0 * unsold_inventory  (waste)
  |     reward += 1.0 * initial_inv              (if cleared early)
  |
  +-- Agent applies reward shaping (if enabled):
  |     shaped_reward = reward + gamma * Phi(s_{t+1}) - Phi(s_t)
  |
  +-- Agent stores (s_t, a_t, shaped_reward, s_{t+1}, done, mask_{t+1})
  |     in replay buffer
  |
  +-- Agent samples mini-batch from buffer and performs gradient step
  |     (see Training Pipeline for details)
  |
  +-- Advance time: step_count++, time_of_day = (tod + 1) % n_blocks
```

### Phase 3: End of Episode

```
Episode complete:
  |
  +-- Decay epsilon: epsilon = max(epsilon_end, epsilon * epsilon_decay)
  +-- Record episode metrics: total_reward, revenue, waste_rate, clearance_rate
  +-- Periodic greedy evaluation (every 50 episodes):
  |     Run 20 episodes with epsilon=0, record mean metrics
  +-- Save best agent checkpoint (by total reward)
```

### Concrete Example: Salmon Fillet, 2h Mode

```
Product: salmon_fillet (seafood category)
  Base price: $12.50, Cost: $7.25, Elasticity: 2.6
  Markdown window: 24h -> 12 steps at 2h intervals
  Initial inventory: ~18 units (with noise)
  Actions: 11 levels [20%, 25%, 30%, ..., 70%]

Step 0 (hour 0-2, Saturday evening, 8pm):
  State: [1.0, 1.0, 0.0, 0.83, 0.83, 0.0, 0.78, 0.0, 0.0, 0.85]
  Mask:  [T, T, T, T, T, T, T, T, T, T, T]   (all valid, first step)
  Agent picks action 2 (30% discount -> $8.75)
  DOW=Sat(1.3) * intraday=evening(1.5) = high traffic
  Demand ~ Poisson(5.0 * exp(-2.6*(-0.3)) * 1.5 * 1.3 * 0.5) = Poisson(~4.7)
  Sells 5 units, revenue=$43.75, inventory=13

Step 3 (hour 6-8, Sunday morning, 2am):
  State: [0.75, 0.72, 0.18, 0.08, 1.0, 0.28, 0.62, 0.15, 0.37, 0.72]
  Mask:  [F, F, T, T, T, T, T, T, T, T, T]   (can't go back to 20% or 25%)
  Agent picks action 4 (40% discount -> $7.50)
  DOW=Sun(1.2) * intraday=night(0.2) = very low traffic
  Demand ~ Poisson(~0.8) -> sells 1 unit
  Agent learned: don't waste deep discounts during low-traffic hours

Step 10 (hour 20-22, Sunday evening):
  State: [0.17, 0.28, 0.55, 0.83, 1.0, 0.22, 0.78, 0.35, 0.72, 0.45]
  Mask:  [F, F, F, F, F, F, T, T, T, T, T]   (locked at 50%+)
  Only 5 units left, 2 steps remaining — urgency is high
  Agent picks action 9 (65% discount -> $4.38)
  Peak traffic helps clear 4 units

Step 11 (final, hour 22-24):
  1 unit remaining, picks action 10 (70% -> $3.75)
  Sells last unit -> clearance bonus!
  Total reward: revenue($89.50) - holding($2.30) + bonus($18.00) = $105.20
```

---

## Neural Network (PyTorch MLP)

The Q-network is a 2-layer multilayer perceptron built with PyTorch `nn.Sequential`.

### Architecture

```python
nn.Sequential(
    nn.Linear(state_dim, hidden_dim),   # Layer 1
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),  # Layer 2
    nn.ReLU(),
    nn.Linear(hidden_dim, n_actions)    # Output
)
```

Default `hidden_dim=128`, `state_dim=10` (per-SKU) or `state_dim=14` (pooled):

```
Input (10-dim per-SKU / 14-dim pooled)
    |
    v
[Linear: state_dim -> 128] -> [ReLU]    (Layer 1)
    |
    v
[Linear: 128 -> 128] -> [ReLU]          (Layer 2)
    |
    v
[Linear: 128 -> n_actions]              (Output)
    |
    v
Q-values (11-dim for 2h mode)
```

Total parameters (2h mode, hidden_dim=128):
- Per-SKU (10-dim): `10*128 + 128 + 128*128 + 128 + 128*11 + 11 = 19,083`
- Pooled (14-dim): `14*128 + 128 + 128*128 + 128 + 128*11 + 11 = 19,595`

### Kaiming Initialization

All Linear layers use Kaiming (He) normal initialization via `nn.init.kaiming_normal_()` with biases zeroed:

```python
for layer in net:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
```

Kaiming initialization is designed for ReLU networks — it accounts for the fact that ReLU zeroes out ~half of activations, keeping variance stable through depth. Without it, deep networks suffer from vanishing/exploding activations.

### Training Step

The loss is MSE over the Q-values for taken actions, optionally weighted by importance-sampling weights (PER):

```
L = (1/B) * sum_i [ w_i * (Q(s_i, a_i) - target_i)^2 ]
```

PyTorch handles backpropagation and optimization automatically:

```python
loss.backward()                                      # autograd computes all gradients
nn.utils.clip_grad_norm_(q_network.parameters(), 1.0) # gradient clipping (max norm)
optimizer.step()                                      # Adam update
```

### Gradient Clipping

Gradient norms are clipped to a max of 1.0 via `clip_grad_norm_()`. This prevents catastrophic gradient explosions from outlier TD errors, which are common in RL due to non-stationary targets and high-variance rewards.

### Adam Optimizer

`torch.optim.Adam` with `lr=5e-4`. Adam adapts the learning rate per-parameter — parameters with sparse gradients get larger effective updates.

### Model Persistence

Agent state is saved and loaded via `torch.save()` / `torch.load()` as `.pt` files containing the Q-network, target network, and optimizer state dicts.

---

## Deep Q-Network (DQN)

### Core Idea

DQN (Mnih et al., 2015) approximates the optimal action-value function `Q*(s, a)` — the expected cumulative reward from state `s`, taking action `a`, and following the optimal policy thereafter.

The agent maintains a neural network `Q_theta(s, a)` and trains it by minimizing the temporal difference (TD) error:

```
Loss = E[(Q_theta(s, a) - target)^2]

target = r + gamma * max_a' Q_target(s', a')   (if not terminal)
target = r                                      (if terminal)
```

### Training Step

```
1. Sample mini-batch {(s, a, r, s', done)} from replay buffer
2. Compute Double DQN targets:
     a* = argmax_a' Q_online(s', a')   (online selects, with mask)
     y = r + gamma^n_step * Q_target(s', a*) * (1 - done)
3. Forward pass: Q_theta(s) for all actions
4. Loss: MSE between Q_theta(s, a) and y for taken actions only
5. loss.backward() + clip_grad_norm_ + optimizer.step()
6. Soft update target network (tau=0.005)
```

### Why Experience Replay?

Without replay, the agent trains on correlated, sequential transitions from the current policy. This violates the i.i.d. assumption of SGD and causes:
- **Catastrophic forgetting**: recent experiences overwrite learned patterns
- **Oscillation**: the policy flip-flops between strategies

Replay breaks temporal correlations by sampling uniformly from a buffer of past experiences, providing a more stable and diverse training signal.

### Why a Target Network?

The TD target `r + gamma * max Q_target(s')` uses the **target** network, not the online network being updated. If we used the online network for both:

```
target = r + gamma * max Q_theta(s')  <-- Q_theta is what we're updating!
```

The targets would shift with every gradient step, creating a "moving target" problem that destabilizes training. The target network provides a stable reference that changes slowly (via periodic hard copies or soft updates).

---

## Double DQN

### The Overestimation Problem

Standard DQN uses `max_a' Q_target(s', a')` as the TD target. The `max` operator introduces a positive bias — when Q-values have estimation errors (which they always do), taking the max over noisy estimates systematically overestimates the true value:

```
E[max(Q + noise)] >= max(E[Q + noise]) = max(Q)
```

This overestimation compounds across the Bellman backup chain, leading to inflated Q-values that diverge from reality. The agent becomes overconfident about suboptimal actions.

### The Fix: Decouple Selection from Evaluation

Double DQN (van Hasselt et al., 2016) splits the max into two steps using two networks:

```
1. SELECT the best action using the ONLINE network:
   a* = argmax_a' Q_online(s', a')

2. EVALUATE that action using the TARGET network:
   target = r + gamma * Q_target(s', a*)
```

Since the online network selects and the target network evaluates, the noise in one doesn't directly amplify the noise in the other. This dramatically reduces overestimation.

### Implementation

```python
with torch.no_grad():
    # Online network SELECTS the best action (with mask)
    online_next_q = q_network(next_states_t)
    masked_online = online_next_q.clone()
    masked_online[~next_masks_t] = -float("inf")
    best_actions = masked_online.argmax(dim=1)

    # Target network EVALUATES that action
    target_next_q = target_network(next_states_t)
    max_next_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)

    targets = rewards_t + gamma * max_next_q * (1 - dones_t)
```

By decoupling selection (online) from evaluation (target), the noise in one doesn't amplify the noise in the other, dramatically reducing Q-value overestimation.

---

## Target Network and Soft Updates

### Soft Updates (Polyak Averaging)

The target network is updated via Polyak averaging every training step:

```python
# Every training step:
Q_target = tau * Q_online + (1 - tau) * Q_target
```

With `tau = 0.005`, the target network moves toward the online network by 0.5% each step. This provides:

- **Smooth target evolution**: no discontinuities (unlike periodic hard copies)
- **Recency**: the target always reflects recent learning
- **Stability**: the target can't change faster than 0.5% per step

The effective "half-life" of old target weights is `ln(0.5) / ln(1 - tau) = ~138 steps` — after ~138 gradient updates, old target weights have decayed to half their influence.

### Implementation

```python
with torch.no_grad():
    for tp, p in zip(target_network.parameters(), q_network.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)
```

---

## Experience Replay

### Uniform Replay Buffer

The basic replay buffer is a circular deque that stores transitions and samples uniformly:

```
Buffer: [(s, a, r, s', done, mask), ...]
Capacity: 10,000 transitions
Sampling: uniform random (each transition equally likely)
```

Each transition includes:
- **State** (10-dim float32): the observation before the action
- **Action** (int): the discount level selected
- **Reward** (float): possibly shaped reward
- **Next state** (10-dim float32): the observation after the action
- **Done** (bool): whether the episode ended
- **Next action mask** (n_actions bool): valid actions from next state

The buffer overwrites the oldest transition when full (FIFO eviction).

### Why Store Action Masks?

The progressive constraint means not all actions are valid from every state. When computing TD targets:

```
max_a' Q(s', a') subject to a' in valid_actions(s')
```

We need to know which actions were valid at `s'`. Storing the mask avoids reconstructing the constraint (which would require knowing the environment's internal state at `s'`).

---

## Prioritized Experience Replay (PER)

### Motivation

Uniform replay treats all experiences equally, but some transitions are more "surprising" or "informative" than others. A transition with a high TD error means the agent's prediction was far off — it has the most to learn from that experience.

PER (Schaul et al., 2016) samples transitions proportionally to their TD error magnitude, so the agent replays its biggest mistakes more often.

### Proportional Prioritization

Each transition `i` has priority:

```
p_i = (|delta_i| + epsilon)^alpha
```

where:
- `delta_i` = TD error = `Q(s, a) - target` — how wrong the prediction was
- `epsilon = 1e-5` — ensures no transition has zero probability
- `alpha = 0.6` — controls prioritization strength (0=uniform, 1=fully prioritized)

The sampling probability is:

```
P(i) = p_i / sum_j(p_j)
```

### Importance Sampling Correction

Non-uniform sampling introduces bias — high-priority transitions are overrepresented. This distorts the gradient:

```
E_uniform[gradient] != E_PER[gradient]
```

Importance-sampling (IS) weights correct for this:

```
w_i = (N * P(i))^(-beta) / max_j(w_j)
```

where:
- `N` = buffer size
- `beta` = correction exponent, annealed from 0.4 to 1.0
- Division by `max(w_j)` normalizes weights to [0, 1]

At `beta=1.0`, the correction is exact and PER is unbiased. At `beta=0.4` (early training), correction is partial — we tolerate some bias for the benefit of faster learning. Beta anneals linearly toward 1.0 as training progresses.

### Stratified Sampling

Rather than drawing `B` independent samples, PER uses stratified sampling for lower variance:

```
Divide [0, total_priority) into B equal segments
From each segment [i*seg, (i+1)*seg), sample one cumulative value uniformly
Find the corresponding leaf in the SumTree
```

This guarantees coverage across the priority distribution — the batch isn't dominated by a few very-high-priority transitions.

### Priority Update Cycle

```
1. Store new transition with max_priority (ensures at least one sample)
2. Sample batch proportional to priorities
3. Compute TD errors for the sampled batch
4. Update priorities: p_i = (|delta_i| + epsilon)^alpha
5. Scale gradients by IS weights w_i
6. Update network with corrected gradients
```

---

## SumTree Data Structure

### Problem

PER needs to sample proportionally to priorities and update individual priorities. Naive approaches are too slow:

| Operation | Array (naive) | SumTree |
|-----------|---------------|---------|
| Insert | O(1) | O(log n) |
| Sample | O(n) | O(log n) |
| Update priority | O(1) | O(log n) |
| Total sum | O(n) | O(1) |
| Min priority | O(n) | O(1) |

For a buffer of 10,000 transitions, the SumTree reduces sampling from 10,000 operations to ~14 per sample.

### Structure

A binary tree stored as a flat array:

```
Capacity = 4 (example)
Array size = 2 * 4 - 1 = 7

Index:  0    1    2    3    4    5    6

             [36]              <- root (total sum)
           /      \
        [16]      [20]         <- internal nodes (child sums)
        /  \      /  \
      [6] [10] [12]  [8]      <- leaves (priorities)

Data array: [transition_0, transition_1, transition_2, transition_3]
Leaf index = data_index + (capacity - 1)
```

### Proportional Sampling (Tree Traversal)

To sample with probability proportional to priority:

```
1. Generate random value v in [0, total_sum)
2. Start at root (index 0)
3. At each internal node:
     if v <= left_child_sum:
       go left
     else:
       v -= left_child_sum
       go right
4. Reach a leaf — return its data and priority
```

This is essentially a cumulative distribution function (CDF) lookup in O(log n).

### Priority Update (Propagation)

When a leaf priority changes:

```
1. Update leaf value
2. Walk up to root, recalculating each parent:
     parent = (child_idx - 1) // 2
     parent_sum = left_child + right_child
3. Stop at root
```

### Parallel Min-Tree

A second tree with the same structure tracks the minimum priority instead of the sum. This enables O(1) `min_priority()` queries, used for computing IS weights.

```python
# Min-tree propagation
min_tree[parent] = min(min_tree[left], min_tree[right])
```

---

## N-Step Returns

### Motivation

Standard 1-step DQN uses `target = r + gamma * Q(s')`, which propagates reward information one step at a time. For our short episodes (12-24 steps), this means the terminal reward signal (waste penalty, clearance bonus) takes many training iterations to propagate backward to early actions. N-step returns accelerate this by looking ahead multiple steps:

```
G_n = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_{n-1}
target = G_n + gamma^n * Q(s_n) * (1 - done_n)
```

For 12-24 step episodes, n=4-8 is the sweet spot (per Rainbow DQN ablation research). This directly propagates rewards across ~1/3 of the episode in a single update, dramatically improving credit assignment.

### NStepAccumulator Design

The `NStepAccumulator` class sits between `store_transition()` and the replay buffer:

```python
class NStepAccumulator:
    def __init__(self, n_step, gamma, buffer):
        self._deque = deque(maxlen=n_step)  # sliding window

    def push(state, action, reward, next_state, done, next_action_mask):
        # Append transition to deque
        # If done: flush all remaining transitions
        # Elif deque full: compute n-step return from window, push oldest
```

**Normal operation** (deque full, not terminal): computes `G_n` from the n transitions in the deque, pushes `(s_0, a_0, G_n, s_n, done_n, mask_n)` to the underlying buffer, then poplefts the oldest transition.

**Episode boundary** (`done=True`): flushes all remaining transitions. Each gets a progressively shorter lookback. Since all these transitions lead to a terminal state (`done=True`), the bootstrap term `gamma^n * Q(s_n)` is zeroed out — no need to track the actual step count.

### Compatibility

- **Reward shaping**: Shaping is applied per-transition *before* the accumulator receives them. The potential-based shaping terms telescope correctly when summed: `sum(gamma^i * [r_i + gamma*Phi(s_{i+1}) - Phi(s_i)])` preserves the PBRS guarantee.
- **PER**: N-step returns are stored in the PER buffer like any other transition — priorities are based on TD error against the n-step target.
- **Historical pre-fill**: `fill_buffer()` calls `agent.store_transition()` with `done` flags at episode boundaries, so n-step accumulation works automatically.
- **Bootstrap discount**: The training step uses `gamma^n_step` instead of `gamma` for the bootstrap term, matching the n-step return formula.

---

## Exploration Bias Correction

### The Problem: Asymmetric Exploration Under Progressive Constraints

Standard epsilon-greedy exploration chooses uniformly among valid actions. Combined with the progressive discount constraint, this creates a systematic exploration bias: at discount index 0 with 11 valid actions, the probability of holding (choosing action 0) is only 1/11 = 9%, while the probability of going deeper is 91%. Over 12 steps of exploration in a 24h episode (2h steps), the agent is frequently pushed toward the maximum discount during training and **never learns that holding at low discounts is optimal** for products where demand exceeds inventory at moderate discounts.

This cumulative bias is more severe with longer episodes — each additional exploration step compounds the probability of going deeper, making it progressively harder for the agent to discover conservative hold strategies.

### Solution: Hold-Action Exploration Bias (`hold_action_prob`)

During the epsilon-greedy exploration phase, a fraction `hold_action_prob` of random actions are replaced with the "hold" action (lowest valid action = current discount index):

```python
if random() < epsilon:
    if hold_action_prob > 0 and random() < hold_action_prob:
        return valid_actions[0]  # hold current discount
    return random_choice(valid_actions)  # uniform over valid
```

With `hold_action_prob=0.5`, the exploration distribution becomes:
- P(hold) = 0.5 + 0.5 × (1/n_valid) ≈ 55% (up from 9%)
- P(go deeper by 1) = 0.5 × (1/n_valid) ≈ 5% each

This generates training episodes where the agent holds at low discounts for many steps, learning that conservative pricing can be optimal when projected clearance is high.

### Complementary Fixes

The exploration bias correction works alongside two other changes:

1. **Conservative prefill mix**: The default baseline mix in `historical_data.py` weights conservative policies more heavily (backloaded_progressive 35% + fixed_20 20% = 55% conservative demonstrations), ensuring the replay buffer starts with ample "hold low" trajectories.

2. **Projected clearance feature** (state dim 9 → 10): The `_projected_clearance()` method extrapolates from observed recent sales velocity to estimate whether remaining inventory will clear before the deadline. Uses only observable data (past sales, time, inventory) — no access to demand model internals.

---

## Action Masking (Progressive Constraint)

### The Constraint

In fresh food markdowns, once a discount is offered to customers, it cannot be reduced (going from 40% to 30% would feel like a price increase). Discounts can only **increase or stay the same** over time.

Formally: `a_t >= a_{t-1}` for all `t`, where `a` is the discount level index.

### Implementation: Mask Invalid Actions

Rather than penalizing constraint violations in the reward function (which the agent might learn to ignore), we **prevent** invalid actions entirely:

```python
def action_masks(self):
    mask = np.zeros(n_actions, dtype=bool)
    mask[current_discount_idx:] = True  # only deeper discounts are valid
    return mask
```

The mask is applied at two points:

**1. Action selection (epsilon-greedy):**

```python
# Random exploration: only over valid actions
if random() < epsilon:
    return random.choice(where(mask)[0])
# Greedy: mask invalid Q-values with -inf
else:
    q_values[~mask] = -inf
    return argmax(q_values)
```

**2. TD target computation:**

```python
# Double DQN: mask invalid next-state actions
masked_online[~next_action_masks] = -inf
best_actions = argmax(masked_online, axis=1)  # only considers valid actions
```

### Why Not Reward Penalties?

A reward-based approach (`reward -= penalty if action < current_idx`) has two problems:
1. The agent must **learn** the constraint through trial and error — wasting training time
2. With function approximation, the agent might occasionally violate the constraint if the penalty-adjusted Q-value is incorrectly estimated

Action masking provides **hard guarantees** — invalid actions are never taken, period.

---

## Reward Shaping

### The Credit Assignment Problem

The main waste penalty arrives only at the **terminal step** when inventory expires. During the preceding 6-12 steps, the agent receives only small revenue and holding cost signals. This creates a sparse, delayed reward that makes it hard for the agent to learn which early decisions led to waste.

### Potential-Based Reward Shaping (PBRS)

PBRS (Ng, Harada, Russell, 1999) adds a supplementary reward signal derived from a potential function `Phi(s)`:

```
shaped_reward = reward + gamma * Phi(s') - Phi(s)
```

The key theorem: **any potential-based shaping preserves the set of optimal policies.** The agent learns faster but converges to the same optimal behavior as without shaping.

### The Potential Function

```python
Phi(s) = -inventory_frac * waste_cost_scale * (1 + time_pressure^2)
```

where:
- `inventory_frac = s[1]` — normalized remaining inventory (0 when empty, 1 when full)
- `time_pressure = 1 - s[0]` — normalized elapsed time (0 at start, 1 at deadline)
- `waste_cost_scale = shaping_ratio * base_price * initial_inventory`

### How It Works

The potential is always negative (or zero when inventory is empty). It represents the "risk" of holding inventory:

| State | Phi(s) | Interpretation |
|-------|--------|----------------|
| Full inventory, early | Low magnitude | Some risk but time to sell |
| Full inventory, late | High magnitude | Dangerous — waste imminent |
| Empty inventory, any time | 0 | No risk — nothing to waste |

The **shaped reward** `gamma * Phi(s') - Phi(s)` creates an instantaneous signal:

- **Selling units** (inventory decreases): `Phi(s')` > `Phi(s)` → positive shaping → "good, you reduced risk"
- **Holding inventory as time passes** (time_pressure increases): `Phi(s')` < `Phi(s)` → negative shaping → "bad, risk is growing"
- **Holding at full inventory near deadline**: very negative shaping → strong urgency signal

### Quadratic Time Pressure

The `(1 + time_pressure^2)` term makes the potential function **nonlinear** in time:

```
time_pressure:  0.0   0.25   0.50   0.75   1.00
multiplier:     1.00  1.06   1.25   1.56   2.00
```

The urgency signal is mild early and intensifies quadratically as the deadline approaches. This mimics the real-world intuition: holding inventory with 20 hours left is mildly concerning; holding inventory with 2 hours left is an emergency.

### Revenue Normalization

The `waste_cost_scale = shaping_ratio * base_price * initial_inventory` scales the shaping signal relative to the product's total revenue potential:

| Product | Price | Inventory | Revenue Scale | Shaping Scale (ratio=0.2) |
|---------|-------|-----------|---------------|---------------------------|
| Salad mix | $3.50 | 25 | $87.50 | $17.50 |
| Salmon fillet | $12.50 | 18 | $225.00 | $45.00 |
| Sushi combo | $15.00 | 10 | $150.00 | $30.00 |

The `shaping_ratio=0.2` means "waste avoidance is worth 20% of expected revenue." This is meaningful for both cheap salad ($3.50) and expensive sushi ($15.00) — the shaping signal is proportional to the stakes.

Without normalization, a fixed scale like `Phi=-5.0` would overwhelm the reward for a $3.50 salad (expected revenue ~$87) but barely register for $15 sushi (expected revenue ~$150).

---

## Historical Data Pre-filling

### Cold Start Problem

A fresh DQN agent starts with random weights and an empty replay buffer. Early training is dominated by random exploration — the agent takes nonsensical actions, stores low-quality transitions, and learns slowly from noise.

### Solution: Pre-fill with Baseline Demonstrations

Before online training begins, we run baseline policies through the environment and store their transitions in the replay buffer:

```
Pre-fill Pipeline:
  1. Create environment with product profile
  2. For 200 episodes:
       a. Sample a baseline policy (weighted random)
       b. Roll out the full episode
       c. Store all transitions in the agent's replay buffer
  3. If using reward shaping, transitions go through agent.store_transition()
     so the shaping function is applied consistently
```

### Baseline Mix

The pre-fill uses a weighted mix of policies to provide diverse behavior:

```
linear_progressive:     15%  — steady, time-based ramp
backloaded_progressive: 35%  — conservative early, aggressive late
demand_responsive:      20%  — adaptive to velocity
fixed_20:               20%  — minimal discounting
fixed_40:               10%  — moderate static discount
```

The mix is weighted toward conservative policies (backloaded 35% + fixed_20 20% = 55%) to counteract the asymmetric exploration bias that pushes toward deep discounts.

This mix creates a diverse "curriculum" — the agent sees conservative strategies, aggressive strategies, and adaptive strategies. It doesn't see the random or immediate-deep baselines because those would pollute the buffer with extreme behaviors.

### PER Priority Boost

Historical transitions are stored with `initial_priority=5.0` (much higher than the default `max_priority=1.0`). This ensures they are heavily sampled during the warmup phase, gradually fading as the agent generates its own high-priority experiences.

### Warmup Phase

After pre-filling the buffer, the agent performs N gradient steps on the buffered data **without interacting with the environment**:

```
Warmup Phase (1000 steps):
  for step in range(1000):
    batch = buffer.sample(32)
    loss = compute_td_loss(batch)
    backprop(loss)
    soft_update_target()
```

This pre-trains the Q-network on reasonable behavior before online exploration begins. Typical warmup loss progression:

```
Step    0: Loss ~1000 (random weights, large TD errors)
Step  200: Loss ~200  (network learning basic patterns)
Step  500: Loss ~100  (convergence on buffered data)
Step 1000: Loss ~70-130 (product-dependent)
```

After warmup, the agent starts online training with a Q-network that already has a rough understanding of the problem, dramatically accelerating convergence.

---

## Demand Model

### Poisson Demand Process

Customer demand at each time step follows a Poisson distribution:

```
demand ~ Poisson(lambda)

lambda = base_markdown_demand * price_effect * intraday_effect * dow_effect * (step_hours/4)
```

The Poisson distribution is natural for modeling count data (number of customers who buy) and provides appropriate stochasticity — the variance equals the mean, so high-demand periods are also more variable.

### Price Elasticity Effect

```
price_effect = exp(-elasticity * (price / base_price - 1))
```

This is an exponential demand curve — as the discount deepens, demand increases exponentially:

| Discount | Price Ratio | Effect (elasticity=3.0) |
|----------|-------------|-------------------------|
| 20% | 0.80 | exp(-3.0 * -0.20) = 1.82x |
| 40% | 0.60 | exp(-3.0 * -0.40) = 3.32x |
| 60% | 0.40 | exp(-3.0 * -0.60) = 6.05x |
| 70% | 0.30 | exp(-3.0 * -0.70) = 8.17x |

Higher elasticity products (vegetables at 3.0-4.5) respond more dramatically to discounts than low-elasticity products (seafood at 2.0-3.0).

### Intraday Traffic Pattern

Demand varies by time of day, reflecting real grocery/ecommerce traffic:

**4h mode (6 blocks/day):**

| Block | Hours | Multiplier | Period |
|-------|-------|------------|--------|
| 0 | 00-04 | 0.3x | Night (minimal) |
| 1 | 04-08 | 0.5x | Early morning |
| 2 | 08-12 | 0.9x | Morning |
| 3 | 12-16 | 1.2x | Afternoon |
| 4 | 16-20 | 1.5x | Evening peak |
| 5 | 20-24 | 0.8x | Late evening |

**2h mode (12 blocks/day):**

| Block | Hours | Multiplier |
|-------|-------|------------|
| 0-1 | 00-04 | 0.2-0.4x |
| 2-3 | 04-08 | 0.4-0.6x |
| 4-5 | 08-12 | 0.8-1.0x |
| 6-7 | 12-16 | 1.1-1.3x |
| 8-9 | 16-20 | 1.5-1.5x |
| 10-11 | 20-24 | 1.0-0.5x |

### Day-of-Week Effect

```
Monday:    0.80x  (lowest)
Tuesday:   0.85x
Wednesday: 0.90x
Thursday:  1.00x  (baseline)
Friday:    1.10x
Saturday:  1.30x  (peak)
Sunday:    1.20x
```

The combination of intraday and DOW effects creates realistic demand variation — a Saturday evening step can have 1.3 * 1.5 = **1.95x** demand vs a Monday night at 0.8 * 0.3 = **0.24x**.

### Inventory Noise

Initial inventory is sampled with Gaussian noise:

```
actual_inventory = max(1, int(base_inventory + N(0, noise_std=2.0)))
```

This variation prevents the agent from memorizing a fixed starting state and forces generalization across inventory levels.

---

## Baseline Policies

Seven rule-based policies serve as benchmarks:

### 1. Immediate Deep 70%

```python
action = n_actions - 1  # always deepest
```

Maximizes demand from the first step but sacrifices all margin. Lower bound for revenue efficiency.

### 2. Linear Progressive

```python
elapsed_frac = step_count / episode_length
action = int(elapsed_frac * n_actions)  # ramp evenly
```

Distributes discounting evenly across the window. A reasonable default strategy.

### 3. Backloaded Progressive

```python
if elapsed_frac < 0.5:
    action = 0                    # stay at 20% first half
else:
    action = 1 + int((elapsed_frac - 0.5) / 0.5 * (n-1))  # ramp second half
```

Conservative early (preserves margin), aggressive late (clears inventory). Often the strongest baseline.

### 4. Demand Responsive

```python
if hours_frac < 0.3 and inv_frac > 0.3:
    action = current_idx + 2      # emergency: deep cut
elif velocity_frac < 0.3:
    action = current_idx + 1      # slow sales: bump
else:
    action = current_idx           # hold
```

The only adaptive baseline — it reacts to real-time sales velocity and urgency. Most sophisticated rule-based approach.

### 5-6. Fixed 20% / Fixed 40%

```python
action = fixed_discount_idx       # never change
```

Static pricing at a fixed level. Useful as lower bounds.

### 7. Random Policy

```python
action = random.choice(valid_actions)
```

Lower bound. Useful for measuring how much better any structured policy performs.

---

## Training Pipeline

### Single Product Training (scripts/train.py)

```
Input: product name, hyperparameters, flags
Output: trained agent (.pt), training history (.json)

Pipeline:
  1. Create environment from product catalog profile
  2. Create DQN agent (PyTorch) with configured hyperparameters
  3. [Optional] Pre-fill buffer with 200 episodes of baseline data
  4. [Optional] Run 1000 warmup gradient steps on buffered data
  5. Online training loop (3000-5000 episodes):
       a. Reset environment
       b. Run episode (step loop with epsilon-greedy + action masking)
       c. Store transitions with reward shaping (if enabled)
       d. Train on sampled mini-batches
       e. Decay epsilon
       f. Every 50 episodes: greedy evaluation (20 episodes, epsilon=0)
       g. Save best agent checkpoint
  6. Save final agent and training history (including losses + epsilon_decay)

Evaluation (scripts/evaluate.py):
  -> Greedy rollouts, per-policy metrics, comparison table

Visualization (scripts/visualize.py):
  Single-product mode (--product):
    -> Training curves, training dashboard, policy comparison bars,
       action distributions, policy heatmap, episode walkthrough,
       revenue-waste Pareto, discount progression
  Portfolio mode (--portfolio):
    -> 6-panel dashboard, DQN-vs-baseline scatter, category win rates,
       reward gap distribution, per-SKU gap dots, baseline difficulty,
       revenue-waste comparison, three-way DQN/shaped/baseline comparison
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_dim` | 128 (recommended) | Sufficient for 10-dim input; balances capacity and generalization |
| `lr` | 5e-4 | Standard for DQN with Adam |
| `gamma` | 0.97 | Balances terminal rewards with step-level signals |
| `epsilon_start` | 1.0 | Fully random initially |
| `epsilon_end` | 0.05 | 5% random exploration at convergence |
| `epsilon_decay` | 0.997 (4h) / 0.998-0.999 (2h) | Slower for larger action spaces |
| `buffer_size` | 10,000 | ~1000 episodes of experience |
| `batch_size` | 32 | Standard mini-batch size |
| `n_step` | 1 (default), 5 (recommended) | N-step returns for faster credit assignment in short episodes |
| `hold_action_prob` | 0.0 (default), 0.5 (recommended) | Fraction of exploration actions that hold current discount — corrects asymmetric exploration bias under progressive constraints |
| `tau` | 0.005 | Soft target update rate |
| `PER alpha` | 0.6 | Moderate prioritization |
| `PER beta` | 0.4 -> 1.0 | Anneal IS correction |
| `shaping_ratio` | 0.2 | 20% of revenue scale |

### Epsilon Schedule

The epsilon decays multiplicatively per episode:

```
epsilon_{t+1} = max(epsilon_end, epsilon_t * epsilon_decay)
```

| Mode | Decay | Episodes to reach 0.1 | Final epsilon (5000 ep) |
|------|-------|------------------------|------------------------|
| 2h, 3000ep | 0.999 | 2302 | 0.05 |
| 2h, 5000ep | 0.999 | 2302 | 0.05 |

The recommended 2h mode uses `epsilon_decay=0.999` because the action space is large (11 actions) and the progressive constraint creates asymmetric exploration — the agent needs ample exploration to discover that holding at low discounts can be optimal.

### Transfer Learning

Two-phase training that pools category-level experience before specializing per SKU:

```
Phase 1: Pre-train per category
  For each category (meats, seafood, ...):
    1. Create one DQN agent (reward_shaping=False)
    2. Train for N episodes, cycling through all products in the category
       - Episode i trains on products[i % len(products)]
       - Different seed per episode (seed + ep) for variety
    3. Save model as pretrained_{category}_{step_hours}h.pt

Phase 2: Fine-tune per SKU
  For each product:
    1. Create fresh DQN agent
    2. Load pre-trained category weights via load_pretrained()
       - Copies q_network weights into both q_network and target_network
       - Ignores optimizer state, epsilon, training history
    3. Set epsilon = 0.3 (lower starting exploration)
    4. Train for M episodes (standard pipeline: both plain and shaped)
```

**Experimental findings (v1.3)**: Old-style transfer learning (10-dim state, no product identity) was tested extensively (v1.3–v1.3.2) but **underperforms direct training**. With 3000 fine-tuning episodes, TL achieves 71% beats-baseline vs 81% without TL. The root cause is high intra-category SKU variance: category-level pre-training learns an overly generic policy that interferes with SKU-specific optimization (negative transfer).

**Pooled TL (v2.1)**: Fixes the core failure by using pooled model weights (14-dim state with product features) as initialization. `AugmentedProductEnv` wraps per-SKU envs to produce 14-dim state matching pooled input, so `load_pretrained()` works directly — no weight surgery. Achieves **95% beats-baseline**, the best result. See [Pooled Category Training](#pooled-category-training).

**Architecture compatibility**: For old TL (v1.3), all products share `state_dim=10`. For pooled TL (v2.1), `AugmentedProductEnv` produces `state_dim=14` matching the pooled model, so weights transfer directly.

**`load_pretrained()` method** (`DQNAgent`):
- Loads only `q_network` state dict from the saved checkpoint
- Copies those weights into both `q_network` and `target_network` (target synced)
- Does NOT restore optimizer state, epsilon, train_step, losses, or episode_rewards
- The agent starts fresh for fine-tuning but with learned representations

---

## Portfolio Runner

### Purpose

Validates that the RL approach generalizes across 150 products with diverse economics. Trains both plain and shaped DQN for each SKU, evaluates against all baselines, and produces aggregate analysis.

### Architecture

**Per-SKU Mode** (default):

```
scripts/run_portfolio.py
  |
  +-- Build product list from catalog (optionally filtered by --category)
  |
  +-- ProcessPoolExecutor (--workers 16)
  |     |
  |     +-- Worker 1: _run_single_product("salmon_fillet")
  |     |     +-- Train plain DQN (5000 episodes, 10-dim state)
  |     |     +-- Train shaped DQN (5000 episodes, 10-dim state)
  |     |     +-- Evaluate both + 7 baselines (100 episodes each)
  |     |     +-- Return summary dict
  |     |
  |     +-- Worker 2: _run_single_product("ground_beef_1lb")
  |     +-- Worker 3: ...
  |     +-- Worker N: ...
  |
  +-- Collect results + generate report + 9 portfolio plots
```

**Pooled Mode** (`--pooled`):

```
scripts/run_portfolio.py --pooled
  |
  +-- Group products by category (7 groups of ~22 SKUs)
  |
  +-- ProcessPoolExecutor (--workers 7)
  |     |
  |     +-- Worker 1: _train_category_pooled("meats", [22 products])
  |     |     +-- Create PooledCategoryEnv (one env per SKU)
  |     |     +-- Prefill with multi-product baseline rollouts (14-dim)
  |     |     +-- Train plain DQN (episodes = 5000 * 22, round-robin, 14-dim state)
  |     |     +-- Train shaped DQN (per-product waste_cost_scale, 14-dim state)
  |     |     +-- Evaluate each SKU vs 7 baselines
  |     |     +-- Return list of 22 result dicts
  |     |
  |     +-- Worker 2: _train_category_pooled("seafood", ...)
  |     +-- ... (7 workers total)
  |
  +-- Collect results + generate report + 9 portfolio plots
```

### Demand and Inventory Multipliers

The portfolio runner supports `--demand-mult` and `--inventory-mult` flags to create harder conditions:

```python
env_overrides = {}
if demand_mult:
    env_overrides["base_markdown_demand"] = profile_demand * demand_mult
if inventory_mult:
    env_overrides["initial_inventory"] = int(profile_inventory * inventory_mult)
```

Example: `--demand-mult 0.5 --inventory-mult 2.0` halves demand and doubles inventory, creating conditions where waste is a serious challenge instead of a non-issue.

---

## Product Catalog

### Generation System

150 products across 7 categories (21-22 SKUs each, all 24h windows):

```python
CategorySpec:
  name:            str       # e.g., "seafood"
  sku_names:       List[str] # 21-22 product names per category
  price_range:     (min, max)
  cost_frac_range: (min, max)
  demand_range:    (min, max)
  inventory_range: (min, max)
  elasticity_range:(min, max)
  window_hours:    List[int] # [24] for all categories
  pack_size_range: (min, max) # units per package, e.g. (1, 6)
```

Each SKU's parameters are generated with a seeded RNG for full reproducibility:

```python
def generate_sku_profile(category, sku_name, sku_index, rng):
    base_price = rng.uniform(*category.price_range)
    cost_frac  = rng.uniform(*category.cost_frac_range)
    return {
        "base_price": round(base_price, 2),
        "cost_per_unit": round(base_price * cost_frac, 2),
        "base_markdown_demand": round(rng.uniform(*category.demand_range), 1),
        "initial_inventory": int(rng.uniform(*category.inventory_range)),
        "price_elasticity": round(rng.uniform(*category.elasticity_range), 1),
        "markdown_window_hours": category.window_hours[sku_index % len(...)],
    }
```

### Category Characteristics

| Category | SKUs | Price | Window | Elasticity | Economics |
|----------|------|-------|--------|------------|-----------|
| **Meats** | 22 | $5-15 | 24h | 2.5-3.5 | Moderate price, moderate sensitivity |
| **Seafood** | 22 | $7-20 | 24h | 2.0-3.0 | High price, least elastic |
| **Vegetables** | 21 | $1.50-5 | 24h | 3.0-4.5 | Cheap, highly elastic |
| **Fruits** | 21 | $2-7 | 24h | 3.0-4.0 | Moderate price, elastic |
| **Dairy** | 21 | $1.50-6 | 24h | 2.5-3.5 | Cheap, moderate sensitivity |
| **Bakery** | 21 | $2.50-7 | 24h | 3.5-4.5 | Moderate, day-fresh, most elastic |
| **Deli prepared** | 22 | $5-14 | 24h | 2.5-3.5 | High price, moderate sensitivity |

### Catalog API

```python
generate_catalog(seed=42) -> Dict[str, dict]  # cached, 150 entries
get_product_names(category=None) -> List[str]  # filter by category
get_profile(product_name) -> dict              # single lookup
get_categories() -> List[str]                  # category names
print_catalog_summary()                        # formatted table to stdout
get_product_features(product_name, inventory_mult=1.0) -> np.ndarray  # 4-dim [0,1] features for pooled training
```

The catalog is generated once (cached) and provides a consistent, reproducible set of products across all experiments.

### Observable Product Features

`get_product_features()` returns a 4-dimensional normalized [0,1] vector used by pooled category training to condition the model on product identity:

| Feature | Normalization | What it proxies |
|---------|---------------|-----------------|
| `price_norm` | `(price - cat.price_range[0]) / range` | Revenue scale, willingness-to-pay |
| `cost_frac_norm` | `(cost_frac - cat.cost_fraction_range[0]) / range` | Margin pressure, waste penalty scale |
| `inventory_norm` | `(inv*mult - cat.inventory_range[0]) / range` | Clearing difficulty |
| `pack_size_norm` | `(pack - cat.pack_size_range[0]) / range` | Purchase dynamics, value-seeking behavior |

**Excluded** (simulator leakage): `price_elasticity` and `base_markdown_demand` are never exposed to the agent. It learns demand characteristics through runtime signals (velocity, sell-through, projected clearance) combined with the observable proxies above.

Pack sizes are generated with a separate RNG (`seed + 10000`) to preserve existing product parameters:

| Category | pack_size_range |
|----------|----------------|
| meats | (1, 4) |
| seafood | (1, 2) |
| vegetables | (1, 4) |
| fruits | (1, 6) |
| dairy | (1, 2) |
| bakery | (1, 12) |
| deli_prepared | (1, 8) |

---

## Pooled Category Training

### Motivation

Per-SKU training (v1.4) achieves 86% beats-baseline but requires 300 models (150 plain + 150 shaped). Each new SKU requires full training from scratch. Pooled category training addresses this by training a single model per category that generalizes across all ~22 SKUs, conditioned on observable product features. Pooled TL (v2.1) combines both: pooled weights as initialization for per-SKU fine-tuning, achieving 95% beats-baseline.

### Why It Works (and Why Transfer Learning Failed)

Old transfer learning (v1.3) pre-trained on 10-dim state with **no product identity** — the model couldn't distinguish between different products during pre-training. It learned a blurred average policy that interfered with per-SKU fine-tuning (negative transfer: 71% vs 81%).

Pooled training solves this by including 4 observable product features in every state (14-dim). The model explicitly knows which product it's pricing and can learn product-conditional strategies. Pooled TL (v2.1) then transfers these product-aware representations to per-SKU fine-tuning:

```
Per-SKU:    state = [10 env features]                        → 1 model per SKU (86%)
Old TL:     state = [10 env features] (pretrain)             → 1 model per SKU (71%)
Pooled:     state = [10 env features] + [4 product features] → 1 model per category (78%)
Pooled TL:  state = [10 env features] + [4 product features] → 1 model per SKU (95%)
```

### PooledCategoryEnv

`PooledCategoryEnv(gym.Env)` wraps multiple `MarkdownProductEnv` instances:

```python
class PooledCategoryEnv(gym.Env):
    def __init__(self, category, step_hours, seed, demand_mult, inventory_mult):
        # Creates one MarkdownProductEnv per SKU in the category
        # Pre-computes 4-dim product features for each SKU
        # observation_space = Box(0, 1, shape=(14,))

    def reset(self, seed=None, options=None):
        # Switch active product via options={"product": name}
        # Returns 14-dim obs (10 env + 4 product features)

    def step(self, action):
        # Delegates to active inner env
        # Appends product features to observation

    def action_masks(self):
        # Delegates to active inner env

    def __getattr__(self, name):
        # Delegates attribute access to active inner env
        # Enables baselines to access env.step_count, etc.
```

### Training Pipeline (Pooled)

```
_train_category_pooled(category, products, episodes_per_sku, ...):
  |
  +-- Create PooledCategoryEnv (one env per SKU)
  +-- Create DQN agent (state_dim=14)
  |
  +-- Prefill: pooled_prefill() runs baseline policies through
  |   PooledCategoryEnv, producing 14-dim transitions
  |
  +-- Warmup: N gradient steps on buffered data
  |
  +-- Training loop (total_episodes = episodes_per_sku * len(products)):
  |     Round-robin through products each episode:
  |       product = products[episode % len(products)]
  |       env.reset(options={"product": product})
  |       [For shaped: agent.waste_cost_scale = shaping_ratio * revenue_scale]
  |       Run episode → store transitions (14-dim) → train
  |
  +-- Evaluation: for each product:
        env.reset(options={"product": product})
        Run greedy DQN rollouts + baseline rollouts
        Record metrics (same format as per-SKU results)
```

### State Space (14-dimensional)

| Index | Feature | Source |
|-------|---------|--------|
| 0-9 | Base env features | `MarkdownProductEnv._get_obs()` |
| 10 | `price_norm` | `get_product_features()` — static per product |
| 11 | `cost_frac_norm` | `get_product_features()` — static per product |
| 12 | `inventory_norm` | `get_product_features()` — static per product |
| 13 | `pack_size_norm` | `get_product_features()` — static per product |

Features 10-13 are constant within an episode and change only when switching products. They enable the network to learn product-conditional Q-values: `Q(s, a | product_features)`.

### AugmentedProductEnv (for Pooled TL)

`AugmentedProductEnv(gym.Wrapper)` wraps a single `MarkdownProductEnv` to produce 14-dim observations matching the pooled model input. Used for pooled→per-SKU transfer learning (v2.1):

```python
class AugmentedProductEnv(gym.Wrapper):
    def __init__(self, env, product_name, inventory_mult=1.0):
        # Appends 4 constant product features to every observation
        # observation_space = Box(0, 1, shape=(14,))

    def reset(self, **kwargs):
        # Returns 14-dim obs (10 env + 4 product features)

    def step(self, action):
        # Appends product features to observation

    def __getattr__(self, name):
        # Delegates attribute access to wrapped env (for baselines)
```

The 4 product features are constant throughout all episodes (same product), unlike `PooledCategoryEnv` where features change on product switch. This lets pooled model weights (`Linear(14, 128)`) load directly into per-SKU agents with matching architecture.

### N-Step Compatibility

All steps within one episode are for the same product — the n-step accumulator never mixes transitions across products. At episode boundaries, the accumulator flushes before switching to the next product.

### Results Comparison

| Approach | Models | Beats Baseline | Zero-Shot New SKUs |
|----------|--------|---------------|-------------------|
| **Pooled TL (v2.1)** | **300** | **95%** | No — requires fine-tuning |
| Per-SKU (v1.4) | 300 | 86% | No — requires training |
| Pooled (v2) | 14 | 78% | **Yes** — compute features, use category model |
| Old TL (v1.3.2) | 300 | 71% | No — requires fine-tuning |

---

## References

1. **Mnih et al. (2015)**. "Human-level control through deep reinforcement learning." *Nature* 518. — Original DQN paper.
2. **van Hasselt et al. (2016)**. "Deep Reinforcement Learning with Double Q-learning." *AAAI*. — Double DQN.
3. **Schaul et al. (2016)**. "Prioritized Experience Replay." *ICLR*. — PER with SumTree.
4. **Ng, Harada, Russell (1999)**. "Policy invariance under reward transformations." *ICML*. — Potential-based reward shaping theory.
5. **Lillicrap et al. (2016)**. "Continuous control with deep reinforcement learning." *ICLR*. — Soft target updates (Polyak averaging).
6. **He et al. (2015)**. "Delving Deep into Rectifiers." *ICCV*. — He weight initialization for ReLU networks.
7. **Kingma and Ba (2015)**. "Adam: A Method for Stochastic Optimization." *ICLR*. — Adam optimizer.
8. **De Moor et al. (2022)**. "Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management." *European Journal of Operational Research*.
