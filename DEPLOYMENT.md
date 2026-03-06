# Production Deployment Guide — Fresh RL

How to take the trained RL markdown pricing agent from POC to production. Covers data requirements, state construction, historical prefill, online learning architecture, rollout phases, and operational guardrails.

---

## Table of Contents

1. [Overview](#overview)
2. [Code Mapping: POC to Production](#code-mapping-poc-to-production)
3. [Data Requirements](#data-requirements)
4. [State Vector Construction](#state-vector-construction)
5. [Historical Data to Replay Buffer](#historical-data-to-replay-buffer)
6. [Online RL Deployment Architecture](#online-rl-deployment-architecture)
7. [Deployment Sequence](#deployment-sequence)
8. [Safety and Guardrails](#safety-and-guardrails)
9. [Monitoring Dashboard](#monitoring-dashboard)
10. [Retraining and Maintenance](#retraining-and-maintenance)
11. [Production Code Package (`deployment/`)](#production-code-package-deployment)

---

## Overview

### What the POC Proved

The POC trained 150 Double DQN agents across 7 perishable food categories to make progressive markdown pricing decisions every 2 hours. The best approach (v2.1 pooled transfer learning) achieves 95% beats-baseline rate — 142 out of 150 SKUs outperform the best rule-based policy on reward (revenue minus waste).

### What Changes for Production

In the POC, a simulator generates demand stochastically. In production, **the simulator is removed entirely**. The agent operates as an online RL system:

```
POC:     Agent  -->  Simulator  -->  Synthetic demand  -->  Agent learns
Prod:    Agent  -->  Real pricing decision  -->  Real sales observed  -->  Agent learns
```

The core algorithm (Double DQN, action masking, PER, n-step returns, reward shaping) stays identical. What changes is the data source: real POS transactions and WMS inventory replace the simulated `_demand_model()`.

### Online RL Loop

Every 2 hours, for each active markdown SKU:

```
                    +-------------------+
                    |   Markdown Clock  |
                    |  (every 2 hours)  |
                    +--------+----------+
                             |
                             v
                  +----------+----------+
                  |  Read Current State  |
                  |  (WMS + POS + clock) |
                  +----------+----------+
                             |
                             v
                  +----------+----------+
                  |   Model Inference    |
                  |  Q(s) -> action mask |
                  |  -> discount level   |
                  +----------+----------+
                             |
                             v
                  +----------+----------+
                  |   Apply Discount     |
                  |  (update POS/website)|
                  +----------+----------+
                             |
                     2 hours pass...
                     sales happen
                             |
                             v
                  +----------+----------+
                  |  Observe Outcome     |
                  |  (units sold, rev)   |
                  +----------+----------+
                             |
                             v
                  +----------+----------+
                  |  Build Transition    |
                  |  (s, a, r, s', mask) |
                  +----------+----------+
                             |
                             v
                  +----------+----------+
                  |  Store + Train Step  |
                  |  (replay buffer +    |
                  |   gradient update)   |
                  +----------+----------+
```

---

## Code Mapping: POC to Production

### Reuse As-Is

These modules contain the core RL algorithm and are used directly in production without modification.

| Module | What It Does | Production Role |
|--------|-------------|-----------------|
| `fresh_rl/dqn_agent.py` | `DQNAgent`, `NStepAccumulator`, `ReplayBuffer` | Core agent: `select_action()` for inference, `store_transition()` + `train_step_fn()` for learning, `load_pretrained()` for cold start, `save()`/`load()` for checkpoints |
| `fresh_rl/prioritized_replay.py` | `PrioritizedReplayBuffer` | Replay buffer with priority sampling |
| `fresh_rl/sumtree.py` | `SumTree` data structure | Internal to PER buffer |
| `fresh_rl/baselines.py` | 7 rule-based policies | `BackloadedProgressive` as fallback policy on model failure |
| `fresh_rl/product_catalog.py` | `get_product_features()`, `CATEGORIES` | Product feature computation (state[10-13]) and normalization ranges |

### Not Used in Production

These modules are the simulator — they are replaced by real-world data and systems.

| Module | POC Role | Production Replacement |
|--------|----------|----------------------|
| `fresh_rl/environment.py` | `MarkdownChannelEnv` — simulates demand, inventory, rewards | Real POS sales + WMS inventory. Constants (`DISCOUNT_LEVELS_BY_STEP`) still referenced for the action-to-discount mapping. |
| `fresh_rl/pooled_env.py` | `PooledCategoryEnv`, `AugmentedProductEnv` — wraps simulator with 14-dim state | State constructor service builds 14-dim vector directly from live data |
| `fresh_rl/historical_data.py` | `HistoricalDataGenerator` — generates synthetic transitions from baseline rollouts | Historical ETL pipeline converts real past sessions to transition tuples |

### Scripts: POC vs Production

| POC Script | What It Does | Production Equivalent |
|-----------|-------------|----------------------|
| `scripts/train.py` | Single-SKU training loop: prefill → warmup → online episodes against simulator | **Training service** — same loop structure, but transitions come from real outcomes instead of `env.step()`. Key functions to port: prefill logic (lines 139-156), warmup loop (lines 159-175), the per-step `store_transition()` + `train_step_fn()` pattern (lines 226-235). |
| `scripts/run_portfolio.py` | Parallel training of 150 SKUs with `ProcessPoolExecutor` | **Orchestrator** — schedules training across SKUs. The `_train_and_eval_product()` worker function is the template for per-SKU lifecycle. Transfer learning init via `--pooled-tl` mode (loads pooled weights, wraps env for 14-dim state) maps to the cold-start path. |
| `scripts/evaluate.py` | `evaluate_policy()` — greedy rollouts against simulator baselines | **Offline evaluation** — adapt to replay historical sessions through the model (not counterfactual, but "what would the agent have recommended?"). The function signature and metrics collection are reusable. |
| `scripts/visualize.py` | Portfolio plots from `portfolio_results.json` | **Monitoring dashboard** — same metrics (reward, waste rate, clearance, per-category breakdown) but fed from live data instead of JSON files. |

### New Code to Build

These components don't exist in the POC and need to be created for production.

| Component | Responsibility | Key Interfaces |
|-----------|---------------|----------------|
| **State constructor** | Builds 14-dim vector from live WMS/POS/clock data. Replaces `env._get_obs()` + `AugmentedProductEnv`. | Input: session state (inventory, sales history, timestamps). Output: `np.float32[14]`. See [State Vector Construction](#state-vector-construction) for exact formulas. |
| **Historical ETL pipeline** | Converts historical markdown session CSVs/DB rows to `(s, a, r, s', done, mask)` tuples. Replaces `HistoricalDataGenerator.generate()`. | Input: session records (Section 3B format). Output: transition tuples fed to `agent.store_transition()`. See [Historical Data to Replay Buffer](#historical-data-to-replay-buffer) for pseudocode. |
| **Pricing service** | Inference endpoint called every 2h per active session. Reads state, runs `agent.select_action()`, returns discount level. | Input: SKU + session context. Output: discount index (0-10). Wraps `DQNAgent.select_action()` with action mask enforcement and fallback. |
| **Session manager** | Tracks active markdown sessions: step count, current discount, cumulative sales, start time. | Maintains per-session state needed by the state constructor. Replaces the `env.step_count`, `env.current_discount_idx`, `env.recent_sales` tracking that the simulator did internally. |
| **Training service** | Receives observed outcomes, builds transitions, calls `agent.store_transition()` + `agent.train_step_fn()`. | Runs asynchronously after each 2h observation. Handles reward computation (revenue - holding - waste), action mask construction, and model checkpointing. |
| **Markdown trigger** | Scans inventory/expiry systems for products entering markdown. Initiates new sessions. | Input: batch expiry data from WMS. Output: new session events to the pricing service. |
| **Monitoring and alerting** | Collects metrics (reward, waste, clearance, loss, epsilon), triggers alerts on threshold violations. | See [Monitoring Dashboard](#monitoring-dashboard) for metric definitions and thresholds. |

### Reuse Summary

```
POC Codebase (fresh_rl/)              Production
================================       ================================
dqn_agent.py ──────────────────────>   Used directly (agent core)
prioritized_replay.py ─────────────>   Used directly (PER buffer)
sumtree.py ────────────────────────>   Used directly (PER internals)
baselines.py ──────────────────────>   Fallback policy only
product_catalog.py ────────────────>   get_product_features() + ranges

environment.py ─────────── X ──────>   Replaced by real world + state constructor
pooled_env.py ──────────── X ──────>   Replaced by state constructor
historical_data.py ─────── X ──────>   Replaced by historical ETL pipeline

scripts/train.py ──────── adapt ───>   Training service (same loop, real data)
scripts/run_portfolio.py ─ adapt ───>   Orchestrator (scheduling, TL init)
scripts/evaluate.py ────── adapt ───>   Offline evaluation (replay-based)
scripts/visualize.py ───── adapt ───>   Monitoring dashboard (live metrics)

                           NEW         State constructor
                           NEW         Historical ETL pipeline
                           NEW         Pricing service (inference)
                           NEW         Session manager
                           NEW         Training service
                           NEW         Markdown trigger
                           NEW         Monitoring + alerting
```

---

## Data Requirements

Three categories of data, each with concrete examples.

### A. Product Master Data

Static per-SKU table, updated when products are added or changed. Maps directly to the product catalog (`fresh_rl/product_catalog.py`).

**Example CSV: `product_master.csv`**

| sku_id | sku_name | category | base_price | cost_per_unit | initial_inventory | pack_size | markdown_window_hours |
|--------|----------|----------|-----------|---------------|-------------------|-----------|----------------------|
| SEA-001 | salmon_fillet | seafood | 12.93 | 6.21 | 10 | 1 | 24 |
| DAI-001 | yogurt_greek_plain | dairy | 5.68 | 2.08 | 15 | 1 | 24 |
| BAK-001 | sourdough_loaf | bakery | 3.44 | 1.30 | 22 | 1 | 24 |

**Field mapping to POC:**

| CSV Field | POC Parameter | Used In |
|-----------|--------------|---------|
| `base_price` | `env.base_price` | Revenue calc, state normalization |
| `cost_per_unit` | `env.cost_per_unit` | Waste penalty, reward shaping scale |
| `initial_inventory` | `env.initial_inventory` | Inventory normalization, episode setup |
| `pack_size` | `catalog["_pack_size"]` | Product feature [3] (state[13] in 14-dim) |
| `markdown_window_hours` | `env.markdown_window_hours` | Episode length (24h = 12 steps at 2h) |
| `category` | `catalog["_category"]` | Pooled model selection, feature normalization ranges |

**Notes:**
- `initial_inventory` is the quantity placed on the markdown channel at the start of each session (batch size), not store-wide stock.
- `base_price` is the regular (non-markdown) shelf price. Markdown discounts are applied on top of this.
- All 150 POC products use 24h markdown windows. Production could support other durations by adjusting `episode_length`.

### B. Historical Markdown Sessions

Per-session, per-step transaction records. This is what replaces the POC's `HistoricalDataGenerator` — real past markdown events used to pre-fill the replay buffer.

**Example CSV: `markdown_sessions.csv`**

A complete 24h markdown session for `salmon_fillet` at 2h step intervals (12 steps):

| session_id | sku_name | step | timestamp | discount_pct | discount_idx | units_sold | inventory_before | inventory_after | revenue | day_of_week | time_block |
|-----------|----------|------|-----------|-------------|-------------|-----------|-----------------|----------------|---------|-------------|-----------|
| S-20260301-SEA001 | salmon_fillet | 0 | 2026-03-01 08:00 | 20 | 0 | 1 | 10 | 9 | 10.34 | 6 | 4 |
| S-20260301-SEA001 | salmon_fillet | 1 | 2026-03-01 10:00 | 20 | 0 | 2 | 9 | 7 | 20.69 | 6 | 5 |
| S-20260301-SEA001 | salmon_fillet | 2 | 2026-03-01 12:00 | 25 | 1 | 1 | 7 | 6 | 9.70 | 6 | 6 |
| S-20260301-SEA001 | salmon_fillet | 3 | 2026-03-01 14:00 | 30 | 2 | 2 | 6 | 4 | 18.10 | 6 | 7 |
| S-20260301-SEA001 | salmon_fillet | 4 | 2026-03-01 16:00 | 30 | 2 | 1 | 4 | 3 | 9.05 | 6 | 8 |
| S-20260301-SEA001 | salmon_fillet | 5 | 2026-03-01 18:00 | 35 | 3 | 1 | 3 | 2 | 8.40 | 6 | 9 |
| S-20260301-SEA001 | salmon_fillet | 6 | 2026-03-01 20:00 | 40 | 4 | 1 | 2 | 1 | 7.76 | 6 | 10 |
| S-20260301-SEA001 | salmon_fillet | 7 | 2026-03-01 22:00 | 45 | 5 | 0 | 1 | 1 | 0.00 | 6 | 11 |
| S-20260301-SEA001 | salmon_fillet | 8 | 2026-03-02 00:00 | 50 | 6 | 0 | 1 | 1 | 0.00 | 0 | 0 |
| S-20260301-SEA001 | salmon_fillet | 9 | 2026-03-02 02:00 | 55 | 7 | 1 | 1 | 0 | 5.82 | 0 | 1 |
| S-20260301-SEA001 | salmon_fillet | 10 | 2026-03-02 04:00 | — | — | — | 0 | 0 | — | 0 | 2 |
| S-20260301-SEA001 | salmon_fillet | 11 | 2026-03-02 06:00 | — | — | — | 0 | 0 | — | 0 | 3 |

**Key observations in this example:**
- Discounts are progressive: 20% -> 20% -> 25% -> 30% -> ... -> 55% (never reverts)
- Inventory cleared at step 9 (sold last unit). Steps 10-11 are no-ops (zero inventory).
- Total: 10 units sold out of 10, revenue = $89.86, waste = 0 (cleared before deadline)
- `discount_idx` maps to the 2h discount levels: `[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]`
- `time_block` = hour-of-day / 2 (0-11), wraps at midnight
- `day_of_week` = 0 (Mon) through 6 (Sun)

**Discount index mapping (2h step configuration):**

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| Discount | 20% | 25% | 30% | 35% | 40% | 45% | 50% | 55% | 60% | 65% | 70% |

### C. Real-Time Signals (Inference Time)

What the pricing service reads every 2h to construct the current state vector.

| Signal | Source System | Example Value | Notes |
|--------|-------------|---------------|-------|
| Current inventory | WMS / Inventory DB | 4 units | Remaining unsold units on markdown channel |
| Units sold this step | POS / Transaction log | 2 units | Sales in the last 2h window |
| Sales last 3 steps | POS (rolling window) | [1, 2, 2] | For velocity calculation |
| Current discount index | Pricing service state | 3 (=35%) | Last discount applied |
| Cumulative units sold | POS (session total) | 6 units | For sell-through rate |
| Session start time | Markdown trigger log | 2026-03-01 08:00 | For hours-remaining calc |
| Current time | System clock | 2026-03-01 16:00 | For time-of-day and day-of-week encoding |

---

## State Vector Construction

The agent uses a 14-dimensional state vector. The first 10 features come from the environment observation (`_get_obs()`), and 4 product features are appended (`get_product_features()`).

All features are normalized to [0, 1].

### Full State Vector (14-dim)

**Worked example**: salmon_fillet, step 4 of the session above (16:00 Saturday, 4 units remaining, discount at 30%).

| Idx | Feature | Formula | Source | Example Value |
|-----|---------|---------|--------|---------------|
| 0 | `hours_remaining` | `(episode_length - step_count) * step_hours / markdown_window_hours` | Clock + session start | `(12-4)*2 / 24 = 0.667` |
| 1 | `inventory_remaining` | `inventory_remaining / actual_initial_inventory` | WMS | `4 / 10 = 0.400` |
| 2 | `current_discount_idx` | `current_discount_idx / (n_actions - 1)` | Pricing service state | `2 / 10 = 0.200` |
| 3 | `tod_sin` | `(sin(2*pi*time_block / 12) + 1) / 2` | Clock | `(sin(2*pi*8/12)+1)/2 = 0.067` |
| 4 | `tod_cos` | `(cos(2*pi*time_block / 12) + 1) / 2` | Clock | `(cos(2*pi*8/12)+1)/2 = 0.250` |
| 5 | `dow_sin` | `(sin(2*pi*day_of_week / 7) + 1) / 2` | Clock | `(sin(2*pi*6/7)+1)/2 = 0.109` |
| 6 | `dow_cos` | `(cos(2*pi*day_of_week / 7) + 1) / 2` | Clock | `(cos(2*pi*6/7)+1)/2 = 0.812` |
| 7 | `recent_velocity` | `min(mean(sales_last_3) / (initial_inventory * 0.5), 1)` | POS (rolling) | `min(mean([2,1,2]) / 5.0, 1) = 0.333` |
| 8 | `sell_through_rate` | `min((total_sold / step_count) / (initial_inventory / episode_length), 1)` | POS (cumulative) | `min((6/4)/(10/12), 1) = 1.000` |
| 9 | `projected_clearance` | `min(velocity * remaining_steps / inventory_remaining, 1)` | Derived | `min(1.67*8/4, 1) = 1.000` |
| 10 | `price_norm` | `(base_price - cat_price_min) / (cat_price_max - cat_price_min)` | Product master | `(12.93-7.0)/(20.0-7.0) = 0.456` |
| 11 | `cost_frac_norm` | `(cost_frac - cat_cfrac_min) / (cat_cfrac_max - cat_cfrac_min)` | Product master | `(0.480-0.45)/(0.60-0.45) = 0.202` |
| 12 | `inventory_norm` | `(initial_inv - cat_inv_min) / (cat_inv_max - cat_inv_min)` | Product master + WMS | `(10-8)/(20-8) = 0.167` |
| 13 | `pack_size_norm` | `(pack_size - cat_pack_min) / (cat_pack_max - cat_pack_min)` | Product master | `(1-1)/(2-1) = 0.000` |

**Feature groups:**

- **Time features** [0, 3-6]: Derived from clock and session start time. No external system needed beyond knowing when the session started.
- **Inventory features** [1, 7-9]: Require real-time WMS inventory counts and POS sales data.
- **Discount feature** [2]: The pricing service's own state (what discount is currently active).
- **Product features** [10-13]: Static per-SKU, computed once from product master data. Constant throughout a session.

### Normalization Ranges

Features [10-13] are normalized within **category-specific** ranges. These ranges come from the product catalog definition:

| Category | Price Range | Cost Frac Range | Inventory Range | Pack Size Range |
|----------|-------------|----------------|-----------------|-----------------|
| meats | $5.00 - $15.00 | 0.45 - 0.55 | 10 - 25 | 1 - 4 |
| seafood | $7.00 - $20.00 | 0.45 - 0.60 | 8 - 20 | 1 - 2 |
| vegetables | $1.50 - $5.00 | 0.30 - 0.45 | 15 - 35 | 1 - 4 |
| fruits | $2.00 - $7.00 | 0.30 - 0.45 | 15 - 30 | 1 - 6 |
| dairy | $1.50 - $6.00 | 0.25 - 0.40 | 15 - 35 | 1 - 2 |
| bakery | $2.50 - $7.00 | 0.30 - 0.45 | 12 - 25 | 1 - 12 |
| deli_prepared | $5.00 - $14.00 | 0.35 - 0.50 | 8 - 20 | 1 - 8 |

For production, these ranges should be set from actual category statistics and kept stable (recomputing ranges would shift all feature values and invalidate trained weights).

---

## Historical Data to Replay Buffer

Before deploying online, the agent is warm-started on historical markdown sessions. This replaces the POC's `HistoricalDataGenerator`, which ran simulated baseline rollouts.

### Pipeline Overview

```
+---------------------+     +---------------------+     +---------------------+
|  Historical Session  |     |   State Constructor  |     |  Transition Tuples   |
|  Records (CSV/DB)    |---->|   (14-dim vectors)   |---->|  (s, a, r, s', done, |
|                      |     |                      |     |   action_mask)       |
+---------------------+     +---------------------+     +----------+----------+
                                                                    |
                                                                    v
                                                         +----------+----------+
                                                         | agent.store_        |
                                                         |   transition()      |
                                                         | (applies shaping,   |
                                                         |  n-step, pushes     |
                                                         |  to PER buffer)     |
                                                         +----------+----------+
                                                                    |
                                                                    v
                                                         +----------+----------+
                                                         |   Warmup Training   |
                                                         |   (gradient steps   |
                                                         |    on buffer, no    |
                                                         |    env interaction)  |
                                                         +---------------------+
```

### Building Transition Tuples from Historical Data

For each consecutive pair of steps within a session, construct one transition:

```python
# Pseudocode: convert one session into transitions
def session_to_transitions(session_rows, product_master, category_ranges):
    """
    session_rows: list of dicts, one per step, sorted by step number
    product_master: dict with base_price, cost_per_unit, etc.
    category_ranges: dict with price_range, cost_frac_range, etc.
    """
    transitions = []
    product_features = compute_product_features(product_master, category_ranges)
    initial_inventory = session_rows[0]["inventory_before"]
    episode_length = 12  # 24h / 2h steps
    n_actions = 11       # 2h discount levels

    recent_sales = []
    total_sold = 0

    for t in range(len(session_rows) - 1):
        row = session_rows[t]
        next_row = session_rows[t + 1]

        # Skip if inventory was already 0
        if row["inventory_before"] == 0:
            continue

        # --- Build state s_t (14-dim) ---
        recent_sales.append(row["units_sold"])
        total_sold += row["units_sold"]
        velocity = mean(recent_sales[-3:])

        s_t = [
            (episode_length - row["step"]) * 2 / 24,         # [0] hours_remaining
            row["inventory_before"] / initial_inventory,       # [1] inventory_frac
            row["discount_idx"] / (n_actions - 1),            # [2] discount_frac
            (sin(2*pi*row["time_block"]/12) + 1) / 2,        # [3] tod_sin
            (cos(2*pi*row["time_block"]/12) + 1) / 2,        # [4] tod_cos
            (sin(2*pi*row["day_of_week"]/7) + 1) / 2,        # [5] dow_sin
            (cos(2*pi*row["day_of_week"]/7) + 1) / 2,        # [6] dow_cos
            min(velocity / (initial_inventory * 0.5), 1),     # [7] velocity
            sell_through_rate(total_sold, row["step"], ...),  # [8] sell_through
            projected_clearance(velocity, row, ...),          # [9] projected_clearance
            *product_features,                                # [10-13]
        ]

        # --- Build state s_{t+1} (14-dim) ---
        # Same construction using next_row values
        s_t1 = build_state(next_row, ...)

        # --- Compute reward ---
        price = product_master["base_price"] * (1 - row["discount_pct"] / 100)
        revenue = price * row["units_sold"]
        holding = 0.02 * price * row["inventory_after"] * (2 / 4)
        reward = revenue - holding

        # Terminal rewards (last step or inventory cleared)
        is_terminal = (next_row["inventory_before"] == 0) or (next_row["step"] >= episode_length)
        if is_terminal and next_row["inventory_after"] > 0:
            waste_cost = product_master["cost_per_unit"] * 3.0 * next_row["inventory_after"]
            reward -= waste_cost
        if next_row["inventory_before"] == 0 and row["step"] + 1 < episode_length:
            bonus = 1.0 * initial_inventory
            reward += bonus

        # --- Action mask for s_{t+1} ---
        if is_terminal:
            next_mask = [True] * n_actions
        else:
            next_mask = [False]*next_row["discount_idx"] + [True]*(n_actions - next_row["discount_idx"])

        transitions.append((s_t, row["discount_idx"], reward, s_t1, is_terminal, next_mask))

    return transitions
```

### Reward Computation — Worked Example

Using step 3 of the salmon_fillet session above (discount 30%, sold 2 units, 6 -> 4 inventory):

```
price       = 12.93 * (1 - 0.30) = 9.051
revenue     = 9.051 * 2 = 18.10
holding     = 0.02 * 9.051 * 4 * (2/4) = 0.36
step_reward = 18.10 - 0.36 = 17.74
```

At terminal step (if waste occurs with 3 unsold units):
```
waste_cost  = cost_per_unit * waste_penalty_multiplier * remaining
            = 6.21 * 3.0 * 3 = 55.89
```

Clearance bonus (if all inventory sold before deadline):
```
bonus = clearance_bonus * initial_inventory = 1.0 * 10 = 10.0
```

### Feeding Transitions into the Agent

Use `agent.store_transition()` (not `buffer.push()` directly) so that:

1. **Reward shaping** is applied consistently: `shaped_reward = reward + gamma * Phi(s') - Phi(s)`
2. **N-step accumulation** computes multi-step returns automatically via `NStepAccumulator`
3. **PER priorities** are set correctly for historical data

```python
# Set elevated priority for historical data (PER)
if hasattr(agent.replay_buffer, 'max_priority'):
    agent.replay_buffer.max_priority = 5.0

# Feed transitions
for s, a, r, s_next, done, next_mask in historical_transitions:
    agent.store_transition(s, a, r, s_next, done, next_mask)

# Restore normal priority
if hasattr(agent.replay_buffer, 'max_priority'):
    agent.replay_buffer.max_priority = 1.0
```

### Warmup Training

After filling the buffer, run gradient steps without environment interaction:

```python
for step in range(warmup_steps):  # typically 1000
    loss = agent.train_step_fn()
```

This gives the network a reasonable initialization before making real pricing decisions.

### Data Volume Guidelines

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Sessions per SKU | 20 | 100+ |
| Total transitions per SKU | ~200 | ~1,000+ |
| Buffer capacity | 10,000 | 50,000+ |
| Warmup gradient steps | 500 | 1,000 |

More historical data is always better, but diminishing returns set in around 100 sessions per SKU. The key is diversity — sessions from different days-of-week, different starting inventories, different discount strategies.

---

## Online RL Deployment Architecture

### Service Architecture

```
+------------------------------------------------------------------+
|                      Markdown Trigger                             |
|  (batch expiry scanner — identifies SKUs entering markdown)      |
+-----+------------------------------------------------------------+
      |  new markdown session: sku_id, initial_inventory, start_time
      v
+-----+------------------------------------------------------------+
|                      Pricing Service                              |
|                                                                   |
|  +------------------+    +-----------------+    +--------------+  |
|  | Session Manager  |    | Model Registry  |    | Action Mask  |  |
|  | (tracks active   |    | (per-SKU or     |    | Enforcer     |  |
|  |  sessions, step  |    |  pooled .pt     |    | (progressive |  |
|  |  counts, state)  |    |  checkpoints)   |    |  constraint) |  |
|  +--------+---------+    +--------+--------+    +------+-------+  |
|           |                       |                     |         |
|           v                       v                     v         |
|  +--------+----------+   +--------+---------+                     |
|  | State Constructor |   | DQN Forward Pass |                     |
|  | (14-dim vector    |-->| Q(s) -> argmax   |---> discount_level  |
|  |  from live data)  |   | over valid acts  |                     |
|  +-------------------+   +------------------+                     |
+-----+------------------------------------------------------------+
      |  discount_level applied
      v
+-----+------------------------------------------------------------+
|                    POS / Website                                  |
|  (price updated on markdown channel — customers see new price)   |
+-----+------------------------------------------------------------+
      |  2 hours pass... customers buy
      v
+-----+------------------------------------------------------------+
|                    Sales Logger                                   |
|  (POS transactions aggregated per 2h window per SKU)             |
+-----+------------------------------------------------------------+
      |  units_sold, revenue, inventory_after
      v
+-----+------------------------------------------------------------+
|                    Training Service                               |
|                                                                   |
|  +-------------------+    +------------------+    +------------+  |
|  | Transition Builder|    | Replay Buffer    |    | DQN Trainer|  |
|  | (s, a, r, s',     |--->| (PER, n-step     |--->| (gradient  |  |
|  |  done, mask)      |    |  accumulator)    |    |  update)   |  |
|  +-------------------+    +------------------+    +-----+------+  |
|                                                         |         |
|                                          model weights updated    |
|                                          (available next cycle)   |
+------------------------------------------------------------------+
```

### Decision Cycle (Every 2 Hours)

For each active markdown session:

**Inference path** (latency-sensitive, ~10ms):

1. Read current inventory from WMS
2. Read sales since last step from POS
3. Construct 14-dim state vector
4. Load model weights (cached in memory)
5. Forward pass: `Q(s)` -> 11 Q-values
6. Apply action mask: set Q-values for invalid actions (below current discount) to -inf
7. Return `argmax` -> discount level index
8. Update price on POS/website: `new_price = base_price * (1 - DISCOUNT_LEVELS[action])`

**Learning path** (not latency-sensitive, runs after observation):

1. Observe outcome: units_sold, revenue for the step just completed
2. Compute step reward: `revenue - holding_cost` (+ terminal penalties/bonuses at session end)
3. Build transition tuple: `(s_t, a_t, r_t, s_{t+1}, done, next_action_mask)`
4. Feed through `agent.store_transition()` (shaping + n-step + buffer)
5. Run one gradient step: `agent.train_step_fn()`
6. Soft-update target network (automatic, tau=0.005)

### Cold Start for New SKUs

When a new product enters the markdown channel for the first time:

```
+---------------------+     +---------------------+     +---------------------+
|  New SKU added to   |     |  Load pooled model  |     |  Zero-shot pricing  |
|  product master     |---->|  for its category   |---->|  (pooled model      |
|  (with category)    |     |  (e.g., seafood)    |     |   generalizes via   |
|                     |     |                     |     |   product features)  |
+---------------------+     +---------------------+     +----------+----------+
                                                                    |
                                                          accumulate sessions
                                                                    |
                                                                    v
                                                         +----------+----------+
                                                         |  Fine-tune per-SKU  |
                                                         |  model (pooled TL)  |
                                                         |  once ~50 sessions  |
                                                         |  are available      |
                                                         +---------------------+
```

The pooled model (v2, 78% beats-baseline with zero-shot) provides reasonable pricing from day one. After enough real sessions accumulate, transfer learning (v2.1 approach) fine-tunes a per-SKU model for that product.

---

## Deployment Sequence

### Phase 1: Data Collection and Validation

**Entry criteria:** Product master data loaded, POS/WMS integration available.

**Actions:**
1. Validate product master data completeness (all fields present for each SKU)
2. Set up the markdown session logger to capture all fields from Section 2B
3. Collect 4-8 weeks of historical markdown sessions under existing pricing rules
4. Validate data quality: no gaps in time series, inventory decrements match sales, progressive discount constraint holds
5. Build and test the state constructor against known sessions (verify 14-dim output matches expected values)

**Exit criteria:** 50+ sessions per SKU, data pipeline validated end-to-end.

**Duration:** 4-8 weeks (collecting data under existing rules).

### Phase 2: Historical Prefill and Warmup Training

**Entry criteria:** Phase 1 complete, validated historical data available.

**Actions:**
1. Run the historical-to-transition ETL pipeline (Section 4)
2. For each SKU: initialize a DQN agent, fill replay buffer, run warmup training
3. For new SKUs with no history: load pooled category model weights via `load_pretrained()`
4. Evaluate offline: compute reward on held-out historical sessions (replay, not counterfactual)
5. Sanity-check learned policies: do discount curves look reasonable? Does the agent differentiate between high-inventory and low-inventory states?

**Exit criteria:** All 150 agents initialized, offline reward within expected range, no degenerate policies (e.g., always max discount or always hold).

**Duration:** 1-2 days (compute + validation).

### Phase 3: Shadow Mode

**Entry criteria:** Phase 2 complete, agents producing sensible policies.

**Actions:**
1. Deploy pricing service in **read-only** mode
2. Every 2h, for each active markdown session:
   - Construct state, run inference, log recommended discount
   - Do NOT apply the recommendation — keep existing pricing rules
3. Compare RL recommendations vs actual decisions in logs
4. Continue online learning from actual (non-RL) pricing decisions
5. Monitor for anomalies: agent recommending extreme discounts, divergent Q-values

**Exit criteria:** RL recommendations are reasonable (within 1-2 levels of human decisions on average), no anomalies for 2+ weeks, learning curves stable.

**Duration:** 2-4 weeks.

### Phase 4: A/B Test on Subset

**Entry criteria:** Phase 3 complete, stakeholder approval.

**Actions:**
1. Select a test cohort: 15-30 SKUs (10-20%) across all 7 categories
2. Randomly assign each markdown session to control (existing rules) or treatment (RL agent)
3. RL agent makes real pricing decisions for treatment sessions
4. Track key metrics: revenue, waste rate, clearance rate, reward
5. Run for sufficient statistical power (depends on session volume)

**Exit criteria:** RL beats control on reward with p < 0.05, waste rate not significantly worse, no customer complaints about pricing.

**Duration:** 4-8 weeks.

### Phase 5: Full Rollout

**Entry criteria:** Phase 4 complete, positive A/B results, business approval.

**Actions:**
1. Gradually expand: 20% -> 50% -> 100% of SKUs over 2-4 weeks
2. Keep fallback to baseline policy ready (one config flag)
3. Monitor closely during ramp-up
4. Transition to steady-state monitoring (Section 8)

**Exit criteria:** All SKUs on RL pricing, metrics stable for 2+ weeks.

**Duration:** 2-4 weeks.

### Rollout Timeline

```
Week:  1----2----3----4----5----6----7----8----9---10---11---12---13---14---15---16
       [---- Phase 1: Data Collection ----]
                                           [P2]
                                            [---- Phase 3: Shadow Mode ----]
                                                                           [- Phase 4: A/B --->
                                                                                              [P5->
```

Total: ~12-16 weeks from start to full rollout (conservative). Can be compressed if historical data already exists.

---

## Safety and Guardrails

### Action Masking (Built-In)

The progressive constraint is enforced at the action selection level, not via reward penalties. Invalid actions (reverting to a shallower discount) are masked out before Q-value argmax. This is already implemented in the POC and carries over to production unchanged.

```python
mask = [False] * current_discount_idx + [True] * (n_actions - current_discount_idx)
q_values[~mask] = -inf
action = argmax(q_values)
```

### Exploration Rate (Epsilon)

| Context | Epsilon | Rationale |
|---------|---------|-----------|
| Fresh agent (no history) | 0.30 | Pooled model provides reasonable base, moderate exploration |
| After warmup from history | 0.05 - 0.10 | Already warm-started, mostly exploit |
| Steady state | 0.05 (floor) | `EPSILON_FLOOR=0.05` prevents over-exploitation in production |
| After safety rollback | 0.15 | `EPSILON_DEGRADATION_RESET=0.15` — re-explore after degradation |
| Emergency override | 0.00 | Pure exploitation when stability is critical |

With `hold_action_prob = 0.5`, half of random explorations choose "hold current discount" rather than a random deeper cut. This prevents needlessly aggressive exploration.

The epsilon floor ensures the agent never stops exploring entirely, which is critical for adapting to demand distribution shifts. After a safety rollback, epsilon resets to 0.15 to increase exploration while the model recovers.

### Fallback Policy

If the RL model fails (inference error, NaN Q-values, model file missing):

1. Catch the exception in the pricing service
2. Fall back to `BackloadedProgressive` baseline (strongest baseline in POC)
3. Log the fallback event for investigation
4. Continue the session — the agent can resume RL decisions at the next step

```python
try:
    action = agent.select_action(state, action_mask)
except Exception:
    action = backloaded_progressive.select_action(state, env=session)
    log_fallback(session_id, error)
```

### Human Override

- **Per-session override**: Operator can lock a specific session to a manual discount schedule
- **Per-SKU disable**: Remove a SKU from RL pricing, revert to rules
- **Global kill switch**: Disable all RL pricing in one action, revert entire fleet to baseline

### Revenue and Waste Guardrails

| Guardrail | Threshold | Action |
|-----------|-----------|--------|
| Session waste rate | > 50% for 3 consecutive sessions | Alert, review SKU config |
| Category waste rate | > 30% rolling 7-day average | Alert, consider reverting category to baseline |
| Revenue per session | < 50% of baseline average | Alert, review learned policy |
| Discount too aggressive | Jump 3+ levels in one step | Block (impose max 2-level jump per step) |
| Q-value divergence | max Q > 1000 or NaN | Halt training, reload last checkpoint |

### Automatic Safety Rollback

Batch training includes automatic safety rollback based on model quality metrics. After each nightly training run, `should_rollback()` compares current and previous metrics:

| Check | Threshold | Action |
|-------|-----------|--------|
| Avg Q-value drop | > 30% (`ROLLBACK_Q_DROP_THRESHOLD`) | Restore `agent_prev.pt`, reset epsilon to 0.15 |
| Mean loss spike | > 3x (`ROLLBACK_LOSS_SPIKE_THRESHOLD`) | Restore `agent_prev.pt`, reset epsilon to 0.15 |

The rollback process:
1. Copy `agent_prev.pt` back to `agent.pt`
2. Copy `metrics_prev.json` back to `metrics.json`
3. Reset `agent.epsilon = EPSILON_DEGRADATION_RESET` (0.15)
4. Re-save checkpoint with reset epsilon
5. Log rollback event and reason in batch report

Rollback is safe because checkpoints are always rotated (`agent.pt` → `agent_prev.pt`) before saving new weights. The `rolled_back` flag in the batch report enables monitoring.

---

## Monitoring Dashboard

### Key Metrics

| Metric | Granularity | Alert Threshold | Rationale |
|--------|------------|-----------------|-----------|
| **Reward (rev - waste)** | Per-SKU, per-category, portfolio | < baseline avg for 7 days | Primary performance metric |
| **Beats-baseline rate** | Per-category, portfolio | < 80% (rolling 30-day) | Overall health check |
| **Waste rate** | Per-SKU, per-category | > 30% category avg | Unsold inventory is direct cost |
| **Clearance rate** | Per-SKU | < 50% (rolling 30-day) | Low clearance means poor discount timing |
| **Average discount depth** | Per-category | > 55% avg across category | Agent may be too aggressive |
| **Revenue per unit** | Per-SKU | < cost_per_unit | Selling below cost consistently |
| **Epsilon** | Per-SKU | Stuck at > 0.20 after 1000 steps | Decay not working or agent reset |
| **Training loss** | Per-SKU | > 100 or NaN, or 3x spike from prev | Network diverging (auto-rollback triggers) |
| **Avg Q-value** | Per-SKU | Drop > 30% from previous run | Model confidence degraded (auto-rollback triggers) |
| **Replay buffer size** | Per-SKU | < 500 after 50 sessions | Buffer not filling (data pipeline issue) |
| **Safety rollbacks** | Per-SKU, portfolio | > 5% of SKUs rolled back in one night | Widespread training instability |
| **Inference latency** | Service-wide | p99 > 50ms | Model or data lookup too slow |
| **Fallback rate** | Service-wide | > 1% of decisions | Model errors too frequent |

### Recommended Views

1. **Portfolio heatmap**: 150 SKUs on y-axis, time on x-axis, color = reward relative to baseline
2. **Category summary**: 7 cards showing beats-baseline %, avg reward, waste rate
3. **Per-SKU drill-down**: Discount trajectory, inventory curve, cumulative reward, learning curve (loss)
4. **Alert feed**: Real-time stream of threshold violations

---

## Retraining and Maintenance

### Continuous Online Learning

The agent learns from every markdown session in production — there is no separate "retraining" job. Each 2h step produces a transition that goes into the replay buffer, and one gradient step runs immediately.

This means the model continuously adapts to:
- Seasonal demand shifts
- Changing customer behavior
- New competitor pricing
- Day-of-week and time-of-day patterns

### When to Retrain from Scratch

| Trigger | Action |
|---------|--------|
| Major product reformulation (new price point, new pack size) | Update product master, reset product features, warm-start from pooled model |
| Category-wide range restructure | Retrain pooled model on new range, re-run TL for affected SKUs |
| Model performance degraded for > 4 weeks despite online learning | Rebuild replay buffer from recent 8 weeks of data, retrain from scratch |
| Algorithm upgrade (new architecture, new features) | Full retrain on historical data with new code |

### Keeping Pooled Models Fresh

Run `--pooled-update` weekly (e.g., Sunday night) to retrain category-level pooled models from the last 30 days of cross-SKU session data. This ensures new SKUs cold-start from models trained on recent production data rather than stale POC models.

### Handling New SKUs

1. Add the SKU to the product master with correct category, price, cost, inventory, pack_size
2. Compute product features using category normalization ranges
3. Load the pooled category model: `agent.load_pretrained("pooled_{category}_plain_2h.pt")`
4. The pooled model provides zero-shot pricing from the first session (78% beats-baseline in POC)
5. After ~50 markdown sessions, nightly batch training automatically fine-tunes from pooled model (cold-start path)

### Handling Removed SKUs

1. Stop scheduling markdown sessions for the SKU
2. Archive the model checkpoint and replay buffer
3. Remove from active monitoring
4. No impact on other SKUs (models are independent)

### Seasonal Considerations

- **Holiday peaks**: Demand surges may cause the agent to under-discount (sees high velocity, holds). Consider temporarily increasing epsilon to 0.10-0.15 to encourage more exploration during novel demand patterns.
- **Post-holiday lulls**: Opposite effect. The agent adapts naturally through online learning, but monitor waste rates closely for 1-2 weeks after demand drops.
- **Seasonal products**: Products that only appear seasonally (e.g., holiday items) should be treated as new SKUs each season — warm-start from pooled model, fine-tune as sessions accumulate.
- **Inventory planning changes**: If `initial_inventory` changes significantly (e.g., store reduces markdown batch sizes), update the product master. The inventory_norm feature will shift, but the model adapts through online learning.

---

## Production Code Package (`deployment/`)

The `deployment/` directory contains a complete production library implementing the architecture described above. It imports from the POC `fresh_rl/` package (DQNAgent, baselines, product_catalog) but never modifies it.

### Package Structure

```
deployment/
  __init__.py          — Package exports
  config.py            — Constants, DQN defaults, ProductionConfig (path management)
  state.py             — StateConstructor: builds 14-dim state from session data
  session.py           — SessionManager + ActiveSession: daytime session tracking
  inference.py         — PricingAgent: model loading with 3-tier fallback
  etl.py               — SessionETL: session CSV → transition tuples
  batch_train.py       — Nightly batch training CLI
  data/
    sessions/{date}/   — Session CSVs: S-{date}-{sku}.csv
    product_master.csv — Product master data
  models/
    {sku_name}/        — Per-SKU checkpoints (agent.pt, agent_prev.pt, metrics.json, buffer.pt)
    _pooled/{category}/— Pooled category models (auto-updated weekly with --pooled-update)
  logs/
    batch_train_{date}.log   — Training logs
    batch_report_{date}.json — Training reports
```

### Mapping to Architecture

| Architecture Component | Implementation | Module |
|----------------------|---------------|--------|
| State constructor | `StateConstructor.build_state()` | `deployment/state.py` |
| Session manager | `SessionManager` + `ActiveSession` | `deployment/session.py` |
| Pricing service (inference) | `PricingAgent.get_discount()` | `deployment/inference.py` |
| Historical ETL pipeline | `SessionETL.session_to_transitions()` | `deployment/etl.py` |
| Training service | `train_single_sku()` | `deployment/batch_train.py` |
| Safety rollback | `should_rollback()` | `deployment/batch_train.py` |
| Pooled model update | `train_pooled_category()` | `deployment/batch_train.py` |
| Model registry | `ProductionConfig` path methods | `deployment/config.py` |
| Action mask enforcer | `ActiveSession.action_masks()` | `deployment/session.py` |

### Daytime Usage (Library API)

The caller's code (pricing service, scheduler) drives the daytime loop:

```python
from datetime import datetime
from deployment import ProductionConfig, SessionManager, PricingAgent

config = ProductionConfig(base_dir="deployment")
config.ensure_dirs()

manager = SessionManager(config)

# 1. Start a session when a product enters markdown
session = manager.start_session(
    session_id="S-20260301-salmon",
    sku_name="salmon_fillet",
    category="seafood",
    initial_inventory=10,
    base_price=12.93,
    cost_per_unit=6.21,
    product_features=np.array([0.456, 0.202, 0.167, 0.000]),
    start_time=datetime(2026, 3, 1, 8, 0),
)

# 2. Load the pricing model
agent = PricingAgent(
    sku_name="salmon_fillet",
    category="seafood",
    base_price=12.93,
    cost_per_unit=6.21,
    initial_inventory=10,
    config=config,
)
source = agent.load_model()  # per-sku → pooled → baseline fallback

# 3. Every 2 hours: get state → get discount → apply → observe → record
state = manager.get_current_state("S-20260301-salmon")
mask = session.action_masks()
decision = agent.get_discount(state, mask)
# decision = {"discount_idx": 2, "discount_pct": 0.30, "price": 9.05, ...}

# ... caller applies price, waits 2h, observes sales ...

manager.record_step(
    session_id="S-20260301-salmon",
    discount_idx=decision["discount_idx"],
    units_sold=2,
    inventory_after=8,
    revenue=18.10,
    timestamp=datetime(2026, 3, 1, 10, 0),
)
```

### Nighttime Batch Training (CLI)

After the day's sessions complete, run batch training:

```bash
# Train all SKUs on last 30 days of data
python -m deployment.batch_train --date 2026-03-01 --workers 16

# Train a single SKU
python -m deployment.batch_train --date 2026-03-01 --sku salmon_fillet

# Custom lookback window
python -m deployment.batch_train --date 2026-03-01 --lookback-days 14 --workers 8

# Weekly: train all SKUs + retrain pooled category models
python -m deployment.batch_train --date 2026-03-01 --workers 16 --pooled-update

# Custom pooled lookback window
python -m deployment.batch_train --date 2026-03-01 --workers 16 --pooled-update --pooled-lookback-days 60
```

**What batch training does for each SKU:**

1. Finds all session CSVs in `data/sessions/{date}/S-*-{sku}.csv` for the lookback window
2. ETL: converts sessions → `(s, a, r, s', done, mask)` transition tuples
3. Loads existing checkpoint, or cold-starts from pooled category model, or creates fresh agent
4. Loads persisted replay buffer (`buffer.pt`) if available — preserves PER priorities across runs
5. Feeds transitions into replay buffer (with elevated PER priority)
6. Runs `training_steps_per_session * n_sessions` gradient steps, tracking losses
7. Decays epsilon once per batch run, clamped to `EPSILON_FLOOR` (0.05)
8. Computes model metrics: `avg_q_value` (mean max Q from sampled states) and `mean_loss` (last 100 steps)
9. Rotates checkpoint: `agent.pt` → `agent_prev.pt` and `metrics.json` → `metrics_prev.json`
10. Saves new checkpoint, replay buffer (`buffer.pt`), and metrics (`metrics.json`)
11. **Safety rollback check**: if avg_q drops >30% or loss spikes >3x, restores previous checkpoint and resets epsilon to 0.15
12. Writes JSON report to `logs/batch_report_{date}.json` (includes metrics, rollback info)

**Weekly pooled model update** (`--pooled-update`):

After per-SKU training, retrains each category's pooled model from cross-SKU session data:
1. Groups SKUs by category
2. Collects all transitions across SKUs in each category (with product features)
3. Trains category-level agent for `POOLED_EPISODES_PER_SKU * n_skus` steps
4. Rotates pooled model: `pooled_{cat}_plain_2h.pt` → `pooled_{cat}_plain_2h_prev.pt`
5. Saves updated pooled model (used for cold-starting new SKUs)

### Product Master CSV Format

Place at `deployment/data/product_master.csv`:

```csv
sku_name,category,base_price,cost_per_unit,initial_inventory,pack_size,markdown_window_hours
salmon_fillet,seafood,12.93,6.21,10,1,24
yogurt_greek_plain,dairy,5.68,2.08,15,1,24
sourdough_loaf,bakery,3.44,1.30,22,1,24
```

### Model Fallback Order

`PricingAgent.load_model()` uses a 3-tier fallback:

1. **Per-SKU checkpoint** (`models/{sku}/agent.pt`) — best, from batch training
2. **Pooled category model** (`models/_pooled/{cat}/pooled_{cat}_plain_2h.pt`) — zero-shot
3. **BackloadedProgressive baseline** — rule-based, no model needed

During inference, any DQN error (NaN Q-values, load failure) automatically falls back to the baseline with logging.
