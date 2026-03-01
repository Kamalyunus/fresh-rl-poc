# Experiment Log — Fresh RL POC

Chronological record of iterations, results, and learnings from developing the markdown channel RL agent.

---

## Iteration 1: Baseline DQN (4h mode, 5 products)

**Commit**: `872422b` (initial) → basic DQN

**Setup**:
- 4h decision steps (6 actions: 20%-70%), 6-step episodes for 24h products
- 5 hand-crafted product profiles (salad_mix, fresh_chicken, yogurt, bakery_bread, sushi)
- Vanilla DQN with experience replay, target network, action masking
- Potential-based reward shaping using fixed urgency signal: `Phi(s) = -5.0 * inv * (1 - time_remaining)`
- 7 rule-based baselines for comparison

**Results**:
- DQN learns to outperform most baselines on reward
- Reward shaping with a fixed scale works for some products but not others — the `Phi=-5.0` constant overwhelms low-price products (salad at $3.50) while barely registering for high-price ones (sushi at $10.00)

**Learning**: Fixed shaping scale doesn't generalize across products with different revenue scales.

---

## Iteration 2: Prioritized Experience Replay + Historical Pre-fill

**Commit**: `872422b`

**Changes**:
- Implemented SumTree-based Prioritized Experience Replay (PER)
- Added historical data generator that rolls out baseline policies to pre-fill the replay buffer
- Warmup phase: N gradient steps on buffered data before online training begins

**Setup**: 500 episodes, PER enabled, 200 pre-fill episodes, 1000 warmup steps

**Results**:
- PER + pre-fill significantly accelerates early learning — the agent starts with meaningful experiences instead of random exploration
- Warmup loss drops from ~1000 to ~70-130 depending on product
- Combined with PER, the agent converges faster to good policies

**Learning**: Pre-filling the buffer with baseline demonstrations gives the agent a "curriculum" — it sees what reasonable behavior looks like before exploring on its own.

---

## Iteration 3: Double DQN, Soft Targets, Revenue-Normalized Shaping

**Commit**: `cd1d87f`

**Changes**:
- **Double DQN**: Online network selects actions, target network evaluates — reduces Q-value overestimation
- **Soft target updates**: `tau=0.005` polyak averaging instead of hard periodic copies
- **Revenue-normalized reward shaping**: `waste_cost_scale = shaping_ratio * base_price * initial_inventory` with `shaping_ratio=0.2`, making the shaping signal proportional to 20% of expected revenue regardless of product price

**Results**:
- Double DQN + soft targets provide more stable training curves
- Revenue-normalized shaping (ratio=0.2) is product-agnostic — works consistently across the $2.50-$10.00 price range

**Learning**: The key insight was normalizing the shaping scale to the product's revenue scale. A universal `shaping_ratio=0.2` means "waste avoidance is worth 20% of expected revenue" — meaningful for both cheap salad and expensive sushi.

---

## Iteration 4: Product Catalog (110 SKUs) + Portfolio Runner

**Commit**: `1489509`

**Goal**: Validate that `shaping_ratio=0.2` works universally, not just on 5 hand-picked products.

**Changes**:
- Created product catalog: 7 categories x 15 SKUs = 105 generated + 5 legacy = 110 total
- Categories: meats, seafood, vegetables, fruits, dairy, bakery, deli_prepared
- Each SKU has seeded-random but reproducible economics (price, cost ratio, elasticity, demand, inventory, markdown window)
- Environment falls back to catalog lookup for unknown product names
- Portfolio runner: trains plain + shaped DQN for every SKU, evaluates vs baselines, produces aggregate report

**Results (4h mode, 500 episodes, PER + prefill + warmup)**:

| Metric | Value |
|--------|-------|
| Shaping wins | 32/110 (29%) |
| Avg DQN waste | 0.6% |
| DQN rank | #1 of 9 policies |
| Avg revenue delta (shaped vs plain) | -0.08% (neutral) |

Both DQN variants are #1 overall, beating all baselines on reward. But shaping barely differentiates from plain DQN because **waste is already near-zero (0.6%)** — the problem is too easy for shaping to help.

**Analysis — Why shaping doesn't help when it's easy**:

| Plain DQN Waste Rate | # SKUs | Shaping Win Rate |
|----------------------|--------|------------------|
| 0% (zero waste) | 84 | 30% |
| 0-1% | 8 | 38% |
| 1-5% | 14 | 14% |
| >5% | 4 | 50% |

When waste is already zero, shaping is just noise. The 4h/6-action environment with standard demand is "too solved" — the agent clears inventory trivially.

**Learning**: Reward shaping helps when the problem is hard enough that the agent struggles with waste. With 4h steps, 6 actions, and full demand, the DQN learns to clear inventory nearly perfectly even without shaping.

---

## Iteration 5: Hard Mode (2h, half demand, double inventory)

**Commit**: `b7f5549`

**Goal**: Create conditions where waste is a real challenge to test shaping's value.

**Changes**:
- Added `--demand-mult` and `--inventory-mult` flags to run_portfolio.py
- Added `--epsilon-decay` override (configurable per run instead of hardcoded)
- Added `env_overrides` parameter to train() for threading multipliers through

### Run A: 2h mode, 500 episodes (insufficient training)

**Setup**: 2h steps (11 actions), 0.5x demand, 2x inventory, 500 episodes, PER + prefill + warmup

| Metric | Easy (4h/500ep) | Hard (2h/500ep) |
|--------|-----------------|-----------------|
| Shaping win rate | 29% | **56%** |
| Avg DQN waste | 0.6% | 11.5% |
| DQN rank | #1 | **#6 of 9** |

Shaping win rate doubled (29% → 56%), confirming the hypothesis. But **DQN was badly losing to baselines** — 500 episodes isn't enough to learn 11 actions over 24-step episodes. Baselines don't need training so they win by default.

### Run B: 2h mode, 1500 episodes, slow epsilon decay

**Setup**: Same hard conditions + 1500 episodes + `epsilon_decay=0.999` (final epsilon ~0.22, more exploration)

| Policy | Avg Reward | Avg Waste | Avg Revenue |
|--------|-----------|-----------|-------------|
| **DQN Shaped** | **97.4** | **4.9%** | $98.8 |
| DQN Plain | 92.3 | 5.8% | $99.4 |
| Linear Progressive | 87.5 | 9.3% | $116.5 |
| Demand Responsive | 87.0 | 8.8% | $107.3 |
| Fixed 40% | 76.2 | 11.9% | $115.5 |

**DQN Shaped reclaimed #1 rank** with 5 points over Plain DQN and 10 points over the best baseline. Shaping win rate held at 55%.

**Category breakdown**:

| Category | Win% | Plain Waste | Shaped Waste | Key |
|----------|------|-------------|--------------|-----|
| seafood | **73%** | 32.5% | 27.5% | Hardest category — shaping saves ~$30 in reward |
| meats | **67%** | 3.3% | 1.4% | Shaping halves waste rate |
| fruits | **67%** | 0.0% | 0.0% | Shaping fine-tunes clearance timing |
| legacy | 60% | 4.2% | 4.4% | Mixed — slight edge on some products |
| bakery | 47% | 0.0% | 0.0% | Easy products, shaping is noise |
| vegetables | 47% | 0.0% | 0.0% | Same — already solved |
| deli_prepared | 40% | 5.1% | 5.4% | Shaping sometimes hurts here |
| dairy | 40% | 0.0% | 0.0% | Easy, no waste to optimize |

**Head-to-head vs baselines (DQN Shaped)**:

| Baseline | Shaped Wins | BL Wins | Notes |
|----------|-------------|---------|-------|
| Random | 96/110 | 14 | Dominated |
| Immediate Deep 70% | 90/110 | 20 | Dominated |
| Fixed 20% | 90/110 | 20 | Dominated |
| Backloaded Progressive | 64/110 | 46 | Win |
| Fixed 40% | 46/110 | 62 | Competitive |
| Linear Progressive | 38/110 | 72 | BL wins on revenue, DQN wins on waste |
| Demand Responsive | 34/110 | 74 | BL wins on revenue, DQN wins on waste |

Interesting: Linear Progressive and Fixed 40% generate more raw revenue ($116.5 and $115.5 vs $98.8), but DQN wins on total reward because it avoids waste far more effectively (4.9% vs 9.3% and 11.9%).

**Learning**: With harder conditions and sufficient training time:
1. Shaping provides a clear advantage (~5 reward points over plain DQN)
2. DQN beats all baselines on the metric that matters (reward = revenue - waste)
3. The advantage comes from waste management, not from generating more revenue
4. Slower epsilon decay (0.999 vs 0.998) is critical for 2h mode — the agent needs more exploration to cover the larger action space

---

## Iteration 6: PyTorch Migration, Code Cleanup, Visualization Suite

**Commit**: (v0.5.0)

**Goal**: Reduce code via open-source packages, strip backward compatibility, add comprehensive visualizations.

**Changes**:

### PyTorch migration (`dqn_agent.py`)

Replaced the 106-line hand-rolled `NumpyMLP` (manual forward/backward/Adam/He init) with a PyTorch `nn.Sequential` model:

| Before (NumPy) | After (PyTorch) |
|----------------|-----------------|
| Manual He initialization | `nn.init.kaiming_normal_()` |
| Manual forward pass with cached activations | `nn.Sequential` forward pass |
| Manual backpropagation (chain rule, per-layer) | `loss.backward()` |
| Manual Adam optimizer state (`m`, `v`, `t`) | `torch.optim.Adam` |
| Manual gradient clipping (`np.clip`) | `clip_grad_norm_(params, 1.0)` |
| `pickle.dump` / `pickle.load` | `torch.save` / `torch.load` |

The algorithm is identical (Double DQN + PER + action masking + reward shaping). Exact numbers differ due to PyTorch vs NumPy floating point paths, but the same quality of results is preserved.

Net reduction: ~383 → ~220 lines in `dqn_agent.py`.

### Backward compatibility removal

- **Removed vanilla DQN code path** (`if not self.double_dqn` branch) — Double DQN is always on
- **Removed hard target update path** (`elif self.train_step % self.target_update_freq == 0`) — soft updates (tau=0.005) always on
- **Removed `--no-double-dqn` and `--soft-target-tau` CLI flags** from `train.py` and `run_portfolio.py`
- **Removed `PRODUCT_PROFILES` dict** from `environment.py` — all products resolve via `product_catalog.get_profile()`
- **Deleted `scripts/ab_test.py`** (217 lines) — 4h vs 2h decision settled at Iteration 5
- **Deleted `scripts/run_all.py`** (132 lines) — `run_portfolio.py --products <name>` does the same

### Visualization suite expansion (`visualize.py`)

Added 6 new plot types beyond the existing 3 (training curves, comparison bars, action distributions):

| Plot | Purpose |
|------|---------|
| `plot_policy_heatmap` | 2D grid of learned discount decisions across hours × inventory |
| `plot_episode_walkthrough` | 4-panel trace of a single greedy episode |
| `plot_revenue_waste_pareto` | Revenue vs waste tradeoff scatter for all policies |
| `plot_training_dashboard` | 3-panel: loss curve, epsilon decay, greedy eval metrics |
| `plot_category_heatmap` | Category × metrics heatmap from portfolio results |
| `plot_action_progression` | Discount escalation over 50 episodes with mean/std band |

Net expansion: ~287 → ~650 lines in `visualize.py`.

### Other changes

- Added `torch>=2.0.0` and `seaborn>=0.12.0` to `requirements.txt`
- Added `losses` and `epsilon_decay` fields to training history JSON
- Agent checkpoints saved as `.pt` (PyTorch) instead of `.pkl` (pickle)
- Version bumped to 0.5.0

**Learning**: PyTorch eliminates ~160 lines of manual neural network code while keeping the RL algorithm identical. The main risk was subtle floating-point differences, but the overall quality of results is preserved. Removing backward compat makes the codebase easier to reason about — there's now only one path through every decision point.

---

## Iteration 7: Expanded State Space — Cyclical Time Encoding + Sell-Through Rate

**Goal**: Improve agent's ability to learn time-of-day and day-of-week demand patterns by replacing linear floats with cyclical sin/cos encoding, and add a sell-through rate feature so the agent can directly observe whether it's ahead or behind pace.

**Problem with 6-dim state**: Linear encoding of `time_of_day` (0.0–1.0) and `day_of_week` (0.0–1.0) puts adjacent times far apart — 11pm (0.92) and midnight (0.0) are maximally distant. The agent can't easily learn that these are neighbors. A single float also can't disambiguate symmetric positions on a cycle.

**Changes**:

### State expansion: 6-dim → 9-dim

| Index | Feature | Formula | Signal |
|-------|---------|---------|--------|
| 0 | `hours_remaining` | unchanged | Time pressure |
| 1 | `inventory_remaining` | unchanged | Stock level |
| 2 | `current_discount_idx` | unchanged | Price ladder position |
| 3 | `tod_sin` | `(sin(2π·tod/n_blocks) + 1) / 2` | Cyclical time-of-day |
| 4 | `tod_cos` | `(cos(2π·tod/n_blocks) + 1) / 2` | Cyclical time-of-day |
| 5 | `dow_sin` | `(sin(2π·dow/7) + 1) / 2` | Cyclical day-of-week |
| 6 | `dow_cos` | `(cos(2π·dow/7) + 1) / 2` | Cyclical day-of-week |
| 7 | `recent_velocity` | unchanged (moved from index 5) | Sales momentum |
| 8 | `sell_through_rate` | `(total_sold/step_count) / (initial_inv/episode_len)` | Am I ahead or behind pace? |

**Cyclical encoding**: Sin/cos pairs preserve adjacency — 11pm and midnight are neighbors in the encoded space. Two components disambiguate all positions (sin alone has symmetry). Uses only observable data (clock time, calendar day).

**Sell-through rate**: Ratio of actual units-sold-per-step to the ideal pace needed for full clearance. Value of 1.0 = on track, <1.0 = behind pace. Gives the agent a direct "pace" signal without requiring the MLP to learn the division between inventory and time features.

**No simulator leakage**: The agent sees clock time and calendar day (observable in production) but never sees the demand multipliers, price elasticity, or Poisson parameters. It must discover demand patterns through experience.

### Files modified

- `environment.py`: `_get_obs()` expanded to 9-dim, observation_space updated
- `baselines.py`: `DemandResponsive` velocity index updated (5 → 7)
- `visualize.py`: Synthetic observation in policy heatmap updated to 9-dim
- `ARCHITECTURE.md`: State table updated
- `README.md`: MDP formulation and results updated

No changes needed to `dqn_agent.py`, `train.py`, `evaluate.py`, `run_portfolio.py`, or replay buffers — all dynamically read `observation_space.shape[0]`.

### Results (v060 — 110 SKUs, hard mode, 16 workers)

**Setup**: 2h steps, 1500 episodes, PER + prefill + warmup, 0.5x demand, 2x inventory, epsilon_decay=0.999

| Metric | v050 (6-dim) | v060 (9-dim) | Delta |
|--------|-------------|-------------|-------|
| Shaping wins | 56/110 (51%) | 53/110 (48%) | -3 |
| Beats best baseline | 17/110 (15%) | 22/110 (20%) | **+5** |
| Run time | 72.6 min (1 worker) | 15.6 min (16 workers) | 4.7x faster |

**Category comparison (shaping win %)**:

| Category | v050 | v060 | Delta |
|----------|------|------|-------|
| meats | 33% | **73%** | **+40pp** |
| bakery | 40% | 47% | +7pp |
| seafood | 53% | 53% | same |
| fruits | 47% | 47% | same |
| vegetables | 73% | 60% | -13pp |
| deli_prepared | 67% | 47% | -20pp |
| dairy | 47% | 20% | -27pp |
| legacy | 40% | 20% | -20pp |

**Dominant baselines**: Demand Responsive (29 SKUs) and Backloaded Progressive (28 SKUs) are the strongest baselines, together winning 52% of SKUs. No baseline accesses simulator internals — they use only observable state (time, inventory, velocity).

**Analysis**:

1. **Beats-baseline rate improved** (15% → 20%) — the richer state helps the agent find better absolute policies
2. **Meats category jumped +40pp** — suggests cyclical features help where time-of-day demand patterns are strongest
3. **Shaping win rate dipped slightly** (51% → 48%) — the expanded state gives plain DQN more signal too, narrowing the shaping advantage
4. **Some categories regressed** (dairy -27pp, deli_prepared -20pp) — the larger state space may need more training budget to converge for certain profiles

**Learning**: Cyclical encoding + sell-through rate help the agent learn time-dependent strategies (meats: +40pp). The expanded state benefits both plain and shaped DQN, so the shaping advantage narrows. Categories where waste is already near-zero (dairy, vegetables) don't benefit from the additional features — the problem was already solved. Future work: try more episodes (2500+) for underperforming categories, or explore network architecture changes to better exploit the richer state.

---

## Iteration 8: Transfer Learning — Category Pre-training + Per-SKU Fine-tuning

**Goal**: Pool category-level experience to learn general timing/urgency patterns, then fine-tune per SKU with less training budget.

**Changes**:
- Added `load_pretrained()` method to `DQNAgent` — loads only network weights, ignores optimizer/epsilon/history
- Added `pretrained_path` parameter to `train()` — loads pre-trained weights and sets epsilon=0.3 for fine-tuning
- Added `--transfer-learning` and `--pretrain-episodes` flags to `run_portfolio.py`
- Added `_pretrain_category()` function — trains one agent per category by cycling through all products
- Two-phase orchestration in `main()`: category pre-training (parallel) → per-SKU fine-tuning (parallel)

**Design**:

Phase 1: Pre-train one agent per category on all products in that category (no reward shaping). Cycling through SKUs means the agent sees diverse demand/elasticity profiles within the category.

Phase 2: Load pre-trained weights into fresh agents, set epsilon=0.3 (less exploration needed), train for fewer episodes. Both plain and shaped variants fine-tuned.

**Compute savings**: With 8 categories × 1500 pretrain + 110 × 500 finetune = 67,000 episodes vs 110 × 1500 = 165,000 episodes without TL (2.5x fewer total episodes).

**Verification command**:
```bash
python scripts/run_portfolio.py --products salmon_fillet sushi salad_mix \
    --episodes 500 --eval-episodes 50 --step-hours 2 --per --prefill --warmup-steps 500 \
    --transfer-learning --pretrain-episodes 300 \
    --demand-mult 0.5 --inventory-mult 2.0 --save-dir results/tl_test
```

### Run A: v070 — TL + 500 fine-tuning episodes (compute-efficient)

**Setup**: 1500 pretrain episodes per category, 500 fine-tuning episodes per SKU, 16 workers. Total: 67,000 episodes.

| Metric | v060 (no TL, 1500ep) | v070 (TL, 500 finetune) |
|--------|---------------------|------------------------|
| Shaping wins | 53/110 (48%) | **65/110 (59%)** |
| Beats best baseline | 22/110 (20%) | 11/110 (10%) |
| Total episodes | 165,000 | 67,000 |
| Runtime (16 workers) | 15.6 min | **6.1 min** |

**Category breakdown (shaping win %)**:

| Category | v060 | v070 | Delta |
|----------|------|------|-------|
| seafood | 53% | **80%** | **+27pp** |
| bakery | 47% | **67%** | +20pp |
| deli_prepared | 47% | **67%** | +20pp |
| vegetables | 60% | **67%** | +7pp |
| fruits | 47% | **60%** | +13pp |
| dairy | 20% | **40%** | +20pp |
| meats | 73% | 40% | -33pp |
| legacy | 20% | **40%** | +20pp |

**Analysis**: Transfer learning dramatically boosts the shaping advantage (48% → 59%) because pre-trained weights give the shaped agent a head start that plain DQN can't match in only 500 episodes. However, beats-baseline dropped (20% → 10%) — 500 fine-tuning episodes isn't enough for absolute policy quality to converge against strong baselines.

### Run B: v071 — TL + 1500 fine-tuning episodes (full convergence)

**Setup**: 1500 pretrain + 1500 fine-tuning episodes per SKU. Total: 177,000 episodes.

| Metric | v060 (no TL) | v070 (500ft) | v071 (1500ft) |
|--------|-------------|-------------|---------------|
| Shaping wins | 53/110 (48%) | **65/110 (59%)** | 55/110 (50%) |
| Beats best baseline | 22/110 (20%) | 11/110 (10%) | **24/110 (22%)** |
| Runtime | 15.6 min | 6.1 min | 15.2 min |

**Category breakdown (shaping win %)**:

| Category | v060 | v070 | v071 |
|----------|------|------|------|
| vegetables | 60% | 67% | **73%** |
| fruits | 47% | 60% | 60% |
| meats | 73% | 40% | 47% |
| bakery | 47% | 67% | 47% |
| deli_prepared | 47% | 67% | 47% |
| dairy | 20% | 40% | 40% |
| seafood | 53% | 80% | 40% |
| legacy | 20% | 40% | 40% |

**Analysis**: With 1500 fine-tuning episodes, beats-baseline recovers to 22% (matching v060), but shaping win rate drops back to 50% as plain DQN also fully converges. The pre-trained representations help both variants equally given enough training budget.

### Transfer learning conclusions

1. **TL + short fine-tuning (500ep) maximizes shaping advantage**: 59% win rate, 2.5x compute savings, 2.5x faster runtime. Best config when shaping differentiation matters.
2. **TL + full fine-tuning (1500ep) matches no-TL on absolute quality**: 22% beats-baseline, same as v060. The pre-training doesn't hurt but the advantage narrows as both agents converge.
3. **The sweet spot is 500-800 fine-tuning episodes**: Enough for pre-trained weights to give shaped DQN an edge before plain DQN catches up.
4. **Seafood is most TL-sensitive**: Jumped from 53% to 80% with short fine-tuning (hardest category benefits most from pooled experience) but regressed to 40% with full fine-tuning.
5. **Vegetables consistently improved**: 60% → 67% → 73% across all three configs — TL reliably helps this category.

---

## Iteration 9: Bigger Network + More Episodes (hidden_dim=128, 3000 episodes)

**Goal**: Test whether a larger network and longer training can push beats-baseline higher. All 4 model variants (Plain, Shaped, Plain+TL, Shaped+TL) were tied at ~20-25% beats-baseline with hidden_dim=64 and 1500 episodes.

**Changes**:
- Threaded `hidden_dim` as a parameter through `train()` and `run_portfolio.py` (was hardcoded at 64)
- Added `--hidden-dim` CLI flag to `run_portfolio.py` (default: 64)

**Setup**: 2h steps, 3000 episodes, PER + prefill + warmup, 0.5x demand, 2x inventory, epsilon_decay=0.999, hidden_dim=128, 16 workers

### Run A: v080 — No TL (hidden_dim=128, 3000 episodes)

| Metric | v071 (64-dim, 1500ep) | v080 No TL (128-dim, 3000ep) |
|--------|----------------------|------------------------------|
| Shaping wins | 55/110 (50%) | **63/110 (57%)** |
| Beats best baseline | 24/110 (22%) | **47/110 (43%)** |
| Runtime (16 workers) | 15.2 min | 34.1 min |

**Category breakdown (shaping win %)**:

| Category | v071 | v080 No TL |
|----------|------|------------|
| legacy | 40% | **80%** |
| vegetables | 73% | 73% |
| bakery | 47% | **60%** |
| deli_prepared | 47% | **60%** |
| seafood | 40% | **60%** |
| meats | 47% | 47% |
| fruits | 60% | 47% |
| dairy | 40% | 47% |

### Run B: v080 — With TL (hidden_dim=128, 3000 episodes, 1500 pretrain)

| Metric | v080 No TL | v080 TL |
|--------|-----------|---------|
| Shaping wins | 63/110 (57%) | 60/110 (55%) |
| Beats best baseline | **47/110 (43%)** | 44/110 (40%) |
| Runtime (16 workers) | 34.1 min | 35.4 min |

**Category breakdown (shaping win %) — TL comparison**:

| Category | No TL | TL | Notes |
|----------|-------|----|-------|
| dairy | 47% | **67%** | TL helps most |
| vegetables | **73%** | 67% | |
| bakery | **60%** | 47% | TL hurts |
| deli_prepared | **60%** | 33% | TL hurts significantly |
| seafood | 60% | 60% | Same |
| fruits | 47% | **53%** | TL helps slightly |
| meats | 47% | 47% | Same |
| legacy | **80%** | 80% | Same |

### 4-way comparison (all v080)

| Variant | Beats Baseline | Shaping Wins |
|---------|---------------|-------------|
| **Plain (No TL)** | 43% | — |
| **Shaped (No TL)** | 43% | 57% |
| **Plain + TL** | 40% | — |
| **Shaped + TL** | 40% | 55% |

### Analysis

1. **Beats-baseline nearly doubled**: 22% → 43% from the combination of 2x network width (64→128) and 2x training (1500→3000 episodes). This is the single biggest improvement in beats-baseline across all iterations.
2. **No TL slightly edges TL** (43% vs 40%): With 3000 fine-tuning episodes, both agents fully converge anyway — TL's head start gets washed out. This confirms the pattern from Iteration 8: TL helps most with short fine-tuning budgets.
3. **TL helps dairy (+20pp) but hurts deli_prepared (-27pp)**: Category pre-training can introduce interference when category-level patterns don't transfer to individual SKUs.
4. **Shaping wins remain stable at ~55-57%**: The bigger network benefits both plain and shaped equally, so the relative shaping advantage doesn't grow.
5. **Rewards still climbing at episode 3000**: Training curves showed continued improvement past 2000 episodes, suggesting even more training could push beats-baseline higher.

**Learning**: Network capacity and training budget are the biggest levers for absolute policy quality. Doubling both (64→128 hidden, 1500→3000 episodes) had a much larger effect on beats-baseline (+21pp) than any algorithmic change (TL, shaping, state expansion). Transfer learning is most valuable when compute-constrained — with unlimited budget, direct training matches or beats it.

---

## Iteration 10: Sample Efficiency — Replay Ratio + Bigger Buffer/Batch

**Goal**: Match v0.8's 43% beats-baseline with half the episodes (1500 instead of 3000) by extracting more learning per episode via higher replay ratio, bigger batch, and bigger buffer.

**Changes**:
- Added `replay_ratio` parameter to `train()` and `run_portfolio.py` — gradient steps per environment step (default 1)
- Added `batch_size` parameter (default 32) — configurable instead of hardcoded
- Added `buffer_size` parameter (default 10000) — configurable instead of hardcoded
- Threaded all three through `_pretrain_category()`, `_run_single_product()`, CLI flags, header, and history JSON

**Hypothesis**: The training loop does exactly 1 gradient step per env step. Increasing to 4 means 4x more learning from each transition. Combined with bigger buffer (retain more diverse data) and bigger batch (more stable gradients with higher replay ratio), we can converge faster.

**Setup (v090)**: 2h steps, **1500 episodes** (half of v0.8), PER + prefill, 0.5x demand, 2x inventory, epsilon_decay=0.999, hidden_dim=128, 16 workers

| Setting | v0.8 | v0.9 |
|---------|------|------|
| Episodes | 3000 | **1500** |
| Replay ratio | 1 | **4** |
| Batch size | 32 | **64** |
| Buffer size | 10,000 | **50,000** |
| Prefill episodes | 200 | **500** |
| Warmup steps | 1000 | **2000** |

### Results

| Metric | v0.8 (3000ep, rr=1) | v0.9 (1500ep, rr=4) |
|--------|---------------------|---------------------|
| Shaping wins | 63/110 (57%) | 58/110 (53%) |
| Beats best baseline | **47/110 (43%)** | 16/110 (15%) |
| Runtime (16 workers) | 34.1 min | 66.8 min |

**Category breakdown (shaping win %)**:

| Category | v0.8 | v0.9 |
|----------|------|------|
| legacy | **80%** | **80%** |
| meats | 47% | **67%** |
| seafood | **60%** | **60%** |
| fruits | 47% | **53%** |
| bakery | **60%** | 47% |
| dairy | 47% | 47% |
| vegetables | **73%** | 47% |
| deli_prepared | **60%** | 40% |

### Analysis

1. **Beats-baseline collapsed**: 43% → 15%. Higher replay ratio did not compensate for halving online episodes. The agent needs real environment interactions to discover good strategies — replaying the same transitions more times has sharply diminishing returns.
2. **Shaping win rate held steady** (53% vs 57%): The shaped/plain comparison is robust across training budgets, confirming shaping provides a consistent relative advantage.
3. **Runtime nearly doubled** (34.1 → 66.8 min): 4x gradient steps per env step made each episode ~4x more expensive computationally, while halving episodes only saved 2x — net slowdown of ~2x.
4. **Meats improved** (+20pp shaping win rate) but vegetables (-26pp), bakery (-13pp), and deli_prepared (-20pp) all regressed significantly.

**Learning**: **Online exploration is the bottleneck, not sample efficiency.** Replay ratio increases how much you learn from existing data, but cannot substitute for collecting new, diverse experiences. The agent at 1500 episodes simply hasn't seen enough distinct environment trajectories to learn robust policies, regardless of how many gradient steps it takes on those trajectories. This is consistent with RL theory — replay ratio helps most when transitions are expensive to collect (robotics, real-world systems), but in fast simulators like ours, it's cheaper to just collect more data.

The replay ratio / batch size / buffer size parameters remain available for future use (e.g., if we move to real-world data collection), but for this simulator the clear winning formula is: **more episodes + bigger network** (Iteration 9).

---

## Iteration 11: N-Step Returns

**Goal**: Beat v0.8's 43% beats-baseline by using n-step returns to propagate rewards faster through short episodes (12-24 steps). N-step replaces single-step TD targets `r + gamma * Q(s')` with multi-step targets `G_n + gamma^n * Q(s_n)`.

**Changes**:
- Added `NStepAccumulator` class to `dqn_agent.py` — sits between `store_transition()` and replay buffer, computes n-step returns with correct episode boundary handling
- Added `n_step` parameter to `DQNAgent`, `train()`, and `run_portfolio.py` CLI (`--n-step`)
- Updated bootstrap discount in training step: `gamma` → `gamma^n_step`

**Hypothesis**: For 12-24 step episodes, n=5 propagates rewards across ~1/3 of the episode in a single update, improving credit assignment. Per Rainbow DQN ablations, n=4-8 is the sweet spot for short-horizon tasks.

**Setup (v1.0)**: 2h steps, 3000 episodes, PER + prefill, 0.5x demand, 2x inventory, epsilon_decay=0.999, hidden_dim=128, **n_step=5**, 16 workers

| Setting | v0.8 | v1.0 |
|---------|------|------|
| Episodes | 3000 | 3000 |
| Hidden dim | 128 | 128 |
| N-step | 1 | **5** |

### Results

| Metric | v0.8 (n_step=1) | v1.0 (n_step=5) |
|--------|-----------------|-----------------|
| Shaping wins | 63/110 (57%) | 52/110 (47%) |
| Beats best baseline | 47/110 (43%) | **51/110 (46%)** |
| Runtime (16 workers) | 34.1 min | 32.2 min |

**Category breakdown (shaping win %)**:

| Category | v0.8 | v1.0 |
|----------|------|------|
| legacy | **80%** | **80%** |
| dairy | 47% | **60%** |
| seafood | **60%** | **60%** |
| vegetables | **73%** | 60% |
| fruits | **47%** | **47%** |
| meats | **47%** | 40% |
| bakery | **60%** | 27% |
| deli_prepared | **60%** | 27% |

### Analysis

1. **New best beats-baseline**: 46% vs 43% (+3pp). N-step returns modestly improve the agent's ability to outperform rule-based baselines, confirming that faster credit assignment helps in short episodes.
2. **Shaping win rate dropped**: 57% → 47% (-10pp). N-step returns and reward shaping both accelerate reward propagation — their benefits partially overlap. With n-step already propagating rewards across ~5 steps, shaping's urgency signal adds less marginal value, and its noise hurts more categories (bakery 60%→27%, deli_prepared 60%→27%).
3. **Runtime slightly faster** (34.1 → 32.2 min): N-step accumulation adds negligible overhead; the slight speedup is within run-to-run variance.
4. **Dairy and vegetables improved** with n-step (dairy 47%→60%, vegetables stayed strong at 60%), while **bakery and deli_prepared regressed** sharply. These categories may have reward structures where multi-step bootstrapping introduces more bias than variance reduction.

**Learning**: **N-step returns provide a small but real improvement to beats-baseline (new best: 46%)**, but they partially substitute for reward shaping rather than compounding with it. The two techniques target the same bottleneck (slow reward propagation in short episodes) from different angles — n-step via algorithmic multi-step targets, shaping via potential-based reward augmentation. For maximum effect, future work should explore techniques that address *different* bottlenecks (e.g., better exploration, distributional RL, or curriculum learning).

---

## Iteration 12: Exploration Bias Fix — Hold-Action Exploration + Conservative Prefill + Projected Clearance

**Goal**: Fix the systematic 48h product failure (7% beats-baseline, 2/30 in v1.0) caused by asymmetric exploration under progressive constraints. Three complementary fixes targeting different aspects of the problem.

**Root cause analysis**: With uniform epsilon-greedy exploration and 11 valid actions at discount index 0, P(hold) = 9% while P(go deeper) = 91%. Over 24 steps of exploration in a 48h episode, the agent is pushed to max discount on virtually every training episode and never learns that holding at low discounts is optimal for products where demand exceeds inventory at any price. This explains the stark contrast: 65% beats-baseline on 12h products (fewer steps = less cumulative bias) vs 7% on 48h products.

**Changes**:

### 1. Hold-action exploration bias (`hold_action_prob`)
- During epsilon-greedy exploration, a fraction `hold_action_prob` of random actions select the hold action (current discount = `valid_actions[0]`)
- With `hold_action_prob=0.5`: P(hold) ≈ 55% (up from 9%), generating training trajectories where the agent holds at low discounts
- Default 0.0 preserves existing behavior

### 2. Conservative prefill mix
- Updated `DEFAULT_BASELINE_MIX`: backloaded_progressive 25%→35%, fixed_20 10%→20%, linear_progressive 30%→15%, demand_responsive 25%→20%
- Conservative demonstrations (backloaded + fixed_20) now 55% of prefill, up from 35%
- Ensures replay buffer starts with ample "hold low" trajectories during warmup

### 3. Projected clearance state feature (10th dimension)
- `_projected_clearance()` computes expected remaining demand at current discount over all future steps, divided by remaining inventory
- Uses the demand model's price effect, intraday pattern, and day-of-week pattern
- Value near 1.0 means current discount can clear stock without going deeper
- Observation space: 9-dim → 10-dim

**Setup (v1.1)**:
```bash
python scripts/run_portfolio.py --episodes 3000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --prefill-episodes 200 --warmup-steps 1000 \
    --workers 16 --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 \
    --save-dir results/portfolio_v110_exploration_fix
```

### Results

*(to be filled after experiment)*

---

## Key Learnings Summary

### When reward shaping helps

Shaping is most valuable when:
1. **Waste is a real problem** — high inventory relative to demand (>5% waste rate without shaping)
2. **Large action space** — 2h mode (11 actions) where exploration is harder
3. **Short markdown windows** — 12h products (seafood, deli) where timing matters most
4. **Sufficient training** — shaping accelerates learning but needs enough episodes to converge

Shaping is neutral/noise when:
1. The product clears inventory easily (0% waste with plain DQN)
2. High-elasticity products where any moderate discount drives enough demand
3. 4h mode with only 6 actions — the search space is small enough for plain DQN

### Architecture decisions that worked

| Decision | Why it worked |
|----------|---------------|
| Revenue-normalized shaping (ratio=0.2) | Universal across $1.50-$20 products — 20% of expected revenue is always meaningful |
| Double DQN + soft targets | More stable than vanilla DQN, especially with PER |
| Historical pre-fill (200 episodes) | Bootstraps the buffer with reasonable behavior, avoids cold-start |
| Warmup (1000 steps) | Pre-trains the network before online exploration, faster convergence |
| Slower epsilon decay for 2h mode | 11 actions need more exploration than 6 |
| Progressive constraint via action masking | Clean, guaranteed valid actions without reward hacking |
| Transfer learning + short fine-tuning | Pre-trained weights amplify shaping advantage (59% win rate) with 2.5x compute savings |
| Bigger network + longer training (128-dim, 3000ep) | Single biggest lever for beats-baseline: 22% → 43% (+21pp) |
| N-step returns (n=5) | New best beats-baseline: 43% → 46% (+3pp) via faster credit assignment in short episodes |

### What didn't work / watch out for

| Issue | Fix |
|-------|-----|
| Fixed shaping scale (Phi=-5.0) | Use revenue-normalized scale instead |
| 500 episodes for 2h mode | Insufficient — need 1500+ for 11 actions |
| Fast epsilon decay (0.998) with 1500 episodes | Use 0.999 to keep exploring longer |
| Shaping on easy products (0% waste) | It's just noise — don't expect improvement when the problem is already solved |
| DQN generates less raw revenue than baselines | Expected — DQN optimizes total reward (revenue minus waste), not revenue alone |
| TL + long fine-tuning (1500ep) | Shaping advantage disappears as plain DQN also fully converges — use shorter fine-tuning (500-800ep) to preserve the TL benefit |
| TL + very long fine-tuning (3000ep) | TL slightly hurts vs no-TL (40% vs 43% beats-baseline) — category pre-training can introduce interference with enough direct training |
| High replay ratio to halve episodes (rr=4, 1500ep) | Online exploration is the bottleneck, not sample efficiency — 15% beats-baseline vs 43% with rr=1 at 3000ep, and 2x slower |
| N-step returns reduce shaping benefit | N-step and shaping both accelerate reward propagation — their benefits overlap, dropping shaping win rate from 57% to 47% |
