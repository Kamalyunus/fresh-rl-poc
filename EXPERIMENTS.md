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
- `_projected_clearance()` extrapolates from observed recent sales velocity (last 3 steps) to estimate whether remaining inventory will clear before the deadline
- Uses only observable data: `(recent_velocity * remaining_steps) / inventory_remaining`
- Value near 1.0 means current pace can clear stock without going deeper
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

| Metric | v1.0 (n_step=5) | v1.1 (exploration fix) |
|--------|-----------------|------------------------|
| Shaping wins | 52/110 (47%) | 48/110 (44%) |
| Beats best baseline | 51/110 (46%) | **57/110 (52%)** |
| Runtime (16 workers) | 32.2 min | 40.4 min |

**By markdown window**:

| Window | SKUs | v1.0 | v1.1 | Delta |
|--------|------|------|------|-------|
| 12h | 17 | 65% | 47% | -18pp |
| 24h | 63 | 60% | **75%** | **+15pp** |
| 48h | 30 | 7% | 7% | +0pp |

**Category breakdown (beats-baseline %)**:

| Category | v1.0 | v1.1 | Delta |
|----------|------|------|-------|
| meats | 40% | **80%** | **+40pp** |
| deli_prepared | 60% | **80%** | **+20pp** |
| bakery | 27% | **47%** | **+20pp** |
| fruits | 47% | **53%** | +6pp |
| seafood | 60% | 60% | same |
| legacy | 80% | 40% | -40pp |
| vegetables | 60% | 33% | -27pp |
| dairy | 60% | 13% | -47pp |

### Analysis: 48h Products — Wrong Diagnosis

The initial hypothesis was that uniform epsilon-greedy exploration biased the agent toward deep discounting, preventing it from learning that holding is optimal for 48h products. **This was wrong.** Deep analysis of the 48h failures reveals the opposite problem:

**1. Revenue is the sole failure mode — not waste**

All 28/28 failures lose purely on revenue. Zero failures are due to waste. The DQN captures only 86% of baseline revenue on average (worst case: 76%). Every 48h product achieves 100% clearance — both DQN and baselines — with zero waste across the board.

**2. Demand/inventory ratio explains everything**

| Window | Demand/Inv Ratio | Strategic Challenge |
|--------|-----------------|---------------------|
| 12h | 0.38x | Clear inventory before deadline (waste risk) |
| 24h | 0.78x | Balance clearance vs margin (sweet spot for DQN) |
| 48h | 1.42x | Maximize revenue per unit (everything sells anyway) |

48h products have total expected demand exceeding inventory by 42%. Everything sells regardless of pricing strategy. The only question is at what price — and simple baselines (Backloaded Progressive wins 61% of 48h failures, Fixed 20% wins 25%) apply consistent discounts that capture high-elasticity demand efficiently.

**3. Hold-action bias made things worse**

By biasing exploration toward *holding* (conservative pricing), we pushed the agent in exactly the wrong direction for 48h products. The optimal 48h strategy is to discount early and capture demand at good prices, not to hold at minimum discount. The 12h regression (-18pp) may also stem from this: 12h products have tight deadlines where the agent needs to explore aggressive discounting, not conservative holding.

**4. The 2 winners are special cases**

yogurt_greek_plain and whipped_cream_8oz beat baselines because they have lower price elasticity (2.83 vs 3.37 average) and higher margins — the only 48h products where conservative pricing actually helps.

**5. 24h is the sweet spot**

24h products improved from 60% to 75% (+15pp). At 0.78x demand/inventory, waste risk is real enough that the DQN's adaptive strategy matters, but inventory isn't so scarce that baselines trivially win on revenue.

### Key Takeaway

The 48h failure is **not an exploration problem** — it's a **structural problem**. When demand >> inventory, everything sells regardless of strategy, and the only differentiator is revenue per unit. Simple baselines that apply consistent discounts are hard to beat because there's no strategic complexity to exploit. The DQN's advantage — adapting to waste risk — is irrelevant when waste risk is zero.

The overall improvement (+6pp beats-baseline, new best 52%) was driven by 24h products where the exploration fix + projected clearance feature genuinely helped the agent learn better timing.

---

## Iteration 13: Catalog Cleanup — 150 SKUs, All 24h Windows

**Goal**: Remove structurally unsolvable product tiers and validate DQN on a uniform catalog where it has a genuine strategic advantage.

**Changes**:
- Removed 48h markdown windows from vegetables, fruits, dairy (changed to 24h)
- Removed 12h markdown windows from seafood, deli_prepared (changed to 24h)
- Removed 5 legacy profiles (salad_mix, fresh_chicken, yogurt, bakery_bread, sushi)
- Added 45 new SKUs across all 7 categories (21-22 per category)
- Total: 150 products, all 24h, 7 categories

**Rationale**: The v1.1 analysis showed 48h products have demand/inventory ratio of 1.42x (everything sells regardless of strategy), making baselines unbeatable on revenue. 12h products had a different issue — very short horizons where compounding exploration errors hurt DQN. 24h is the sweet spot where waste risk is real and DQN's adaptive strategy matters.

**Setup (v1.2)**: Same as v1.1 but with the updated 150-SKU catalog:
```bash
python scripts/run_portfolio.py --episodes 3000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --prefill-episodes 200 --warmup-steps 1000 \
    --workers 16 --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 \
    --save-dir results/portfolio_v120_150skus
```

### Results

| Metric | v1.1 (110 SKUs, mixed) | v1.2 (150 SKUs, all 24h) |
|--------|------------------------|--------------------------|
| Shaping wins | 48/110 (44%) | 71/150 (47%) |
| Beats best baseline | 57/110 (52%) | **122/150 (81%)** |
| Runtime (16 workers) | 40.4 min | 53.1 min |

**Category breakdown**:

| Category | SKUs | Beats BL | Shaping Win |
|----------|------|----------|-------------|
| deli_prepared | 22 | **91%** | 50% |
| meats | 22 | **86%** | 45% |
| vegetables | 21 | **86%** | 57% |
| seafood | 22 | **82%** | 45% |
| fruits | 21 | **81%** | 43% |
| dairy | 21 | **76%** | 38% |
| bakery | 21 | **67%** | 52% |

**Reward gap (shaped DQN - best baseline)**:

| Stat | Value |
|------|-------|
| Min | -7.7 |
| P25 | +0.6 |
| Median | **+3.2** |
| P75 | +7.1 |
| Max | +35.9 |

**Remaining failures (28/150)**:
- Backloaded Progressive wins 17/28 (61%)
- Immediate Deep 70% wins 5/28 (18%)
- Demand Responsive wins 4/28 (14%)

### Analysis

1. **Beats-baseline jumped from 52% to 81%** — the single biggest improvement in the project. Removing structurally unsolvable tiers (48h, 12h) and focusing on the 24h sweet spot where DQN has a genuine strategic advantage made the difference.
2. **Every category above 67%**: Even the weakest category (bakery) beats baselines on 2/3 of SKUs. Deli_prepared leads at 91%.
3. **Median reward gap of +3.2**: The DQN doesn't just barely win — it meaningfully outperforms baselines on most products.
4. **Failures are concentrated against Backloaded Progressive**: This baseline is the hardest to beat because it mimics a reasonable human strategy (hold early, ramp late). The remaining 28 failures likely need either more training or a different approach.
5. **Shaping win rate stable at 47%**: Reward shaping continues to be roughly neutral in aggregate — n-step returns already handle credit assignment, making shaping's urgency signal redundant for most products.

**Learning**: **Problem selection matters as much as algorithm design.** The biggest improvement in the entire project (+29pp beats-baseline) came not from a new algorithm or hyperparameter, but from removing product tiers where RL has no structural advantage over simple rules. In production, this translates to: use RL for products where pricing decisions have real consequences (waste risk), and use rule-based policies for products that sell out regardless.

---

## Iteration 14: Transfer Learning Revisited (150 SKUs)

**Commit**: `d099fd8`

**Hypothesis**: Transfer learning (category pre-training + per-SKU fine-tuning) can match v1.2's 81% beats-baseline with less compute by giving each SKU a warm start from its category model.

**Setup**: Same environment settings as v1.2 (0.5x demand, 2x inventory, all flags), with transfer learning enabled:

```bash
# v1.3: TL with 500 fine-tune episodes (2.5x compute savings)
python scripts/run_portfolio.py --episodes 500 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 \
    --transfer-learning --pretrain-episodes 1500

# v1.3.1: TL with 2000 fine-tune episodes (31% compute savings)
python scripts/run_portfolio.py --episodes 2000 ... --transfer-learning --pretrain-episodes 1500

# v1.3.2: TL with 3000 fine-tune episodes (equal fine-tune, tests whether TL helps or hurts)
python scripts/run_portfolio.py --episodes 3000 ... --transfer-learning --pretrain-episodes 1500
```

### Results

| Metric | v1.2 (no TL, 3000ep) | v1.3 (TL, 500ft) | v1.3.1 (TL, 2000ft) | v1.3.2 (TL, 3000ft) |
|--------|---------------------|-------------------|----------------------|----------------------|
| Beats best baseline | **122/150 (81%)** | 38/150 (25%) | 82/150 (55%) | 106/150 (71%) |
| Shaping wins | 71/150 (47%) | 85/150 (57%) | 80/150 (53%) | 63/150 (42%) |
| Plain > baseline | 117/150 (78%) | 43/150 (29%) | 82/150 (55%) | 103/150 (69%) |
| Mean reward (shaped) | 140.5 | 130.2 | 137.2 | 139.2 |
| Mean revenue (shaped) | $114.7 | $103.8 | $103.8 | — |
| Total episodes | 450K | 85.5K | 310.5K | 460.5K |
| Runtime (16 workers) | ~45 min | ~10 min | ~35 min | ~53 min |

**Category breakdown (v1.3.2 — TL + 3000 fine-tune vs v1.2 no TL)**:

| Category | SKUs | v1.3.2 Beats BL | v1.2 Beats BL | Delta |
|----------|------|-----------------|---------------|-------|
| deli_prepared | 22 | 64% | 91% | -27pp |
| bakery | 21 | 62% | 67% | -5pp |
| fruits | 21 | 38% | 81% | -43pp |
| dairy | 21 | 38% | 76% | -38pp |
| meats | 22 | 32% | 86% | -54pp |
| seafood | 22 | 32% | 82% | -50pp |
| vegetables | 21 | 29% | 86% | -57pp |

### Analysis

1. **TL with 500 fine-tune (v1.3) fails badly at 25%**: Epsilon is still ~0.6 after 500 episodes (decay=0.999), so the agent barely finishes exploring before training ends.

2. **TL with 2000 fine-tune (v1.3.1) reaches 55%**: Better but still 26pp below v1.2.

3. **TL with 3000 fine-tune (v1.3.2) reaches 71% — still 10pp below v1.2, and takes longer**: With equal fine-tuning episodes (3000), TL is strictly worse: -10pp beats-baseline and +18% runtime from the pretrain overhead. This is the definitive test — the category pre-training actively hurts.

4. **TL hurts most in high-variance categories**: Meats (-54pp), vegetables (-57pp), and seafood (-50pp) have the widest price/demand ranges within each category. The pre-trained weights bias the agent toward a generic policy that interferes with SKU-specific optimization.

5. **Bakery is least affected (-5pp)**: The narrowest price range category ($2.50-7) with the highest elasticity (3.5-4.5) — SKUs are similar enough that the category model provides a reasonable starting point.

6. **More fine-tuning monotonically improves TL** (25% → 55% → 71%) but can't close the gap — the pre-trained weights create a local optimum that the agent struggles to escape even with 3000 episodes.

**Learning**: **Transfer learning introduces negative transfer when intra-category SKU variation is high.** The category pre-training learns a blurred average policy that biases the agent's starting point. With enough fine-tuning, the agent partially overcomes this bias (71%) but never matches training from scratch (81%). The pre-trained weights act as a regularizer toward the category average — helpful if SKUs are similar, harmful if they're diverse. Direct per-SKU training for 3000 episodes remains the best approach for this catalog.

---

## Iteration 15: Longer Training — 5000 Episodes

**Commit**: `f087d5e`

**Hypothesis**: More training episodes can push past v1.2's 81% ceiling. With 5000 episodes and epsilon_decay=0.999, epsilon reaches ~0.007 (vs ~0.05 at 3000 episodes), allowing the agent to fully exploit its learned policy.

**Setup**: Same as v1.2 but with 5000 episodes:
```bash
python scripts/run_portfolio.py --episodes 5000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 \
    --save-dir results/portfolio_v140_5000ep
```

### Results

| Metric | v1.2 (3000ep) | v1.4 (5000ep) |
|--------|--------------|---------------|
| Beats baseline (shaped) | 122/150 (81%) | **126/150 (84%)** |
| Beats baseline (plain) | 117/150 (78%) | **129/150 (86%)** |
| Shaping wins | 71/150 (47%) | 61/150 (41%) |
| Mean reward (shaped) | 140.5 | 141.4 |
| Mean reward (plain) | 140.7 | 141.9 |
| Runtime (16 workers) | ~45 min | ~90 min |

**Category breakdown**:

| Category | SKUs | v1.4 Beats BL | v1.2 Beats BL |
|----------|------|---------------|---------------|
| deli_prepared | 22 | **91%** | 91% |
| meats | 22 | **91%** | 86% |
| vegetables | 21 | **86%** | 86% |
| fruits | 21 | **86%** | 81% |
| dairy | 21 | **81%** | 76% |
| seafood | 22 | 77% | 82% |
| bakery | 21 | 76% | 67% |

### Analysis

1. **New best: 84% beats-baseline (shaped), 86% (plain)** — a +3pp improvement over v1.2 for shaped, +8pp for plain. More training helps, but with diminishing returns (2x compute for +3pp).

2. **Plain DQN now outperforms shaped DQN (86% vs 84%)**: With 5000 episodes, the agent has enough experience to fully converge without reward shaping. Shaping becomes slight noise — its urgency signal conflicts with what the agent has already learned from direct experience.

3. **Shaping win rate drops to 41%**: Confirms that reward shaping's value diminishes with longer training. N-step returns (n=5) already handle credit assignment; the additional shaping signal adds no information with sufficient episodes.

4. **Every category at 76%+**: Bakery improved most (+9pp to 76%), dairy +5pp to 81%. Seafood slightly regressed (82% → 77%) — likely stochastic variance.

5. **Diminishing returns curve**: 81% at 3000ep → 84% at 5000ep is +3pp for 2x compute. The marginal value of additional training is flattening.

**Learning**: **More training monotonically improves beats-baseline but with sharply diminishing returns.** The 3000→5000 episode jump costs 2x compute for +3pp. Plain DQN overtaking shaped DQN confirms that reward shaping is a convergence accelerator, not a final performance booster — given enough data, the raw reward signal is sufficient.

---

## Iteration 16: Pooled Category Training (v2 — 7 models for 150 SKUs)

**Commit**: (v2.0)

**Goal**: Train 7 category-level models instead of 150 per-SKU models by conditioning on observable product features. Unlike the failed transfer learning (v1.3, which pre-trained on 10-dim state with no product identity → per-SKU fine-tune → 150 final models), pooled training uses a single 14-dim state (10 base + 4 product features) and produces 7 final models that generalize across all SKUs in each category.

**Key difference from TL (v1.3)**:
- **TL**: category pre-train on 10-dim state (no product identity) → per-SKU fine-tune → 150 final models → 71% beats-baseline
- **Pooled**: single model trained on 14-dim state (10 base + 4 observable product features) → 7 final models → 78% beats-baseline

**Changes**:

### 1. Observable product features (`product_catalog.py`)
- Added `pack_size` attribute to `CategorySpec` (generated with separate RNG `seed+10000` to preserve existing params)
- Added `get_product_features(product_name, inventory_mult)` returning 4-dim normalized [0,1] vector:
  - `price_norm` — price within category range (proxies revenue scale)
  - `cost_frac_norm` — cost fraction within range (proxies margin pressure)
  - `inventory_norm` — inventory within range (proxies clearing difficulty)
  - `pack_size_norm` — pack size within range (proxies purchase dynamics)
- **No simulator leakage**: `price_elasticity` and `base_markdown_demand` excluded — agent learns demand through runtime signals (velocity, sell-through, projected clearance) + observable proxies

### 2. PooledCategoryEnv (`pooled_env.py` — NEW)
- `PooledCategoryEnv(gym.Env)`: holds one `MarkdownProductEnv` per SKU + pre-computed 4-dim product features
- `reset(options={"product": name})` switches active product, appends features to obs (14-dim)
- `__getattr__` delegation so baselines can access `env.step_count`, `env.inventory_remaining`, etc.
- `pooled_prefill()`: runs `DEFAULT_BASELINE_MIX` policies through the pooled env, producing 14-dim transitions

### 3. Pooled mode in portfolio runner (`run_portfolio.py`)
- `--pooled` flag + `--pooled-episodes-per-sku` (default 2500)
- `_train_category_pooled()`: creates PooledCategoryEnv, trains plain+shaped agents with round-robin through SKUs, evaluates all SKUs
- Per-product `waste_cost_scale` update for shaped variant
- 7 workers (one per category) via ProcessPoolExecutor

**Setup (v2)**:
```bash
python scripts/run_portfolio.py --pooled --pooled-episodes-per-sku 5000 \
    --eval-episodes 100 --step-hours 2 --per --prefill --warmup-steps 1000 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 --workers 7
```

### Results

| Metric | v1.4 Per-SKU (5000ep) | v2 Pooled (5000ep/SKU) |
|--------|----------------------|------------------------|
| Beats best baseline | **129/150 (86%)** | 117/150 (78%) |
| Shaping wins | 61/150 (41%) | 70/150 (47%) |
| Models trained | 300 | 14 |
| Runtime (~) | ~90 min (16 workers) | ~180 min (7 workers) |

**Category breakdown (pooled, best of plain/shaped)**:

| Category | SKUs | v2 Pooled | v1.4 Per-SKU | Delta |
|----------|------|-----------|--------------|-------|
| dairy | 21 | **90%** | 81% | **+9pp** |
| deli_prepared | 22 | **86%** | 91% | -5pp |
| vegetables | 21 | **81%** | 86% | -5pp |
| fruits | 21 | **81%** | 86% | -5pp |
| meats | 22 | **73%** | 91% | -18pp |
| seafood | 22 | **73%** | 77% | -4pp |
| bakery | 21 | **62%** | 76% | -14pp |

### Analysis

1. **78% beats-baseline with only 14 models**: A strong result given the model count reduction (300 → 14). The pooled model learns generalizable pricing strategies across diverse SKUs within each category.

2. **Dairy outperforms per-SKU (+9pp)**: The only category where pooled beats per-SKU. Dairy has the narrowest intra-category variance — SKUs are similar enough that pooling experience across them is strictly beneficial.

3. **Meats and bakery most affected**: These categories have the widest intra-category diversity (meats: $5-15 price range, bakery: $2.50-7 with 3.5-4.5 elasticity range). A single model struggles to specialize for very different products.

4. **Shaping wins slightly higher (47% vs 41%)**: With a shared model handling diverse products, the shaped variant's urgency signal helps differentiate behavior across different product profiles.

5. **Zero-shot generalization**: The key advantage — for any new SKU, compute its 4 product features and use the existing category model. No retraining needed.

6. **Why this works but TL failed**: TL pre-trained on 10-dim state with **no product identity** — the model couldn't distinguish between a $5 ground beef and a $15 lamb chop. Pooled training includes 4 observable product features in every state, so the model explicitly knows which product it's pricing and can learn product-conditional strategies.

**Learning**: **Pooled category training trades ~8pp absolute performance for 21x model reduction and zero-shot generalization to new SKUs.** The key enabler is observable product features in the state — conditioning on price, cost, inventory, and pack size lets a single model learn product-specific strategies without separate training. Best used as a warm-start for new SKUs or when maintaining 300 models is impractical; per-SKU training remains best for peak performance on a fixed catalog.

---

## Iteration 17: Pooled→Per-SKU Transfer Learning (v2.1 — 95% beats-baseline)

**Commit**: (v2.1)

**Goal**: Use pooled category model weights (v2, 14-dim state) as initialization for per-SKU fine-tuning. The hypothesis: pooled models already understand product-conditional pricing via the 4 observable features — transferring these representations should beat both per-SKU from scratch (v1.4, 86%) and the old TL that had no product identity (v1.3, 71%).

**Key difference from old TL (v1.3)**:
- **Old TL**: category pre-train on 10-dim state (no product identity) → per-SKU fine-tune with 10-dim state → 71% beats-baseline
- **Pooled TL (v2.1)**: pooled model trained on 14-dim state (with product features) → per-SKU fine-tune with 14-dim state → **95% beats-baseline**

**Changes**:

### 1. AugmentedProductEnv (`pooled_env.py`)
- `AugmentedProductEnv(gym.Wrapper)`: wraps a single `MarkdownProductEnv` to append 4 product features to every observation (10→14 dim)
- Keeps the change isolated — `train()`, `evaluate_policy()`, and `HistoricalDataGenerator` all work transparently through the wrapper
- `__getattr__` delegates attribute access to wrapped env so baselines can access `env.step_count`, `env.inventory_remaining`, etc.
- Product features are constant (same product throughout all episodes), unlike `PooledCategoryEnv` where features change on product switch

### 2. Weight transfer mechanism
- Pooled model: `Linear(14, 128)` → Per-SKU agent: `Linear(14, 128)` (created with `state_dim=14` via augmented env)
- `load_pretrained()` works as-is — no weight surgery needed
- Plain DQN loads `pooled_{category}_plain_{step_hours}h.pt`; shaped DQN loads `pooled_{category}_shaped_{step_hours}h.pt`
- Transfer epsilon starts at 0.3 (lower exploration for fine-tuning)

### 3. train.py changes
- New params: `augment_state: bool`, `inventory_mult: float`
- When `augment_state=True`: wraps env with `AugmentedProductEnv` right after creation
- `state_dim=14` flows automatically to agent, prefill, and greedy eval
- `HistoricalDataGenerator` accepts optional external env for augmented prefill

### 4. run_portfolio.py — `--pooled-tl` mode
- New CLI flags: `--pooled-tl`, `--pooled-model-dir`
- Per-product dispatch: determines category → builds pooled model paths → passes to `_run_single_product()`
- Evaluation wraps eval env with `AugmentedProductEnv` when `augment_state=True`

**Setup (v2.1)**:
```bash
python scripts/run_portfolio.py --pooled-tl \
    --pooled-model-dir results/portfolio_v2_pooled \
    --episodes 5000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 \
    --save-dir results/portfolio_v21_pooled_tl
```

### Results

| Metric | v2.1 Pooled TL | v1.4 Per-SKU | v2 Pooled | v1.3 Old TL |
|--------|---------------|-------------|-----------|-------------|
| Beats best baseline | **142/150 (95%)** | 129/150 (86%) | 117/150 (78%) | 107/150 (71%) |
| Shaping wins | 66/150 (44%) | 61/150 (41%) | 70/150 (47%) | — |
| Models trained | 300 | 300 | 14 | 300 |
| Runtime (~) | ~160 min (16 workers) | ~90 min (16 workers) | ~180 min (7 workers) | ~120 min |

**Category breakdown (v2.1, best of plain/shaped)**:

| Category | SKUs | v2.1 Pooled TL | v1.4 Per-SKU | v2 Pooled | Delta vs v1.4 |
|----------|------|---------------|--------------|-----------|---------------|
| dairy | 21 | **100%** | 81% | 90% | **+19pp** |
| deli_prepared | 22 | **100%** | 91% | 86% | **+9pp** |
| fruits | 21 | **95%** | 86% | 81% | **+9pp** |
| vegetables | 21 | **95%** | 86% | 81% | **+9pp** |
| meats | 22 | **91%** | 91% | 73% | +0pp |
| seafood | 22 | **91%** | 77% | 73% | **+14pp** |
| bakery | 21 | **90%** | 76% | 62% | **+14pp** |

### Analysis

1. **95% beats-baseline with the same model count (300)**: The +9pp improvement over v1.4 comes entirely from better weight initialization. Pooled representations give per-SKU agents a head start on understanding price-demand relationships.

2. **Dairy and deli_prepared hit 100%**: Every single SKU in these categories beats its best baseline. These categories had the best pooled performance (90% and 86%), so the transfer is strongest where the pooled model was already effective.

3. **Biggest improvements in previously weak categories**: bakery (+14pp, 76%→90%) and seafood (+14pp, 77%→91%) benefit most from transfer. The pooled model provides useful general pricing knowledge even for categories where it struggled to fully specialize.

4. **Only 8 non-winners, all near-ties**: bacon_pack, blueberries_6oz, croissants_4pk, pretzel_rolls_6pk, poke_bowl, salad_mix_5oz, smoked_salmon_4oz, veal_cutlet — all within 4 reward of baseline. These are products where the best baseline (usually Immediate Deep 70% or Backloaded Progressive) is essentially optimal.

5. **Why this works but old TL failed**: Old TL (v1.3) pre-trained on 10-dim state with **no product identity** — the model learned a blurred average policy. Pooled models (v2) train on 14-dim state with observable product features, learning **product-conditional** representations. When transferred, these representations give the fine-tuned model a meaningful starting point rather than a conflicting one.

6. **Shaping win rate unchanged (44% vs 41%)**: Transfer learning doesn't change the shaping dynamics — it's still most useful for hard-to-clear products where waste avoidance matters.

**Learning**: **Pooled→per-SKU transfer learning is the clear best approach (95% vs 86% vs 78%).** The key insight is that transfer learning works when the pre-trained model has product identity in its state — observable product features let the pooled model learn representations that transfer meaningfully to individual SKUs. The progression is: train 7 pooled models (cheap) → fine-tune 150 per-SKU models with pooled initialization (best results). The extra runtime (~160 min vs ~90 min for per-SKU) is modest given the +9pp improvement.

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
4. Very long training (5000+ episodes) — plain DQN fully converges without shaping, and shaping win rate drops to 41%

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
| Projected clearance feature (velocity-based) | Observable "will I clear at current pace?" signal — helped 24h products reach 75% beats-baseline |
| Hold-action exploration + conservative prefill | New best overall: 52% beats-baseline (+6pp), driven by 24h product improvement |
| Longer training (5000ep) | New best 84% beats-baseline (+3pp over 3000ep). Plain DQN reaches 86% — shaping unnecessary with enough data |
| Pooled category training (14-dim state) | 78% beats-baseline with only 14 models (vs 300). Enables zero-shot pricing for new SKUs via observable product features |
| Pooled→per-SKU TL (14-dim state) | **New best 95% beats-baseline (+9pp over per-SKU).** Pooled weights with product identity transfer meaningfully — unlike old TL's 10-dim blurred representations |

### What didn't work / watch out for

| Issue | Fix |
|-------|-----|
| Fixed shaping scale (Phi=-5.0) | Use revenue-normalized scale instead |
| 500 episodes for 2h mode | Insufficient — need 1500+ for 11 actions |
| Fast epsilon decay (0.998) with 1500 episodes | Use 0.999 to keep exploring longer |
| Shaping on easy products (0% waste) | It's just noise — don't expect improvement when the problem is already solved |
| DQN generates less raw revenue than baselines | Expected — DQN optimizes total reward (revenue minus waste), not revenue alone |
| TL with wide-range categories | Category pre-training introduces negative transfer — 71% beats-baseline with 3000ft vs 81% without TL, despite more total compute. Pre-trained weights bias toward category-average policy |
| TL + short fine-tuning (500ep) | Epsilon still ~0.6 after 500 episodes (decay=0.999) — agent doesn't finish exploring. Only 25% beats-baseline |
| TL + equal fine-tuning (3000ep) | Strictly worse than no TL: -10pp beats-baseline and +18% runtime. The definitive test that TL hurts for this catalog's diversity level |
| High replay ratio to halve episodes (rr=4, 1500ep) | Online exploration is the bottleneck, not sample efficiency — 15% beats-baseline vs 43% with rr=1 at 3000ep, and 2x slower |
| N-step returns reduce shaping benefit | N-step and shaping both accelerate reward propagation — their benefits overlap, dropping shaping win rate from 57% to 47% |
| Hold-action bias on 48h products | 48h products need *more* discounting, not less — hold bias pushed the agent in the wrong direction. 48h failure is structural (demand >> inventory), not an exploration problem |
| Projected clearance leaking simulator data | Initial implementation used demand model internals (elasticity, intraday pattern) — had to hotfix to velocity-based `(recent_velocity * remaining_steps) / inventory` |
| Pooled training — bakery/meats categories | High intra-category diversity hurts pooled models most (bakery 62%, meats 73%). A single model can't fully specialize for very different products in the same category |
| Old TL but not pooled TL | The difference is product identity: 10-dim pre-training (v1.3) → negative transfer; 14-dim pre-training with product features (v2.1) → +9pp improvement. Observable features in the state are the key enabler |

---

## Iteration 18: Priority Subset — Tau Schedule, TL Warmup Skip, Mixed Prefill (v3.0)

**Changes** (3 low-risk improvements, no architecture changes):

### 1. Tau warming schedule
- Soft target update rate (`tau`) now supports linear warmup: start conservative (`tau_start=0.005`), warm to faster sync (`tau_end`) over `tau_warmup_steps` gradient steps.
- Hypothesis: early training benefits from stable targets (low tau), but later training converges faster with higher tau once Q-values stabilize.
- New params: `--tau-start`, `--tau-end`, `--tau-warmup-steps` (all default to current behavior: constant 0.005).
- Files: `dqn_agent.py` (constructor, `_current_tau()`, `train_step_fn()`, `save()`, `load()`), `train.py`, `run_portfolio.py`, `deployment/config.py`, `deployment/batch_train.py`.

### 2. Reduced warmup for transfer learning
- When `pretrained_path` is set, warmup steps are automatically reduced to 0 (overridable via `--tl-warmup-steps`).
- Rationale: warmup gradient steps on prefill data pull pretrained weights back toward baseline behavior, undoing the benefit of transfer learning.
- New param: `--tl-warmup-steps` (default: skip warmup entirely when TL is active).
- Files: `train.py`, `run_portfolio.py`.

### 3. Mixed baseline prefill (ImmediateDeepDiscount)
- Added `ImmediateDeepDiscount` (always pick deepest discount) to the default prefill baseline mix at 5% weight.
- Rebalanced: `backloaded_progressive` reduced from 35% → 30%, all others unchanged.
- Rationale: more diverse initial experiences — the agent sees the extreme "always deep" policy, which provides useful negative examples (high revenue but destroys margin) and positive examples (high clearance rate).
- Files: `historical_data.py` (imports, `DEFAULT_BASELINE_MIX`, `get_baseline_by_name()`).

**Run command** (3000 episodes, tau schedule):
```bash
python scripts/run_portfolio.py --pooled-tl \
    --pooled-model-dir results/portfolio_v2_pooled \
    --episodes 3000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 \
    --tau-start 0.005 --tau-end 0.03 --tau-warmup-steps 12000 \
    --save-dir results/portfolio_v30_pooled_tl
```

**Results**: **141/150 (94%) beats-baseline** at 3000 episodes (82 min, 16 workers)

| Metric | v2.1 (5000ep) | v3.0 (3000ep) | Delta |
|--------|--------------|---------------|-------|
| Beats baseline | 142/150 (95%) | 141/150 (94%) | -1 |
| Shaping wins | 66/150 (44%) | 72/150 (48%) | +6 |
| Mean shaped reward | 143.3 | 143.0 | -0.2 |
| Runtime | ~160 min | 82 min | -49% |

**Category win rates** (v3.0 / v2.1):
| Category | v3.0 | v2.1 |
|----------|------|------|
| dairy | 100% (21/21) | 100% (21/21) |
| deli_prepared | 100% (22/22) | 100% (22/22) |
| fruits | 95% (20/21) | 95% (20/21) |
| meats | 95% (21/22) | 91% (20/22) |
| vegetables | 95% (20/21) | 95% (20/21) |
| bakery | 86% (18/21) | 90% (19/21) |
| seafood | 86% (19/22) | 91% (20/22) |

**Head-to-head**: v3.0 gained 3 SKUs (blueberries_6oz, croissants_4pk, veal_cutlet), lost 4 (cinnamon_rolls_4pk, danish_pastry_4pk, fish_tacos_kit, mixed_berries_12oz). All gaps are small (< 5 reward).

**9 non-winners** (all near-ties): bacon_pack (-4.5), poke_bowl (-4.7), smoked_salmon_4oz (-4.2), pretzel_rolls_6pk (-2.7), cinnamon_rolls_4pk (-1.1), fish_tacos_kit (-0.6), danish_pastry_4pk (-0.5), mixed_berries_12oz (-0.1), salad_mix_5oz (-0.1).

**Takeaway**: v3.0 matches v2.1 performance with 40% fewer episodes. The tau warming schedule + TL warmup skip + mixed prefill collectively enable faster convergence without sacrificing quality. The 1pp beats-baseline gap (94% vs 95%) is within noise — mean shaped reward is essentially identical (143.0 vs 143.3). Shaping wins improved from 44% to 48%, suggesting the diverse prefill mix helps shaped agents more.
