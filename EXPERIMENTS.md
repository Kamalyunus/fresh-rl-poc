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

### What didn't work / watch out for

| Issue | Fix |
|-------|-----|
| Fixed shaping scale (Phi=-5.0) | Use revenue-normalized scale instead |
| 500 episodes for 2h mode | Insufficient — need 1500+ for 11 actions |
| Fast epsilon decay (0.998) with 1500 episodes | Use 0.999 to keep exploring longer |
| Shaping on easy products (0% waste) | It's just noise — don't expect improvement when the problem is already solved |
| DQN generates less raw revenue than baselines | Expected — DQN optimizes total reward (revenue minus waste), not revenue alone |
