# Fresh RL POC — Markdown Channel Intraday Progressive Discounting

A proof-of-concept demonstrating how Reinforcement Learning (DQN) can optimize progressive markdown pricing for perishable products on an ecommerce markdown channel, outperforming rule-based approaches on **revenue**, **waste reduction**, and **clearance rate** across 150 product SKUs.

## Problem

Perishable products (produce, dairy, bakery, meat, prepared foods) that are 1-2 days from expiry are moved to a markdown landing page. The retailer must decide how deeply to discount throughout the day at 2-hour intervals across a 24-hour markdown window. Discounts can only go deeper (never revert) — this is the **progressive constraint**. Pricing too conservatively risks waste at deadline; pricing too aggressively sacrifices margin.

## Approach

This POC models the markdown channel as a **Markov Decision Process (MDP)** and trains a **Double DQN** agent with **action masking**, **Prioritized Experience Replay (PER)**, **historical data pre-filling**, and **revenue-normalized reward shaping** to enforce the progressive discount constraint and learn optimal policies across diverse product categories.

### MDP Formulation

| Component | Design |
|-----------|--------|
| **State** | `[hours_remaining, inventory_remaining, current_discount_idx, tod_sin, tod_cos, dow_sin, dow_cos, recent_velocity, sell_through_rate, projected_clearance]` (10-dim, normalized to [0,1]) |
| **Action** | 11 discrete levels {20%..70% by 5%} — with progressive constraint (discounts can only go deeper) |
| **Reward** | Revenue - waste penalty - holding cost + clearance bonus |
| **Transition** | Stochastic demand (Poisson) with price elasticity, intraday pattern, day-of-week effect |

### Key Features

- **Product catalog**: 150 SKUs across 7 categories with realistic economics (seeded, reproducible)
- **Custom Gymnasium environment** with configurable product profiles via product catalog
- **Progressive discount constraint** enforced via action masking in both agent and baselines
- **Double DQN** with soft target updates (tau=0.005) — PyTorch-based
- **Prioritized Experience Replay** (SumTree-based) for sample-efficient learning
- **Historical data pre-filling** from baseline policies to bootstrap the replay buffer
- **N-step returns** (n=5) for faster credit assignment in short 12-step episodes
- **Hold-action exploration bias** to correct asymmetric exploration under progressive constraints
- **Projected clearance feature** enabling the agent to reason about whether holding current discount can clear inventory
- **Revenue-normalized reward shaping** (shaping_ratio=0.2) for waste-aware learning
- **7 baseline policies** for rigorous comparison
- **Pooled category training** (v2): trains 7 category-level models conditioned on 4 observable product features (14-dim state) — enables zero-shot pricing for new SKUs without retraining
- **Pooled transfer learning** (v2.1): uses pooled model weights as initialization for per-SKU fine-tuning via `AugmentedProductEnv` (14-dim state) — best overall at 95% beats-baseline
- **Portfolio runner** for cross-category validation with parallel workers (per-SKU and pooled modes)
- **Visualization suite**: per-product plots (training curves, policy heatmaps, episode walkthroughs, revenue-waste Pareto) + comprehensive portfolio plots (dashboard, DQN-vs-baseline scatter, category win rates, reward gap distribution, per-SKU gaps, baseline difficulty, revenue-waste comparison, three-way DQN/shaped/baseline comparison)
- **Production hardening**: model quality metrics (avg Q-value, mean loss), automatic safety rollback on degradation, epsilon floor (0.05) with degradation reset (0.15), PER buffer persistence across training runs, weekly pooled model retraining from production data

## Project Structure

```
fresh-rl-poc/
├── fresh_rl/
│   ├── __init__.py
│   ├── environment.py          # MarkdownChannelEnv + MarkdownProductEnv
│   ├── dqn_agent.py            # Double DQN with action masking, buffer persistence (PyTorch)
│   ├── baselines.py            # 7 rule-based markdown policies
│   ├── product_catalog.py      # 150 SKUs across 7 categories
│   ├── pooled_env.py           # PooledCategoryEnv for category-level training
│   ├── prioritized_replay.py   # PER with SumTree, save/load persistence
│   ├── sumtree.py              # SumTree data structure for PER
│   └── historical_data.py      # Baseline replay buffer pre-filling
├── scripts/
│   ├── train.py                # Training script (single product)
│   ├── evaluate.py             # Evaluation: DQN vs baselines
│   ├── visualize.py            # Visualization suite (per-product + portfolio)
│   └── run_portfolio.py        # Portfolio runner (all SKUs, parallel)
├── deployment/
│   ├── config.py                  # Constants, safety thresholds, ProductionConfig
│   ├── state.py                   # StateConstructor: 14-dim state from session data
│   ├── session.py                 # SessionManager + ActiveSession: daytime tracking
│   ├── inference.py               # PricingAgent: 3-tier model fallback
│   ├── etl.py                     # SessionETL: session CSV → transitions
│   └── batch_train.py             # Nightly training with metrics, rollback, buffer persistence
├── results/                       # Output directory (gitignored)
├── requirements.txt
├── DEPLOYMENT.md                  # Production deployment guide
├── EXPERIMENTS.md                 # Iteration history and learnings
├── ARCHITECTURE.md                # Technical architecture documentation
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# List all 150 available products
python scripts/train.py --list-products

# Train a single product with all features
python scripts/train.py --product salmon_fillet --episodes 5000 --step-hours 2 \
    --reward-shaping --per --prefill --warmup-steps 1000 --shaping-ratio 0.2 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5

# Generate single-product visualizations
python scripts/visualize.py --product salmon_fillet --step-hours 2 --per

# Generate comprehensive portfolio visualizations (9 plots)
python scripts/visualize.py --portfolio results/portfolio/portfolio_results.json

# Run pooled->per-SKU transfer learning — best mode (v2.1, 95%)
python scripts/run_portfolio.py --pooled-tl \
    --pooled-model-dir results/portfolio_v2_pooled \
    --episodes 5000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5

# Run per-SKU portfolio (v1.4, 86%)
python scripts/run_portfolio.py --episodes 5000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5

# Run pooled category training — 7 models for all 150 SKUs (v2, 78%)
python scripts/run_portfolio.py --pooled --pooled-episodes-per-sku 5000 \
    --eval-episodes 100 --step-hours 2 --per --prefill --warmup-steps 1000 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 --workers 7

# Run a single product via portfolio runner
python scripts/run_portfolio.py --products salmon_fillet --episodes 5000
```

## Product Catalog

150 products across 7 categories:

| Category | SKUs | Price Range | Window | Elasticity | Example SKUs |
|----------|------|-------------|--------|------------|--------------|
| **meats** | 22 | $5-15 | 24h | 2.5-3.5 | ground_beef_1lb, chicken_breast, lamb_chop |
| **seafood** | 22 | $7-20 | 24h | 2.0-3.0 | salmon_fillet, shrimp_1lb, lobster_tail |
| **vegetables** | 21 | $1.50-5 | 24h | 3.0-4.5 | salad_mix_5oz, asparagus_bunch, mushroom_8oz |
| **fruits** | 21 | $2-7 | 24h | 3.0-4.0 | strawberries_1lb, avocado_3pk, blueberries_6oz |
| **dairy** | 21 | $1.50-6 | 24h | 2.5-3.5 | yogurt_greek_plain, fresh_mozzarella, brie_wheel_8oz |
| **bakery** | 21 | $2.50-7 | 24h | 3.5-4.5 | sourdough_loaf, croissants_4pk, bagels_6pk |
| **deli_prepared** | 22 | $5-14 | 24h | 2.5-3.5 | rotisserie_chicken, sushi_roll_california, quiche_lorraine |

## How It Works

### Demand Model

Demand is modeled as a Poisson process modulated by three factors:

```
demand = Poisson(base_markdown_demand * price_effect * intraday_effect * dow_effect)
```

- **Price effect**: `exp(-elasticity * (price/base_price - 1))` — deeper discounts exponentially increase demand
- **Intraday pattern**: night 0.3x → morning 0.5x → lunch 0.9x → afternoon 1.2x → evening peak 1.5x → late 0.8x
- **Day-of-week effect**: weekend multiplier (Sat 1.3x, Sun 1.2x, Mon 0.8x)

### Reward Function

```
R = revenue - waste_penalty - holding_cost + clearance_bonus
```

| Component | Signal | Purpose |
|-----------|--------|---------|
| Revenue | price * qty_sold | Drives profitable sales |
| Waste penalty | cost * 3.0 * unsold_at_deadline | Strongly penalizes leftover inventory |
| Holding cost | 0.02 * price * remaining_inventory | Encourages timely sell-through |
| Clearance bonus | 1.0 * initial_inventory (if all sold) | Rewards full clearance before deadline |

### Revenue-Normalized Reward Shaping

Potential-based shaping using cost-aware quadratic urgency, scaled relative to the product's revenue scale:

```
waste_cost_scale = shaping_ratio * base_price * initial_inventory
Phi(s) = -inventory_frac * waste_cost_scale * (1 + time_pressure^2)
shaped_reward = reward + gamma * Phi(s') - Phi(s)
```

The `shaping_ratio=0.2` normalizes the shaping signal to 20% of expected revenue, preventing it from overwhelming the actual reward signal across products with different price points.

### DQN Architecture

- **Double DQN**: Uses online network to select actions, target network to evaluate — reduces overestimation
- **Soft target updates**: `tau=0.005` for smooth target tracking instead of hard periodic copies
- **Prioritized Experience Replay**: SumTree-based, prioritizes high-TD-error transitions
- **Historical pre-fill**: Seeds replay buffer with conservative baseline policy rollouts before training begins
- **N-step returns**: Propagates rewards across multiple steps (n=5) for faster convergence in short episodes
- **Hold-action exploration bias**: Corrects asymmetric epsilon-greedy under progressive constraints (hold_action_prob=0.5)
- **Warmup gradient steps**: Runs N gradient steps on buffered data before online training

## Baselines

| Policy | Strategy |
|--------|----------|
| Immediate Deep 70% | Always picks the deepest discount |
| Linear Progressive | Evenly steps 20%→70% over the window |
| Backloaded Progressive | 20% first half, ramp 30%→70% second half |
| Demand Responsive | Adjusts based on velocity and urgency |
| Fixed 20% / Fixed 40% | Stays at a constant discount level |
| Random | Random from valid (progressive) actions |

## Results

### Pooled Transfer Learning (v2.1 — best overall, 95%)

Uses pooled category model weights (v2) as initialization for per-SKU fine-tuning. `AugmentedProductEnv` wraps per-SKU envs to produce 14-dim state matching pooled model input, so weights transfer directly. Fine-tunes with 5000 episodes per SKU under hard mode (0.5x demand, 2x inventory).

| Metric | v2.1 Pooled TL | v1.4 Per-SKU | v2 Pooled |
|--------|---------------|-------------|-----------|
| **Beats best baseline** | **142/150 (95%)** | 129/150 (86%) | 117/150 (78%) |
| Models trained | 300 | 300 | 14 |
| Shaping wins | 66/150 (44%) | 61/150 (41%) | 70/150 (47%) |

**Category breakdown (v2.1, best of plain/shaped)**:

| Category | SKUs | v2.1 TL Win% | v1.4 Per-SKU | v2 Pooled |
|----------|------|-------------|--------------|-----------|
| dairy | 21 | **100%** | 81% | 90% |
| deli_prepared | 22 | **100%** | 91% | 86% |
| fruits | 21 | **95%** | 86% | 81% |
| vegetables | 21 | **95%** | 86% | 81% |
| meats | 22 | **91%** | 91% | 73% |
| seafood | 22 | **91%** | 77% | 73% |
| bakery | 21 | **90%** | 76% | 62% |

Only 8 products didn't beat baseline, all near-ties (gaps < 4 reward).

**Why v2.1 works**: Pooled models learn product-conditional pricing via the 4 observable features. The hidden layers encode representations that account for product differences — this transfers directly to per-SKU fine-tuning. Unlike old TL (v1.3, 71%), the model *knows* which product it's pricing during pre-training.

### Per-SKU Training (v1.4 — 150 SKUs)

Best per-SKU configuration: 2h steps, 5000 episodes, PER + prefill + warmup, 0.5x demand, 2x inventory, epsilon_decay=0.999, hidden_dim=128, n_step=5, hold_action_prob=0.5, 10-dim state with projected clearance. Trains 300 models (150 plain + 150 shaped).

| Metric | Value |
|--------|-------|
| Beats best baseline (plain DQN) | **129/150 (86%)** |
| Beats best baseline (shaped DQN) | **126/150 (84%)** |

### Pooled Category Training (v2 — 7 category-level models)

Trains 14 models total (7 categories x 2 variants) instead of 300. Each model sees all ~22 SKUs in its category during training, conditioned on 4 observable product features appended to the state (14-dim). Enables **zero-shot pricing for new SKUs** — just compute product features and use the existing category model.

| Metric | Per-SKU (v1.4) | Pooled (v2) |
|--------|---------------|-------------|
| Beats best baseline | **129/150 (86%)** | 117/150 (78%) |
| Models trained | 300 | 14 |
| Shaping wins | 61/150 (41%) | 70/150 (47%) |

**When to use which**:
- **Pooled TL (v2.1)**: Best absolute performance (95%) — use when pooled models are available and per-SKU training is feasible
- **Per-SKU**: Good performance (86%) without needing pooled pre-training
- **Pooled**: Instant zero-shot policy for new SKUs, 21x fewer models, good performance (78%)
- **Hybrid**: Use pooled model on day 1 of a new SKU, fine-tune per-SKU with pooled TL for peak performance

### Production-Realistic Evaluation (v4.2 — 2-phase, 57%)

Separates training from deployment to mirror production reality:
- **Phase 1 (Historical)**: Train on 365 days of historical data, evaluate baselines on the same seeds, pick best DQN variant per SKU
- **Phase 2 (Deployment)**: Deploy selected model on 365 days of fresh demand (epsilon=0.10), continue online learning. Same demand given to both DQN and baseline each day.

| Metric | v4.2 (2-phase deploy) | v4.1 (daily eval) | v4.0 (greedy eval) |
|--------|----------------------|-------------------|-------------------|
| **Beats baseline** | **86/150 (57%)** | 58/150 (39%) | 124/150 (83%) |
| Win rate > 40% | 124/150 (83%) | 85/150 (57%) | — |
| Runtime | 17.9 min | 11.4 min | 10.3 min |

The 57% represents what a business would experience after deploying trained models with low exploration noise.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full iteration history (25 iterations) and learnings.

## References

1. MDPI (2025). Deep RL for Dynamic Pricing and Ordering of Perishables
2. De Moor et al. (2022). Reward Shaping in Perishable Inventory Management (EJOR)
3. KickstartAI / Albert Heijn. Dynamic Pricing for the Supermarket
4. Wasteless. AI-Powered Dynamic Pricing for Perishable Goods

## License

MIT
