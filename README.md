# Fresh RL POC — Markdown Channel Intraday Progressive Discounting

A proof-of-concept demonstrating how Reinforcement Learning (DQN) can optimize progressive markdown pricing for perishable products on an ecommerce markdown channel, outperforming rule-based approaches on **revenue**, **waste reduction**, and **clearance rate** across 110 product SKUs.

## Problem

Perishable products (produce, dairy, bakery, meat, prepared foods) that are 1-2 days from expiry are moved to a markdown landing page. The retailer must decide how deeply to discount throughout the day at configurable intervals (2h or 4h). Discounts can only go deeper (never revert) — this is the **progressive constraint**. Pricing too conservatively risks waste at deadline; pricing too aggressively sacrifices margin.

## Approach

This POC models the markdown channel as a **Markov Decision Process (MDP)** and trains a **Double DQN** agent with **action masking**, **Prioritized Experience Replay (PER)**, **historical data pre-filling**, and **revenue-normalized reward shaping** to enforce the progressive discount constraint and learn optimal policies across diverse product categories.

### MDP Formulation

| Component | Design |
|-----------|--------|
| **State** | `[hours_remaining, inventory_remaining, current_discount_idx, tod_sin, tod_cos, dow_sin, dow_cos, recent_velocity, sell_through_rate, projected_clearance]` (10-dim, normalized to [0,1]) |
| **Action** | 4h mode: 6 levels {20%..70%}, 2h mode: 11 levels {20%..70% by 5%} — with progressive constraint |
| **Reward** | Revenue - waste penalty - holding cost + clearance bonus |
| **Transition** | Stochastic demand (Poisson) with price elasticity, intraday pattern, day-of-week effect |

### Key Features

- **Product catalog**: 110 SKUs across 7 categories with realistic economics (seeded, reproducible)
- **Custom Gymnasium environment** with configurable product profiles via product catalog
- **Progressive discount constraint** enforced via action masking in both agent and baselines
- **Double DQN** with soft target updates (tau=0.005) — PyTorch-based
- **Prioritized Experience Replay** (SumTree-based) for sample-efficient learning
- **Historical data pre-filling** from baseline policies to bootstrap the replay buffer
- **N-step returns** for faster credit assignment in short episodes (12-24 steps)
- **Hold-action exploration bias** to correct asymmetric exploration under progressive constraints
- **Projected clearance feature** enabling the agent to reason about whether holding current discount can clear inventory
- **Revenue-normalized reward shaping** (shaping_ratio=0.2) for waste-aware learning
- **7 baseline policies** for rigorous comparison
- **Transfer learning**: category pre-training pools experience across SKUs, per-SKU fine-tuning adapts to individual demand profiles (2.5x compute savings)
- **Portfolio runner** for cross-category validation with parallel workers
- **Visualization suite**: training curves, policy comparison, policy heatmaps, episode walkthroughs, revenue-waste Pareto, training dashboard, category heatmap, and discount progression

## Project Structure

```
fresh-rl-poc/
├── fresh_rl/
│   ├── __init__.py
│   ├── environment.py          # MarkdownChannelEnv + MarkdownProductEnv
│   ├── dqn_agent.py            # Double DQN with action masking (PyTorch)
│   ├── baselines.py            # 7 rule-based markdown policies
│   ├── product_catalog.py      # 110 SKUs across 7 categories
│   ├── prioritized_replay.py   # PER with SumTree
│   ├── sumtree.py              # SumTree data structure for PER
│   └── historical_data.py      # Baseline replay buffer pre-filling
├── scripts/
│   ├── train.py                # Training script (single product)
│   ├── evaluate.py             # Evaluation: DQN vs baselines
│   ├── visualize.py            # Visualization suite (10 plot types)
│   └── run_portfolio.py        # Portfolio runner (all SKUs, parallel)
├── results/                    # Output directory (gitignored)
├── requirements.txt
├── EXPERIMENTS.md              # Iteration history and learnings
├── ARCHITECTURE.md             # Technical architecture documentation
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# List all 110 available products
python scripts/train.py --list-products

# Train a single product with all features
python scripts/train.py --product salmon_fillet --episodes 1500 --step-hours 2 \
    --reward-shaping --per --prefill --warmup-steps 1000 --shaping-ratio 0.2

# Generate visualizations for a trained product
python scripts/visualize.py --product salmon_fillet --step-hours 2 --per

# Run portfolio across all 110 SKUs
python scripts/run_portfolio.py --episodes 1500 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 4

# Run portfolio under harder conditions (scarce demand, excess inventory)
python scripts/run_portfolio.py --episodes 1500 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 4 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999

# Run a single product via portfolio runner
python scripts/run_portfolio.py --products salmon_fillet --episodes 1500

# Run portfolio with transfer learning (category pre-training + per-SKU fine-tuning)
python scripts/run_portfolio.py --episodes 500 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 4 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --transfer-learning --pretrain-episodes 1500
```

## Product Catalog

110 products across 8 categories (7 generated + 1 legacy):

| Category | SKUs | Price Range | Window | Elasticity | Example SKUs |
|----------|------|-------------|--------|------------|--------------|
| **meats** | 15 | $5-15 | 24h | 2.5-3.5 | ground_beef_1lb, chicken_breast, lamb_chop |
| **seafood** | 15 | $7-20 | 12-24h | 2.0-3.0 | salmon_fillet, shrimp_1lb, lobster_tail |
| **vegetables** | 15 | $1.50-5 | 24h | 3.0-4.5 | salad_mix_5oz, asparagus_bunch, mushroom_8oz |
| **fruits** | 15 | $2-7 | 24h | 3.0-4.0 | strawberries_1lb, avocado_3pk, blueberries_6oz |
| **dairy** | 15 | $1.50-6 | 24h | 2.5-3.5 | yogurt_greek_plain, fresh_mozzarella, brie_wheel_8oz |
| **bakery** | 15 | $2.50-7 | 24h | 3.5-4.5 | sourdough_loaf, croissants_4pk, bagels_6pk |
| **deli_prepared** | 15 | $5-14 | 12-24h | 2.5-3.5 | rotisserie_chicken, sushi_roll_california, quiche_lorraine |
| **legacy** | 5 | $2.50-10 | 12-24h | 2.5-4.0 | salad_mix, fresh_chicken, yogurt, bakery_bread, sushi |

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
- **Historical pre-fill**: Seeds replay buffer with baseline policy rollouts before training begins
- **N-step returns**: Propagates rewards across multiple steps (n=5) for faster convergence in short episodes
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

### Portfolio Validation (110 SKUs, hard mode, 9-dim state)

Best configuration: 2h steps, 1500 episodes, PER + prefill + warmup, 0.5x demand, 2x inventory, epsilon_decay=0.999, 9-dim cyclical state

| Metric | Value |
|--------|-------|
| Shaping wins | 53/110 (48%) |
| Beats best baseline | 22/110 (20%) |

**Category breakdown**:

| Category | SKUs | Win% | Avg Rev Delta | Avg Waste Delta |
|----------|------|------|---------------|-----------------|
| meats | 15 | **73%** | +0.0% | +1.4pp |
| vegetables | 15 | 60% | +0.2% | +0.0pp |
| seafood | 15 | 53% | -2.2% | +2.0pp |
| bakery | 15 | 47% | -0.5% | +0.0pp |
| deli_prepared | 15 | 47% | -1.6% | +0.1pp |
| fruits | 15 | 47% | -0.2% | +0.0pp |
| dairy | 15 | 20% | -1.1% | +0.0pp |
| legacy | 5 | 20% | -1.1% | +0.3pp |

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full iteration history and learnings.

## References

1. MDPI (2025). Deep RL for Dynamic Pricing and Ordering of Perishables
2. De Moor et al. (2022). Reward Shaping in Perishable Inventory Management (EJOR)
3. KickstartAI / Albert Heijn. Dynamic Pricing for the Supermarket
4. Wasteless. AI-Powered Dynamic Pricing for Perishable Goods

## License

MIT
