# Fresh RL POC — Markdown Channel Intraday Progressive Discounting

A proof-of-concept demonstrating how Reinforcement Learning (DQN) can optimize progressive markdown pricing for perishable products on an ecommerce markdown channel, outperforming rule-based approaches on **revenue**, **waste reduction**, and **clearance rate**.

## Problem

Perishable products (produce, dairy, bakery, meat, prepared foods) that are 1-2 days from expiry are moved to a markdown landing page. The retailer must decide how deeply to discount throughout the day at 4-hour intervals. Discounts can only go deeper (never revert) — this is the **progressive constraint**. Pricing too conservatively risks waste at deadline; pricing too aggressively sacrifices margin.

## Approach

This POC models the markdown channel as a **Markov Decision Process (MDP)** and trains a **Deep Q-Network (DQN)** agent with **action masking** to enforce the progressive discount constraint.

### MDP Formulation

| Component | Design |
|-----------|--------|
| **State** | `[hours_remaining, inventory_remaining, current_discount_idx, time_of_day, day_of_week, recent_velocity]` (6-dim, normalized to [0,1]) |
| **Action** | Discount levels: {20%, 30%, 40%, 50%, 60%, 70%} with progressive constraint |
| **Reward** | Revenue - waste penalty - holding cost + clearance bonus |
| **Transition** | Stochastic demand (Poisson) with price elasticity, intraday pattern, day-of-week effect |

### Key Features

- **Custom Gymnasium environment** with configurable product profiles (salad, chicken, yogurt, bread, sushi)
- **Progressive discount constraint** enforced via action masking in both agent and baselines
- **From-scratch DQN** with experience replay, target network, and masked epsilon-greedy exploration (pure NumPy)
- **Potential-based reward shaping** using urgency heuristic to accelerate learning
- **7 baseline policies** for rigorous comparison
- **Visualization suite** for training curves, policy comparison, action distributions, and clearance rates

## Project Structure

```
fresh-rl-poc/
├── fresh_rl/
│   ├── __init__.py
│   ├── environment.py      # MarkdownChannelEnv + MarkdownProductEnv
│   ├── baselines.py        # 7 rule-based markdown policies
│   └── dqn_agent.py        # DQN agent with action masking (pure NumPy)
├── scripts/
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation: DQN vs baselines
│   ├── visualize.py         # Generate comparison plots
│   └── run_all.py           # Full pipeline: train → eval → viz
├── results/                 # Output directory (gitignored)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (train + evaluate + visualize)
python scripts/run_all.py --product salad_mix --episodes 500

# Or run individual steps:
python scripts/train.py --product salad_mix --episodes 500 --reward-shaping
python scripts/evaluate.py --product salad_mix --episodes 200
python scripts/visualize.py --product salad_mix

# Quick smoke test
python scripts/train.py --product sushi --episodes 50
python scripts/evaluate.py --product sushi --episodes 20
```

### Product Profiles

| Product | Markdown Window | Init. Inventory | Base Price | Demand/Step | Elasticity | Cost |
|---------|----------------|-----------------|------------|-------------|------------|------|
| `salad_mix` | 24h (6 steps) | ~25 | $3.50 | 5.0 | 3.5 | $1.20 |
| `fresh_chicken` | 24h (6 steps) | ~15 | $8.00 | 3.0 | 2.8 | $4.00 |
| `yogurt` | 48h (12 steps) | ~30 | $2.50 | 6.0 | 3.0 | $0.80 |
| `bakery_bread` | 24h (6 steps) | ~18 | $4.00 | 4.5 | 4.0 | $1.50 |
| `sushi` | 12h (3 steps) | ~10 | $10.00 | 2.5 | 2.5 | $5.00 |

## How It Works

### Demand Model

Demand is modeled as a Poisson process modulated by three factors:

```
demand = Poisson(base_markdown_demand × price_effect × intraday_effect × dow_effect)
```

- **Price effect**: `exp(-elasticity × (price/base_price - 1))` — deeper discounts exponentially increase demand
- **Intraday pattern**: night 0.3× → morning 0.5× → lunch 0.9× → afternoon 1.2× → evening peak 1.5× → late 0.8×
- **Day-of-week effect**: weekend multiplier (Sat 1.3×, Sun 1.2×, Mon 0.8×)

### Reward Function

```
R = revenue − waste_penalty − holding_cost + clearance_bonus
```

| Component | Signal | Purpose |
|-----------|--------|---------|
| Revenue | price × qty_sold | Drives profitable sales |
| Waste penalty | cost × 3.0 × unsold_at_deadline | Strongly penalizes leftover inventory |
| Holding cost | 0.02 × price × remaining_inventory | Encourages timely sell-through |
| Clearance bonus | 1.0 × initial_inventory (if all sold) | Rewards full clearance before deadline |

### Reward Shaping

Optional potential-based reward shaping using an urgency signal:

```
Φ(s) = -5.0 × inventory_normalized × (1 - hours_remaining_normalized)
shaped_reward = reward + γ × Φ(s') − Φ(s)
```

Penalizes high inventory with little time remaining. Preserves optimal policy (Ng et al., 1999).

## Baselines

| Policy | Strategy |
|--------|----------|
| Immediate Deep 70% | Always picks the deepest discount |
| Linear Progressive | Evenly steps 20%→70% over the window |
| Backloaded Progressive | 20% first half, ramp 30%→70% second half |
| Demand Responsive | Adjusts based on velocity and urgency |
| Fixed 20% / Fixed 40% | Stays at a constant discount level |
| Random | Random from valid (progressive) actions |

## References

1. MDPI (2025). Deep RL for Dynamic Pricing and Ordering of Perishables
2. De Moor et al. (2022). Reward Shaping in Perishable Inventory Management (EJOR)
3. KickstartAI / Albert Heijn. Dynamic Pricing for the Supermarket
4. Wasteless. AI-Powered Dynamic Pricing for Perishable Goods

## License

MIT
