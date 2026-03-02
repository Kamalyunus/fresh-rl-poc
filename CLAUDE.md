# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based progressive markdown pricing for perishable products. Double DQN agents learn to price 150 SKUs across 7 categories on an ecommerce markdown channel, outperforming rule-based baselines on reward (revenue - waste).

Three training modes:
- **Pooled TL (v2.1)**: Pooled model weights as initialization for per-SKU fine-tuning, 14-dim state, **95% beats-baseline** — best overall
- **Per-SKU (v1.4)**: 150 separate models, 10-dim state, 86% beats-baseline
- **Pooled (v2)**: 7 category-level models, 14-dim state (10 base + 4 product features), 78% beats-baseline — zero-shot generalization to new SKUs

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train single product
python scripts/train.py --product salmon_fillet --episodes 5000 --step-hours 2 \
    --reward-shaping --per --prefill --warmup-steps 1000 --shaping-ratio 0.2 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5

# Run per-SKU portfolio (150 models)
python scripts/run_portfolio.py --episodes 5000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5

# Run pooled portfolio (7 category models)
python scripts/run_portfolio.py --pooled --pooled-episodes-per-sku 5000 \
    --eval-episodes 100 --step-hours 2 --per --prefill --warmup-steps 1000 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5 --workers 7

# Run pooled->per-SKU transfer learning (best: v2.1, 95%)
python scripts/run_portfolio.py --pooled-tl \
    --pooled-model-dir results/portfolio_v2_pooled \
    --episodes 5000 --eval-episodes 100 \
    --step-hours 2 --per --prefill --warmup-steps 1000 --workers 16 \
    --demand-mult 0.5 --inventory-mult 2.0 --epsilon-decay 0.999 \
    --hidden-dim 128 --n-step 5 --hold-action-prob 0.5

# Visualize portfolio results
python scripts/visualize.py --portfolio results/portfolio_v21_pooled_tl/portfolio_results.json

# List all products
python scripts/train.py --list-products

# --- Production (deployment/) ---

# Nightly batch training (all SKUs)
python -m deployment.batch_train --date 2026-03-01 --workers 16

# Nightly batch training (single SKU)
python -m deployment.batch_train --date 2026-03-01 --sku salmon_fillet
```

## Architecture

```
fresh_rl/
  product_catalog.py    — 150 SKUs, 7 categories, get_product_features() for pooled mode
  environment.py        — MarkdownChannelEnv (base), MarkdownProductEnv (per-product wrapper)
  pooled_env.py         — PooledCategoryEnv, AugmentedProductEnv (14-dim state wrappers)
  dqn_agent.py          — Double DQN, action masking, PER, n-step, reward shaping
  baselines.py          — 7 rule-based policies
  historical_data.py    — Replay buffer pre-filling from baseline rollouts
  prioritized_replay.py — PER buffer with SumTree
  sumtree.py            — SumTree data structure

scripts/
  train.py              — Single-product training loop
  evaluate.py           — evaluate_policy() for greedy rollouts
  run_portfolio.py      — Portfolio runner (per-SKU, pooled, and pooled-TL modes)
  visualize.py          — Single-product + portfolio visualizations (10 portfolio plots)

deployment/
  config.py             — Constants, DQN defaults, ProductionConfig (path management)
  state.py              — StateConstructor: 14-dim state from session data
  session.py            — SessionManager + ActiveSession: daytime session tracking + CSV logging
  inference.py          — PricingAgent: 3-tier model fallback (per-SKU → pooled → baseline)
  etl.py                — SessionETL: session CSV → (s,a,r,s',done,mask) transitions
  batch_train.py        — Nightly batch training CLI (python -m deployment.batch_train)
```

## Key Design Patterns

- **State space**: 10-dim (per-SKU) or 14-dim (pooled). All features normalized to [0,1]. The 4 extra pooled features are observable product attributes (price, cost fraction, inventory, pack size) — no simulator leakage.
- **Action masking**: Progressive constraint (discounts only go deeper) enforced via boolean masks, not reward penalties. Masks propagated through replay buffer for correct TD targets.
- **Reward shaping**: Potential-based (preserves optimal policy). Scale = `shaping_ratio * base_price * initial_inventory`. Set per-product in pooled mode via `agent.waste_cost_scale`.
- **N-step returns**: `NStepAccumulator` sits between `store_transition()` and replay buffer. Flushes at episode boundaries. Compatible with PER and shaping.
- **Pooled training**: `PooledCategoryEnv` holds one `MarkdownProductEnv` per SKU. `reset(options={"product": name})` switches active product. `__getattr__` delegates to active env so baselines work unchanged.
- **Pooled TL (v2.1)**: `AugmentedProductEnv` wraps per-SKU env to append 4 product features (10->14 dim), matching pooled model input. Pooled weights transfer directly via `load_pretrained()` — no weight surgery needed.
- **Evaluation**: `evaluate_policy()` in `scripts/evaluate.py` runs greedy rollouts. Same function used for both per-SKU and pooled modes.
- **Results format**: `portfolio_results.json` has same schema for both modes — `visualize.py` works on either.

## Important Constraints

- Agent runs on **CPU only** (`self.device = torch.device("cpu")`). GPU was tested and found slower for the small 2-layer MLP.
- `pack_size` uses separate RNG (`seed + 10000`) to preserve existing product parameters.
- Old transfer learning (v1.3) is superseded — TL underperforms due to no product identity in pre-training state. Pooled TL (v2.1) fixes this by using 14-dim state with product features.
- Hard mode: `--demand-mult 0.5 --inventory-mult 2.0` is the standard test setting (halved demand, doubled inventory).
- All 150 products have 24h markdown windows (12h/48h removed in v1.2 as structurally unsolvable).

## Documentation

- **DEPLOYMENT.md** — Production deployment guide: data requirements, state vector construction, historical prefill pipeline, online RL architecture, rollout phases, safety guardrails, monitoring, and maintenance.
- **ARCHITECTURE.md** — Technical architecture deep-dive: MDP formulation, algorithms, data structures, training pipeline.
- **EXPERIMENTS.md** — Full experiment log (17 iterations from v1.0 to v2.1).

## Results Directory Structure

```
results/portfolio_v21_pooled_tl/       — Best results (v2.1, 95%)
  portfolio_results.json               — All 150 SKU results
  portfolio_*.png                      — 10 portfolio-level plots (incl. training progress)
  {product_name}/eval_summary.json     — Per-product evaluation

results/portfolio_v2_pooled/           — Pooled category models (v2, 78%)
  _pooled_{category}/                  — Category model checkpoints (.pt, used by v2.1 TL)
```
