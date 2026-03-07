# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based progressive markdown pricing for perishable products. Double DQN agents learn to price 150 SKUs across 7 categories on an ecommerce markdown channel, outperforming rule-based baselines on reward (revenue - waste).

Two training modes:
- **Pooled TL (v5.0)**: 2-phase pipeline — offline Phase 1 (gradient steps on baseline data, no env interaction) + online Phase 2 deployment (eps=0.05). Pooled model weights initialize per-SKU fine-tuning. 14-dim state.
- **Pooled (v5.0)**: 7 category-level models trained offline on baseline data, 14-dim state (10 base + 4 product features), generates checkpoints for TL

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run pooled category training (generates checkpoints for TL)
python scripts/run_portfolio.py --config configs/pooled.json

# Run 2-phase pooled TL pipeline (greedy deployment)
python scripts/run_portfolio.py --config configs/pooled_tl.json

# Override config values with CLI args
python scripts/run_portfolio.py --config configs/pooled_tl.json \
    --products salmon_fillet --episodes 10 --save-dir results/test

# Train single product
python scripts/train.py --product salmon_fillet --episodes 5000 --step-hours 2 \
    --reward-shaping --per --prefill --warmup-steps 1000 --shaping-ratio 0.2 \
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

# Custom lookback window
python -m deployment.batch_train --date 2026-03-01 --lookback-days 14 --workers 8

# Nightly batch training + weekly pooled model update
python -m deployment.batch_train --date 2026-03-01 --workers 16 --pooled-update

# Pooled update with custom lookback
python -m deployment.batch_train --date 2026-03-01 --workers 16 \
    --pooled-update --pooled-lookback-days 60
```

## Architecture

```
fresh_rl/
  product_catalog.py    — 150 SKUs, 7 categories, get_product_features() for pooled mode
  environment.py        — MarkdownChannelEnv (base), MarkdownProductEnv (per-product wrapper)
  pooled_env.py         — PooledCategoryEnv, AugmentedProductEnv (14-dim state wrappers)
  dqn_agent.py          — Double DQN, action masking, PER, n-step, reward shaping, buffer persistence
  baselines.py          — 7 rule-based policies
  historical_data.py    — Replay buffer pre-filling from baseline rollouts
  prioritized_replay.py — PER buffer with SumTree, save/load persistence
  sumtree.py            — SumTree data structure

configs/
  pooled.json           — Pooled category training config
  pooled_tl.json        — 2-phase pooled-TL pipeline config

scripts/
  train.py              — Single-product training loop
  evaluate.py           — evaluate_policy() for greedy rollouts
  run_portfolio.py      — Portfolio runner (pooled and pooled-TL modes, --config support)
  visualize.py          — Portfolio visualizations (11 plots)

deployment/
  config.py             — Constants, DQN defaults, metrics/safety thresholds, ProductionConfig (path management)
  state.py              — StateConstructor: 14-dim state from session data
  session.py            — SessionManager + ActiveSession: daytime session tracking + CSV logging
  inference.py          — PricingAgent: 3-tier model fallback (per-SKU → pooled → baseline)
  etl.py                — SessionETL: session CSV → (s,a,r,s',done,mask) transitions
  batch_train.py        — Nightly batch training CLI with metrics, safety rollback, buffer persistence, pooled update
```

## Key Design Patterns

- **State space**: 10-dim (per-SKU) or 14-dim (pooled). All features normalized to [0,1]. The 4 extra pooled features are observable product attributes (price, cost fraction, inventory, pack size) — no simulator leakage.
- **Action masking**: Progressive constraint (discounts only go deeper) enforced via boolean masks, not reward penalties. Masks propagated through replay buffer for correct TD targets.
- **Reward shaping**: Potential-based (preserves optimal policy). Scale = `shaping_ratio * base_price * initial_inventory`. Set per-product in pooled mode via `agent.waste_cost_scale`.
- **N-step returns**: `NStepAccumulator` sits between `store_transition()` and replay buffer. Flushes at episode boundaries. Compatible with PER and shaping.
- **Pooled training**: `PooledCategoryEnv` holds one `MarkdownProductEnv` per SKU. `reset(options={"product": name})` switches active product. `__getattr__` delegates to active env so baselines work unchanged.
- **Pooled TL (v2.1)**: `AugmentedProductEnv` wraps per-SKU env to append 4 product features (10->14 dim), matching pooled model input. Pooled weights transfer directly via `load_pretrained()` — no weight surgery needed.
- **Evaluation (greedy)**: `evaluate_policy()` in `scripts/evaluate.py` runs greedy rollouts. Used for baseline identification in pooled-TL pipeline.
- **Offline training (v5.0)**: `offline_steps` param in `train()` runs gradient steps on buffered data instead of online episodes. `-1` = auto-compute from buffer size (1 epoch). Used for both pooled category training and TL Phase 1. No epsilon decay, no baseline replay during offline training.
- **Pooled prefill with per-product collection**: `pooled_prefill(collect_per_product=True)` returns per-product transitions alongside total count. Transitions saved to `_pooled_{cat}/transitions/` for per-SKU TL fine-tuning. Uses deterministic seeds (`seed + ep`).
- **Greedy backtest**: `_greedy_backtest()` runs trained agents greedily on 365 training seeds after offline Phase 1 to select best variant (plain vs shaped). `_greedy_backtest_baseline()` runs the best baseline on same seeds for comparison. Replaces online reward comparison which isn't available in offline mode.
- **2-phase pipeline (v5.0)**: Phase 1 (offline) loads pooled transitions, runs N gradient steps, saves checkpoint. Phase 2 (online) loads best Phase 1 checkpoint, runs 365 fresh episodes (seed + 10000) with epsilon=0.05. Buffer carried from Phase 1 to Phase 2. `beats_baseline` = Phase 2 last-30-day win rate > 50%. Saves `eval_deployment.json` per product.
- **Results format**: `portfolio_results.json` has same schema for both modes — `visualize.py` works on either. v4.2+ results include `best_win_rate`, `best_variant`, `deploy_mean_reward`.
- **Model metrics**: `compute_avg_q()` and `mean_loss` tracked as proxy quality signals in production (no simulator). Written to `metrics.json` alongside checkpoints.
- **Safety rollback**: Auto-reverts to `agent_prev.pt` when avg_q drops >30% or loss spikes >3x. Resets epsilon to 0.15 for re-exploration.
- **Epsilon floor**: Production epsilon clamped to `EPSILON_FLOOR=0.05` to prevent over-exploitation.
- **Buffer persistence**: PER buffer saved/loaded across nightly training runs via `save_buffer()`/`load_buffer()`. Preserves priorities and sampling state.
- **Weekly pooled update**: `--pooled-update` flag retrains category-level pooled models from cross-SKU session data, keeping cold-start models fresh.
- **Early stopping**: `--early-stop-patience N` stops training after N episodes without greedy reward improvement. Saves `best_greedy_{suffix}.pt` on each improvement, reloads best checkpoint at end. History includes `early_stopped` and `early_stop_episode`.
- **Baseline caching**: Deterministic baseline evaluations cached to `baseline_cache.json` per product (keyed by seed, step_hours, eval_episodes, demand/inventory mults). Skips 7 baseline evals on re-runs.
- **Configurable TL epsilon**: `--tl-epsilon-start` and `--tl-epsilon-decay` override hardcoded TL exploration schedule. Defaults resolve based on mode (pooled-TL: 0.15/0.997, standard: 0.3/agent default).
- **Adaptive replay ratio**: `--replay-ratio` defaults to 2 for pooled-TL mode, 1 otherwise. **Caution**: 2x replay ratio combined with aggressive epsilon schedule regressed performance in v3.1 (87% vs v3.0's 94%).
- **Config files**: `configs/*.json` define experiment parameters. `--config` loads a JSON file as defaults; CLI args override. `effective_config.json` saved to results dir for reproducibility.

## Important Constraints

- Agent runs on **CPU only** (`self.device = torch.device("cpu")`). GPU was tested and found slower for the small 2-layer MLP.
- `pack_size` uses separate RNG (`seed + 10000`) to preserve existing product parameters.
- Old transfer learning (v1.3) is superseded — TL underperforms due to no product identity in pre-training state. Pooled TL (v2.1) fixes this by using 14-dim state with product features.
- Hard mode: `--demand-mult 0.5 --inventory-mult 2.0` is the standard test setting (halved demand, doubled inventory).
- All 150 products have 24h markdown windows (12h/48h removed in v1.2 as structurally unsolvable).

## Documentation

- **DEPLOYMENT.md** — Production deployment guide: data requirements, state vector construction, historical prefill pipeline, online RL architecture, rollout phases, safety guardrails, monitoring, and maintenance.
- **ARCHITECTURE.md** — Technical architecture deep-dive: MDP formulation, algorithms, data structures, training pipeline.
- **EXPERIMENTS.md** — Full experiment log (27 iterations from v1.0 through v5.0, production hardening, time-to-value visualization, offline training, and 2-phase production-realistic evaluation).

## Results Directory Structure

```
results/portfolio_v21_pooled_tl/       — Best results (v2.1, 95%)
  portfolio_results.json               — All 150 SKU results
  portfolio_*.png                      — 11 portfolio-level plots (incl. training progress, time-to-value)
  {product_name}/eval_summary.json     — Per-product evaluation

results/portfolio_v30_pooled_tl/       — v3.0 results (94%, 3000ep, tau schedule)

results/portfolio_v31_sample_eff/      — v3.1 results (87%, sample efficiency experiment — regression)

results/portfolio_v32_conservative_es/ — v3.2 results (92%, conservative early stopping)

results/portfolio_v40_365ep_pooled/    — v4.0 pooled models (53%, 365ep/SKU realistic budget)
  _pooled_{category}/                  — Category model checkpoints (.pt)

results/portfolio_v40_365ep_pooled_tl/ — v4.0 TL results (83%, 365ep realistic budget)

results/portfolio_v41_production_eval/ — v4.1 TL results (39%, production-realistic daily eval)
  portfolio_results.json               — Includes per-episode baseline_rewards for paired comparison
  {product_name}/training_history_*.json — Contains baseline_rewards array

results/portfolio_v42_2phase_eval/     — v4.2 TL results (57%, 2-phase production-realistic eval)
  portfolio_results.json               — Includes best_variant, deploy_mean_reward
  {product_name}/eval_deployment.json  — Phase 2 deployment DQN + baseline reward arrays
  {product_name}/deploy/               — Phase 2 training files (agent, history)

results/portfolio_v2_pooled/           — Pooled category models (v2, 78%)
  _pooled_{category}/                  — Category model checkpoints (.pt, used by TL)

results/portfolio_v50_offline_pooled/  — v5.0 offline pooled models
  _pooled_{category}/                  — Category model checkpoints (.pt)
  _pooled_{category}/transitions/      — Per-product transitions for TL

results/portfolio_v50_offline_pooled_tl/ — v5.0 offline TL results (65%)
  portfolio_results.json               — All 150 SKU results
  {product_name}/eval_deployment.json  — Phase 2 deployment DQN + baseline reward arrays
  {product_name}/deploy/               — Phase 2 training files
```
