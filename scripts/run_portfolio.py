"""
Portfolio runner: train and evaluate DQN across all 150 catalog SKUs.

Two modes:
  --pooled      Train 7 category-level pooled models (generates checkpoints for TL)
  --pooled-tl   2-phase pipeline: pooled weights → per-SKU fine-tuning → deployment eval

Usage:
    python scripts/run_portfolio.py --config configs/pooled.json --step-hours 2 --workers 7
    python scripts/run_portfolio.py --config configs/pooled_tl.json --step-hours 2 --workers 16
"""

import sys
import os
import argparse
import json
import csv
import time
import traceback
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fresh_rl.product_catalog import (
    generate_catalog,
    get_product_names,
    get_profile,
    get_categories,
)


# ── Pooled category pipeline (runs in worker process) ─────────────────────

def _train_category_pooled(
    category,
    products,
    episodes_per_sku,
    step_hours,
    seed,
    save_dir,
    shaping_ratio,
    use_per,
    prefill,
    prefill_episodes,
    warmup_steps,
    demand_mult=1.0,
    inventory_mult=1.0,
    epsilon_decay=None,
    hidden_dim=64,
    replay_ratio=1,
    batch_size=32,
    buffer_size=10000,
    n_step=1,
    hold_action_prob=0.0,
    tau_start=0.005,
    tau_end=0.005,
    tau_warmup_steps=0,
):
    """Train plain + shaped pooled DQN for one category, evaluate per-SKU."""
    from fresh_rl.pooled_env import PooledCategoryEnv, pooled_prefill
    from fresh_rl.dqn_agent import DQNAgent
    from fresh_rl.product_catalog import generate_catalog

    cat_dir = os.path.join(save_dir, f"_pooled_{category}")
    os.makedirs(cat_dir, exist_ok=True)

    catalog = generate_catalog()

    # Create pooled env
    env = PooledCategoryEnv(
        category=category,
        step_hours=step_hours,
        seed=seed,
        demand_mult=demand_mult,
        inventory_mult=inventory_mult,
    )

    state_dim = env.observation_space.shape[0]  # 14
    n_actions = env.action_space.n

    if epsilon_decay is None:
        epsilon_decay = 0.998 if step_hours == 2 else 0.997

    total_episodes = episodes_per_sku * len(products)

    print(f"  [POOLED] {category}: {len(products)} products, "
          f"{episodes_per_sku} eps/SKU = {total_episodes} total, state_dim={state_dim}")

    # Helper to create a fresh agent
    def _make_agent(reward_shaping, waste_cost_scale=None):
        return DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            lr=5e-4,
            gamma=0.97,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            reward_shaping=reward_shaping,
            waste_cost_scale=waste_cost_scale,
            seed=seed,
            use_per=use_per,
            n_step=n_step,
            hold_action_prob=hold_action_prob,
            tau_start=tau_start,
            tau_end=tau_end,
            tau_warmup_steps=tau_warmup_steps,
        )

    # Collect transitions per product during plain variant training
    from collections import defaultdict
    transitions_by_product = defaultdict(list)

    for variant, shaping in [("plain", False), ("shaped", True)]:
        # Compute initial waste_cost_scale from first product (will be updated per-episode)
        first_env = env._envs[products[0]]
        revenue_scale = first_env.base_price * first_env.initial_inventory
        waste_cost_scale = shaping_ratio * revenue_scale if shaping else None

        agent = _make_agent(reward_shaping=shaping, waste_cost_scale=waste_cost_scale)

        # Prefill
        if prefill:
            print(f"    [{variant}] Prefilling {prefill_episodes} eps/product...")
            n_trans = pooled_prefill(
                env, agent, episodes_per_product=prefill_episodes,
                products=products, seed=seed,
            )
            print(f"    [{variant}] Added {n_trans} transitions")

        # Warmup
        if warmup_steps > 0 and len(agent.replay_buffer) >= agent.batch_size:
            print(f"    [{variant}] Warming up {warmup_steps} steps...")
            for step in range(warmup_steps):
                agent.train_step_fn()

        # Training: round-robin through products
        collect_transitions = (variant == "plain")
        for ep in range(total_episodes):
            product = products[ep % len(products)]
            inner_env = env._envs[product]

            # Update waste_cost_scale per-product for shaped variant
            if shaping:
                rev_scale = inner_env.base_price * inner_env.initial_inventory
                agent.waste_cost_scale = shaping_ratio * rev_scale

            obs, _ = env.reset(
                seed=seed + ep,
                options={"product": product},
            )
            done = False

            while not done:
                mask = env.action_masks()
                action = agent.select_action(obs, action_mask=mask)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_mask = env.action_masks() if not done else np.ones(n_actions, dtype=bool)

                # Collect raw transitions for per-SKU TL prefill (plain only)
                if collect_transitions:
                    transitions_by_product[product].append(
                        (obs.copy(), action, reward, next_obs.copy(), done, next_mask.copy())
                    )

                agent.store_transition(obs, action, reward, next_obs, done, next_mask)
                for _ in range(replay_ratio):
                    agent.train_step_fn()
                obs = next_obs

            agent.decay_epsilon()

            if (ep + 1) % 500 == 0:
                print(f"    [{variant}] {category}: ep {ep+1}/{total_episodes}, eps={agent.epsilon:.3f}")

        # Save model
        model_path = os.path.join(cat_dir, f"pooled_{category}_{variant}_{step_hours}h.pt")
        agent.save(model_path)
        print(f"    [{variant}] Saved: {model_path}")

        # Save per-product transitions after plain variant training
        if collect_transitions and transitions_by_product:
            trans_dir = os.path.join(cat_dir, "transitions")
            os.makedirs(trans_dir, exist_ok=True)
            for prod, trans_list in transitions_by_product.items():
                states, actions, rewards, next_states, dones, masks = zip(*trans_list)
                np.savez_compressed(os.path.join(trans_dir, f"{prod}.npz"),
                    states=np.array(states, dtype=np.float32),
                    actions=np.array(actions, dtype=np.int64),
                    rewards=np.array(rewards, dtype=np.float32),
                    next_states=np.array(next_states, dtype=np.float32),
                    dones=np.array(dones, dtype=bool),
                    masks=np.array(masks, dtype=bool),
                )
            print(f"    [plain] Saved transitions for {len(transitions_by_product)} products to {trans_dir}")

        if variant == "plain":
            agent_plain = agent
        else:
            agent_shaped = agent

    print(f"  [POOLED] {category}: training complete — checkpoints saved")
    return []


# ── Single-product pipeline (runs in worker process) ─────────────────────

def _run_single_product(
    product: str,
    episodes: int,
    eval_episodes: int,
    step_hours: int,
    seed: int,
    save_dir: str,
    shaping_ratio: float,
    use_per: bool,
    prefill: bool,
    prefill_episodes: int,
    warmup_steps: int,
    demand_mult: float = 1.0,
    inventory_mult: float = 1.0,
    epsilon_decay: float = None,
    hidden_dim: int = 64,
    replay_ratio: int = 1,
    batch_size: int = 32,
    buffer_size: int = 10000,
    n_step: int = 1,
    hold_action_prob: float = 0.0,
    pooled_tl_plain_path: str = None,
    pooled_tl_shaped_path: str = None,
    prefill_transitions_path: str = None,
    tau_start: float = 0.005,
    tau_end: float = 0.005,
    tau_warmup_steps: int = 0,
    tl_warmup_steps: int = None,
    tl_epsilon_start: float = None,
    tl_epsilon_decay: float = None,
    early_stop_patience: int = None,
):
    """Train plain + shaped DQN for one product, evaluate, return summary dict.

    2-phase pipeline (v4.2):
      Phase 1: Train plain + shaped variants with per-episode baseline replay
      Phase 2: Deploy best variant on fresh demand seeds with low epsilon
    """
    # Imports inside worker to avoid pickling issues
    from scripts.train import train
    from scripts.evaluate import evaluate_policy
    from fresh_rl.environment import MarkdownProductEnv
    from fresh_rl.baselines import get_all_baselines

    product_dir = os.path.join(save_dir, product)
    os.makedirs(product_dir, exist_ok=True)

    catalog = generate_catalog()
    profile = catalog.get(product, {})
    category = profile.get("_category", "unknown")

    # Build env overrides from multipliers
    env_overrides = {}
    if demand_mult != 1.0:
        base_demand = profile.get("base_markdown_demand", 5.0)
        env_overrides["base_markdown_demand"] = round(base_demand * demand_mult, 1)
    if inventory_mult != 1.0:
        base_inv = profile.get("initial_inventory", 20)
        env_overrides["initial_inventory"] = int(base_inv * inventory_mult)

    per_kwargs = dict(
        use_per=use_per,
        prefill=prefill,
        prefill_episodes=prefill_episodes,
        warmup_steps=warmup_steps,
        shaping_ratio=shaping_ratio,
        env_overrides=env_overrides if env_overrides else None,
        epsilon_decay=epsilon_decay,
        hidden_dim=hidden_dim,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        buffer_size=buffer_size,
        n_step=n_step,
        hold_action_prob=hold_action_prob,
        augment_state=True,
        inventory_mult=inventory_mult,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_warmup_steps=tau_warmup_steps,
        tl_warmup_steps=tl_warmup_steps,
        tl_epsilon_start=tl_epsilon_start,
        tl_epsilon_decay=tl_epsilon_decay,
        early_stop_patience=early_stop_patience,
        greedy_eval_n=0,  # Use rolling win rate for checkpoint selection, not simulator greedy eval
        prefill_transitions_path=prefill_transitions_path,
    )

    effective_inv = int(profile.get("initial_inventory", 20) * inventory_mult)
    effective_demand = round(profile.get("base_markdown_demand", 5.0) * demand_mult, 1)

    result = {
        "product": product,
        "category": category,
        "base_price": profile.get("base_price", 0),
        "cost_per_unit": profile.get("cost_per_unit", 0),
        "markdown_window_hours": profile.get("markdown_window_hours", 0),
        "initial_inventory": effective_inv,
        "base_markdown_demand": effective_demand,
        "price_elasticity": profile.get("price_elasticity", 0),
    }

    try:
        # ── Step 1: Evaluate baselines BEFORE training ─────────────────────
        eval_env_kwargs = dict(product_name=product, step_hours=step_hours, seed=seed)
        if env_overrides:
            eval_env_kwargs.update(env_overrides)
        env = MarkdownProductEnv(**eval_env_kwargs)
        from fresh_rl.pooled_env import AugmentedProductEnv
        env = AugmentedProductEnv(env, product, inventory_mult=inventory_mult)
        n_actions = env.action_space.n

        baselines = get_all_baselines(n_actions=n_actions, seed=seed)

        # Baseline caching: deterministic baselines can be cached across runs
        # Phase 1: evaluate baselines on same seeds as training (not separate eval_episodes)
        cache_key = {
            "seed": seed, "step_hours": step_hours, "eval_episodes": episodes,
            "demand_mult": demand_mult, "inventory_mult": inventory_mult,
        }
        cache_path = os.path.join(product_dir, "baseline_cache.json")
        cached_baselines = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cache_data = json.load(f)
                if cache_data.get("cache_key") == cache_key:
                    cached_baselines = cache_data["baselines"]
                    print(f"    {product}: using cached baseline results")
            except Exception:
                pass  # cache corrupt, re-evaluate

        baseline_evals = {}
        if cached_baselines is not None:
            baseline_evals = cached_baselines
        else:
            for policy in baselines:
                name = policy.name if hasattr(policy, "name") else str(policy)
                er = evaluate_policy(env, policy, n_episodes=episodes, seed=seed, is_dqn=False)
                baseline_evals[name] = er
            # Save baseline cache
            try:
                with open(cache_path, "w") as f:
                    json.dump({"cache_key": cache_key, "baselines": baseline_evals}, f, indent=2, default=str)
            except Exception:
                pass  # non-critical

        # Find best baseline policy object + metrics
        best_bl_name = max(baseline_evals, key=lambda k: baseline_evals[k]["mean_reward"])
        best_bl = baseline_evals[best_bl_name]
        best_bl_policy = next(b for b in baselines if (b.name if hasattr(b, "name") else str(b)) == best_bl_name)

        result["best_baseline"] = best_bl_name
        result["best_baseline_reward"] = best_bl["mean_reward"]
        result["best_baseline_revenue"] = best_bl["mean_revenue"]
        result["best_baseline_waste"] = best_bl["mean_waste_rate"]

        # ── Step 2: Train with per-episode baseline replay ─────────────────
        agent_plain, hist_plain = train(
            n_episodes=episodes,
            product=product,
            step_hours=step_hours,
            reward_shaping=False,
            seed=seed,
            save_dir=product_dir,
            pretrained_path=pooled_tl_plain_path,
            best_baseline=best_bl_policy,
            **per_kwargs,
        )

        agent_shaped, hist_shaped = train(
            n_episodes=episodes,
            product=product,
            step_hours=step_hours,
            reward_shaping=True,
            seed=seed,
            save_dir=product_dir,
            pretrained_path=pooled_tl_shaped_path,
            best_baseline=best_bl_policy,
            **per_kwargs,
        )

        # ── Step 3: Select best variant from Phase 1 training ─────────────
        def _daily_metrics(hist):
            ep_r = hist["episode_rewards"]
            bl_r = hist["baseline_rewards"]
            n = len(ep_r)
            window = min(30, n)
            last_ep = ep_r[-window:]
            last_bl = bl_r[-window:]
            wins = sum(1 for e, b in zip(last_ep, last_bl) if b is not None and e > b)
            total = sum(1 for b in last_bl if b is not None)
            win_rate = wins / total if total > 0 else 0
            mean_reward = float(np.mean(last_ep))
            mean_revenue = float(np.mean(hist["episode_revenues"][-window:]))
            mean_waste = float(np.mean(hist["episode_wastes"][-window:]))
            mean_clearance = float(np.mean(hist["episode_clearance"][-window:]))
            return win_rate, mean_reward, mean_revenue, mean_waste, mean_clearance

        plain_wr, plain_mean, plain_rev, plain_wst, plain_clr = _daily_metrics(hist_plain)
        shaped_wr, shaped_mean, shaped_rev, shaped_wst, shaped_clr = _daily_metrics(hist_shaped)

        result["plain_reward"] = plain_mean
        result["plain_revenue"] = plain_rev
        result["plain_waste"] = plain_wst
        result["plain_clearance"] = plain_clr
        result["plain_win_rate"] = plain_wr

        result["shaped_reward"] = shaped_mean
        result["shaped_revenue"] = shaped_rev
        result["shaped_waste"] = shaped_wst
        result["shaped_clearance"] = shaped_clr
        result["shaped_win_rate"] = shaped_wr

        # Deltas
        result["shaped_vs_plain_reward"] = shaped_mean - plain_mean
        result["shaped_vs_plain_revenue_pct"] = (
            (shaped_rev - plain_rev)
            / max(abs(plain_rev), 0.01) * 100
        )
        result["shaped_vs_baseline_revenue_pct"] = (
            (shaped_rev - best_bl["mean_revenue"])
            / max(abs(best_bl["mean_revenue"]), 0.01) * 100
        )
        result["shaping_wins"] = shaped_mean > plain_mean

        # Pick best variant by win rate for deployment
        if shaped_wr >= plain_wr:
            best_variant = "shaped"
            best_suffix = f"{product}_{step_hours}h_per_shaped" if use_per else f"{product}_{step_hours}h_shaped"
            deploy_shaping = True
            best_agent = agent_shaped
        else:
            best_variant = "plain"
            best_suffix = f"{product}_{step_hours}h_per" if use_per else f"{product}_{step_hours}h"
            deploy_shaping = False
            best_agent = agent_plain

        best_checkpoint = os.path.join(product_dir, f"best_greedy_{best_suffix}.pt")

        # ── Step 4: Phase 2 — Deployment with fresh demand ────────────
        # Carry forward Phase 1 buffer (production-realistic: agent keeps its experience)
        deploy_dir = os.path.join(product_dir, "deploy")
        _, hist_deploy = train(
            n_episodes=episodes,
            product=product,
            step_hours=step_hours,
            reward_shaping=deploy_shaping,
            seed=seed + 10000,  # Fresh demand seeds
            save_dir=deploy_dir,
            pretrained_path=best_checkpoint,
            best_baseline=best_bl_policy,
            use_per=use_per,
            prefill=False,  # No synthetic prefill — buffer carried from Phase 1
            warmup_steps=0,  # No gradient warmup (weights already tuned)
            shaping_ratio=shaping_ratio,
            env_overrides=env_overrides if env_overrides else None,
            epsilon_decay=epsilon_decay,
            hidden_dim=hidden_dim,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            buffer_size=buffer_size,
            n_step=n_step,
            hold_action_prob=hold_action_prob,
            augment_state=True,
            inventory_mult=inventory_mult,
            tau_start=tau_start,
            tau_end=tau_end,
            tau_warmup_steps=tau_warmup_steps,
            tl_warmup_steps=0,  # No warmup (weights already tuned)
            tl_epsilon_start=0.0,  # Greedy deployment (no exploration noise)
            tl_epsilon_decay=tl_epsilon_decay,
            epsilon_end=0.0,  # Allow epsilon to stay at 0 (no floor)
            greedy_eval_n=0,  # No greedy eval in deployment (already greedy)
            initial_buffer=best_agent.replay_buffer,  # Carry Phase 1 experience
        )

        # ── Step 5: Derive final metrics from Phase 2 deployment ──────
        deploy_wr, deploy_mean, deploy_rev, deploy_wst, deploy_clr = _daily_metrics(hist_deploy)

        result["beats_baseline"] = deploy_wr > 0.5
        result["best_win_rate"] = deploy_wr
        result["best_variant"] = best_variant
        result["deploy_mean_reward"] = deploy_mean
        result["deploy_revenue"] = deploy_rev
        result["deploy_waste"] = deploy_wst
        result["deploy_clearance"] = deploy_clr
        result["status"] = "ok"

        # Save deployment summary for visualization
        with open(os.path.join(product_dir, "eval_deployment.json"), "w") as f:
            json.dump({
                "variant": best_variant,
                "epsilon": 0.0,
                "seed": seed + 10000,
                "dqn_rewards": hist_deploy["episode_rewards"],
                "baseline_rewards": hist_deploy["baseline_rewards"],
                "win_rate": deploy_wr,
                "mean_reward": deploy_mean,
            }, f, indent=2)

        # Save per-product eval
        with open(os.path.join(product_dir, "eval_summary.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ── Aggregate reporting ──────────────────────────────────────────────────

def print_aggregate_report(results: list):
    """Print console summary table by category."""
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("\n  No successful results to report.")
        return

    # Group by category
    by_cat = {}
    for r in ok_results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)

    # Header
    total_beats = sum(1 for r in ok_results if r["beats_baseline"])
    total = len(ok_results)

    # v4.2 fields: best_variant counts
    has_variant = any("best_variant" in r for r in ok_results)
    if has_variant:
        plain_picks = sum(1 for r in ok_results if r.get("best_variant") == "plain")
        shaped_picks = sum(1 for r in ok_results if r.get("best_variant") == "shaped")
        avg_win_rate = np.mean([r.get("best_win_rate", 0) for r in ok_results]) * 100

    print(f"\n{'='*90}")
    print(f"  PORTFOLIO RESULTS — {total} SKUs")
    print(f"{'='*90}")
    print(f"  Beats best baseline on {total_beats}/{total} SKUs ({total_beats/total*100:.0f}%)")
    if has_variant:
        print(f"  Best variant picks: plain={plain_picks}, shaped={shaped_picks}")
        print(f"  Avg deployment win rate: {avg_win_rate:.0f}%")

    print(f"\n  {'Category':<18} {'SKUs':>5} {'Beats%':>7} {'Avg WR':>7} {'Avg Deploy':>11}")
    print(f"  {'-'*52}")

    for cat in sorted(by_cat.keys()):
        cat_results = by_cat[cat]
        n = len(cat_results)
        beats = sum(1 for r in cat_results if r["beats_baseline"])
        avg_wr = np.mean([r.get("best_win_rate", 0) for r in cat_results]) * 100
        avg_deploy = np.mean([r.get("deploy_mean_reward", r.get("shaped_reward", 0)) for r in cat_results])
        print(
            f"  {cat:<18} {n:>5} {beats/n*100:>6.0f}% "
            f"{avg_wr:>6.0f}% {avg_deploy:>+10.1f}"
        )

    # Errors
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors:
            print(f"    {r['product']}: {r.get('error', 'unknown')}")

    print(f"\n{'='*90}\n")


def save_results(results: list, save_dir: str):
    """Save CSV and JSON summaries."""
    os.makedirs(save_dir, exist_ok=True)
    ok_results = [r for r in results if r["status"] == "ok"]

    # JSON
    json_path = os.path.join(save_dir, "portfolio_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_products": len(results),
            "n_ok": len(ok_results),
            "results": results,
        }, f, indent=2, default=str)

    # CSV
    if ok_results:
        csv_path = os.path.join(save_dir, "portfolio_results.csv")
        fieldnames = [
            "product", "category", "base_price", "markdown_window_hours",
            "plain_reward", "shaped_reward", "shaped_vs_plain_reward",
            "plain_revenue", "shaped_revenue", "shaped_vs_plain_revenue_pct",
            "plain_waste", "shaped_waste",
            "best_baseline", "best_baseline_reward", "shaped_vs_baseline_revenue_pct",
            "shaping_wins", "beats_baseline",
            "best_win_rate", "best_variant", "deploy_mean_reward",
            "deploy_revenue", "deploy_waste", "deploy_clearance",
            "plain_win_rate", "shaped_win_rate",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ok_results)
        print(f"  Saved: {csv_path}")

    print(f"  Saved: {json_path}")


# ── Config loading ────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    """Load JSON config file and flatten nested sections into argparse-compatible dict."""
    with open(config_path) as f:
        cfg = json.load(f)

    flat = {}

    # Mode
    mode = cfg.get("mode", "")
    if mode == "pooled":
        flat["pooled"] = True
    elif mode == "pooled-tl":
        flat["pooled_tl"] = True

    # Top-level scalars
    for key in ("episodes", "eval_episodes", "step_hours", "seed", "workers",
                "save_dir", "pooled_model_dir"):
        if key in cfg:
            flat[key] = cfg[key]

    # Pooled-specific
    if "episodes_per_sku" in cfg:
        flat["pooled_episodes_per_sku"] = cfg["episodes_per_sku"]

    # DQN section
    for key, val in cfg.get("dqn", {}).items():
        flat[key] = val

    # PER section
    per_cfg = cfg.get("per", {})
    if per_cfg.get("enabled"):
        flat["per"] = True
    if per_cfg.get("prefill"):
        flat["prefill"] = True
    for key in ("prefill_episodes", "warmup_steps"):
        if key in per_cfg:
            flat[key] = per_cfg[key]

    # Environment section
    for key, val in cfg.get("environment", {}).items():
        flat[key] = val

    # Tau schedule section
    for key, val in cfg.get("tau_schedule", {}).items():
        flat[key] = val

    # Transfer learning section
    for key, val in cfg.get("transfer_learning", {}).items():
        if val is not None:
            flat[key] = val

    return flat


def _save_effective_config(args, save_dir: str):
    """Save the effective configuration used for this run."""
    os.makedirs(save_dir, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if v is not None}
    path = os.path.join(save_dir, "effective_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio runner: pooled category training or 2-phase pooled-TL pipeline"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (CLI args override config values)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single category (e.g., meats)")
    parser.add_argument("--products", nargs="+", default=None,
                        help="Explicit list of product names")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Training episodes per product")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per product")
    parser.add_argument("--step-hours", type=int, default=4, choices=[2, 4])
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (ProcessPoolExecutor)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip products that already have eval_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results/portfolio")

    # DQN options
    parser.add_argument("--shaping-ratio", type=float, default=0.2)
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--prefill-episodes", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=0)

    # Environment difficulty multipliers
    parser.add_argument("--demand-mult", type=float, default=1.0,
                        help="Multiply base_markdown_demand (e.g., 0.5 = half demand)")
    parser.add_argument("--inventory-mult", type=float, default=1.0,
                        help="Multiply initial_inventory (e.g., 2.0 = double inventory)")
    parser.add_argument("--epsilon-decay", type=float, default=None,
                        help="Epsilon decay rate per episode (default: 0.998 for 2h, 0.997 for 4h)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden layer size for DQN network (default: 64)")
    parser.add_argument("--replay-ratio", type=int, default=None,
                        help="Gradient steps per environment step (default: 2 for pooled-TL, 1 otherwise)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DQN training (default: 32)")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Replay buffer size (default: 10000)")
    parser.add_argument("--n-step", type=int, default=1,
                        help="N-step returns for DQN (default: 1, try 5 for faster credit assignment)")
    parser.add_argument("--hold-action-prob", type=float, default=0.0,
                        help="Probability of choosing hold (current discount) during exploration (default: 0.0)")

    # Tau schedule
    parser.add_argument("--tau-start", type=float, default=0.005,
                        help="Initial tau for soft target updates (default: 0.005)")
    parser.add_argument("--tau-end", type=float, default=0.005,
                        help="Final tau after warmup (default: 0.005)")
    parser.add_argument("--tau-warmup-steps", type=int, default=0,
                        help="Steps to linearly warm tau from tau-start to tau-end (default: 0 = constant)")

    # Transfer learning warmup
    parser.add_argument("--tl-warmup-steps", type=int, default=None,
                        help="Override warmup steps when using pretrained weights (default: 0 = skip warmup)")

    # Transfer learning epsilon
    parser.add_argument("--tl-epsilon-start", type=float, default=None,
                        help="Starting epsilon for TL fine-tuning (default: 0.15 for pooled-TL, 0.3 otherwise)")
    parser.add_argument("--tl-epsilon-decay", type=float, default=None,
                        help="Epsilon decay for TL fine-tuning (default: 0.997 for pooled-TL, standard otherwise)")

    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="Stop training after this many episodes without greedy reward improvement (default: disabled)")

    # Pooled category training
    parser.add_argument("--pooled", action="store_true",
                        help="Train 7 category-level pooled models instead of 150 per-SKU models")
    parser.add_argument("--pooled-episodes-per-sku", type=int, default=2500,
                        help="Training episodes per SKU in pooled mode (default: 2500)")

    # Pooled→per-SKU transfer learning (v2.1+)
    parser.add_argument("--pooled-tl", action="store_true",
                        help="Use pooled category models as initialization for per-SKU fine-tuning")
    parser.add_argument("--pooled-model-dir", type=str, default="results/portfolio_v2_pooled",
                        help="Path to pooled results directory (default: results/portfolio_v2_pooled)")

    # Two-pass parse: first get --config, then apply config as defaults
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        cfg_defaults = _load_config(pre_args.config)
        parser.set_defaults(**cfg_defaults)
        print(f"  Loaded config: {pre_args.config}")

    args = parser.parse_args()

    # Require --pooled or --pooled-tl
    if not args.pooled and not args.pooled_tl:
        parser.error("Must specify --pooled or --pooled-tl mode")

    # Resolve mode-dependent defaults
    if args.replay_ratio is None:
        args.replay_ratio = 2 if args.pooled_tl else 1
    if args.pooled_tl:
        if args.tl_epsilon_start is None:
            args.tl_epsilon_start = 0.15
        if args.tl_epsilon_decay is None:
            args.tl_epsilon_decay = 0.997

    # Build product list
    if args.products:
        products = args.products
    elif args.category:
        products = get_product_names(args.category)
        if not products:
            print(f"  No products found for category '{args.category}'.")
            print(f"  Available categories: {get_categories()}")
            sys.exit(1)
    else:
        products = get_product_names()

    # Resume: skip already-done products
    done = []
    if args.resume:
        remaining = []
        for p in products:
            summary_path = os.path.join(args.save_dir, p, "eval_summary.json")
            if os.path.exists(summary_path):
                done.append(p)
            else:
                remaining.append(p)
        if done:
            print(f"  [RESUME] Skipping {len(done)} already-completed products")
        products = remaining

    if not products:
        print("  No products to run.")
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"  PORTFOLIO RUNNER — {len(products)} products")
    print(f"{'='*70}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Step hours:     {args.step_hours}h")
    print(f"  Shaping ratio:  {args.shaping_ratio}")
    print(f"  Hidden dim:     {args.hidden_dim}")
    print(f"  Replay ratio:   {args.replay_ratio}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Buffer size:    {args.buffer_size}")
    print(f"  N-step returns: {args.n_step}")
    print(f"  Hold action:    {args.hold_action_prob}")
    if args.demand_mult != 1.0:
        print(f"  Demand mult:    {args.demand_mult}x")
    if args.inventory_mult != 1.0:
        print(f"  Inventory mult: {args.inventory_mult}x")
    if args.pooled:
        print(f"  Pooled mode:    ON ({args.pooled_episodes_per_sku} eps/SKU)")
    if args.pooled_tl:
        print(f"  Pooled TL:      ON (from {args.pooled_model_dir})")
        print(f"  TL epsilon:     {args.tl_epsilon_start} (decay: {args.tl_epsilon_decay})")
    if args.early_stop_patience is not None:
        print(f"  Early stopping: patience={args.early_stop_patience} episodes")
    print(f"  Workers:        {args.workers}")
    print(f"  Save dir:       {args.save_dir}")
    if pre_args.config:
        print(f"  Config file:    {pre_args.config}")
    print(f"{'='*70}\n")

    # Save effective config for reproducibility
    _save_effective_config(args, args.save_dir)

    # ── Pooled category training mode ─────────────────────────────────────
    if args.pooled:
        catalog = generate_catalog()
        categories_to_train = {}
        for p in products:
            cat = catalog[p].get("_category", "unknown")
            categories_to_train.setdefault(cat, []).append(p)

        print(f"  Pooled mode: {len(categories_to_train)} categories, "
              f"{len(products)} total SKUs\n")

        pooled_kwargs = dict(
            episodes_per_sku=args.pooled_episodes_per_sku,
            step_hours=args.step_hours,
            seed=args.seed,
            save_dir=args.save_dir,
            shaping_ratio=args.shaping_ratio,
            use_per=args.per,
            prefill=args.prefill,
            prefill_episodes=args.prefill_episodes,
            warmup_steps=args.warmup_steps,
            demand_mult=args.demand_mult,
            inventory_mult=args.inventory_mult,
            epsilon_decay=args.epsilon_decay,
            hidden_dim=args.hidden_dim,
            replay_ratio=args.replay_ratio,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            n_step=args.n_step,
            hold_action_prob=args.hold_action_prob,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_warmup_steps=args.tau_warmup_steps,
        )

        start_time = time.time()

        if args.workers > 1:
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(categories_to_train))
            ) as executor:
                futures = {
                    executor.submit(
                        _train_category_pooled,
                        category=cat,
                        products=cat_products,
                        **pooled_kwargs,
                    ): cat
                    for cat, cat_products in categories_to_train.items()
                }
                for future in as_completed(futures):
                    cat = futures[future]
                    try:
                        future.result()
                        print(f"  [{cat}] Done")
                    except Exception as e:
                        print(f"  [{cat}] FAILED: {e}")
                        traceback.print_exc()
        else:
            for cat, cat_products in categories_to_train.items():
                _train_category_pooled(
                    category=cat,
                    products=cat_products,
                    **pooled_kwargs,
                )

        elapsed = time.time() - start_time
        print(f"\n  Pooled training complete: {len(categories_to_train)} categories, {elapsed/60:.1f} minutes")
        print(f"  Checkpoints saved to: {args.save_dir}")

        return

    # ── Pooled→per-SKU transfer learning mode (v2.1+) ────────────────────
    catalog = generate_catalog()

    # Build per-product pooled model paths and transition paths
    pooled_tl_paths = {}  # product -> (plain_path, shaped_path)
    pooled_tl_trans = {}  # product -> transitions_path or None
    for p in products:
        cat = catalog[p].get("_category", "unknown")
        cat_dir = os.path.join(args.pooled_model_dir, f"_pooled_{cat}")
        plain_path = os.path.join(cat_dir, f"pooled_{cat}_plain_{args.step_hours}h.pt")
        shaped_path = os.path.join(cat_dir, f"pooled_{cat}_shaped_{args.step_hours}h.pt")
        pooled_tl_paths[p] = (plain_path, shaped_path)
        trans_path = os.path.join(cat_dir, "transitions", f"{p}.npz")
        pooled_tl_trans[p] = trans_path if os.path.exists(trans_path) else None

    n_with_trans = sum(1 for v in pooled_tl_trans.values() if v is not None)
    if n_with_trans > 0:
        print(f"  Pooled transitions: found for {n_with_trans}/{len(products)} products")

    # Verify at least one model exists
    missing = [p for p, (pp, sp) in pooled_tl_paths.items()
               if not os.path.exists(pp) and not os.path.exists(sp)]
    if missing:
        cats_missing = sorted(set(catalog[p].get("_category", "unknown") for p in missing))
        print(f"  WARNING: Missing pooled models for {len(missing)} products "
              f"(categories: {cats_missing})")
        print(f"  Looking in: {args.pooled_model_dir}")

    tl_kwargs = dict(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        step_hours=args.step_hours,
        seed=args.seed,
        save_dir=args.save_dir,
        shaping_ratio=args.shaping_ratio,
        use_per=args.per,
        prefill=args.prefill,
        prefill_episodes=args.prefill_episodes,
        warmup_steps=args.warmup_steps,
        demand_mult=args.demand_mult,
        inventory_mult=args.inventory_mult,
        epsilon_decay=args.epsilon_decay,
        hidden_dim=args.hidden_dim,
        replay_ratio=args.replay_ratio,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        n_step=args.n_step,
        hold_action_prob=args.hold_action_prob,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_warmup_steps=args.tau_warmup_steps,
        tl_warmup_steps=args.tl_warmup_steps,
        tl_epsilon_start=args.tl_epsilon_start,
        tl_epsilon_decay=args.tl_epsilon_decay,
        early_stop_patience=args.early_stop_patience,
    )

    all_results = []
    start_time = time.time()

    # Load previously completed results for resume mode
    if args.resume:
        for p in done:
            summary_path = os.path.join(args.save_dir, p, "eval_summary.json")
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                all_results.append({k: v for k, v in data.items() if k != "eval_details"})
            except Exception:
                pass

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_single_product, product=p,
                    pooled_tl_plain_path=pooled_tl_paths[p][0],
                    pooled_tl_shaped_path=pooled_tl_paths[p][1],
                    prefill_transitions_path=pooled_tl_trans[p],
                    **tl_kwargs,
                ): p
                for p in products
            }
            for i, future in enumerate(as_completed(futures), 1):
                product = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    status = "OK" if result["status"] == "ok" else "ERROR"
                    win = "WIN" if result.get("beats_baseline") else "---"
                    print(
                        f"  [{i}/{len(products)}] {product:<30s} "
                        f"{status}  {win}  "
                        f"reward: {result.get('shaped_reward', 0):.1f}"
                    )
                except Exception as e:
                    all_results.append({"product": product, "status": "error", "error": str(e)})
                    print(f"  [{i}/{len(products)}] {product:<30s} FAILED: {e}")
    else:
        for i, p in enumerate(products, 1):
            print(f"\n  [{i}/{len(products)}] Running: {p}")
            plain_path, shaped_path = pooled_tl_paths[p]
            result = _run_single_product(
                product=p,
                pooled_tl_plain_path=plain_path,
                pooled_tl_shaped_path=shaped_path,
                prefill_transitions_path=pooled_tl_trans[p],
                **tl_kwargs,
            )
            all_results.append(result)

            status = "OK" if result["status"] == "ok" else "ERROR"
            win = "WIN" if result.get("beats_baseline") else "---"
            print(
                f"  [{i}/{len(products)}] {p:<30s} "
                f"{status}  {win}  "
                f"reward: {result.get('shaped_reward', 0):.1f}"
            )
            save_results(all_results, args.save_dir)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    save_results(all_results, args.save_dir)
    print_aggregate_report(all_results)

    portfolio_json = os.path.join(args.save_dir, "portfolio_results.json")
    if os.path.exists(portfolio_json):
        from scripts.visualize import generate_portfolio_plots
        generate_portfolio_plots(portfolio_json, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
