"""
Portfolio runner: train and evaluate DQN across all catalog SKUs.

Trains plain DQN and shaped DQN for each product, evaluates against baselines,
then produces aggregate analysis showing shaping win rate across categories.

Usage:
    python scripts/run_portfolio.py [--category meats] [--episodes 500] [--workers 4] [--resume]
    python scripts/run_portfolio.py --category legacy --episodes 200 --eval-episodes 50
    python scripts/run_portfolio.py --episodes 500 --eval-episodes 100
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


# ── Category pre-training (runs in worker process) ───────────────────────

def _pretrain_category(
    category,
    products,
    total_episodes,
    step_hours,
    seed,
    save_dir,
    use_per,
    demand_mult,
    inventory_mult,
    epsilon_decay,
    hidden_dim=64,
    replay_ratio=1,
    batch_size=32,
    buffer_size=10000,
    n_step=1,
):
    """Pre-train one agent on all products in a category."""
    from fresh_rl.environment import MarkdownProductEnv
    from fresh_rl.dqn_agent import DQNAgent
    from fresh_rl.product_catalog import generate_catalog

    os.makedirs(save_dir, exist_ok=True)
    catalog = generate_catalog()

    # Get dims from first product
    sample_env = MarkdownProductEnv(products[0], step_hours=step_hours, seed=seed)
    state_dim = sample_env.observation_space.shape[0]
    n_actions = sample_env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim, n_actions=n_actions, hidden_dim=hidden_dim,
        lr=5e-4, gamma=0.97, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=epsilon_decay or 0.999,
        buffer_size=buffer_size, batch_size=batch_size,
        reward_shaping=False,  # no shaping during pre-training
        seed=seed, use_per=use_per,
        n_step=n_step,
    )

    print(f"  [PRETRAIN] {category}: {len(products)} products, {total_episodes} episodes")

    for ep in range(total_episodes):
        product = products[ep % len(products)]
        profile = catalog[product]

        env_kwargs = dict(product_name=product, step_hours=step_hours, seed=seed + ep)
        if demand_mult != 1.0:
            env_kwargs["base_markdown_demand"] = round(
                profile.get("base_markdown_demand", 5.0) * demand_mult, 1)
        if inventory_mult != 1.0:
            env_kwargs["initial_inventory"] = int(
                profile.get("initial_inventory", 20) * inventory_mult)

        env = MarkdownProductEnv(**env_kwargs)
        obs, _ = env.reset()
        done = False

        while not done:
            mask = env.action_masks()
            action = agent.select_action(obs, action_mask=mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_mask = env.action_masks() if not done else np.ones(n_actions, dtype=bool)
            agent.store_transition(obs, action, reward, next_obs, done, next_mask)
            for _ in range(replay_ratio):
                agent.train_step_fn()
            obs = next_obs

        agent.decay_epsilon()

        if (ep + 1) % 500 == 0:
            print(f"    {category}: episode {ep+1}/{total_episodes}, eps={agent.epsilon:.3f}")

    model_path = os.path.join(save_dir, f"pretrained_{category}_{step_hours}h.pt")
    agent.save(model_path)
    print(f"  [PRETRAIN] {category}: saved to {model_path}")
    return model_path


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
    pretrained_path: str = None,
    hidden_dim: int = 64,
    replay_ratio: int = 1,
    batch_size: int = 32,
    buffer_size: int = 10000,
    n_step: int = 1,
):
    """Train plain + shaped DQN for one product, evaluate, return summary dict."""
    # Imports inside worker to avoid pickling issues
    from scripts.train import train
    from scripts.evaluate import evaluate_policy
    from fresh_rl.environment import MarkdownProductEnv
    from fresh_rl.dqn_agent import DQNAgent
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
        # Train plain DQN
        agent_plain, hist_plain = train(
            n_episodes=episodes,
            product=product,
            step_hours=step_hours,
            reward_shaping=False,
            seed=seed,
            save_dir=product_dir,
            pretrained_path=pretrained_path,
            **per_kwargs,
        )

        # Train shaped DQN
        agent_shaped, hist_shaped = train(
            n_episodes=episodes,
            product=product,
            step_hours=step_hours,
            reward_shaping=True,
            seed=seed,
            save_dir=product_dir,
            pretrained_path=pretrained_path,
            **per_kwargs,
        )

        # Evaluate both + baselines
        eval_env_kwargs = dict(product_name=product, step_hours=step_hours, seed=seed)
        if env_overrides:
            eval_env_kwargs.update(env_overrides)
        env = MarkdownProductEnv(**eval_env_kwargs)
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Set agents to greedy
        agent_plain.epsilon = 0.0
        agent_plain.name = "DQN Plain"
        agent_shaped.epsilon = 0.0
        agent_shaped.name = "DQN Shaped"

        baselines = get_all_baselines(n_actions=n_actions, seed=seed)

        eval_results = {}
        for policy in [agent_plain, agent_shaped] + baselines:
            is_dqn = isinstance(policy, DQNAgent)
            name = policy.name if hasattr(policy, "name") else str(policy)
            er = evaluate_policy(env, policy, n_episodes=eval_episodes, seed=seed, is_dqn=is_dqn)
            eval_results[name] = er

        # Extract key metrics
        plain_r = eval_results["DQN Plain"]
        shaped_r = eval_results["DQN Shaped"]

        result["plain_reward"] = plain_r["mean_reward"]
        result["plain_revenue"] = plain_r["mean_revenue"]
        result["plain_waste"] = plain_r["mean_waste_rate"]
        result["plain_clearance"] = plain_r["mean_clearance_rate"]

        result["shaped_reward"] = shaped_r["mean_reward"]
        result["shaped_revenue"] = shaped_r["mean_revenue"]
        result["shaped_waste"] = shaped_r["mean_waste_rate"]
        result["shaped_clearance"] = shaped_r["mean_clearance_rate"]

        # Best baseline
        baseline_names = [b.name for b in baselines]
        baseline_evals = {k: v for k, v in eval_results.items() if k in baseline_names}
        best_bl_name = max(baseline_evals, key=lambda k: baseline_evals[k]["mean_reward"])
        best_bl = baseline_evals[best_bl_name]

        result["best_baseline"] = best_bl_name
        result["best_baseline_reward"] = best_bl["mean_reward"]
        result["best_baseline_revenue"] = best_bl["mean_revenue"]
        result["best_baseline_waste"] = best_bl["mean_waste_rate"]

        # Deltas
        result["shaped_vs_plain_reward"] = shaped_r["mean_reward"] - plain_r["mean_reward"]
        result["shaped_vs_plain_revenue_pct"] = (
            (shaped_r["mean_revenue"] - plain_r["mean_revenue"])
            / max(abs(plain_r["mean_revenue"]), 0.01) * 100
        )
        result["shaped_vs_baseline_revenue_pct"] = (
            (shaped_r["mean_revenue"] - best_bl["mean_revenue"])
            / max(abs(best_bl["mean_revenue"]), 0.01) * 100
        )
        result["shaping_wins"] = shaped_r["mean_reward"] > plain_r["mean_reward"]
        result["beats_baseline"] = shaped_r["mean_reward"] > best_bl["mean_reward"]
        result["status"] = "ok"

        # Save per-product eval
        with open(os.path.join(product_dir, "eval_summary.json"), "w") as f:
            json.dump({**result, "eval_details": eval_results}, f, indent=2, default=str)

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
    total_wins = sum(1 for r in ok_results if r["shaping_wins"])
    total_beats = sum(1 for r in ok_results if r["beats_baseline"])
    total = len(ok_results)

    print(f"\n{'='*90}")
    print(f"  PORTFOLIO RESULTS — {total} SKUs")
    print(f"{'='*90}")
    print(f"\n  Shaping wins on {total_wins}/{total} SKUs ({total_wins/total*100:.0f}%)")
    print(f"  Beats best baseline on {total_beats}/{total} SKUs ({total_beats/total*100:.0f}%)")

    print(f"\n  {'Category':<18} {'SKUs':>5} {'Win%':>6} {'Avg Rev Delta':>14} {'Avg Waste Delta':>16}")
    print(f"  {'-'*62}")

    for cat in sorted(by_cat.keys()):
        cat_results = by_cat[cat]
        n = len(cat_results)
        wins = sum(1 for r in cat_results if r["shaping_wins"])
        avg_rev = np.mean([r.get("shaped_vs_plain_revenue_pct", 0) for r in cat_results])
        avg_waste = np.mean([
            (r.get("plain_waste", 0) - r.get("shaped_waste", 0)) * 100
            for r in cat_results
        ])
        print(
            f"  {cat:<18} {n:>5} {wins/n*100:>5.0f}% "
            f"{avg_rev:>+13.1f}% {avg_waste:>+15.1f}pp"
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
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ok_results)
        print(f"  Saved: {csv_path}")

    print(f"  Saved: {json_path}")


# ── Portfolio visualization ──────────────────────────────────────────────

def plot_portfolio_summary(results: list, save_dir: str):
    """Generate 4-panel portfolio summary plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if r["status"] == "ok"]
    if len(ok) < 2:
        print("  Skipping plot: need at least 2 successful results.")
        return

    categories = sorted(set(r["category"] for r in ok))
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(categories))
    cat_colors = {cat: cmap(i) for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Portfolio Summary — {len(ok)} SKUs across {len(categories)} categories",
        fontsize=16, fontweight="bold",
    )

    # Panel 1: Scatter — shaped vs plain reward
    ax = axes[0, 0]
    for r in ok:
        ax.scatter(
            r["plain_reward"], r["shaped_reward"],
            c=[cat_colors[r["category"]]], s=40, alpha=0.7,
            edgecolors="white", linewidth=0.5,
        )
    # Diagonal reference line
    all_rewards = [r["plain_reward"] for r in ok] + [r["shaped_reward"] for r in ok]
    lo, hi = min(all_rewards), max(all_rewards)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Plain DQN Reward")
    ax.set_ylabel("Shaped DQN Reward")
    ax.set_title("Shaped vs Plain Reward per SKU")
    # Legend
    for cat in categories:
        ax.scatter([], [], c=[cat_colors[cat]], label=cat, s=40)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Histogram — shaped vs best baseline revenue improvement %
    ax = axes[0, 1]
    deltas = [r["shaped_vs_baseline_revenue_pct"] for r in ok]
    ax.hist(deltas, bins=20, color="#2E86C1", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(np.median(deltas), color="green", linestyle="--", alpha=0.7,
               label=f"median={np.median(deltas):.1f}%")
    ax.set_xlabel("Revenue Improvement vs Best Baseline (%)")
    ax.set_ylabel("Count")
    ax.set_title("Shaped DQN vs Best Baseline Revenue")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Bar — waste rate by category (plain vs shaped)
    ax = axes[1, 0]
    by_cat = {}
    for r in ok:
        by_cat.setdefault(r["category"], []).append(r)
    cat_names = sorted(by_cat.keys())
    x_pos = np.arange(len(cat_names))
    width = 0.35
    plain_wastes = [np.mean([r["plain_waste"] for r in by_cat[c]]) * 100 for c in cat_names]
    shaped_wastes = [np.mean([r["shaped_waste"] for r in by_cat[c]]) * 100 for c in cat_names]
    ax.bar(x_pos - width / 2, plain_wastes, width, label="Plain DQN", color="#E74C3C", alpha=0.7)
    ax.bar(x_pos + width / 2, shaped_wastes, width, label="Shaped DQN", color="#27AE60", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Waste Rate (%)")
    ax.set_title("Waste Rate by Category")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 4: Box plot — shaping reward delta by category
    ax = axes[1, 1]
    box_data = [
        [r["shaped_vs_plain_reward"] for r in by_cat[c]]
        for c in cat_names
    ]
    bp = ax.boxplot(box_data, tick_labels=cat_names, patch_artist=True)
    for patch, cat in zip(bp["boxes"], cat_names):
        patch.set_facecolor(cat_colors[cat])
        patch.set_alpha(0.6)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Shaping Reward Delta")
    ax.set_title("Reward Improvement by Category")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio runner: train/evaluate DQN across all catalog SKUs"
    )
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single category (e.g., meats, legacy)")
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
    parser.add_argument("--replay-ratio", type=int, default=1,
                        help="Gradient steps per environment step (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DQN training (default: 32)")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Replay buffer size (default: 10000)")
    parser.add_argument("--n-step", type=int, default=1,
                        help="N-step returns for DQN (default: 1, try 5 for faster credit assignment)")

    # Transfer learning
    parser.add_argument("--transfer-learning", action="store_true",
                        help="Pre-train per category, then fine-tune per SKU")
    parser.add_argument("--pretrain-episodes", type=int, default=1500,
                        help="Pre-training episodes per category (default: 1500)")

    args = parser.parse_args()

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
    if args.resume:
        done = []
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
    print(f"  Eval episodes:  {args.eval_episodes}")
    print(f"  Step hours:     {args.step_hours}h")
    print(f"  Shaping ratio:  {args.shaping_ratio}")
    print(f"  Hidden dim:     {args.hidden_dim}")
    print(f"  Replay ratio:   {args.replay_ratio}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Buffer size:    {args.buffer_size}")
    print(f"  N-step returns: {args.n_step}")
    if args.demand_mult != 1.0:
        print(f"  Demand mult:    {args.demand_mult}x")
    if args.inventory_mult != 1.0:
        print(f"  Inventory mult: {args.inventory_mult}x")
    if args.transfer_learning:
        print(f"  Transfer learn: ON ({args.pretrain_episodes} pretrain episodes)")
    print(f"  Workers:        {args.workers}")
    print(f"  Save dir:       {args.save_dir}")
    print(f"{'='*70}\n")

    common_kwargs = dict(
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
    )

    # ── Phase 1: Category pre-training (if transfer learning enabled) ────
    pretrained_paths = {}  # product_name -> model_path
    if args.transfer_learning:
        catalog = generate_catalog()
        categories_to_train = {}
        for p in products:
            cat = catalog[p].get("_category", "unknown")
            categories_to_train.setdefault(cat, []).append(p)

        pretrain_dir = os.path.join(args.save_dir, "_pretrained")

        print(f"\n  Phase 1: Pre-training {len(categories_to_train)} categories "
              f"({args.pretrain_episodes} episodes each)\n")

        pretrain_kwargs = dict(
            total_episodes=args.pretrain_episodes,
            step_hours=args.step_hours,
            seed=args.seed,
            save_dir=pretrain_dir,
            use_per=args.per,
            demand_mult=args.demand_mult,
            inventory_mult=args.inventory_mult,
            epsilon_decay=args.epsilon_decay,
            hidden_dim=args.hidden_dim,
            replay_ratio=args.replay_ratio,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            n_step=args.n_step,
        )

        if args.workers > 1:
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(categories_to_train))
            ) as executor:
                futures = {
                    executor.submit(
                        _pretrain_category,
                        category=cat,
                        products=cat_products,
                        **pretrain_kwargs,
                    ): (cat, cat_products)
                    for cat, cat_products in categories_to_train.items()
                }
                for future in as_completed(futures):
                    cat, cat_products = futures[future]
                    model_path = future.result()
                    for p in cat_products:
                        pretrained_paths[p] = model_path
        else:
            for cat, cat_products in categories_to_train.items():
                model_path = _pretrain_category(
                    category=cat,
                    products=cat_products,
                    **pretrain_kwargs,
                )
                for p in cat_products:
                    pretrained_paths[p] = model_path

        print(f"\n  Phase 1 complete. Pre-trained {len(categories_to_train)} category models.")
        print(f"  Phase 2: Fine-tuning {len(products)} SKUs ({args.episodes} episodes each)\n")

    all_results = []
    start_time = time.time()

    # Load previously completed results for aggregate report (resume mode)
    if args.resume:
        for p in done:
            summary_path = os.path.join(args.save_dir, p, "eval_summary.json")
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                # Keep only the top-level keys we need
                all_results.append({k: v for k, v in data.items() if k != "eval_details"})
            except Exception:
                pass

    if args.workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_single_product, product=p,
                    pretrained_path=pretrained_paths.get(p),
                    **common_kwargs,
                ): p
                for p in products
            }
            for i, future in enumerate(as_completed(futures), 1):
                product = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    status = "OK" if result["status"] == "ok" else "ERROR"
                    win = "WIN" if result.get("shaping_wins") else "---"
                    print(
                        f"  [{i}/{len(products)}] {product:<30s} "
                        f"{status}  {win}  "
                        f"reward: {result.get('shaped_reward', 0):.1f}"
                    )
                except Exception as e:
                    all_results.append({"product": product, "status": "error", "error": str(e)})
                    print(f"  [{i}/{len(products)}] {product:<30s} FAILED: {e}")
    else:
        # Sequential execution
        for i, p in enumerate(products, 1):
            print(f"\n  [{i}/{len(products)}] Running: {p}")
            result = _run_single_product(
                product=p,
                pretrained_path=pretrained_paths.get(p),
                **common_kwargs,
            )
            all_results.append(result)

            status = "OK" if result["status"] == "ok" else "ERROR"
            win = "WIN" if result.get("shaping_wins") else "---"
            print(
                f"  [{i}/{len(products)}] {p:<30s} "
                f"{status}  {win}  "
                f"reward: {result.get('shaped_reward', 0):.1f}"
            )

            # Incremental save after each product
            save_results(all_results, args.save_dir)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Final save + report + plot
    save_results(all_results, args.save_dir)
    print_aggregate_report(all_results)
    plot_portfolio_summary(all_results, args.save_dir)


if __name__ == "__main__":
    main()
