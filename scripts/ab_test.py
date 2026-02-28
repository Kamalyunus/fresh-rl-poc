"""
A/B test script: compare 4-hour vs 2-hour step configurations.

Trains DQN agents for both configs, evaluates against baselines,
and produces a head-to-head comparison table and chart.

Usage:
    python scripts/ab_test.py [--product salad_mix] [--episodes 500] [--eval-episodes 200] [--seed 42]
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train
from scripts.evaluate import run_evaluation


def run_ab_test(product="salad_mix", n_episodes=500, eval_episodes=200, seed=42, save_dir="results", reward_shaping=False):
    """Train and evaluate both 4h and 2h configs, then compare."""

    configs = [
        {"step_hours": 4, "label": "4h steps (6 actions)"},
        {"step_hours": 2, "label": "2h steps (11 actions)"},
    ]

    ab_results = {}

    for cfg in configs:
        step_hours = cfg["step_hours"]
        label = cfg["label"]
        cfg_dir = os.path.join(save_dir, f"ab_{step_hours}h")

        print(f"\n{'#'*70}")
        print(f"  A/B Test — Training: {label}")
        print(f"{'#'*70}")

        # Train
        agent, history = train(
            n_episodes=n_episodes,
            product=product,
            step_hours=step_hours,
            reward_shaping=reward_shaping,
            seed=seed,
            save_dir=cfg_dir,
        )

        # Evaluate
        print(f"\n  Evaluating {label}...")
        eval_results = run_evaluation(
            product=product,
            step_hours=step_hours,
            n_episodes=eval_episodes,
            seed=seed,
            save_dir=cfg_dir,
        )

        # Find best DQN result
        dqn_results = [r for r in eval_results if "DQN" in r["policy_name"]]
        best_dqn = max(dqn_results, key=lambda x: x["mean_reward"]) if dqn_results else None

        # Find best baseline result
        baseline_results = [r for r in eval_results if "DQN" not in r["policy_name"]]
        best_baseline = max(baseline_results, key=lambda x: x["mean_reward"]) if baseline_results else None

        ab_results[step_hours] = {
            "label": label,
            "best_dqn": best_dqn,
            "best_baseline": best_baseline,
            "all_results": eval_results,
            "training_history": history,
        }

    # Print head-to-head comparison
    print(f"\n\n{'='*80}")
    print(f"  A/B TEST RESULTS — {product}")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<25} {'4h steps':>15} {'2h steps':>15} {'Diff':>12}")
    print(f"  {'-'*67}")

    metrics = [
        ("DQN Reward", "mean_reward", "{:.1f}"),
        ("DQN Revenue ($)", "mean_revenue", "${:.1f}"),
        ("DQN Waste %", "mean_waste_rate", "{:.1f}%"),
        ("DQN Clearance %", "mean_clearance_rate", "{:.1f}%"),
    ]

    for label, key, fmt in metrics:
        val_4h = ab_results[4]["best_dqn"][key] if ab_results[4]["best_dqn"] else 0
        val_2h = ab_results[2]["best_dqn"][key] if ab_results[2]["best_dqn"] else 0

        # Scale percentage metrics for display
        if "%" in fmt:
            disp_4h = fmt.format(val_4h * 100)
            disp_2h = fmt.format(val_2h * 100)
            diff = val_2h * 100 - val_4h * 100
            diff_str = f"{diff:+.1f}pp"
        elif "$" in fmt:
            disp_4h = fmt.format(val_4h)
            disp_2h = fmt.format(val_2h)
            diff = (val_2h - val_4h) / max(abs(val_4h), 0.01) * 100
            diff_str = f"{diff:+.1f}%"
        else:
            disp_4h = fmt.format(val_4h)
            disp_2h = fmt.format(val_2h)
            diff = val_2h - val_4h
            diff_str = f"{diff:+.1f}"

        print(f"  {label:<25} {disp_4h:>15} {disp_2h:>15} {diff_str:>12}")

    # Best baseline comparison
    print(f"\n  {'Best Baseline':<25} ", end="")
    for sh in [4, 2]:
        bl = ab_results[sh]["best_baseline"]
        if bl:
            print(f"{bl['policy_name']:>15}", end=" ")
        else:
            print(f"{'N/A':>15}", end=" ")
    print()

    print(f"{'='*80}")

    # Generate side-by-side comparison chart
    _plot_ab_comparison(ab_results, product, save_dir)

    # Save AB results summary
    summary = {
        "product": product,
        "n_episodes": n_episodes,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "configs": {},
    }
    for sh, data in ab_results.items():
        summary["configs"][str(sh)] = {
            "label": data["label"],
            "best_dqn": data["best_dqn"],
            "best_baseline": data["best_baseline"],
        }
    with open(os.path.join(save_dir, f"ab_comparison_{product}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  A/B results saved to: {save_dir}/ab_comparison_{product}.json")
    return ab_results


def _plot_ab_comparison(ab_results, product, save_dir):
    """Generate side-by-side bar chart comparing 4h vs 2h configs."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"A/B Test: 4h vs 2h Steps — {product}",
                 fontsize=16, fontweight="bold", y=1.02)

    configs = ["4h", "2h"]
    colors = ["#2E86C1", "#E67E22"]
    x = np.arange(len(configs))
    width = 0.5

    metric_defs = [
        ("Mean Reward", "mean_reward", False),
        ("Mean Revenue ($)", "mean_revenue", False),
        ("Waste Rate (%)", "mean_waste_rate", True),
        ("Clearance Rate (%)", "mean_clearance_rate", True),
    ]

    for ax, (title, key, is_pct) in zip(axes, metric_defs):
        vals = []
        for sh in [4, 2]:
            dqn = ab_results[sh]["best_dqn"]
            v = dqn[key] if dqn else 0
            if is_pct:
                v *= 100
            vals.append(v)

        bars = ax.bar(x, vals, width, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            fmt = f"{val:.1f}%" if is_pct else (f"${val:.0f}" if "$" in title else f"{val:.1f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    fmt, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, f"ab_comparison_{product}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test: 4h vs 2h step configurations")
    parser.add_argument("--product", type=str, default="salad_mix",
                        choices=["salad_mix", "fresh_chicken", "yogurt", "bakery_bread", "sushi"])
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes per config")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Evaluation episodes per config")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable reward shaping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()

    run_ab_test(
        product=args.product,
        n_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        save_dir=args.save_dir,
        reward_shaping=args.reward_shaping,
    )
