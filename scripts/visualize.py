"""
Visualization script: generate plots from training and evaluation results.

Usage:
    python scripts/visualize.py [--product salad_mix] [--step-hours 4] [--save-dir results]
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

# Style config
COLORS = {
    "DQN Agent": "#2E86C1",
    "DQN + Reward Shaping": "#1B4F72",
    "Immediate Deep 70%": "#E74C3C",
    "Linear Progressive": "#E67E22",
    "Backloaded Progressive": "#F39C12",
    "Demand Responsive": "#27AE60",
    "Fixed 20%": "#8E44AD",
    "Fixed 40%": "#9B59B6",
    "Random": "#95A5A6",
}


def smooth(data, window=20):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training_curves(history, save_dir):
    """Plot training curves: reward, revenue, waste, clearance over episodes."""
    step_hours = history.get("step_hours", 4)
    suffix = f"{history['product']}_{step_hours}h"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Markdown Channel RL — Training Progress ({history['product']}, {step_hours}h steps)",
                 fontsize=16, fontweight="bold", y=0.98)

    # Reward curve
    ax = axes[0, 0]
    rewards = history["episode_rewards"]
    ax.plot(rewards, alpha=0.2, color="#2E86C1", linewidth=0.5)
    ax.plot(smooth(rewards), color="#2E86C1", linewidth=2, label="Smoothed (20-ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Revenue curve
    ax = axes[0, 1]
    revenues = history["episode_revenues"]
    ax.plot(revenues, alpha=0.2, color="#27AE60", linewidth=0.5)
    ax.plot(smooth(revenues), color="#27AE60", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Episode Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Waste rate curve
    ax = axes[1, 0]
    wastes = [w * 100 for w in history["episode_wastes"]]
    ax.plot(wastes, alpha=0.2, color="#E74C3C", linewidth=0.5)
    ax.plot(smooth(wastes), color="#E74C3C", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Waste Rate (%)")
    ax.set_title("Episode Waste Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Clearance rate curve
    ax = axes[1, 1]
    clearance = [c * 100 for c in history["episode_clearance"]]
    ax.plot(clearance, alpha=0.2, color="#8E44AD", linewidth=0.5)
    ax.plot(smooth(clearance), color="#8E44AD", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Clearance Rate (%)")
    ax.set_title("Episode Clearance Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"training_curves_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison_bars(eval_results, save_dir):
    """Bar chart comparing all policies on key metrics (4 panels)."""
    results = eval_results["results"]
    product = eval_results["product"]
    step_hours = eval_results.get("step_hours", 4)
    suffix = f"{product}_{step_hours}h"

    names = [r["policy_name"] for r in results]
    revenues = [r["mean_revenue"] for r in results]
    waste_rates = [r["mean_waste_rate"] * 100 for r in results]
    clearance_rates = [r.get("mean_clearance_rate", 0) * 100 for r in results]
    rewards = [r["mean_reward"] for r in results]
    colors = [COLORS.get(n, "#95A5A6") for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Markdown Channel RL — Policy Comparison ({product}, {step_hours}h steps)",
                 fontsize=16, fontweight="bold", y=1.02)

    # Revenue
    ax = axes[0]
    bars = ax.barh(names, revenues, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Revenue ($)")
    ax.set_title("Revenue", fontweight="bold")
    for bar, val in zip(bars, revenues):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"${val:.0f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    # Waste Rate (lower is better)
    ax = axes[1]
    bars = ax.barh(names, waste_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Waste Rate (%)")
    ax.set_title("Waste Rate (lower is better)", fontweight="bold")
    for bar, val in zip(bars, waste_rates):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    # Clearance Rate (higher is better)
    ax = axes[2]
    bars = ax.barh(names, clearance_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Clearance Rate (%)")
    ax.set_title("Clearance Rate (higher is better)", fontweight="bold")
    for bar, val in zip(bars, clearance_rates):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    # Total Reward
    ax = axes[3]
    bars = ax.barh(names, rewards, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Total Reward")
    ax.set_title("Overall Reward (higher is better)", fontweight="bold")
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_width() + 0.3 if val >= 0 else bar.get_width() - 3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"comparison_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_action_distributions(eval_results, save_dir):
    """Plot action (discount) distributions for each policy."""
    results = eval_results["results"]
    product = eval_results["product"]
    step_hours = eval_results.get("step_hours", 4)
    suffix = f"{product}_{step_hours}h"

    # Derive discount labels from saved metadata (fallback to 6-level default)
    discount_levels = eval_results.get("discount_levels", [0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    discount_labels = [f"{int(d*100)}%" for d in discount_levels]
    n_actions = len(discount_labels)

    n_policies = len(results)
    ncols = min(4, n_policies)
    nrows = (n_policies + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_policies == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    fig.suptitle(f"Markdown Channel RL — Discount Distributions ({product}, {step_hours}h steps)",
                 fontsize=14, fontweight="bold")

    for i, r in enumerate(results):
        if i >= len(axes):
            break
        ax = axes[i]
        dist = r.get("action_distribution", [1/n_actions]*n_actions)
        # Pad or truncate distribution to match labels
        if len(dist) < n_actions:
            dist = dist + [0.0] * (n_actions - len(dist))
        elif len(dist) > n_actions:
            dist = dist[:n_actions]
        color = COLORS.get(r["policy_name"], "#95A5A6")
        ax.bar(discount_labels, dist, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(r["policy_name"], fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Frequency")
        ax.grid(True, axis="y", alpha=0.3)
        if n_actions > 6:
            ax.tick_params(axis="x", rotation=45, labelsize=7)

    # Hide unused axes
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, f"action_dist_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_plots(product="salad_mix", step_hours=4, save_dir="results"):
    """Generate all visualization plots."""
    suffix = f"{product}_{step_hours}h"
    print(f"\n  Generating visualizations for: {product} ({step_hours}h steps)")
    print(f"  {'='*50}")

    # Training curves
    history_path = os.path.join(save_dir, f"training_history_{suffix}.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, save_dir)
    else:
        print(f"  [SKIP] No training history found at {history_path}")

    # Evaluation comparison
    eval_path = os.path.join(save_dir, f"evaluation_{suffix}.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
        plot_comparison_bars(eval_results, save_dir)
        plot_action_distributions(eval_results, save_dir)
    else:
        print(f"  [SKIP] No evaluation results found at {eval_path}")

    print(f"\n  Done! Check {save_dir}/ for output PNGs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=str, default="salad_mix")
    parser.add_argument("--step-hours", type=int, default=4, choices=[2, 4],
                        help="Hours per decision step (2 or 4)")
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()
    generate_all_plots(product=args.product, step_hours=args.step_hours, save_dir=args.save_dir)
