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
    "DQN Agent + PER": "#2E86C1",
    "DQN + Reward Shaping": "#1B4F72",
    "DQN + Shaping + PER": "#1B4F72",
    "DQN Plain": "#2E86C1",
    "DQN Shaped": "#1B4F72",
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


def _suffix(product, step_hours, use_per=False):
    """Build standard file suffix."""
    s = f"{product}_{step_hours}h"
    if use_per:
        s += "_per"
    return s


# ── Existing plot functions ──────────────────────────────────────────────

def plot_training_curves(history, save_dir):
    """Plot training curves: reward, revenue, waste, clearance over episodes."""
    step_hours = history.get("step_hours", 4)
    suffix = _suffix(history["product"], step_hours, history.get("use_per", False))

    n_episodes = len(history["episode_rewards"])
    window = max(20, min(100, n_episodes // 20))

    greedy_eval = history.get("greedy_eval", [])
    has_greedy = len(greedy_eval) > 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Markdown Channel RL — Training Progress ({history['product']}, {step_hours}h steps)",
                 fontsize=16, fontweight="bold", y=0.98)

    # Reward curve
    ax = axes[0, 0]
    rewards = history["episode_rewards"]
    ax.plot(rewards, alpha=0.15, color="#2E86C1", linewidth=0.5)
    ax.plot(smooth(rewards, window), color="#2E86C1", linewidth=2, label=f"Smoothed ({window}-ep)")
    if has_greedy:
        g_eps = [g["episode"] for g in greedy_eval]
        g_rew = [g["reward"] for g in greedy_eval]
        ax.plot(g_eps, g_rew, "o-", color="#E74C3C", linewidth=2, markersize=4, label="Greedy eval")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Revenue curve
    ax = axes[0, 1]
    revenues = history["episode_revenues"]
    ax.plot(revenues, alpha=0.15, color="#27AE60", linewidth=0.5)
    ax.plot(smooth(revenues, window), color="#27AE60", linewidth=2, label=f"Smoothed ({window}-ep)")
    if has_greedy:
        g_rev = [g["revenue"] for g in greedy_eval]
        ax.plot(g_eps, g_rev, "o-", color="#E74C3C", linewidth=2, markersize=4, label="Greedy eval")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Episode Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Waste rate curve
    ax = axes[1, 0]
    wastes = [w * 100 for w in history["episode_wastes"]]
    ax.plot(wastes, alpha=0.15, color="#E74C3C", linewidth=0.5)
    ax.plot(smooth(wastes, window), color="#E74C3C", linewidth=2, label=f"Smoothed ({window}-ep)")
    if has_greedy:
        g_waste = [g["waste"] * 100 for g in greedy_eval]
        ax.plot(g_eps, g_waste, "o-", color="#1B4F72", linewidth=2, markersize=4, label="Greedy eval")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Waste Rate (%)")
    ax.set_title("Episode Waste Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Clearance rate curve
    ax = axes[1, 1]
    clearance = [c * 100 for c in history["episode_clearance"]]
    ax.plot(clearance, alpha=0.15, color="#8E44AD", linewidth=0.5)
    ax.plot(smooth(clearance, window), color="#8E44AD", linewidth=2, label=f"Smoothed ({window}-ep)")
    if has_greedy:
        g_clear = [g["clearance"] * 100 for g in greedy_eval]
        ax.plot(g_eps, g_clear, "o-", color="#E74C3C", linewidth=2, markersize=4, label="Greedy eval")
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
    suffix = _suffix(product, step_hours, eval_results.get("use_per", False))

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
    suffix = _suffix(product, step_hours, eval_results.get("use_per", False))

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

    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, f"action_dist_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── New visualization functions ──────────────────────────────────────────

def plot_policy_heatmap(agent, env, save_dir, suffix):
    """
    2D heatmap of learned policy: X=hours_remaining, Y=inventory_remaining,
    color=chosen discount%. Reveals the strategy surface.
    """
    import seaborn as sns

    discount_levels = env.DISCOUNT_LEVELS
    grid_size = 20
    hours_range = np.linspace(0, 1, grid_size)
    inv_range = np.linspace(0, 1, grid_size)

    policy_grid = np.zeros((grid_size, grid_size))

    old_eps = agent.epsilon
    agent.epsilon = 0.0

    for i, inv_norm in enumerate(reversed(inv_range)):
        for j, hours_norm in enumerate(hours_range):
            obs = np.array([hours_norm, inv_norm, 0.0, 0.5, 0.5, 0.3], dtype=np.float32)
            mask = np.ones(len(discount_levels), dtype=bool)
            action = agent.select_action(obs, action_mask=mask)
            policy_grid[i, j] = discount_levels[action] * 100

    agent.epsilon = old_eps

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        policy_grid, ax=ax,
        xticklabels=[f"{h:.0%}" for h in hours_range[::2]],
        yticklabels=[f"{v:.0%}" for v in reversed(inv_range[::2])],
        cmap="YlOrRd", cbar_kws={"label": "Discount %"},
        vmin=20, vmax=70,
    )
    ax.set_xticks(np.arange(0, grid_size, 2) + 0.5)
    ax.set_yticks(np.arange(0, grid_size, 2) + 0.5)
    ax.set_xlabel("Hours Remaining (normalized)")
    ax.set_ylabel("Inventory Remaining (normalized)")
    ax.set_title(f"Learned Policy Heatmap — {suffix}")

    plt.tight_layout()
    path = os.path.join(save_dir, f"policy_heatmap_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_episode_walkthrough(env, agent, save_dir, suffix):
    """
    4-panel (2x2) trace of one greedy episode showing:
    - Inventory per step (colored by discount)
    - Discount % over steps
    - Units sold per step
    - Cumulative revenue
    """
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    obs, _ = env.reset(seed=123)
    inventories, discounts, sales, cum_revenues = [], [], [], []
    total_rev = 0.0
    done = False

    while not done:
        mask = env.action_masks()
        action = agent.select_action(obs, action_mask=mask)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        inventories.append(info["total_inventory"])
        discounts.append(info["discount"] * 100)
        sales.append(info["units_sold"])
        total_rev += info["revenue"]
        cum_revenues.append(total_rev)

    agent.epsilon = old_eps

    steps = np.arange(1, len(inventories) + 1)
    discount_colors = plt.cm.YlOrRd(np.array(discounts) / 70.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Episode Walkthrough — {suffix}", fontsize=16, fontweight="bold")

    # Inventory remaining
    ax = axes[0, 0]
    ax.bar(steps, inventories, color=discount_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Inventory Remaining")
    ax.set_title("Inventory (colored by discount)")
    ax.grid(True, axis="y", alpha=0.3)

    # Discount over steps
    ax = axes[0, 1]
    ax.step(steps, discounts, where="mid", color="#E74C3C", linewidth=2)
    ax.fill_between(steps, discounts, step="mid", alpha=0.2, color="#E74C3C")
    ax.set_xlabel("Step")
    ax.set_ylabel("Discount (%)")
    ax.set_title("Discount Progression")
    ax.set_ylim(15, 75)
    ax.grid(True, alpha=0.3)

    # Units sold per step
    ax = axes[1, 0]
    ax.bar(steps, sales, color="#2E86C1", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Units Sold")
    ax.set_title("Units Sold per Step")
    ax.grid(True, axis="y", alpha=0.3)

    # Cumulative revenue
    ax = axes[1, 1]
    ax.plot(steps, cum_revenues, color="#27AE60", linewidth=2)
    ax.fill_between(steps, cum_revenues, alpha=0.2, color="#27AE60")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.set_title("Cumulative Revenue")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"episode_walkthrough_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_revenue_waste_pareto(eval_results, save_dir, suffix):
    """
    Scatter: X=waste_rate%, Y=revenue for each policy.
    DQN variants as diamonds, baselines as circles. Annotated with names.
    """
    results = eval_results["results"]

    fig, ax = plt.subplots(figsize=(10, 7))

    dqn_names = {"DQN Agent", "DQN Agent + PER", "DQN + Reward Shaping",
                 "DQN + Shaping + PER", "DQN Plain", "DQN Shaped"}

    for r in results:
        name = r["policy_name"]
        waste = r["mean_waste_rate"] * 100
        revenue = r["mean_revenue"]
        is_dqn = name in dqn_names
        color = COLORS.get(name, "#95A5A6")
        marker = "D" if is_dqn else "o"
        size = 120 if is_dqn else 80

        ax.scatter(waste, revenue, c=color, marker=marker, s=size,
                   edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(name, (waste, revenue), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, alpha=0.8)

    ax.set_xlabel("Waste Rate (%)")
    ax.set_ylabel("Mean Revenue ($)")
    ax.set_title(f"Revenue vs Waste Tradeoff — {suffix}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"pareto_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_dashboard(history, save_dir):
    """
    3-panel horizontal training dashboard:
    - Loss curve (raw + smoothed, log scale)
    - Epsilon decay over episodes
    - Greedy eval reward + waste% (dual y-axis)
    """
    step_hours = history.get("step_hours", 4)
    suffix = _suffix(history["product"], step_hours, history.get("use_per", False))
    n_episodes = len(history["episode_rewards"])
    window = max(20, min(100, n_episodes // 20))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Training Dashboard — {history['product']} ({step_hours}h steps)",
                 fontsize=16, fontweight="bold")

    # Panel 1: Loss curve
    ax = axes[0]
    losses = history.get("losses", [])
    if losses:
        # Filter out any zero/negative for log scale
        losses_plot = [max(l, 1e-8) for l in losses]
        ax.plot(losses_plot, alpha=0.15, color="#E67E22", linewidth=0.5)
        if len(losses_plot) >= window:
            ax.plot(
                np.arange(window - 1, len(losses_plot)),
                smooth(losses_plot, window),
                color="#E67E22", linewidth=2, label=f"Smoothed ({window})"
            )
        ax.set_yscale("log")
        ax.legend()
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # Panel 2: Epsilon decay
    ax = axes[1]
    eps_decay = history.get("epsilon_decay", 0.997)
    epsilons = []
    eps = 1.0
    for _ in range(n_episodes):
        epsilons.append(eps)
        eps = max(0.05, eps * eps_decay)
    ax.plot(epsilons, color="#8E44AD", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title(f"Epsilon Decay (rate={eps_decay})")
    ax.grid(True, alpha=0.3)

    # Panel 3: Greedy eval reward + waste (dual y-axis)
    ax = axes[2]
    greedy_eval = history.get("greedy_eval", [])
    if greedy_eval:
        g_eps = [g["episode"] for g in greedy_eval]
        g_rew = [g["reward"] for g in greedy_eval]
        g_waste = [g["waste"] * 100 for g in greedy_eval]

        color_r = "#2E86C1"
        ax.plot(g_eps, g_rew, "o-", color=color_r, linewidth=2, markersize=4, label="Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Greedy Reward", color=color_r)
        ax.tick_params(axis="y", labelcolor=color_r)

        ax2 = ax.twinx()
        color_w = "#E74C3C"
        ax2.plot(g_eps, g_waste, "s--", color=color_w, linewidth=2, markersize=4, label="Waste %")
        ax2.set_ylabel("Waste Rate (%)", color=color_w)
        ax2.tick_params(axis="y", labelcolor=color_w)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax.set_title("Greedy Evaluation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"training_dashboard_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_category_heatmap(portfolio_results, save_dir):
    """
    Seaborn heatmap: Categories (rows) x Metrics (columns) from portfolio JSON.
    Metrics: revenue, waste rate, clearance rate, shaping delta.
    """
    import seaborn as sns
    import pandas as pd

    results = portfolio_results.get("results", portfolio_results)
    if isinstance(results, dict):
        results = results.get("results", [])

    ok = [r for r in results if r.get("status") == "ok"]
    if len(ok) < 2:
        print("  [SKIP] Category heatmap needs at least 2 successful results.")
        return

    # Group by category
    by_cat = {}
    for r in ok:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    categories = sorted(by_cat.keys())
    metrics = {
        "Shaped Revenue": lambda rs: np.mean([r.get("shaped_revenue", 0) for r in rs]),
        "Waste Rate (%)": lambda rs: np.mean([r.get("shaped_waste", 0) * 100 for r in rs]),
        "Clearance (%)": lambda rs: np.mean([r.get("shaped_clearance", 0) * 100 for r in rs]),
        "Shaping Delta": lambda rs: np.mean([r.get("shaped_vs_plain_reward", 0) for r in rs]),
    }

    data = []
    for cat in categories:
        row = {}
        for mname, mfn in metrics.items():
            row[mname] = mfn(by_cat[cat])
        data.append(row)

    df = pd.DataFrame(data, index=categories)

    # Normalize each column to [0,1] for comparable heatmap colors
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(10, max(6, len(categories) * 0.6)))
    sns.heatmap(
        df_norm, ax=ax, annot=df.values, fmt=".1f",
        cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Normalized Score"},
    )
    ax.set_title("Category Performance Heatmap")
    ax.set_ylabel("Category")

    plt.tight_layout()
    path = os.path.join(save_dir, "category_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_action_progression(env, agent, save_dir, suffix, n_episodes=50):
    """
    How discounts escalate over episode steps: run greedy episodes,
    plot individual traces (faint) + mean (bold) + std band.
    """
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    all_traces = []
    max_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        trace = []
        done = False
        while not done:
            mask = env.action_masks()
            action = agent.select_action(obs, action_mask=mask)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            trace.append(info["discount"] * 100)
        all_traces.append(trace)
        max_steps = max(max_steps, len(trace))

    agent.epsilon = old_eps

    # Pad traces to max_steps with NaN
    padded = np.full((n_episodes, max_steps), np.nan)
    for i, trace in enumerate(all_traces):
        padded[i, :len(trace)] = trace

    steps = np.arange(1, max_steps + 1)
    mean_discount = np.nanmean(padded, axis=0)
    std_discount = np.nanstd(padded, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual traces (faint)
    for i in range(min(n_episodes, 30)):
        trace = all_traces[i]
        ax.plot(np.arange(1, len(trace) + 1), trace,
                color="#2E86C1", alpha=0.1, linewidth=0.8)

    # Mean + std band
    ax.plot(steps, mean_discount, color="#1B4F72", linewidth=2.5, label="Mean discount")
    ax.fill_between(steps,
                    np.clip(mean_discount - std_discount, 15, 75),
                    np.clip(mean_discount + std_discount, 15, 75),
                    alpha=0.2, color="#2E86C1", label="\u00b11 std")

    ax.set_xlabel("Step")
    ax.set_ylabel("Discount (%)")
    ax.set_title(f"Discount Progression — {suffix} ({n_episodes} episodes)")
    ax.set_ylim(15, 75)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"action_progression_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Orchestrator ─────────────────────────────────────────────────────────

def generate_all_plots(product="salad_mix", step_hours=4, save_dir="results", use_per=False):
    """Generate all visualization plots."""
    suffix = _suffix(product, step_hours, use_per)
    print(f"\n  Generating visualizations for: {product} ({step_hours}h steps{', PER' if use_per else ''})")
    print(f"  {'='*50}")

    # Training curves + dashboard
    history_path = os.path.join(save_dir, f"training_history_{suffix}.json")
    history = None
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, save_dir)
        plot_training_dashboard(history, save_dir)
    else:
        print(f"  [SKIP] No training history found at {history_path}")

    # Evaluation comparison + pareto
    eval_path = os.path.join(save_dir, f"evaluation_{suffix}.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
        plot_comparison_bars(eval_results, save_dir)
        plot_action_distributions(eval_results, save_dir)
        plot_revenue_waste_pareto(eval_results, save_dir, suffix)
    else:
        print(f"  [SKIP] No evaluation results found at {eval_path}")

    # Agent-dependent plots: load trained agent
    agent_path = os.path.join(save_dir, f"best_agent_{suffix}.pt")
    if os.path.exists(agent_path):
        from fresh_rl.environment import MarkdownProductEnv
        from fresh_rl.dqn_agent import DQNAgent

        env = MarkdownProductEnv(product_name=product, step_hours=step_hours, seed=42)
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        agent = DQNAgent(state_dim=state_dim, n_actions=n_actions, seed=42)
        agent.load(agent_path)
        agent.epsilon = 0.0

        plot_policy_heatmap(agent, env, save_dir, suffix)
        plot_episode_walkthrough(env, agent, save_dir, suffix)
        plot_action_progression(env, agent, save_dir, suffix)
    else:
        print(f"  [SKIP] No trained agent found at {agent_path}")

    # Portfolio category heatmap
    portfolio_path = os.path.join(save_dir, "portfolio", "portfolio_results.json")
    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            portfolio_data = json.load(f)
        plot_category_heatmap(portfolio_data, os.path.join(save_dir, "portfolio"))
    else:
        # Also check save_dir directly
        portfolio_path2 = os.path.join(save_dir, "portfolio_results.json")
        if os.path.exists(portfolio_path2):
            with open(portfolio_path2) as f:
                portfolio_data = json.load(f)
            plot_category_heatmap(portfolio_data, save_dir)

    print(f"\n  Done! Check {save_dir}/ for output PNGs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=str, default="salad_mix")
    parser.add_argument("--step-hours", type=int, default=4, choices=[2, 4],
                        help="Hours per decision step (2 or 4)")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--per", action="store_true", help="Look for PER result files")
    args = parser.parse_args()
    generate_all_plots(product=args.product, step_hours=args.step_hours, save_dir=args.save_dir, use_per=args.per)
