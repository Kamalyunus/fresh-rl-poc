"""
Visualization suite for Fresh RL: single-product and portfolio-level plots.

Usage:
    # Single product plots (training curves, policy heatmap, episode walkthrough, etc.)
    python scripts/visualize.py --product salmon_fillet --step-hours 2 --per

    # Comprehensive portfolio visualization (8 plots from portfolio_results.json)
    python scripts/visualize.py --portfolio results/portfolio_v120/portfolio_results.json
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
            obs = np.array([hours_norm, inv_norm, 0.0, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5], dtype=np.float32)
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


# ── Portfolio visualization functions ────────────────────────────────────

def _load_portfolio_results(path):
    """Load and validate portfolio results from JSON."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", data if isinstance(data, list) else [])
    return [r for r in results if r.get("status") == "ok"]


def _get_category_colors(categories):
    """Get consistent category color mapping."""
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(categories), 1))
    return {cat: cmap(i) for i, cat in enumerate(sorted(categories))}


def plot_dqn_vs_baseline_scatter(ok_results, save_dir):
    """Scatter: shaped DQN reward vs best baseline reward per SKU, colored by category."""
    categories = sorted(set(r["category"] for r in ok_results))
    cat_colors = _get_category_colors(categories)

    fig, ax = plt.subplots(figsize=(10, 10))

    for r in ok_results:
        color = cat_colors[r["category"]]
        ax.scatter(
            r["best_baseline_reward"], r["shaped_reward"],
            c=[color], s=50, alpha=0.7,
            edgecolors="white", linewidth=0.5,
        )

    # Diagonal reference line
    all_vals = ([r["best_baseline_reward"] for r in ok_results]
                + [r["shaped_reward"] for r in ok_results])
    lo, hi = min(all_vals), max(all_vals)
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.3, linewidth=1)

    # Shade win/loss regions
    ax.fill_between([lo - margin, hi + margin], [lo - margin, hi + margin],
                    hi + margin, alpha=0.03, color="green")
    ax.fill_between([lo - margin, hi + margin], lo - margin,
                    [lo - margin, hi + margin], alpha=0.03, color="red")

    wins = sum(1 for r in ok_results if r["beats_baseline"])
    total = len(ok_results)
    ax.text(0.05, 0.95, f"DQN wins: {wins}/{total} ({wins/total*100:.0f}%)",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    for cat in categories:
        ax.scatter([], [], c=[cat_colors[cat]], label=cat, s=50)
    ax.legend(fontsize=9, loc="lower right")

    ax.set_xlabel("Best Baseline Reward", fontsize=12)
    ax.set_ylabel("DQN Shaped Reward", fontsize=12)
    ax.set_title("DQN vs Best Baseline — Per-SKU Reward",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_dqn_vs_baseline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_category_win_rates(ok_results, save_dir):
    """Horizontal bar chart: beats-baseline win% per category, sorted."""
    by_cat = {}
    for r in ok_results:
        by_cat.setdefault(r["category"], []).append(r)

    cats = sorted(by_cat.keys(),
                  key=lambda c: sum(1 for r in by_cat[c] if r["beats_baseline"]) / len(by_cat[c]))
    win_pcts = [sum(1 for r in by_cat[c] if r["beats_baseline"]) / len(by_cat[c]) * 100
                for c in cats]
    n_wins = [sum(1 for r in by_cat[c] if r["beats_baseline"]) for c in cats]
    n_total = [len(by_cat[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [plt.cm.RdYlGn(pct / 100) for pct in win_pcts]
    bars = ax.barh(cats, win_pcts, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)

    for bar, pct, nw, nt in zip(bars, win_pcts, n_wins, n_total):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}% ({nw}/{nt})", va="center", fontsize=10, fontweight="bold")

    overall_pct = sum(n_wins) / sum(n_total) * 100
    ax.axvline(overall_pct, color="navy", linestyle="--", alpha=0.5, linewidth=1.5,
               label=f"Overall: {overall_pct:.0f}%")

    ax.set_xlabel("Beats Best Baseline (%)", fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_title("DQN Win Rate by Category", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_category_win_rates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reward_gap_distribution(ok_results, save_dir):
    """Histogram of reward gap (DQN - best baseline), green=wins, red=losses."""
    gaps = [r["shaped_reward"] - r["best_baseline_reward"] for r in ok_results]
    wins = [g for g in gaps if g > 0]
    losses = [g for g in gaps if g <= 0]

    fig, ax = plt.subplots(figsize=(12, 6))

    bin_edges = np.histogram_bin_edges(gaps, bins=30)
    ax.hist(wins, bins=bin_edges, color="#27AE60", alpha=0.8, edgecolor="white",
            label=f"DQN wins ({len(wins)})")
    ax.hist(losses, bins=bin_edges, color="#E74C3C", alpha=0.8, edgecolor="white",
            label=f"Baseline wins ({len(losses)})")

    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    median_gap = np.median(gaps)
    ax.axvline(median_gap, color="navy", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Median: {median_gap:+.1f}")

    ax.set_xlabel("Reward Gap (DQN - Best Baseline)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of DQN vs Best Baseline Reward Gap",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_reward_gap_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_sku_reward_gaps(ok_results, save_dir):
    """Dot plot: reward gap per SKU, grouped by category, sorted within each."""
    by_cat = {}
    for r in ok_results:
        by_cat.setdefault(r["category"], []).append(r)

    fig, ax = plt.subplots(figsize=(14, 7))

    x_pos = 0
    tick_positions = []
    tick_labels = []

    for cat in sorted(by_cat.keys()):
        results_cat = sorted(by_cat[cat],
                             key=lambda r: r["shaped_reward"] - r["best_baseline_reward"])
        gaps = [r["shaped_reward"] - r["best_baseline_reward"] for r in results_cat]
        xs = np.arange(x_pos, x_pos + len(gaps))

        colors = ["#27AE60" if g > 0 else "#E74C3C" for g in gaps]
        ax.scatter(xs, gaps, c=colors, s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.3, zorder=3)

        tick_positions.append(x_pos + len(gaps) / 2)
        tick_labels.append(cat)

        if x_pos > 0:
            ax.axvline(x_pos - 1, color="gray", linestyle=":", alpha=0.3)

        x_pos += len(gaps) + 2

    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Reward Gap (DQN - Best Baseline)", fontsize=12)
    ax.set_title("Per-SKU Reward Gap by Category", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_per_sku_gaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_baseline_difficulty(ok_results, save_dir):
    """Which baselines are hardest to beat: frequency as best + DQN win rate against each."""
    from collections import Counter

    baseline_counts = Counter(r["best_baseline"] for r in ok_results)
    baselines = sorted(baseline_counts.keys(),
                       key=lambda b: baseline_counts[b], reverse=True)

    counts = [baseline_counts[b] for b in baselines]
    win_rates = []
    for b in baselines:
        skus = [r for r in ok_results if r["best_baseline"] == b]
        win_rates.append(
            sum(1 for r in skus if r["beats_baseline"]) / len(skus) * 100
            if skus else 0
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: frequency as the best baseline
    colors_1 = [COLORS.get(b, "#95A5A6") for b in baselines]
    bars = ax1.barh(baselines, counts, color=colors_1, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(c), va="center", fontsize=10)
    ax1.set_xlabel("# SKUs Where This Baseline Is Best", fontsize=11)
    ax1.set_title("Most Common Best Baseline", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.3)

    # Panel 2: DQN win rate against each
    colors_2 = [plt.cm.RdYlGn(wr / 100) for wr in win_rates]
    bars = ax2.barh(baselines, win_rates, color=colors_2, edgecolor="white", linewidth=0.5)
    for bar, wr, b in zip(bars, win_rates, baselines):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{wr:.0f}% (n={baseline_counts[b]})", va="center", fontsize=10)
    ax2.set_xlabel("DQN Win Rate (%)", fontsize=11)
    ax2.set_xlim(0, 110)
    ax2.set_title("DQN Win Rate vs Each Baseline Type", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_baseline_difficulty.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_revenue_waste_by_category(ok_results, save_dir):
    """Grouped bar: DQN vs best baseline revenue and waste per category."""
    by_cat = {}
    for r in ok_results:
        by_cat.setdefault(r["category"], []).append(r)

    cats = sorted(by_cat.keys())
    x_pos = np.arange(len(cats))
    width = 0.35

    dqn_rev = [np.mean([r["shaped_revenue"] for r in by_cat[c]]) for c in cats]
    bl_rev = [np.mean([r["best_baseline_revenue"] for r in by_cat[c]]) for c in cats]
    dqn_waste = [np.mean([r["shaped_waste"] for r in by_cat[c]]) * 100 for c in cats]
    bl_waste = [np.mean([r["best_baseline_waste"] for r in by_cat[c]]) * 100 for c in cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(x_pos - width / 2, dqn_rev, width, label="DQN Shaped",
            color="#2E86C1", alpha=0.8)
    ax1.bar(x_pos + width / 2, bl_rev, width, label="Best Baseline",
            color="#E67E22", alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Revenue ($)", fontsize=11)
    ax1.set_title("Revenue: DQN vs Best Baseline", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x_pos - width / 2, dqn_waste, width, label="DQN Shaped",
            color="#2E86C1", alpha=0.8)
    ax2.bar(x_pos + width / 2, bl_waste, width, label="Best Baseline",
            color="#E67E22", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Waste Rate (%)", fontsize=11)
    ax2.set_title("Waste: DQN vs Best Baseline", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_revenue_waste_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_portfolio_dashboard(ok_results, save_dir):
    """6-panel comprehensive portfolio dashboard combining all key views."""
    from collections import Counter

    categories = sorted(set(r["category"] for r in ok_results))
    cat_colors = _get_category_colors(categories)
    by_cat = {}
    for r in ok_results:
        by_cat.setdefault(r["category"], []).append(r)

    fig = plt.figure(figsize=(24, 16), layout="constrained")
    wins_total = sum(1 for r in ok_results if r["beats_baseline"])
    fig.suptitle(
        f"Portfolio Dashboard — {len(ok_results)} SKUs, "
        f"{wins_total}/{len(ok_results)} beat baseline "
        f"({wins_total/len(ok_results)*100:.0f}%)",
        fontsize=18, fontweight="bold",
    )

    gs = fig.add_gridspec(2, 3, hspace=0.08, wspace=0.08)

    # Panel 1 (top-left): DQN vs Baseline scatter
    ax = fig.add_subplot(gs[0, 0])
    for r in ok_results:
        ax.scatter(r["best_baseline_reward"], r["shaped_reward"],
                   c=[cat_colors[r["category"]]], s=25, alpha=0.7,
                   edgecolors="white", linewidth=0.3)
    all_vals = ([r["best_baseline_reward"] for r in ok_results]
                + [r["shaped_reward"] for r in ok_results])
    lo, hi = min(all_vals), max(all_vals)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax.set_title(f"DQN vs Baseline Reward", fontweight="bold")
    ax.set_xlabel("Best Baseline Reward")
    ax.set_ylabel("DQN Shaped Reward")
    for cat in categories:
        ax.scatter([], [], c=[cat_colors[cat]], label=cat, s=25)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2 (top-mid): Category win rates
    ax = fig.add_subplot(gs[0, 1])
    cats_sorted = sorted(
        by_cat.keys(),
        key=lambda c: sum(1 for r in by_cat[c] if r["beats_baseline"]) / len(by_cat[c]),
    )
    win_pcts = [
        sum(1 for r in by_cat[c] if r["beats_baseline"]) / len(by_cat[c]) * 100
        for c in cats_sorted
    ]
    colors = [plt.cm.RdYlGn(pct / 100) for pct in win_pcts]
    ax.barh(cats_sorted, win_pcts, color=colors, edgecolor="white", height=0.6)
    for i, (pct, c) in enumerate(zip(win_pcts, cats_sorted)):
        nw = sum(1 for r in by_cat[c] if r["beats_baseline"])
        nt = len(by_cat[c])
        ax.text(pct + 1, i, f"{pct:.0f}% ({nw}/{nt})", va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_title("Win Rate by Category", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 3 (top-right): Reward gap histogram
    ax = fig.add_subplot(gs[0, 2])
    gaps = [r["shaped_reward"] - r["best_baseline_reward"] for r in ok_results]
    wins_g = [g for g in gaps if g > 0]
    losses_g = [g for g in gaps if g <= 0]
    bin_edges = np.histogram_bin_edges(gaps, bins=25)
    ax.hist(wins_g, bins=bin_edges, color="#27AE60", alpha=0.8, edgecolor="white",
            label=f"Wins ({len(wins_g)})")
    ax.hist(losses_g, bins=bin_edges, color="#E74C3C", alpha=0.8, edgecolor="white",
            label=f"Losses ({len(losses_g)})")
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(np.median(gaps), color="navy", linestyle="--",
               label=f"Median: {np.median(gaps):+.1f}")
    ax.set_title("Reward Gap Distribution", fontweight="bold")
    ax.set_xlabel("DQN - Baseline Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4 (bottom-left): Per-SKU dot plot
    ax = fig.add_subplot(gs[1, 0])
    x_pos = 0
    ticks, tlabels = [], []
    for cat in sorted(by_cat.keys()):
        results_cat = sorted(
            by_cat[cat],
            key=lambda r: r["shaped_reward"] - r["best_baseline_reward"],
        )
        gaps_cat = [r["shaped_reward"] - r["best_baseline_reward"] for r in results_cat]
        xs = np.arange(x_pos, x_pos + len(gaps_cat))
        c = ["#27AE60" if g > 0 else "#E74C3C" for g in gaps_cat]
        ax.scatter(xs, gaps_cat, c=c, s=20, alpha=0.7,
                   edgecolors="white", linewidth=0.2, zorder=3)
        ticks.append(x_pos + len(gaps_cat) / 2)
        tlabels.append(cat)
        x_pos += len(gaps_cat) + 2
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=8, rotation=30)
    ax.set_title("Per-SKU Reward Gaps", fontweight="bold")
    ax.set_ylabel("DQN - Baseline")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 5 (bottom-mid): Revenue comparison
    ax = fig.add_subplot(gs[1, 1])
    cats_rev = sorted(by_cat.keys())
    x_r = np.arange(len(cats_rev))
    w = 0.35
    dqn_rev = [np.mean([r["shaped_revenue"] for r in by_cat[c]]) for c in cats_rev]
    bl_rev = [np.mean([r["best_baseline_revenue"] for r in by_cat[c]]) for c in cats_rev]
    ax.bar(x_r - w / 2, dqn_rev, w, label="DQN", color="#2E86C1", alpha=0.8)
    ax.bar(x_r + w / 2, bl_rev, w, label="Baseline", color="#E67E22", alpha=0.8)
    ax.set_xticks(x_r)
    ax.set_xticklabels(cats_rev, rotation=30, ha="right", fontsize=8)
    ax.set_title("Revenue by Category", fontweight="bold")
    ax.set_ylabel("Mean Revenue ($)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 6 (bottom-right): Baseline difficulty
    ax = fig.add_subplot(gs[1, 2])
    bl_counts = Counter(r["best_baseline"] for r in ok_results)
    bls = sorted(bl_counts.keys(), key=lambda b: bl_counts[b], reverse=True)
    bl_wr = []
    for b in bls:
        skus = [r for r in ok_results if r["best_baseline"] == b]
        bl_wr.append(sum(1 for r in skus if r["beats_baseline"]) / len(skus) * 100)
    colors_bl = [plt.cm.RdYlGn(wr / 100) for wr in bl_wr]
    ax.barh(bls, bl_wr, color=colors_bl, edgecolor="white", height=0.6)
    for i, (wr, b) in enumerate(zip(bl_wr, bls)):
        ax.text(wr + 1, i, f"{wr:.0f}% (n={bl_counts[b]})", va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_title("DQN Win Rate vs Each Baseline", fontweight="bold")
    ax.set_xlabel("Win Rate (%)")
    ax.grid(True, axis="x", alpha=0.3)

    path = os.path.join(save_dir, "portfolio_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_three_way_comparison(ok_results, save_dir):
    """6-panel comparison of DQN Plain vs DQN Shaped vs Best Baseline."""
    from collections import Counter

    by_cat = {}
    for r in ok_results:
        by_cat.setdefault(r["category"], []).append(r)
    cats = sorted(by_cat.keys())
    x_pos = np.arange(len(cats))
    width = 0.25

    # Compute per-category means
    plain_reward = [np.mean([r["plain_reward"] for r in by_cat[c]]) for c in cats]
    shaped_reward = [np.mean([r["shaped_reward"] for r in by_cat[c]]) for c in cats]
    bl_reward = [np.mean([r["best_baseline_reward"] for r in by_cat[c]]) for c in cats]

    plain_rev = [np.mean([r["plain_revenue"] for r in by_cat[c]]) for c in cats]
    shaped_rev = [np.mean([r["shaped_revenue"] for r in by_cat[c]]) for c in cats]
    bl_rev = [np.mean([r["best_baseline_revenue"] for r in by_cat[c]]) for c in cats]

    plain_waste = [np.mean([r["plain_waste"] for r in by_cat[c]]) * 100 for c in cats]
    shaped_waste = [np.mean([r["shaped_waste"] for r in by_cat[c]]) * 100 for c in cats]
    bl_waste = [np.mean([r["best_baseline_waste"] for r in by_cat[c]]) * 100 for c in cats]

    C_PLAIN = "#2E86C1"
    C_SHAPED = "#1B4F72"
    C_BASELINE = "#E67E22"

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(
        f"DQN Plain vs DQN Shaped vs Best Baseline — {len(ok_results)} SKUs",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # Panel 1: Mean reward by category
    ax = axes[0, 0]
    ax.bar(x_pos - width, plain_reward, width, label="DQN Plain",
           color=C_PLAIN, alpha=0.85)
    ax.bar(x_pos, shaped_reward, width, label="DQN Shaped",
           color=C_SHAPED, alpha=0.85)
    ax.bar(x_pos + width, bl_reward, width, label="Best Baseline",
           color=C_BASELINE, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward by Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Mean revenue by category
    ax = axes[0, 1]
    ax.bar(x_pos - width, plain_rev, width, label="DQN Plain",
           color=C_PLAIN, alpha=0.85)
    ax.bar(x_pos, shaped_rev, width, label="DQN Shaped",
           color=C_SHAPED, alpha=0.85)
    ax.bar(x_pos + width, bl_rev, width, label="Best Baseline",
           color=C_BASELINE, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Revenue ($)")
    ax.set_title("Revenue by Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Mean waste by category
    ax = axes[0, 2]
    ax.bar(x_pos - width, plain_waste, width, label="DQN Plain",
           color=C_PLAIN, alpha=0.85)
    ax.bar(x_pos, shaped_waste, width, label="DQN Shaped",
           color=C_SHAPED, alpha=0.85)
    ax.bar(x_pos + width, bl_waste, width, label="Best Baseline",
           color=C_BASELINE, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Waste Rate (%)")
    ax.set_title("Waste Rate by Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 4: Who wins most often (per-SKU best)
    ax = axes[1, 0]
    winner_counts = Counter()
    for r in ok_results:
        rewards = {
            "DQN Plain": r["plain_reward"],
            "DQN Shaped": r["shaped_reward"],
            "Best Baseline": r["best_baseline_reward"],
        }
        winner = max(rewards, key=rewards.get)
        winner_counts[winner] += 1

    labels = ["DQN Plain", "DQN Shaped", "Best Baseline"]
    counts = [winner_counts.get(l, 0) for l in labels]
    pct = [c / len(ok_results) * 100 for c in counts]
    bar_colors = [C_PLAIN, C_SHAPED, C_BASELINE]
    bars = ax.bar(labels, counts, color=bar_colors, edgecolor="white", alpha=0.85)
    for bar, c, p in zip(bars, counts, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{c} ({p:.0f}%)", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("# SKUs Where This Policy Is Best")
    ax.set_title("Per-SKU Winner Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 5: Per-SKU reward (sorted, all 3 overlaid)
    ax = axes[1, 1]
    sorted_results = sorted(ok_results,
                            key=lambda r: r["shaped_reward"] - r["best_baseline_reward"])
    xs = np.arange(len(sorted_results))
    ax.scatter(xs, [r["plain_reward"] for r in sorted_results],
               c=C_PLAIN, s=12, alpha=0.6, label="DQN Plain", zorder=2)
    ax.scatter(xs, [r["shaped_reward"] for r in sorted_results],
               c=C_SHAPED, s=12, alpha=0.6, label="DQN Shaped", zorder=3)
    ax.scatter(xs, [r["best_baseline_reward"] for r in sorted_results],
               c=C_BASELINE, s=12, alpha=0.6, marker="x", label="Best Baseline", zorder=2)
    ax.set_xlabel(f"SKUs (sorted by shaped - baseline gap)")
    ax.set_ylabel("Reward")
    ax.set_title("All SKU Rewards (3-way)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 6: Pairwise win rates
    ax = axes[1, 2]
    shaped_beats_plain = sum(
        1 for r in ok_results if r["shaped_reward"] > r["plain_reward"]
    )
    shaped_beats_bl = sum(1 for r in ok_results if r["beats_baseline"])
    plain_beats_bl = sum(
        1 for r in ok_results if r["plain_reward"] > r["best_baseline_reward"]
    )
    n = len(ok_results)

    matchups = ["Shaped > Plain", "Shaped > Baseline", "Plain > Baseline"]
    rates = [shaped_beats_plain / n * 100, shaped_beats_bl / n * 100,
             plain_beats_bl / n * 100]
    bar_colors_pw = [C_SHAPED, C_SHAPED, C_PLAIN]
    bars = ax.barh(matchups, rates, color=bar_colors_pw, edgecolor="white",
                   alpha=0.85, height=0.5)
    for bar, r_val in zip(bars, rates):
        count = int(r_val * n / 100)
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{r_val:.0f}% ({count}/{n})", va="center", fontsize=11,
                fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("Pairwise Win Rates", fontsize=13, fontweight="bold")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "portfolio_three_way_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _load_training_histories(portfolio_dir, results):
    """Load greedy eval checkpoints from per-product training history files.

    Returns dict with keys 'plain' and 'shaped', each mapping product name to
    a list of {episode, reward, waste, clearance} dicts.
    """
    histories = {"plain": {}, "shaped": {}}
    for r in results:
        product = r["product"]
        product_dir = os.path.join(portfolio_dir, product)
        if not os.path.isdir(product_dir):
            continue
        # Find training history files by scanning for matching pattern
        for fname in os.listdir(product_dir):
            if not fname.startswith(f"training_history_{product}_"):
                continue
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(product_dir, fname)
            try:
                with open(fpath) as f:
                    hist = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            evals = hist.get("greedy_eval", [])
            if not evals:
                continue
            variant = "shaped" if "_shaped" in fname else "plain"
            histories[variant][product] = evals
    return histories


def plot_portfolio_training_progress(ok_results, portfolio_dir, save_dir):
    """2x2 plot showing aggregate training progress across the portfolio.

    Uses greedy eval checkpoints (epsilon=0 evaluations) from per-product
    training histories for clean, comparable learning curves.
    """
    histories = _load_training_histories(portfolio_dir, ok_results)
    if not histories["plain"] and not histories["shaped"]:
        print("  [SKIP] No training histories found for portfolio training progress plot.")
        return

    # Build per-product baseline lookup
    baseline_by_product = {r["product"]: r["best_baseline_reward"] for r in ok_results}
    category_by_product = {r["product"]: r["category"] for r in ok_results}

    # Collect all episode numbers across all products
    all_episodes = set()
    for variant in ("plain", "shaped"):
        for evals in histories[variant].values():
            for e in evals:
                all_episodes.add(e["episode"])
    episodes = sorted(all_episodes)

    # Aggregate per-episode stats for each variant.
    # For products that early-stopped, carry forward their last greedy eval
    # to all subsequent episodes to avoid survivorship bias.
    def aggregate(variant):
        # Build per-product lookup: episode -> eval, and track last eval
        product_last = {}  # product -> last eval dict
        for product, evals in histories[variant].items():
            if evals:
                product_last[product] = max(evals, key=lambda e: e["episode"])

        rewards, wastes, beats = [], [], []
        for ep in episodes:
            ep_rewards, ep_wastes, ep_beats = [], [], []
            for product, evals in histories[variant].items():
                # Find checkpoint matching this episode, or carry forward last if past end
                matched = None
                for e in evals:
                    if e["episode"] == ep:
                        matched = e
                        break
                if matched is None and ep > product_last.get(product, {}).get("episode", float("inf")):
                    matched = product_last[product]
                if matched is not None:
                    ep_rewards.append(matched["reward"])
                    ep_wastes.append(matched.get("waste", 0.0))
                    bl = baseline_by_product.get(product)
                    if bl is not None:
                        ep_beats.append(1 if matched["reward"] > bl else 0)
            if ep_rewards:
                rewards.append((np.mean(ep_rewards), np.percentile(ep_rewards, 25),
                                np.percentile(ep_rewards, 75)))
            else:
                rewards.append((np.nan, np.nan, np.nan))
            wastes.append((np.mean(ep_wastes), np.percentile(ep_wastes, 25),
                           np.percentile(ep_wastes, 75)) if ep_wastes else (np.nan, np.nan, np.nan))
            beats.append(np.mean(ep_beats) * 100 if ep_beats else np.nan)
        return (np.array(rewards), np.array(wastes), np.array(beats))

    agg = {}
    for variant in ("plain", "shaped"):
        if histories[variant]:
            agg[variant] = aggregate(variant)

    # Per-category beats-baseline curves (best of plain/shaped per product per episode)
    # Carry forward last eval for early-stopped products.
    categories = sorted(set(r["category"] for r in ok_results))
    cat_colors = _get_category_colors(categories)

    # Build per-product last-eval lookup across variants
    product_last_any = {}  # product -> {variant: last_eval}
    for variant in ("plain", "shaped"):
        for product, evals in histories.get(variant, {}).items():
            if evals:
                last = max(evals, key=lambda e: e["episode"])
                product_last_any.setdefault(product, {})[variant] = last

    cat_beats = {cat: [] for cat in categories}
    for ep in episodes:
        cat_ep = {cat: [] for cat in categories}
        for r in ok_results:
            product = r["product"]
            cat = r["category"]
            bl = baseline_by_product.get(product)
            if bl is None:
                continue
            best_reward = -float("inf")
            for variant in ("plain", "shaped"):
                evals = histories.get(variant, {}).get(product, [])
                matched = None
                for e in evals:
                    if e["episode"] == ep:
                        matched = e
                        break
                if matched is None:
                    last = product_last_any.get(product, {}).get(variant)
                    if last and ep > last["episode"]:
                        matched = last
                if matched is not None:
                    best_reward = max(best_reward, matched["reward"])
            if best_reward > -float("inf"):
                cat_ep[cat].append(1 if best_reward > bl else 0)
        for cat in categories:
            cat_beats[cat].append(np.mean(cat_ep[cat]) * 100 if cat_ep[cat] else np.nan)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Portfolio Training Progress (Greedy Eval Checkpoints)", fontsize=14, fontweight="bold")
    variant_style = {"plain": ("#2E86C1", "Plain DQN"), "shaped": ("#E67E22", "Shaped DQN")}
    ep_arr = np.array(episodes)

    # Panel 1: Mean greedy reward
    ax = axes[0, 0]
    for v, (color, label) in variant_style.items():
        if v not in agg:
            continue
        rewards = agg[v][0]
        ax.plot(ep_arr, rewards[:, 0], color=color, label=label, linewidth=1.5)
        ax.fill_between(ep_arr, rewards[:, 1], rewards[:, 2], color=color, alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Greedy Reward")
    ax.set_title("Mean Greedy Reward Over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: % beating baseline
    ax = axes[0, 1]
    for v, (color, label) in variant_style.items():
        if v not in agg:
            continue
        ax.plot(ep_arr, agg[v][2], color=color, label=label, linewidth=1.5)
    final_pct = sum(1 for r in ok_results if r.get("beats_baseline")) / len(ok_results) * 100
    ax.axhline(final_pct, color="gray", linestyle="--", alpha=0.5, label=f"Final: {final_pct:.0f}%")
    ax.set_xlabel("Episode")
    ax.set_ylabel("% Beating Baseline")
    ax.set_title("% Products Beating Baseline Over Training")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Mean waste rate
    ax = axes[1, 0]
    for v, (color, label) in variant_style.items():
        if v not in agg:
            continue
        wastes = agg[v][1]
        ax.plot(ep_arr, wastes[:, 0] * 100, color=color, label=label, linewidth=1.5)
        ax.fill_between(ep_arr, wastes[:, 1] * 100, wastes[:, 2] * 100, color=color, alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Waste Rate (%)")
    ax.set_title("Mean Waste Rate Over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Per-category learning curves
    ax = axes[1, 1]
    for cat in categories:
        ax.plot(ep_arr, cat_beats[cat], color=cat_colors[cat], label=cat, linewidth=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("% Beating Baseline")
    ax.set_title("Per-Category % Beating Baseline")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_training_progress.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _load_daily_rewards(portfolio_dir, results):
    """Load per-episode training rewards from training history files.

    Returns dict with keys 'plain' and 'shaped', each mapping product name to
    a list of daily rewards (one per episode/day).
    """
    daily = {"plain": {}, "shaped": {}}
    for r in results:
        product = r["product"]
        product_dir = os.path.join(portfolio_dir, product)
        if not os.path.isdir(product_dir):
            continue
        for fname in os.listdir(product_dir):
            if not fname.startswith(f"training_history_{product}_"):
                continue
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(product_dir, fname)
            try:
                with open(fpath) as f:
                    hist = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            rewards = hist.get("episode_rewards", [])
            if not rewards:
                continue
            variant = "shaped" if "_shaped" in fname else "plain"
            daily[variant][product] = rewards
    return daily


def plot_time_to_value(ok_results, portfolio_dir, save_dir):
    """2x2 stakeholder-ready chart showing how quickly RL delivers value.

    Uses actual daily training rewards (1 session per day, with exploration)
    to show realistic production performance over time.
    """
    daily = _load_daily_rewards(portfolio_dir, ok_results)
    if not daily["plain"] and not daily["shaped"]:
        print("  [SKIP] No training histories found for time-to-value plot.")
        return

    baseline_by_product = {r["product"]: r["best_baseline_reward"] for r in ok_results}
    category_by_product = {r["product"]: r["category"] for r in ok_results}

    # Find products with data and max episode count
    products_with_data = set()
    for variant in ("plain", "shaped"):
        products_with_data.update(daily[variant].keys())
    max_days = max(
        max((len(daily[v].get(p, [])) for p in products_with_data), default=0)
        for v in ("plain", "shaped")
    )
    if max_days == 0:
        print("  [SKIP] No daily reward data found for time-to-value plot.")
        return
    days = np.arange(max_days)

    # Per-product, per-day: best daily reward = max(plain, shaped)
    # Each day is one real session with exploration noise
    best_daily = {}  # product -> array of daily rewards
    for product in products_with_data:
        plain_r = daily["plain"].get(product, [])
        shaped_r = daily["shaped"].get(product, [])
        n = max(len(plain_r), len(shaped_r))
        best = np.full(n, np.nan)
        for i in range(n):
            vals = []
            if i < len(plain_r):
                vals.append(plain_r[i])
            if i < len(shaped_r):
                vals.append(shaped_r[i])
            if vals:
                best[i] = max(vals)
        best_daily[product] = best

    # Per-day: raw % of products beating baseline + reward gaps
    smooth_window = 30  # 30-day rolling average for trend
    raw_pct = np.full(max_days, np.nan)
    smooth_pct = np.full(max_days, np.nan)
    gap_mean = np.full(max_days, np.nan)
    gap_p25 = np.full(max_days, np.nan)
    gap_p75 = np.full(max_days, np.nan)

    pct_history = []
    for day in range(max_days):
        beats, n = 0, 0
        gaps = []
        for product in products_with_data:
            bl = baseline_by_product.get(product)
            if bl is None:
                continue
            arr = best_daily[product]
            if day >= len(arr) or np.isnan(arr[day]):
                continue
            n += 1
            if arr[day] > bl:
                beats += 1
            gaps.append(arr[day] - bl)
        if n > 0:
            raw_pct[day] = beats / n * 100
            pct_history.append(raw_pct[day])
            window = pct_history[-smooth_window:]
            smooth_pct[day] = np.mean(window)
        if gaps:
            gap_mean[day] = np.mean(gaps)
            gap_p25[day] = np.percentile(gaps, 25)
            gap_p75[day] = np.percentile(gaps, 75)

    # First day each product's 30-day rolling avg beats baseline
    first_beat = {}
    for product in products_with_data:
        bl = baseline_by_product.get(product)
        if bl is None:
            continue
        arr = best_daily[product]
        rolling = []
        for day in range(len(arr)):
            if not np.isnan(arr[day]):
                rolling.append(arr[day])
            window = rolling[-smooth_window:]
            if window and np.mean(window) > bl:
                first_beat[product] = day
                break

    # Limit to first 500 days for focus
    max_plot = min(max_days, 500)
    plot_days = days[:max_plot]

    # --- Plot ---
    categories = sorted(set(r["category"] for r in ok_results))
    cat_colors = _get_category_colors(categories)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Time to Value — Actual Daily Performance After Deployment",
                 fontsize=14, fontweight="bold")

    # Panel 1: % beating baseline over time (hero chart)
    ax = axes[0, 0]
    ax.plot(plot_days, raw_pct[:max_plot], color="#27AE60", alpha=0.25,
            linewidth=0.8, label="Daily (raw)")
    ax.plot(plot_days, smooth_pct[:max_plot], color="#27AE60", linewidth=2.5,
            zorder=3, label=f"{smooth_window}-day avg")
    ax.fill_between(plot_days, smooth_pct[:max_plot], alpha=0.12, color="#27AE60")
    # Annotate key milestones on the smooth line
    valid_smooth = ~np.isnan(smooth_pct[:max_plot])
    if np.any(valid_smooth):
        # Day 1
        first_valid = np.argmax(valid_smooth)
        day1_val = smooth_pct[first_valid]
        ax.annotate(f"Day 1: {day1_val:.0f}%", xy=(first_valid, day1_val),
                    xytext=(first_valid + 30, day1_val - 10),
                    fontsize=9, fontweight="bold", color="#27AE60",
                    arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))
        # Last day
        last_valid = max_plot - 1 - np.argmax(valid_smooth[:max_plot][::-1])
        final_val = smooth_pct[last_valid]
        ax.annotate(f"Day {last_valid + 1}: {final_val:.0f}%",
                    xy=(last_valid, final_val),
                    xytext=(last_valid - 80, final_val - 10),
                    fontsize=9, fontweight="bold", color="#27AE60",
                    arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))
    ax.set_xlabel("Day in Production")
    ax.set_ylabel("% of Products Beating Baseline")
    ax.set_title("% Beating Baseline Over Time (1 session/day)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Day-1 performance by category (raw, single session)
    ax = axes[0, 1]
    cat_day1 = {}
    for cat in categories:
        cat_products = [r["product"] for r in ok_results
                        if r["category"] == cat and r["product"] in products_with_data]
        if not cat_products:
            cat_day1[cat] = 0
            continue
        beats = 0
        for p in cat_products:
            bl = baseline_by_product.get(p)
            arr = best_daily.get(p)
            if arr is not None and len(arr) > 0 and bl is not None and arr[0] > bl:
                beats += 1
        cat_day1[cat] = beats / len(cat_products) * 100

    sorted_cats = sorted(categories, key=lambda c: cat_day1[c])
    bars = ax.barh(sorted_cats, [cat_day1[c] for c in sorted_cats],
                   color=[cat_colors[c] for c in sorted_cats],
                   edgecolor="white", linewidth=0.5)
    for bar, cat in zip(bars, sorted_cats):
        val = cat_day1[cat]
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("% Beating Baseline at Day 1")
    ax.set_title("Day-1 Performance by Category")
    ax.set_xlim(0, 110)
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 3: Days-to-beat-baseline histogram (using 30-day rolling avg)
    ax = axes[1, 0]
    beat_eps = list(first_beat.values())
    never_beat = len(products_with_data) - len(beat_eps)
    if beat_eps:
        max_ep = max(beat_eps)
        bins = [0] + list(range(1, min(max_ep + 50, 501), max(1, min(max_ep + 50, 501) // 25)))
        if bins[-1] < max_ep:
            bins.append(max_ep + 1)
        ax.hist(beat_eps, bins=bins, color="#2E86C1", edgecolor="white", linewidth=0.5)
        median_val = np.median(beat_eps)
        mean_val = np.mean(beat_eps)
        ax.axvline(median_val, color="#E74C3C", linestyle="--", linewidth=1.5,
                   label=f"Median: {median_val:.0f} days")
        ax.axvline(mean_val, color="#E67E22", linestyle=":", linewidth=1.5,
                   label=f"Mean: {mean_val:.0f} days")
        ax.legend(fontsize=9)
    if never_beat > 0:
        ax.text(0.95, 0.95, f"{never_beat} products never beat baseline",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#E74C3C", fontstyle="italic")
    ax.set_xlabel("Days to Beat Baseline (30-day rolling avg)")
    ax.set_ylabel("Number of Products")
    ax.set_title("Distribution: Days to Beat Baseline")
    ax.grid(True, alpha=0.3)

    # Panel 4: Reward gap over time (smoothed with rolling mean)
    ax = axes[1, 1]
    gm = gap_mean[:max_plot].copy()
    gp25 = gap_p25[:max_plot].copy()
    gp75 = gap_p75[:max_plot].copy()
    # Rolling mean using cumsum trick (numpy-only)
    def _rolling_mean(arr, w):
        valid = ~np.isnan(arr)
        if np.sum(valid) <= w:
            return arr
        out = arr.copy()
        vals = arr[valid]
        cs = np.cumsum(vals)
        cs = np.insert(cs, 0, 0)
        smoothed = np.empty_like(vals)
        for i in range(len(vals)):
            start = max(0, i + 1 - w)
            smoothed[i] = (cs[i + 1] - cs[start]) / (i + 1 - start)
        out[valid] = smoothed
        return out
    gm = _rolling_mean(gm, smooth_window)
    gp25 = _rolling_mean(gp25, smooth_window)
    gp75 = _rolling_mean(gp75, smooth_window)
    ax.plot(plot_days, gm, color="#2E86C1", linewidth=2, label="Mean gap")
    ax.fill_between(plot_days, gp25, gp75, color="#2E86C1", alpha=0.15,
                    label="25th-75th pctl")
    ax.axhline(0, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.7, label="Break-even")
    ax.set_xlabel("Day in Production")
    ax.set_ylabel("Reward - Baseline Reward")
    ax.set_title("Reward Gap vs Baseline Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_time_to_value.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_portfolio_plots(portfolio_path, save_dir=None):
    """Generate all portfolio-level visualizations from portfolio_results.json.

    Args:
        portfolio_path: Path to portfolio_results.json
        save_dir: Output directory (defaults to same directory as portfolio_path)
    """
    ok_results = _load_portfolio_results(portfolio_path)
    if len(ok_results) < 2:
        print("  Need at least 2 successful results for portfolio plots.")
        return

    if save_dir is None:
        save_dir = os.path.dirname(portfolio_path)
    os.makedirs(save_dir, exist_ok=True)

    n = len(ok_results)
    wins = sum(1 for r in ok_results if r.get("beats_baseline"))
    cats = len(set(r["category"] for r in ok_results))
    print(f"\n  Generating portfolio visualizations: "
          f"{n} SKUs, {wins}/{n} wins ({wins/n*100:.0f}%), {cats} categories")
    print(f"  {'='*60}")

    plot_portfolio_dashboard(ok_results, save_dir)
    plot_three_way_comparison(ok_results, save_dir)
    plot_dqn_vs_baseline_scatter(ok_results, save_dir)
    plot_category_win_rates(ok_results, save_dir)
    plot_reward_gap_distribution(ok_results, save_dir)
    plot_per_sku_reward_gaps(ok_results, save_dir)
    plot_baseline_difficulty(ok_results, save_dir)
    plot_revenue_waste_by_category(ok_results, save_dir)
    plot_category_heatmap({"results": ok_results}, save_dir)

    portfolio_dir = os.path.dirname(portfolio_path)
    plot_portfolio_training_progress(ok_results, portfolio_dir, save_dir)
    plot_time_to_value(ok_results, portfolio_dir, save_dir)

    print(f"\n  Done! 11 portfolio plots saved to {save_dir}/")


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

    # Agent-dependent plots: load trained agent (try shaped first, then plain)
    agent_path = os.path.join(save_dir, f"best_agent_{suffix}_shaped.pt")
    if not os.path.exists(agent_path):
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
    parser = argparse.ArgumentParser(
        description="Visualization suite for Fresh RL training and portfolio results"
    )
    parser.add_argument("--product", type=str, default="salmon_fillet",
                        help="Product name for single-product plots")
    parser.add_argument("--step-hours", type=int, default=2, choices=[2, 4],
                        help="Hours per decision step (default: 2)")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Output directory for plots")
    parser.add_argument("--per", action="store_true",
                        help="Look for PER result files")
    parser.add_argument("--portfolio", type=str, default=None,
                        help="Path to portfolio_results.json for comprehensive portfolio plots")
    args = parser.parse_args()

    if args.portfolio:
        generate_portfolio_plots(args.portfolio, save_dir=args.save_dir)
    else:
        generate_all_plots(product=args.product, step_hours=args.step_hours,
                           save_dir=args.save_dir, use_per=args.per)
