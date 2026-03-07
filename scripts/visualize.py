"""
Portfolio visualization suite for Fresh RL.

Generates 10 portfolio-level plots from portfolio_results.json:
  - Dashboard (6-panel), three-way comparison, scatter, category win rates,
    reward gap distribution, per-SKU gaps, baseline difficulty, revenue/waste
    comparison, category heatmap, time-to-value, per-category time-to-value

Usage:
    python scripts/visualize.py --portfolio results/portfolio_v42_2phase_eval/portfolio_results.json
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


def _load_daily_rewards(portfolio_dir, results):
    """Load Phase 2 deployment rewards from eval_deployment.json files.

    Returns dict mapping product name to {"dqn": [...], "baseline": [...]}.
    Each product has a single DQN array and a single baseline array (no cherry-picking).
    """
    daily = {}
    for r in results:
        product = r["product"]
        product_dir = os.path.join(portfolio_dir, product)
        if not os.path.isdir(product_dir):
            continue

        deploy_path = os.path.join(product_dir, "eval_deployment.json")
        if os.path.exists(deploy_path):
            try:
                with open(deploy_path) as f:
                    deploy = json.load(f)
                dqn_rewards = deploy.get("dqn_rewards", [])
                bl_rewards = deploy.get("baseline_rewards", [])
                if dqn_rewards and bl_rewards:
                    daily[product] = {"dqn": dqn_rewards, "baseline": bl_rewards}
            except (json.JSONDecodeError, OSError):
                pass

    return daily


def plot_time_to_value(ok_results, portfolio_dir, save_dir):
    """2x2 stakeholder-ready chart showing how quickly RL delivers value.

    Uses Phase 2 deployment data: each product has a single deployed DQN
    (selected during Phase 1) and a fixed baseline, both evaluated on the
    same fresh demand seeds. No cherry-picking between variants.
    """
    daily = _load_daily_rewards(portfolio_dir, ok_results)
    if not daily:
        print("  [SKIP] No daily reward data found for time-to-value plot.")
        return

    category_by_product = {r["product"]: r["category"] for r in ok_results}

    products_with_data = set(daily.keys())
    max_days = max(len(daily[p]["dqn"]) for p in products_with_data)
    if max_days == 0:
        print("  [SKIP] No daily reward data found for time-to-value plot.")
        return
    days = np.arange(max_days)

    # Per-product, per-day: DQN reward and baseline reward
    best_daily = {}  # product -> array of DQN rewards
    best_daily_bl = {}  # product -> array of baseline rewards
    for product in products_with_data:
        dqn_r = daily[product]["dqn"]
        bl_r = daily[product]["baseline"]
        n = len(dqn_r)
        best = np.array(dqn_r, dtype=float)
        bl_arr = np.full(n, np.nan)
        for i in range(min(n, len(bl_r))):
            if bl_r[i] is not None:
                bl_arr[i] = bl_r[i]
        best_daily[product] = best
        best_daily_bl[product] = bl_arr

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
            arr = best_daily[product]
            bl_arr = best_daily_bl[product]
            if day >= len(arr) or np.isnan(arr[day]):
                continue
            if day >= len(bl_arr) or np.isnan(bl_arr[day]):
                continue
            n += 1
            if arr[day] > bl_arr[day]:
                beats += 1
            gaps.append(arr[day] - bl_arr[day])
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
        arr = best_daily[product]
        bl_arr = best_daily_bl[product]
        rolling_dqn = []
        rolling_bl = []
        for day in range(len(arr)):
            if not np.isnan(arr[day]):
                rolling_dqn.append(arr[day])
            if day < len(bl_arr) and not np.isnan(bl_arr[day]):
                rolling_bl.append(bl_arr[day])
            win_dqn = rolling_dqn[-smooth_window:]
            win_bl = rolling_bl[-smooth_window:]
            if win_dqn and win_bl and np.mean(win_dqn) > np.mean(win_bl):
                first_beat[product] = day
                break

    # Limit to first 500 days for focus
    max_plot = min(max_days, 500)
    plot_days = days[:max_plot]

    # --- Plot ---
    categories = sorted(set(r["category"] for r in ok_results))
    cat_colors = _get_category_colors(categories)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Time to Value — Phase 2 Deployment Performance",
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
            arr = best_daily.get(p)
            bl_arr = best_daily_bl.get(p)
            if (arr is not None and len(arr) > 0 and
                    bl_arr is not None and len(bl_arr) > 0 and
                    not np.isnan(arr[0]) and not np.isnan(bl_arr[0]) and
                    arr[0] > bl_arr[0]):
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


def plot_category_time_to_value(ok_results, portfolio_dir, save_dir):
    """Per-category time-to-value: % beating baseline and reward gap over days.

    Uses Phase 2 deployment data: single deployed DQN vs fixed baseline per product.
    Two panels per category row: win rate trend + reward gap trend.
    """
    daily = _load_daily_rewards(portfolio_dir, ok_results)
    if not daily:
        print("  [SKIP] No daily reward data found for category time-to-value plot.")
        return

    category_by = {r["product"]: r["category"] for r in ok_results}

    products_with_data = set(daily.keys())

    # DQN reward and baseline reward per product
    best_daily = {}
    best_daily_bl = {}
    for p in products_with_data:
        dqn_r = daily[p]["dqn"]
        bl_r = daily[p]["baseline"]
        n = len(dqn_r)
        best_daily[p] = dqn_r
        bl_arr = []
        for i in range(n):
            if i < len(bl_r) and bl_r[i] is not None:
                bl_arr.append(bl_r[i])
            else:
                bl_arr.append(float("nan"))
        best_daily_bl[p] = bl_arr

    max_days = max(len(v) for v in best_daily.values())
    smooth_window = 30
    categories = sorted(set(r["category"] for r in ok_results))
    cat_colors = _get_category_colors(categories)

    fig, axes = plt.subplots(len(categories), 2, figsize=(14, 3 * len(categories)),
                             sharex=True)
    fig.suptitle("Time to Value by Category — Phase 2 Deployment (1 session/day)",
                 fontsize=14, fontweight="bold", y=1.0)

    for row, cat in enumerate(categories):
        cat_products = [p for p in products_with_data if category_by.get(p) == cat]
        n_sku = len(cat_products)
        color = cat_colors[cat]

        # Compute daily stats for this category
        pct_raw = np.full(max_days, np.nan)
        pct_smooth = np.full(max_days, np.nan)
        gap_raw = np.full(max_days, np.nan)
        gap_smooth = np.full(max_days, np.nan)

        pct_hist = []
        gap_hist = []
        for day in range(max_days):
            beats, n = 0, 0
            gaps = []
            for p in cat_products:
                arr = best_daily.get(p, [])
                bl_arr = best_daily_bl.get(p, [])
                if day >= len(arr) or day >= len(bl_arr):
                    continue
                dqn_r = arr[day]
                bl_r = bl_arr[day]
                if dqn_r != dqn_r or bl_r != bl_r:  # NaN check
                    continue
                n += 1
                if dqn_r > bl_r:
                    beats += 1
                gaps.append(dqn_r - bl_r)
            if n > 0:
                pct_raw[day] = beats / n * 100
                pct_hist.append(pct_raw[day])
                pct_smooth[day] = np.mean(pct_hist[-smooth_window:])
            if gaps:
                gap_raw[day] = np.mean(gaps)
                gap_hist.append(gap_raw[day])
                gap_smooth[day] = np.mean(gap_hist[-smooth_window:])

        days = np.arange(max_days)

        # Left panel: % beating baseline
        ax = axes[row, 0]
        ax.plot(days, pct_raw, color=color, alpha=0.2, linewidth=0.5)
        ax.plot(days, pct_smooth, color=color, linewidth=2.5)
        # Annotate start and end
        valid = ~np.isnan(pct_smooth)
        if np.any(valid):
            first_i = np.argmax(valid)
            last_i = max_days - 1 - np.argmax(valid[::-1])
            ax.text(0.02, 0.95, "Day 1: %.0f%%" % pct_smooth[first_i],
                    transform=ax.transAxes, fontsize=8, fontweight="bold",
                    va="top", color=color)
            ax.text(0.98, 0.95, "Day %d: %.0f%%" % (last_i + 1, pct_smooth[last_i]),
                    transform=ax.transAxes, fontsize=8, fontweight="bold",
                    va="top", ha="right", color=color)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%s (%d)" % (cat, n_sku), fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("% Beating Baseline (30-day avg)", fontsize=11)

        # Right panel: reward gap
        ax = axes[row, 1]
        ax.plot(days, gap_raw, color=color, alpha=0.2, linewidth=0.5)
        ax.plot(days, gap_smooth, color=color, linewidth=2.5)
        ax.axhline(0, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.5)
        if np.any(~np.isnan(gap_smooth)):
            first_i = np.argmax(~np.isnan(gap_smooth))
            last_i = max_days - 1 - np.argmax(~np.isnan(gap_smooth)[::-1])
            ax.text(0.02, 0.95, "%+.1f" % gap_smooth[first_i],
                    transform=ax.transAxes, fontsize=8, fontweight="bold",
                    va="top", color=color)
            ax.text(0.98, 0.95, "%+.1f" % gap_smooth[last_i],
                    transform=ax.transAxes, fontsize=8, fontweight="bold",
                    va="top", ha="right", color=color)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Mean Reward Gap vs Baseline (30-day avg)", fontsize=11)

    axes[-1, 0].set_xlabel("Day in Production")
    axes[-1, 1].set_xlabel("Day in Production")

    plt.tight_layout()
    path = os.path.join(save_dir, "portfolio_category_time_to_value.png")
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
    plot_time_to_value(ok_results, portfolio_dir, save_dir)
    plot_category_time_to_value(ok_results, portfolio_dir, save_dir)

    print(f"\n  Done! 11 portfolio plots saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Portfolio visualization for Fresh RL results"
    )
    parser.add_argument("--portfolio", type=str, required=True,
                        help="Path to portfolio_results.json")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory for plots (default: same as portfolio file)")
    args = parser.parse_args()

    generate_portfolio_plots(args.portfolio, save_dir=args.save_dir)
