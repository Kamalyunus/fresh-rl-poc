"""
Compare training approaches: Per-SKU (v1.4) vs Pooled (v2) vs Pooled TL (v2.1).

Generates comparison plots showing:
1. Category win rate comparison (grouped bars)
2. Per-SKU reward gap scatter
3. Reward distribution comparison (box plot)
4. Win/lose overlap (stacked bars)
5. Summary dashboard

Supports 2-way (per-SKU vs pooled) or 3-way (+ pooled TL) comparison.

Usage:
    python scripts/compare_pooled_vs_persku.py
    python scripts/compare_pooled_vs_persku.py --pooled-tl results/portfolio_v21_pooled_tl/portfolio_results.json
    python scripts/compare_pooled_vs_persku.py --persku results/portfolio_v140_5000ep/portfolio_results.json \
        --pooled results/portfolio_v2_pooled/portfolio_results.json \
        --pooled-tl results/portfolio_v21_pooled_tl/portfolio_results.json \
        --save-dir results/comparison_v14_v2_v21
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CATEGORY_ORDER = ["dairy", "deli_prepared", "fruits", "vegetables", "meats", "seafood", "bakery"]
COLORS = {
    "persku": "#2E86C1",
    "pooled": "#E67E22",
    "pooled_tl": "#27AE60",
    "both_win": "#27AE60",
    "persku_only": "#2E86C1",
    "pooled_only": "#E67E22",
    "both_lose": "#E74C3C",
}

# Dataset configs: (key, label, color)
DATASET_PERSKU = ("persku", "Per-SKU (v1.4)", COLORS["persku"])
DATASET_POOLED = ("pooled", "Pooled (v2)", COLORS["pooled"])
DATASET_POOLED_TL = ("pooled_tl", "Pooled TL (v2.1)", COLORS["pooled_tl"])


def load_results(path):
    """Load portfolio results JSON and index by product name."""
    with open(path) as f:
        data = json.load(f)
    by_product = {}
    for r in data["results"]:
        if r.get("status") == "ok":
            by_product[r["product"]] = r
    return data, by_product


def _best_reward(result):
    """Best of plain/shaped reward for a result."""
    return max(result["plain_reward"], result["shaped_reward"])


def _wins_baseline(result):
    """Whether best DQN reward beats best baseline."""
    return _best_reward(result) > result["best_baseline_reward"]


def plot_category_win_rates(datasets, common, save_dir):
    """Grouped bar chart: category win rates for each approach."""
    n_ds = len(datasets)
    categories = defaultdict(lambda: {k: 0 for k in [d[0] for d in datasets]} | {"total": 0})

    for name in common:
        cat = datasets[0][1][name].get("category", "unknown")
        categories[cat]["total"] += 1
        for key, by_product, label, color in datasets:
            if _wins_baseline(by_product[name]):
                categories[cat][key] += 1

    cats = [c for c in CATEGORY_ORDER if c in categories]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cats))
    w = 0.8 / n_ds

    for i, (key, by_product, label, color) in enumerate(datasets):
        rates = [categories[c][key] / max(categories[c]["total"], 1) * 100 for c in cats]
        offset = (i - (n_ds - 1) / 2) * w
        bars = ax.bar(x + offset, rates, w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

    ax.set_ylabel("Beats Best Baseline (%)", fontsize=12)
    title_parts = " vs ".join(label for _, _, label, _ in datasets)
    ax.set_title(f"Category Win Rates: {title_parts}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Overall totals
    total_n = sum(categories[c]["total"] for c in cats)
    parts = []
    for key, _, label, _ in datasets:
        total = sum(categories[c][key] for c in cats)
        parts.append(f"{label.split('(')[1].rstrip(')')}: {total}/{total_n} ({total/total_n*100:.0f}%)")
    ax.text(0.02, 0.95, "  |  ".join(parts),
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_category_win_rates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reward_scatter(datasets, common, save_dir):
    """Scatter plots: pairwise reward comparison colored by category."""
    cat_colors = {
        "bakery": "#E74C3C", "dairy": "#3498DB", "deli_prepared": "#2ECC71",
        "fruits": "#F39C12", "meats": "#9B59B6", "seafood": "#1ABC9C",
        "vegetables": "#E67E22",
    }

    # Generate all pairwise scatter plots
    pairs = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            pairs.append((datasets[i], datasets[j]))

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(10 * n_pairs, 10))
    if n_pairs == 1:
        axes = [axes]

    for ax, (ds_x, ds_y) in zip(axes, pairs):
        key_x, bp_x, label_x, color_x = ds_x
        key_y, bp_y, label_y, color_y = ds_y

        for name in sorted(common):
            x_best = _best_reward(bp_x[name])
            y_best = _best_reward(bp_y[name])
            cat = bp_x[name].get("category", "unknown")
            ax.scatter(x_best, y_best, c=cat_colors.get(cat, "#95A5A6"), s=40, alpha=0.7,
                       edgecolors="white", linewidth=0.5)

        lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

        for cat, color in sorted(cat_colors.items()):
            ax.scatter([], [], c=color, s=60, label=cat)
        ax.legend(fontsize=9, loc="lower right")

        ax.set_xlabel(f"{label_x} Reward", fontsize=12)
        ax.set_ylabel(f"{label_y} Reward", fontsize=12)
        ax.set_title(f"{label_x} vs {label_y}", fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        above = sum(1 for n in common if _best_reward(bp_y[n]) > _best_reward(bp_x[n]))
        below = len(common) - above
        short_y = label_y.split("(")[1].rstrip(")")
        short_x = label_x.split("(")[1].rstrip(")")
        ax.text(0.02, 0.95, f"{short_y} better: {above}/{len(common)}  |  {short_x} better: {below}/{len(common)}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_reward_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reward_gap_by_category(datasets, common, save_dir):
    """Box plot: reward gap (DQN - baseline) by category for each approach."""
    n_ds = len(datasets)
    cat_gaps = defaultdict(lambda: {d[0]: [] for d in datasets})

    for name in sorted(common):
        cat = datasets[0][1][name].get("category", "unknown")
        for key, by_product, label, color in datasets:
            best = _best_reward(by_product[name])
            cat_gaps[cat][key].append(best - by_product[name]["best_baseline_reward"])

    cats = [c for c in CATEGORY_ORDER if c in cat_gaps]
    spacing = n_ds + 1

    fig, ax = plt.subplots(figsize=(14, 6))
    bp_handles = []

    for i, (key, _, label, color) in enumerate(datasets):
        positions = [c_idx * spacing + i for c_idx in range(len(cats))]
        bp = ax.boxplot([cat_gaps[c][key] for c in cats], positions=positions,
                         widths=0.7, patch_artist=True,
                         boxprops=dict(facecolor=color, alpha=0.6),
                         medianprops=dict(color="black", linewidth=2),
                         flierprops=dict(marker="o", markersize=4, alpha=0.5))
        bp_handles.append((bp["boxes"][0], label))

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks([c_idx * spacing + (n_ds - 1) / 2 for c_idx in range(len(cats))])
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Reward Gap (DQN - Best Baseline)", fontsize=12)
    ax.set_title("Reward Gap Distribution by Category", fontsize=14, fontweight="bold")
    ax.legend([h for h, _ in bp_handles], [l for _, l in bp_handles], fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_reward_gap_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_venn_wins(datasets, common, save_dir):
    """Stacked bar showing win/lose overlap between first two datasets."""
    # Use first two datasets for overlap analysis
    ds1_key, ds1_bp, ds1_label, _ = datasets[0]
    ds2_key, ds2_bp, ds2_label, _ = datasets[1]

    cat_counts = defaultdict(lambda: {"both_win": 0, "ds1_only": 0, "ds2_only": 0, "both_lose": 0})

    for name in common:
        cat = ds1_bp[name].get("category", "unknown")
        ds1_wins = _wins_baseline(ds1_bp[name])
        ds2_wins = _wins_baseline(ds2_bp[name])

        if ds1_wins and ds2_wins:
            cat_counts[cat]["both_win"] += 1
        elif ds1_wins and not ds2_wins:
            cat_counts[cat]["ds1_only"] += 1
        elif not ds1_wins and ds2_wins:
            cat_counts[cat]["ds2_only"] += 1
        else:
            cat_counts[cat]["both_lose"] += 1

    cats = [c for c in CATEGORY_ORDER if c in cat_counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cats))

    both_w = [cat_counts[c]["both_win"] for c in cats]
    d1_only = [cat_counts[c]["ds1_only"] for c in cats]
    d2_only = [cat_counts[c]["ds2_only"] for c in cats]
    both_l = [cat_counts[c]["both_lose"] for c in cats]

    short1 = ds1_label.split("(")[1].rstrip(")")
    short2 = ds2_label.split("(")[1].rstrip(")")

    ax.bar(x, both_w, label="Both win", color=COLORS["both_win"], alpha=0.85)
    ax.bar(x, d1_only, bottom=both_w, label=f"{short1} only", color=COLORS["persku"], alpha=0.85)
    cumul = [a + b for a, b in zip(both_w, d1_only)]
    ax.bar(x, d2_only, bottom=cumul, label=f"{short2} only", color=COLORS["pooled"], alpha=0.85)
    cumul2 = [a + b for a, b in zip(cumul, d2_only)]
    ax.bar(x, both_l, bottom=cumul2, label="Both lose", color=COLORS["both_lose"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Number of SKUs", fontsize=12)
    ax.set_title(f"Win/Lose Overlap: {ds1_label} vs {ds2_label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    t_both = sum(both_w)
    t_d1 = sum(d1_only)
    t_d2 = sum(d2_only)
    t_lose = sum(both_l)
    ax.text(0.02, 0.95, f"Both win: {t_both}  |  {short1} only: {t_d1}  |  {short2} only: {t_d2}  |  Both lose: {t_lose}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_win_overlap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_dashboard(datasets, common, save_dir):
    """4-panel summary dashboard."""
    n = len(common)
    n_ds = len(datasets)

    # Compute per-dataset metrics
    ds_metrics = []
    for key, by_product, label, color in datasets:
        wins = sum(1 for name in common if _wins_baseline(by_product[name]))
        gaps = [_best_reward(by_product[name]) - by_product[name]["best_baseline_reward"] for name in common]
        rewards = [_best_reward(by_product[name]) for name in common]
        ds_metrics.append({
            "key": key, "label": label, "color": color,
            "wins": wins, "gaps": gaps, "rewards": rewards,
        })

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    title_parts = " vs ".join(label for _, _, label, _ in datasets)
    fig.suptitle(f"{title_parts}", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Key metrics table
    ax = axes[0, 0]
    model_counts = {"persku": "300", "pooled": "14", "pooled_tl": "300"}
    metrics = ["Beats Baseline", "Models Trained", "Median Gap", "Mean Reward"]
    col_labels = ["Metric"] + [d["label"] for d in ds_metrics]
    rows = []
    for m_idx, metric in enumerate(metrics):
        row = [metric]
        for d in ds_metrics:
            if m_idx == 0:
                row.append(f"{d['wins']}/{n} ({d['wins']/n*100:.0f}%)")
            elif m_idx == 1:
                row.append(model_counts.get(d["key"], "?"))
            elif m_idx == 2:
                row.append(f"{np.median(d['gaps']):.1f}")
            else:
                row.append(f"{np.mean(d['rewards']):.1f}")
        rows.append(row)

    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    col_bg = ["#D5D8DC", "#D6EAF8", "#FDEBD0", "#D5F5E3"]
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#D5D8DC")
            cell.set_fontsize(11)
            cell.get_text().set_fontweight("bold")
        elif key[1] >= 1 and key[1] < len(col_bg):
            cell.set_facecolor(col_bg[key[1]])
    ax.set_title("Key Metrics", fontsize=13, fontweight="bold", pad=20)

    # Panel 2: Reward gap histogram overlay
    ax = axes[0, 1]
    all_gaps = [g for d in ds_metrics for g in d["gaps"]]
    bins = np.linspace(min(all_gaps) - 2, max(all_gaps) + 2, 30)
    for d in ds_metrics:
        short = d["label"].split("(")[1].rstrip(")")
        ax.hist(d["gaps"], bins=bins, alpha=0.5, color=d["color"],
                label=f"{short} (med={np.median(d['gaps']):.1f})")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Reward Gap (DQN - Best Baseline)")
    ax.set_ylabel("Count")
    ax.set_title("Reward Gap Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 3: Reward correlation (best two datasets)
    ax = axes[1, 0]
    cat_colors_map = {
        "bakery": "#E74C3C", "dairy": "#3498DB", "deli_prepared": "#2ECC71",
        "fruits": "#F39C12", "meats": "#9B59B6", "seafood": "#1ABC9C",
        "vegetables": "#E67E22",
    }
    # Plot best vs second-best (or first vs last if 3-way)
    d_x = ds_metrics[0]
    d_y = ds_metrics[-1]
    x_rewards = [_best_reward(datasets[0][1][name]) for name in sorted(common)]
    y_rewards = [_best_reward(datasets[-1][1][name]) for name in sorted(common)]
    cats_list = [datasets[0][1][name].get("category", "unknown") for name in sorted(common)]

    for xr, yr, cat in zip(x_rewards, y_rewards, cats_list):
        ax.scatter(xr, yr, c=cat_colors_map.get(cat, "#95A5A6"), s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.3)
    lims = [0, max(max(x_rewards), max(y_rewards)) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel(f"{d_x['label']} Reward")
    ax.set_ylabel(f"{d_y['label']} Reward")
    ax.set_title("Reward Correlation (per product)", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    for cat, color in sorted(cat_colors_map.items()):
        ax.scatter([], [], c=color, s=40, label=cat)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    # Panel 4: Version progression bar chart
    ax = axes[1, 1]
    versions = [d["label"] for d in ds_metrics]
    win_rates = [d["wins"] / n * 100 for d in ds_metrics]
    colors = [d["color"] for d in ds_metrics]
    bars = ax.bar(range(len(versions)), win_rates, color=colors, alpha=0.85)
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{rate:.0f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=10)
    ax.set_ylabel("Beats Best Baseline (%)", fontsize=12)
    ax.set_title("Version Progression", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "comparison_summary_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare portfolio results across training approaches")
    parser.add_argument("--persku", default="results/portfolio_v140_5000ep/portfolio_results.json",
                        help="Per-SKU (v1.4) results path")
    parser.add_argument("--pooled", default="results/portfolio_v2_pooled/portfolio_results.json",
                        help="Pooled (v2) results path")
    parser.add_argument("--pooled-tl", default=None,
                        help="Pooled TL (v2.1) results path (optional, enables 3-way comparison)")
    parser.add_argument("--save-dir", default=None,
                        help="Output directory (default: auto-named)")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = "results/comparison_v14_v2_v21" if args.pooled_tl else "results/comparison_v14_vs_v2"
    os.makedirs(args.save_dir, exist_ok=True)

    # Load datasets
    print(f"Loading per-SKU results: {args.persku}")
    _, persku_bp = load_results(args.persku)
    print(f"  {len(persku_bp)} products")

    print(f"Loading pooled results: {args.pooled}")
    _, pooled_bp = load_results(args.pooled)
    print(f"  {len(pooled_bp)} products")

    datasets = [
        (DATASET_PERSKU[0], persku_bp, DATASET_PERSKU[1], DATASET_PERSKU[2]),
        (DATASET_POOLED[0], pooled_bp, DATASET_POOLED[1], DATASET_POOLED[2]),
    ]

    if args.pooled_tl:
        print(f"Loading pooled TL results: {args.pooled_tl}")
        _, pooled_tl_bp = load_results(args.pooled_tl)
        print(f"  {len(pooled_tl_bp)} products")
        datasets.append(
            (DATASET_POOLED_TL[0], pooled_tl_bp, DATASET_POOLED_TL[1], DATASET_POOLED_TL[2]),
        )

    # Find common products across all datasets
    common = set(persku_bp) & set(pooled_bp)
    if args.pooled_tl:
        common &= set(pooled_tl_bp)
    print(f"Common products: {len(common)}")

    print("\nGenerating comparison plots...")
    plot_category_win_rates(datasets, common, args.save_dir)
    plot_reward_scatter(datasets, common, args.save_dir)
    plot_reward_gap_by_category(datasets, common, args.save_dir)
    plot_venn_wins(datasets, common, args.save_dir)
    plot_summary_dashboard(datasets, common, args.save_dir)

    print(f"\nDone! All plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
