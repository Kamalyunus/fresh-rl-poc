"""
Run the complete pipeline: train (plain + shaped) → evaluate → visualize.

Usage:
    python scripts/run_all.py [--product salad_mix] [--episodes 500] [--step-hours 4]
"""

import sys
import os
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train
from scripts.evaluate import run_evaluation
from scripts.visualize import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Run complete Markdown Channel RL pipeline")
    parser.add_argument("--product", type=str, default="salad_mix",
                        choices=["salad_mix", "fresh_chicken", "yogurt", "bakery_bread", "sushi"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--step-hours", type=int, default=4, choices=[2, 4],
                        help="Hours per decision step (2 or 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results")

    # DQN variants
    parser.add_argument("--no-double-dqn", action="store_true",
                        help="Disable Double DQN (enabled by default)")
    parser.add_argument("--soft-target-tau", type=float, default=0.005,
                        help="Soft target update rate (0 = hard updates only)")

    # PER and pre-fill options
    parser.add_argument("--per", action="store_true", help="Enable Prioritized Experience Replay")
    parser.add_argument("--prefill", action="store_true",
                        help="Pre-fill replay buffer with historical baseline data")
    parser.add_argument("--prefill-episodes", type=int, default=200,
                        help="Number of historical episodes for pre-fill")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Gradient steps on buffered data before online training")
    parser.add_argument("--warmup-epsilon", type=float, default=None,
                        help="Starting epsilon after warmup")
    parser.add_argument("--shaping-ratio", type=float, default=0.2,
                        help="Shaping strength relative to revenue scale (default: 0.2)")

    args = parser.parse_args()

    suffix = f"{args.product}_{args.step_hours}h"

    print("\n" + "=" * 70)
    print("  MARKDOWN CHANNEL RL — Complete Pipeline")
    print(f"  Config: {args.product}, {args.step_hours}h steps")
    print("=" * 70)

    # Common DQN/PER/prefill kwargs
    per_kwargs = dict(
        use_per=args.per,
        prefill=args.prefill,
        prefill_episodes=args.prefill_episodes,
        warmup_steps=args.warmup_steps,
        warmup_epsilon=args.warmup_epsilon,
        double_dqn=not args.no_double_dqn,
        soft_target_tau=args.soft_target_tau if args.soft_target_tau > 0 else None,
        shaping_ratio=args.shaping_ratio,
    )

    per_suffix = "_per" if args.per else ""

    # Phase 1: Train without reward shaping
    print("\n[PHASE 1] Training DQN (no reward shaping)...")
    agent, history = train(
        n_episodes=args.episodes,
        product=args.product,
        step_hours=args.step_hours,
        reward_shaping=False,
        seed=args.seed,
        save_dir=args.save_dir,
        **per_kwargs,
    )

    # Save plain agent as the default best
    src = os.path.join(args.save_dir, f"best_agent_{suffix}{per_suffix}.pkl")
    plain_dst = os.path.join(args.save_dir, f"best_agent_{suffix}{per_suffix}_plain.pkl")
    if os.path.exists(src):
        shutil.copy2(src, plain_dst)

    # Phase 2: Train with reward shaping
    print("\n[PHASE 2] Training DQN (with reward shaping)...")
    agent_shaped, history_shaped = train(
        n_episodes=args.episodes,
        product=args.product,
        step_hours=args.step_hours,
        reward_shaping=True,
        seed=args.seed,
        save_dir=args.save_dir,
        **per_kwargs,
    )

    # Save shaped agent separately, restore plain as default
    shaped_dst = os.path.join(args.save_dir, f"best_agent_{suffix}{per_suffix}_shaped.pkl")
    if os.path.exists(src):
        shutil.copy2(src, shaped_dst)
    if os.path.exists(plain_dst):
        shutil.copy2(plain_dst, src)

    # Phase 3: Evaluate
    print("\n[PHASE 3] Evaluating all policies...")
    run_evaluation(
        product=args.product,
        step_hours=args.step_hours,
        n_episodes=args.eval_episodes,
        seed=args.seed,
        save_dir=args.save_dir,
        use_per=args.per,
    )

    # Phase 4: Visualize
    print("\n[PHASE 4] Generating visualizations...")
    generate_all_plots(product=args.product, step_hours=args.step_hours, save_dir=args.save_dir, use_per=args.per)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print(f"  Results in: {args.save_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
