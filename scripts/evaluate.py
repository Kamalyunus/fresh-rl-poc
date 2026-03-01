"""
Evaluation script: compare DQN agent against all baselines.

Usage:
    python scripts/evaluate.py [--product salad_mix] [--step-hours 4] [--episodes 200] [--seed 42]
"""

import sys
import os
import argparse
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fresh_rl.environment import MarkdownProductEnv
from fresh_rl.dqn_agent import DQNAgent
from fresh_rl.baselines import get_all_baselines


def evaluate_policy(env, policy, n_episodes=200, seed=42, is_dqn=False):
    """Evaluate a policy over multiple episodes and return detailed metrics."""
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    rewards = []
    revenues = []
    waste_rates = []
    clearance_rates = []
    waste_units = []
    sold_units = []
    avg_discounts = []
    action_counts = np.zeros(n_actions)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100000)))
        total_reward = 0.0
        ep_actions = []
        done = False

        while not done:
            if is_dqn:
                action_mask = env.action_masks()
                action = policy.select_action(obs, action_mask=action_mask)
            else:
                action = policy.select_action(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            ep_actions.append(action)
            action_counts[action] += 1

        rewards.append(total_reward)
        stats = info.get("episode_stats", {})
        revenues.append(stats.get("total_revenue", 0))
        waste_rates.append(stats.get("waste_rate", 0))
        clearance_rates.append(stats.get("clearance_rate", 0))
        waste_units.append(stats.get("total_waste", 0))
        sold_units.append(stats.get("total_sold", 0))
        avg_discounts.append(np.mean([env.DISCOUNT_LEVELS[a] for a in ep_actions]))

    return {
        "policy_name": policy.name if hasattr(policy, "name") else "DQN Agent",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_revenue": float(np.mean(revenues)),
        "std_revenue": float(np.std(revenues)),
        "mean_waste_rate": float(np.mean(waste_rates)),
        "std_waste_rate": float(np.std(waste_rates)),
        "mean_clearance_rate": float(np.mean(clearance_rates)),
        "mean_waste_units": float(np.mean(waste_units)),
        "mean_sold_units": float(np.mean(sold_units)),
        "mean_discount": float(np.mean(avg_discounts)),
        "action_distribution": (action_counts / max(action_counts.sum(), 1)).tolist(),
    }


def run_evaluation(product="salad_mix", step_hours=4, n_episodes=200, seed=42, save_dir="results", use_per=False):
    """Run full evaluation comparing DQN against all baselines."""
    os.makedirs(save_dir, exist_ok=True)

    env = MarkdownProductEnv(product_name=product, step_hours=step_hours, seed=seed)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    suffix = f"{product}_{step_hours}h"
    if use_per:
        suffix += "_per"

    print(f"\n{'='*75}")
    print(f"  Markdown Channel RL — Evaluation: DQN vs Baselines")
    print(f"{'='*75}")
    print(f"  Product:    {product} (window: {env.markdown_window_hours}h, ~{env.initial_inventory} units)")
    print(f"  Step hours: {step_hours}h ({n_actions} actions, {env.episode_length} steps/episode)")
    print(f"  Episodes:   {n_episodes}")
    print(f"{'='*75}\n")

    # Load trained agents
    results = []
    agent_variants = []
    per_label = " + PER" if use_per else ""

    # Try loading DQN (no shaping)
    agent_path = os.path.join(save_dir, f"best_agent_{suffix}.pt")
    if os.path.exists(agent_path):
        agent = DQNAgent(state_dim=state_dim, n_actions=n_actions, seed=seed)
        agent.load(agent_path)
        agent.epsilon = 0.0
        agent.name = f"DQN Agent{per_label}"
        agent_variants.append(agent)

    # Try loading DQN (with shaping)
    agent_path_shaped = os.path.join(save_dir, f"best_agent_{suffix}_shaped.pt")
    if os.path.exists(agent_path_shaped):
        agent_shaped = DQNAgent(state_dim=state_dim, n_actions=n_actions, reward_shaping=True, seed=seed)
        agent_shaped.load(agent_path_shaped)
        agent_shaped.epsilon = 0.0
        agent_shaped.name = f"DQN + Shaping{per_label}"
        agent_variants.append(agent_shaped)

    # Baselines
    baselines = get_all_baselines(n_actions=n_actions, seed=seed)

    # Evaluate all policies
    all_policies = agent_variants + baselines
    for policy in all_policies:
        name = policy.name if hasattr(policy, "name") else str(policy)
        is_dqn = isinstance(policy, DQNAgent)
        print(f"  Evaluating: {name}...", end=" ", flush=True)
        result = evaluate_policy(env, policy, n_episodes=n_episodes, seed=seed, is_dqn=is_dqn)
        results.append(result)
        print(
            f"Revenue: ${result['mean_revenue']:7.1f} | "
            f"Waste: {result['mean_waste_rate']*100:5.1f}% | "
            f"Clear: {result['mean_clearance_rate']*100:5.1f}% | "
            f"Reward: {result['mean_reward']:8.1f}"
        )

    # Print comparison table
    print(f"\n{'='*85}")
    print(f"  {'Policy':<25} {'Revenue':>10} {'Waste %':>10} {'Clear %':>10} {'Reward':>10}")
    print(f"  {'-'*75}")

    results_sorted = sorted(results, key=lambda x: x["mean_reward"], reverse=True)
    dqn_names = {v.name for v in agent_variants}
    for r in results_sorted:
        marker = " *" if r["policy_name"] in dqn_names else "  "
        print(
            f"{marker}{r['policy_name']:<25} "
            f"${r['mean_revenue']:>8.1f} "
            f"{r['mean_waste_rate']*100:>9.1f}% "
            f"{r['mean_clearance_rate']*100:>9.1f}% "
            f"{r['mean_reward']:>9.1f}"
        )
    print(f"{'='*85}")

    # Compute improvements over best baseline
    baseline_results = [r for r in results if r["policy_name"] not in dqn_names]
    if baseline_results and agent_variants:
        best_baseline = max(baseline_results, key=lambda x: x["mean_reward"])
        best_agent = max(
            [r for r in results if r["policy_name"] in dqn_names],
            key=lambda x: x["mean_reward"],
        )

        rev_improvement = (best_agent["mean_revenue"] - best_baseline["mean_revenue"]) / max(best_baseline["mean_revenue"], 0.01) * 100
        waste_improvement = (best_baseline["mean_waste_rate"] - best_agent["mean_waste_rate"]) / max(best_baseline["mean_waste_rate"], 0.001) * 100

        print(f"\n  Best DQN vs Best Baseline ({best_baseline['policy_name']}):")
        print(f"    Revenue improvement:  {rev_improvement:+.1f}%")
        print(f"    Waste reduction:      {waste_improvement:+.1f}%")

    # Save results
    eval_results = {
        "product": product,
        "step_hours": step_hours,
        "n_actions": int(n_actions),
        "discount_levels": env.DISCOUNT_LEVELS.tolist(),
        "n_episodes": n_episodes,
        "use_per": use_per,
        "seed": seed,
        "results": results_sorted,
        "timestamp": datetime.now().isoformat(),
    }
    eval_path = os.path.join(save_dir, f"evaluation_{suffix}.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n  Results saved to: {eval_path}")
    return results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN agent vs baselines")
    parser.add_argument("--product", type=str, default="salad_mix")
    parser.add_argument("--step-hours", type=int, default=4, choices=[2, 4],
                        help="Hours per decision step (2 or 4)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--per", action="store_true", help="Look for PER-trained agent models")
    args = parser.parse_args()

    run_evaluation(
        product=args.product,
        step_hours=args.step_hours,
        n_episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir,
        use_per=args.per,
    )
