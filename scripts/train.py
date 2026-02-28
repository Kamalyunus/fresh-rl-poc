"""
Training script for the DQN agent on the markdown channel environment.

Usage:
    python scripts/train.py [--episodes 500] [--product salad_mix] [--step-hours 4] [--reward-shaping] [--seed 42]
    python scripts/train.py --product sushi --episodes 500 --step-hours 2 --per --prefill --warmup-steps 1000
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


def train(
    n_episodes: int = 500,
    product: str = "salad_mix",
    step_hours: int = 4,
    reward_shaping: bool = False,
    seed: int = 42,
    save_dir: str = "results",
    use_per: bool = False,
    prefill: bool = False,
    prefill_episodes: int = 200,
    prefill_baselines: dict = None,
    warmup_steps: int = 0,
    warmup_epsilon: float = None,
):
    """Train a DQN agent and save results."""

    os.makedirs(save_dir, exist_ok=True)

    # Create environment
    env = MarkdownProductEnv(product_name=product, step_hours=step_hours, seed=seed)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Suffix for file naming
    suffix = f"{product}_{step_hours}h"
    if use_per:
        suffix += "_per"

    print(f"{'='*60}")
    print(f"  Markdown Channel RL — Training DQN Agent")
    print(f"{'='*60}")
    print(f"  Product:           {product}")
    print(f"  Step hours:        {step_hours}h ({env.n_time_blocks} blocks/day)")
    print(f"  Markdown window:   {env.markdown_window_hours}h ({env.episode_length} steps)")
    print(f"  Initial inventory: ~{env.initial_inventory} units")
    print(f"  Base price:        ${env.base_price:.2f}")
    print(f"  State dim:         {state_dim}")
    print(f"  Actions:           {n_actions} ({list(env.DISCOUNT_LEVELS)})")
    print(f"  Episodes:          {n_episodes}")
    print(f"  Reward shaping:    {reward_shaping}")
    print(f"  PER:               {use_per}")
    if prefill:
        print(f"  Pre-fill:          {prefill_episodes} episodes")
        print(f"  Warm-up steps:     {warmup_steps}")
        if warmup_epsilon is not None:
            print(f"  Warm-up epsilon:   {warmup_epsilon}")
    print(f"  Seed:              {seed}")
    print(f"{'='*60}\n")

    # Epsilon decay: slower for 2h mode (more steps per episode)
    epsilon_decay = 0.998 if step_hours == 2 else 0.997

    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=64,
        lr=5e-4,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=50,
        reward_shaping=reward_shaping,
        seed=seed,
        use_per=use_per,
    )

    # Pre-fill phase: load historical data into replay buffer
    if prefill:
        from fresh_rl.historical_data import HistoricalDataGenerator
        print("  [PRE-FILL] Generating historical data...")
        generator = HistoricalDataGenerator(
            product=product,
            step_hours=step_hours,
            baseline_mix=prefill_baselines,
            seed=seed,
        )
        n_transitions = generator.fill_buffer(
            agent.replay_buffer,
            n_episodes=prefill_episodes,
            initial_priority=5.0,
        )
        print(f"  [PRE-FILL] Added {n_transitions} transitions to buffer\n")

    # Warm-up phase: gradient steps on buffered data before online training
    if warmup_steps > 0 and len(agent.replay_buffer) >= agent.batch_size:
        print(f"  [WARM-UP] Running {warmup_steps} gradient steps...")
        warmup_losses = []
        for step in range(warmup_steps):
            loss = agent.train_step_fn()
            if loss is not None:
                warmup_losses.append(loss)
            if (step + 1) % 200 == 0 and warmup_losses:
                recent = warmup_losses[-200:]
                print(f"    Step {step+1:5d}/{warmup_steps} | Loss: {np.mean(recent):.4f}")
        if warmup_losses:
            print(f"  [WARM-UP] Final loss: {np.mean(warmup_losses[-100:]):.4f}\n")

    # Set warm-start epsilon if specified
    if warmup_epsilon is not None:
        agent.epsilon = warmup_epsilon
        print(f"  [WARM-START] Epsilon set to {warmup_epsilon}\n")

    # Training loop
    episode_rewards = []
    episode_revenues = []
    episode_wastes = []
    episode_clearance = []
    best_reward = -float("inf")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action_mask = env.action_masks()
            action = agent.select_action(obs, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get next action mask for replay buffer
            next_action_mask = env.action_masks() if not done else np.ones(n_actions, dtype=bool)
            agent.store_transition(obs, action, reward, next_obs, done, next_action_mask)

            # Train
            loss = agent.train_step_fn()

            obs = next_obs
            total_reward += reward

        # End of episode
        agent.decay_epsilon()
        agent.episode_rewards.append(total_reward)
        episode_rewards.append(total_reward)

        stats = info.get("episode_stats", {})
        episode_revenues.append(stats.get("total_revenue", 0))
        episode_wastes.append(stats.get("waste_rate", 0))
        episode_clearance.append(stats.get("clearance_rate", 0))

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(save_dir, f"best_agent_{suffix}.pkl"))

        # Logging
        if (ep + 1) % 50 == 0 or ep == 0:
            recent_rewards = episode_rewards[-50:]
            recent_revenue = episode_revenues[-50:]
            recent_waste = episode_wastes[-50:]
            recent_clear = episode_clearance[-50:]
            print(
                f"  Episode {ep+1:4d}/{n_episodes} | "
                f"Reward: {np.mean(recent_rewards):8.1f} | "
                f"Revenue: ${np.mean(recent_revenue):7.1f} | "
                f"Waste: {np.mean(recent_waste)*100:5.1f}% | "
                f"Clear: {np.mean(recent_clear)*100:5.1f}% | "
                f"Eps: {agent.epsilon:.3f}"
            )

    # Save final agent
    agent.save(os.path.join(save_dir, f"final_agent_{suffix}.pkl"))

    # Save training history
    history = {
        "product": product,
        "step_hours": step_hours,
        "n_actions": int(n_actions),
        "discount_levels": env.DISCOUNT_LEVELS.tolist(),
        "n_episodes": n_episodes,
        "reward_shaping": reward_shaping,
        "use_per": use_per,
        "prefill": prefill,
        "prefill_episodes": prefill_episodes if prefill else 0,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_revenues": episode_revenues,
        "episode_wastes": episode_wastes,
        "episode_clearance": episode_clearance,
        "best_reward": best_reward,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(save_dir, f"training_history_{suffix}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best reward:  {best_reward:.1f}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Results saved to: {save_dir}/")
    print(f"{'='*60}")

    return agent, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for markdown channel discounting")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--product", type=str, default="salad_mix",
                        choices=["salad_mix", "fresh_chicken", "yogurt", "bakery_bread", "sushi"])
    parser.add_argument("--step-hours", type=int, default=4, choices=[2, 4],
                        help="Hours per decision step (2 or 4)")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable reward shaping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results")

    # PER and pre-fill options
    parser.add_argument("--per", action="store_true", help="Enable Prioritized Experience Replay")
    parser.add_argument("--prefill", action="store_true",
                        help="Pre-fill replay buffer with historical baseline data")
    parser.add_argument("--prefill-episodes", type=int, default=200,
                        help="Number of historical episodes for pre-fill")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Gradient steps on buffered data before online training")
    parser.add_argument("--warmup-epsilon", type=float, default=None,
                        help="Starting epsilon after warmup (default: use standard decay)")

    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        product=args.product,
        step_hours=args.step_hours,
        reward_shaping=args.reward_shaping,
        seed=args.seed,
        save_dir=args.save_dir,
        use_per=args.per,
        prefill=args.prefill,
        prefill_episodes=args.prefill_episodes,
        warmup_steps=args.warmup_steps,
        warmup_epsilon=args.warmup_epsilon,
    )
