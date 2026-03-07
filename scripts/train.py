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
    n_episodes: int = 1000,
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
    shaping_ratio: float = 0.2,
    env_overrides: dict = None,
    epsilon_decay: float = None,
    pretrained_path: str = None,
    hidden_dim: int = 64,
    replay_ratio: int = 1,
    batch_size: int = 32,
    buffer_size: int = 10000,
    n_step: int = 1,
    hold_action_prob: float = 0.0,
    augment_state: bool = False,
    inventory_mult: float = 1.0,
    tau_start: float = 0.005,
    tau_end: float = 0.005,
    tau_warmup_steps: int = 0,
    tl_warmup_steps: int = None,
    tl_epsilon_start: float = None,
    tl_epsilon_decay: float = None,
    epsilon_end: float = 0.05,
    early_stop_patience: int = None,
    greedy_eval_n: int = 20,
    prefill_transitions_path: str = None,
    best_baseline=None,
):
    """Train a DQN agent and save results."""

    os.makedirs(save_dir, exist_ok=True)

    # Create environment
    env_kwargs = dict(product_name=product, step_hours=step_hours, seed=seed)
    if env_overrides:
        env_kwargs.update(env_overrides)
    env = MarkdownProductEnv(**env_kwargs)

    # Wrap with augmented state for pooled→per-SKU transfer learning
    if augment_state:
        from fresh_rl.pooled_env import AugmentedProductEnv
        env = AugmentedProductEnv(env, product, inventory_mult=inventory_mult)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Suffix for file naming
    suffix = f"{product}_{step_hours}h"
    if use_per:
        suffix += "_per"
    if reward_shaping:
        suffix += "_shaped"

    # Compute waste cost scale: normalize relative to revenue scale
    revenue_scale = env.base_price * env.initial_inventory
    waste_cost_scale = shaping_ratio * revenue_scale

    # Epsilon decay: slower for 2h mode (more steps per episode)
    if epsilon_decay is None:
        epsilon_decay = 0.998 if step_hours == 2 else 0.997

    print(f"{'='*60}")
    print(f"  Markdown Channel RL — Training DQN Agent")
    print(f"{'='*60}")
    print(f"  Product:           {product}")
    print(f"  Step hours:        {step_hours}h ({env.n_time_blocks} blocks/day)")
    print(f"  Markdown window:   {env.markdown_window_hours}h ({env.episode_length} steps)")
    print(f"  Initial inventory: ~{env.initial_inventory} units")
    print(f"  Base price:        ${env.base_price:.2f}")
    print(f"  State dim:         {state_dim}")
    print(f"  Hidden dim:        {hidden_dim}")
    print(f"  Replay ratio:      {replay_ratio}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Buffer size:       {buffer_size}")
    print(f"  N-step returns:    {n_step}")
    print(f"  Hold action prob:  {hold_action_prob}")
    print(f"  Actions:           {n_actions} ({list(env.DISCOUNT_LEVELS)})")
    print(f"  Episodes:          {n_episodes}")
    print(f"  Reward shaping:    {reward_shaping}" +
          (f" (ratio={shaping_ratio}, scale={waste_cost_scale:.1f})" if reward_shaping else ""))
    print(f"  Epsilon decay:     {epsilon_decay}")
    print(f"  PER:               {use_per}")
    if tau_warmup_steps > 0:
        print(f"  Tau schedule:      {tau_start} → {tau_end} over {tau_warmup_steps} steps")
    if prefill_transitions_path and os.path.exists(prefill_transitions_path):
        print(f"  Pre-fill:          from pooled transitions")
        print(f"  Warm-up steps:     {warmup_steps}")
    elif prefill:
        print(f"  Pre-fill:          {prefill_episodes} episodes")
        print(f"  Warm-up steps:     {warmup_steps}")
    print(f"  Seed:              {seed}")
    if pretrained_path:
        print(f"  Pre-trained:       {os.path.basename(pretrained_path)}")
    if early_stop_patience is not None:
        print(f"  Early stopping:    patience={early_stop_patience} episodes")
    print(f"{'='*60}\n")

    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr=5e-4,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        reward_shaping=reward_shaping,
        waste_cost_scale=waste_cost_scale,
        seed=seed,
        use_per=use_per,
        n_step=n_step,
        hold_action_prob=hold_action_prob,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_warmup_steps=tau_warmup_steps,
    )

    # Transfer learning: load pre-trained weights
    if pretrained_path:
        agent.load_pretrained(pretrained_path)
        agent.epsilon = tl_epsilon_start if tl_epsilon_start is not None else 0.10
        if tl_epsilon_decay is not None:
            agent.epsilon_decay = tl_epsilon_decay
        print(f"  [TRANSFER] Loaded pre-trained weights from {os.path.basename(pretrained_path)}")
        print(f"  [TRANSFER] Fine-tuning epsilon: {agent.epsilon}, decay: {agent.epsilon_decay}")
        # Reduce warmup for TL — pretrained weights don't need full warmup
        original_warmup = warmup_steps
        warmup_steps = tl_warmup_steps if tl_warmup_steps is not None else 0
        print(f"  [TRANSFER] Warmup steps adjusted: {original_warmup} → {warmup_steps}\n")

    # Pre-fill phase: load historical data into replay buffer
    if prefill_transitions_path and os.path.exists(prefill_transitions_path):
        data = np.load(prefill_transitions_path)
        has_per = hasattr(agent.replay_buffer, 'max_priority')
        if has_per:
            old_max = agent.replay_buffer.max_priority
            agent.replay_buffer.max_priority = 5.0
        for i in range(len(data['actions'])):
            agent.store_transition(
                data['states'][i], data['actions'][i], float(data['rewards'][i]),
                data['next_states'][i], bool(data['dones'][i]), data['masks'][i],
            )
        if has_per:
            agent.replay_buffer.max_priority = old_max
        print(f"  [PRE-FILL] Loaded {len(data['actions'])} transitions from pooled training\n")
    elif prefill:
        from fresh_rl.historical_data import HistoricalDataGenerator
        print("  [PRE-FILL] Generating historical data...")
        generator = HistoricalDataGenerator(
            product=product,
            step_hours=step_hours,
            baseline_mix=prefill_baselines,
            seed=seed,
            env=env if augment_state else None,
        )
        n_transitions = generator.fill_buffer(
            agent.replay_buffer,
            n_episodes=prefill_episodes,
            initial_priority=5.0,
            agent=agent,
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

    # Training loop
    episode_rewards = []
    episode_revenues = []
    episode_wastes = []
    episode_clearance = []
    baseline_rewards = []  # per-episode baseline replay on same seed (None if no baseline)
    greedy_eval_episodes = []  # (episode_idx, mean_reward, mean_revenue, mean_waste, mean_clearance)
    best_reward = -float("inf")

    # Early stopping state
    best_greedy_reward = -float("inf")
    greedy_no_improve_count = 0
    early_stopped = False
    early_stop_episode = None

    # Greedy eval frequency: every 50 episodes
    eval_freq = 50
    eval_n = greedy_eval_n

    def _greedy_eval():
        """Run greedy policy (epsilon=0) and return mean metrics."""
        eval_kwargs = dict(product_name=product, step_hours=step_hours, seed=seed + 10000)
        if env_overrides:
            eval_kwargs.update(env_overrides)
        eval_env = MarkdownProductEnv(**eval_kwargs)
        if augment_state:
            from fresh_rl.pooled_env import AugmentedProductEnv
            eval_env = AugmentedProductEnv(eval_env, product, inventory_mult=inventory_mult)
        old_eps = agent.epsilon
        agent.epsilon = 0.0
        eval_rewards, eval_revenues, eval_wastes, eval_clears = [], [], [], []
        for _ in range(eval_n):
            obs_e, _ = eval_env.reset()
            total_r = 0.0
            done_e = False
            while not done_e:
                a = agent.select_action(obs_e, action_mask=eval_env.action_masks())
                obs_e, r, term, trunc, info_e = eval_env.step(a)
                done_e = term or trunc
                total_r += r
            eval_rewards.append(total_r)
            stats_e = info_e.get("episode_stats", {})
            eval_revenues.append(stats_e.get("total_revenue", 0))
            eval_wastes.append(stats_e.get("waste_rate", 0))
            eval_clears.append(stats_e.get("clearance_rate", 0))
        agent.epsilon = old_eps
        return np.mean(eval_rewards), np.mean(eval_revenues), np.mean(eval_wastes), np.mean(eval_clears)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
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

            # Train (replay_ratio gradient steps per env step)
            for _ in range(replay_ratio):
                agent.train_step_fn()

            obs = next_obs
            total_reward += reward

        # Replay same demand with baseline policy (production-realistic comparison)
        bl_reward = None
        if best_baseline is not None:
            bl_obs, _ = env.reset(seed=seed + ep)
            bl_done = False
            bl_total = 0.0
            while not bl_done:
                bl_action = best_baseline.select_action(bl_obs, env=env)
                bl_obs, bl_r, bl_term, bl_trunc, _ = env.step(bl_action)
                bl_done = bl_term or bl_trunc
                bl_total += bl_r
            bl_reward = bl_total
        baseline_rewards.append(bl_reward)

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

        # Periodic evaluation + logging
        if (ep + 1) % eval_freq == 0 or ep == 0:
            recent_rewards = episode_rewards[-50:]
            recent_revenue = episode_revenues[-50:]
            recent_waste = episode_wastes[-50:]
            recent_clear = episode_clearance[-50:]

            # Checkpoint selection: rolling win rate (production-realistic)
            win_str = ""
            current_wr = 0.0
            if best_baseline is not None:
                recent_bl = baseline_rewards[-50:]
                recent_ep = episode_rewards[-50:]
                daily_wins = sum(1 for e, b in zip(recent_ep, recent_bl) if b is not None and e > b)
                daily_total = sum(1 for b in recent_bl if b is not None)
                if daily_total > 0:
                    current_wr = daily_wins / daily_total
                    win_str = f" | Win%: {current_wr*100:.0f}%"
                if current_wr > best_greedy_reward:  # reusing var as best_win_rate
                    best_greedy_reward = current_wr
                    greedy_no_improve_count = 0
                    agent.save(os.path.join(save_dir, f"best_greedy_{suffix}.pt"))
                else:
                    greedy_no_improve_count += 1

            # Greedy eval (optional, for monitoring only — not used for checkpoint selection)
            g_str = ""
            if eval_n > 0:
                g_reward, g_revenue, g_waste, g_clear = _greedy_eval()
                greedy_eval_episodes.append((ep, g_reward, g_revenue, g_waste, g_clear))
                g_str = f" | Greedy: {g_reward:6.1f}"
                # Fallback: if no baseline provided, use greedy eval for checkpoint
                if best_baseline is None:
                    if g_reward > best_greedy_reward:
                        best_greedy_reward = g_reward
                        greedy_no_improve_count = 0
                        agent.save(os.path.join(save_dir, f"best_greedy_{suffix}.pt"))
                    else:
                        greedy_no_improve_count += 1

            print(
                f"  Episode {ep+1:4d}/{n_episodes} | "
                f"Reward: {np.mean(recent_rewards):8.1f} | "
                f"Revenue: ${np.mean(recent_revenue):7.1f} | "
                f"Waste: {np.mean(recent_waste)*100:5.1f}% | "
                f"Clear: {np.mean(recent_clear)*100:5.1f}% | "
                f"Eps: {agent.epsilon:.3f}"
                f"{g_str}{win_str}"
            )

            # Check early stopping condition
            if early_stop_patience is not None and greedy_no_improve_count * eval_freq >= early_stop_patience:
                early_stopped = True
                early_stop_episode = ep + 1
                print(f"\n  [EARLY STOP] No improvement for {early_stop_patience} episodes. "
                      f"Stopping at episode {ep+1}.")
                break

    # Reload best greedy checkpoint (early stop or normal completion)
    best_greedy_path = os.path.join(save_dir, f"best_greedy_{suffix}.pt")
    if os.path.exists(best_greedy_path):
        agent.load_pretrained(best_greedy_path)
        agent.epsilon = 0.0

    # Save training history
    history = {
        "product": product,
        "step_hours": step_hours,
        "n_actions": int(n_actions),
        "discount_levels": env.DISCOUNT_LEVELS.tolist(),
        "n_episodes": n_episodes,
        "reward_shaping": reward_shaping,
        "shaping_ratio": shaping_ratio,
        "use_per": use_per,
        "prefill": prefill,
        "prefill_episodes": prefill_episodes if prefill else 0,
        "warmup_steps": warmup_steps,
        "epsilon_decay": epsilon_decay,
        "replay_ratio": replay_ratio,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "n_step": n_step,
        "hold_action_prob": hold_action_prob,
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_revenues": episode_revenues,
        "episode_wastes": episode_wastes,
        "episode_clearance": episode_clearance,
        "baseline_rewards": baseline_rewards,
        "losses": agent.losses[-2000:],
        "greedy_eval": [
            {"episode": e, "reward": r, "revenue": rev, "waste": w, "clearance": c}
            for e, r, rev, w, c in greedy_eval_episodes
        ],
        "best_reward": best_reward,
        "early_stopped": early_stopped,
        "early_stop_episode": early_stop_episode,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(save_dir, f"training_history_{suffix}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!" + (f" (early stopped at ep {early_stop_episode})" if early_stopped else ""))
    if best_baseline is not None:
        print(f"  Best win rate: {best_greedy_reward*100:.0f}%")
    elif eval_n > 0:
        print(f"  Best greedy reward: {best_greedy_reward:.1f}")
    print(f"  Best episode reward: {best_reward:.1f}")
    print(f"  Results saved to: {save_dir}/")
    print(f"{'='*60}")

    return agent, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for markdown channel discounting")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--product", type=str, default="salad_mix")
    parser.add_argument("--list-products", action="store_true",
                        help="List all available products and exit")
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
    parser.add_argument("--shaping-ratio", type=float, default=0.2,
                        help="Shaping strength relative to revenue scale (default: 0.2)")

    # Tau schedule
    parser.add_argument("--tau-start", type=float, default=0.005,
                        help="Initial tau for soft target updates (default: 0.005)")
    parser.add_argument("--tau-end", type=float, default=0.005,
                        help="Final tau after warmup (default: 0.005)")
    parser.add_argument("--tau-warmup-steps", type=int, default=0,
                        help="Steps to linearly warm tau from tau-start to tau-end (default: 0 = constant)")

    # Transfer learning warmup
    parser.add_argument("--tl-warmup-steps", type=int, default=None,
                        help="Override warmup steps when using pretrained weights (default: 0 = skip warmup)")

    # Transfer learning epsilon
    parser.add_argument("--tl-epsilon-start", type=float, default=None,
                        help="Starting epsilon for TL fine-tuning (default: 0.3)")
    parser.add_argument("--tl-epsilon-decay", type=float, default=None,
                        help="Epsilon decay for TL fine-tuning (default: use standard decay)")

    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="Stop training after this many episodes without greedy reward improvement (default: disabled)")

    args = parser.parse_args()

    if args.list_products:
        from fresh_rl.product_catalog import print_catalog_summary
        print_catalog_summary()
        sys.exit(0)

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
        shaping_ratio=args.shaping_ratio,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_warmup_steps=args.tau_warmup_steps,
        tl_warmup_steps=args.tl_warmup_steps,
        tl_epsilon_start=args.tl_epsilon_start,
        tl_epsilon_decay=args.tl_epsilon_decay,
        early_stop_patience=args.early_stop_patience,
    )
