"""
Nightly batch training CLI.

During the day, agents make inference-only pricing decisions and log session
data to CSV. At night, this script trains agents on the day's data.

Usage:
    python -m deployment.batch_train --date 2026-03-01 --workers 16
    python -m deployment.batch_train --date 2026-03-01 --sku salmon_fillet
    python -m deployment.batch_train --date 2026-03-01 --lookback-days 14
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np

from deployment.config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    ELEVATED_PER_PRIORITY,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    HIDDEN_DIM,
    HOLD_ACTION_PROB,
    LOOKBACK_DAYS,
    LR,
    N_ACTIONS,
    N_STEP,
    PER_ALPHA,
    PER_BETA_END,
    PER_BETA_START,
    PER_EPSILON,
    STATE_DIM,
    TRAINING_STEPS_PER_SESSION,
    TRANSFER_EPSILON,
    USE_PER,
    WARMUP_STEPS,
    ProductionConfig,
)
from deployment.etl import SessionETL

logger = logging.getLogger(__name__)


def _compute_product_features(product: dict) -> np.ndarray:
    """Compute 4-dim product features from product master row.

    Uses category ranges from product_catalog.CATEGORIES for normalization.
    """
    from fresh_rl.product_catalog import CATEGORIES

    cat = CATEGORIES.get(product["category"])
    if cat is None:
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    def _norm(val, lo, hi):
        return float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    base_price = float(product["base_price"])
    cost_per_unit = float(product["cost_per_unit"])
    initial_inventory = int(product["initial_inventory"])
    pack_size = int(product.get("pack_size", 1))

    price_norm = _norm(base_price, *cat.price_range)
    cost_frac = cost_per_unit / max(base_price, 1e-6)
    cost_frac_norm = _norm(cost_frac, *cat.cost_fraction_range)
    inventory_norm = _norm(initial_inventory, *cat.inventory_range)
    pack_size_norm = _norm(pack_size, *cat.pack_size_range)

    return np.array([price_norm, cost_frac_norm, inventory_norm, pack_size_norm], dtype=np.float32)


def train_single_sku(
    sku_name: str,
    config: ProductionConfig,
    product: dict,
    end_date: str,
    lookback_days: int = LOOKBACK_DAYS,
    training_steps_per_session: int = TRAINING_STEPS_PER_SESSION,
) -> dict:
    """Train a single SKU on recent session data.

    Parameters
    ----------
    sku_name : str
        SKU to train.
    config : ProductionConfig
        Paths.
    product : dict
        Product master row (base_price, cost_per_unit, category, etc.).
    end_date : str
        Last date to include (YYYY-MM-DD).
    lookback_days : int
        How many days of history to use.
    training_steps_per_session : int
        Gradient steps per session of data.

    Returns
    -------
    dict
        Training summary with keys: sku_name, n_sessions, n_transitions,
        training_steps, final_loss, model_source, checkpoint_path.
    """
    from fresh_rl.dqn_agent import DQNAgent

    base_price = float(product["base_price"])
    cost_per_unit = float(product["cost_per_unit"])
    category = product["category"]
    initial_inventory = int(product["initial_inventory"])
    product_features = _compute_product_features(product)

    # Find session CSVs in date range
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    start = end - timedelta(days=lookback_days)
    start_date = start.isoformat()

    etl = SessionETL(config)
    csv_paths = etl.load_sessions_for_date_range(start_date, end_date, sku_name)

    if not csv_paths:
        logger.info("No sessions found for %s in [%s, %s]", sku_name, start_date, end_date)
        return {
            "sku_name": sku_name,
            "n_sessions": 0,
            "n_transitions": 0,
            "training_steps": 0,
            "final_loss": None,
            "model_source": "none",
            "checkpoint_path": None,
        }

    # ETL: sessions → transitions
    transitions = etl.etl_sessions(csv_paths, base_price, cost_per_unit, product_features)
    n_sessions = len(csv_paths)
    n_transitions = len(transitions)

    if n_transitions == 0:
        logger.info("No valid transitions for %s", sku_name)
        return {
            "sku_name": sku_name,
            "n_sessions": n_sessions,
            "n_transitions": 0,
            "training_steps": 0,
            "final_loss": None,
            "model_source": "none",
            "checkpoint_path": None,
        }

    # Load or create agent
    checkpoint_path = config.checkpoint_path(sku_name)
    model_source = "fresh"

    agent = DQNAgent(
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        reward_shaping=True,
        waste_cost_scale=cost_per_unit * 3.0 * initial_inventory,
        use_per=USE_PER,
        per_alpha=PER_ALPHA,
        per_beta_start=PER_BETA_START,
        per_beta_end=PER_BETA_END,
        per_epsilon=PER_EPSILON,
        n_step=N_STEP,
        hold_action_prob=HOLD_ACTION_PROB,
    )

    if os.path.exists(checkpoint_path):
        # Continue training from existing checkpoint
        try:
            agent.load(checkpoint_path)
            model_source = "continued"
            logger.info("Loaded existing checkpoint for %s", sku_name)
        except Exception as e:
            logger.warning("Failed to load checkpoint for %s: %s, starting fresh", sku_name, e)
    else:
        # Cold-start: try pooled category model
        pooled_path = config.pooled_model_path(category, "plain")
        if os.path.exists(pooled_path):
            try:
                agent.load_pretrained(pooled_path)
                agent.epsilon = TRANSFER_EPSILON
                model_source = "pooled-tl"
                logger.info("Cold-starting %s from pooled model (%s)", sku_name, category)
            except Exception as e:
                logger.warning("Failed to load pooled model for %s: %s", sku_name, e)

    # Feed transitions into replay buffer with elevated PER priority
    has_per = hasattr(agent.replay_buffer, "max_priority")
    if has_per:
        old_max = agent.replay_buffer.max_priority
        agent.replay_buffer.max_priority = ELEVATED_PER_PRIORITY

    for s, a, r, s_next, done, next_mask in transitions:
        agent.store_transition(s, a, r, s_next, done, next_mask)

    if has_per:
        agent.replay_buffer.max_priority = old_max

    # Training: gradient steps proportional to data volume
    total_steps = training_steps_per_session * n_sessions
    final_loss = None

    for step in range(total_steps):
        loss = agent.train_step_fn()
        if loss is not None:
            final_loss = loss

    # Decay epsilon once per batch run
    agent.decay_epsilon()

    # Rotate checkpoint: current → prev
    prev_path = config.prev_checkpoint_path(sku_name)
    if os.path.exists(checkpoint_path):
        shutil.copy2(checkpoint_path, prev_path)

    agent.save(checkpoint_path)
    logger.info(
        "Trained %s: %d sessions, %d transitions, %d steps, loss=%.4f",
        sku_name, n_sessions, n_transitions, total_steps,
        final_loss if final_loss is not None else 0.0,
    )

    return {
        "sku_name": sku_name,
        "n_sessions": n_sessions,
        "n_transitions": n_transitions,
        "training_steps": total_steps,
        "final_loss": final_loss,
        "model_source": model_source,
        "checkpoint_path": checkpoint_path,
    }


def _train_worker(args):
    """Worker function for ProcessPoolExecutor (must be top-level for pickling)."""
    sku_name, config_base_dir, product, end_date, lookback_days, steps_per_session = args
    config = ProductionConfig(base_dir=config_base_dir)
    return train_single_sku(sku_name, config, product, end_date, lookback_days, steps_per_session)


def load_product_master(path: str) -> dict:
    """Load product master CSV into {sku_name: product_dict}."""
    products = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products[row["sku_name"]] = row
    return products


def main():
    parser = argparse.ArgumentParser(
        description="Nightly batch training for markdown pricing agents",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Training date (YYYY-MM-DD), uses sessions up to this date",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS,
        help=f"Days of history to include (default: {LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--sku",
        default=None,
        help="Train a single SKU (default: all SKUs in product master)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--steps-per-session",
        type=int,
        default=TRAINING_STEPS_PER_SESSION,
        help=f"Gradient steps per session (default: {TRAINING_STEPS_PER_SESSION})",
    )
    parser.add_argument(
        "--base-dir",
        default="deployment",
        help="Base directory for data/models/logs (default: deployment)",
    )
    args = parser.parse_args()

    config = ProductionConfig(base_dir=args.base_dir)
    config.ensure_dirs()

    # Set up logging
    log_path = config.batch_log_path(args.date)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("Batch training started: date=%s, lookback=%d, workers=%d",
                args.date, args.lookback_days, args.workers)

    # Load product master
    if not os.path.exists(config.product_master_path):
        logger.error("Product master not found: %s", config.product_master_path)
        sys.exit(1)

    products = load_product_master(config.product_master_path)
    logger.info("Loaded %d products from %s", len(products), config.product_master_path)

    # Filter to single SKU if specified
    if args.sku:
        if args.sku not in products:
            logger.error("SKU '%s' not found in product master", args.sku)
            sys.exit(1)
        sku_list = [args.sku]
    else:
        sku_list = sorted(products.keys())

    # Train
    results = []
    if args.workers <= 1:
        # Sequential
        for sku_name in sku_list:
            result = train_single_sku(
                sku_name, config, products[sku_name],
                args.date, args.lookback_days, args.steps_per_session,
            )
            results.append(result)
    else:
        # Parallel
        work_items = [
            (sku, config.base_dir, products[sku], args.date,
             args.lookback_days, args.steps_per_session)
            for sku in sku_list
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_train_worker, work_items))

    # Write report
    trained = [r for r in results if r["n_transitions"] > 0]
    skipped = [r for r in results if r["n_transitions"] == 0]

    report = {
        "date": args.date,
        "lookback_days": args.lookback_days,
        "total_skus": len(sku_list),
        "trained": len(trained),
        "skipped": len(skipped),
        "total_sessions": sum(r["n_sessions"] for r in results),
        "total_transitions": sum(r["n_transitions"] for r in results),
        "total_training_steps": sum(r["training_steps"] for r in results),
        "results": results,
    }

    report_path = config.batch_report_path(args.date)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Batch training complete: %d/%d trained, %d skipped, report: %s",
        len(trained), len(sku_list), len(skipped), report_path,
    )


if __name__ == "__main__":
    main()
