"""
Training entry point.

Loads preprocessed data, builds model, and trains with tqdm progress tracking.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    TrainConfig,
    OUTPUT_DIR,
)
from src.data.dataset import AffinityDataset, create_balanced_sampler
from src.model.model import AffinityModel
from src.model.encoder import load_esm2_tokenizer
from src.training.trainer import Trainer

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train binding affinity model")
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=str(OUTPUT_DIR / "preprocessed"),
        help="Path to preprocessed data directory",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run quick smoke test with tiny subset"
    )
    args = parser.parse_args()

    # ── Configs ──
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    eval_config = EvalConfig()

    if args.run_name:
        logger.info(f"Run name: {args.run_name}")
    if args.epochs:
        train_config.num_epochs = args.epochs
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.num_workers is not None:
        data_config.num_workers = args.num_workers

    set_seed(train_config.seed)

    # ── Load preprocessed data ──
    preprocessed_dir = Path(args.preprocessed_dir)
    logger.info(f"Loading preprocessed data from {preprocessed_dir}")

    train_df = pd.read_csv(preprocessed_dir / "train.csv")
    val_df = pd.read_csv(preprocessed_dir / "val.csv")

    if args.smoke_test:
        logger.info("SMOKE TEST: using 500 train, 100 val samples")
        train_df = train_df.head(500)
        val_df = val_df.head(100)
        train_config.num_epochs = 2

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Load antigen family info to set num_families
    families_path = preprocessed_dir / "antigen_families.json"
    if families_path.exists():
        with open(families_path) as f:
            family_info = json.load(f)
        model_config.num_antigen_families = len(family_info)
        logger.info(f"Number of antigen families: {model_config.num_antigen_families}")

    # Load normalization stats for z-scoring and loss weighting
    norm_stats = None
    type_weights = None
    stats_path = preprocessed_dir / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            norm_stats = json.load(f)
        type_weights = {k: v.get("loss_weight", 1.0) for k, v in norm_stats.items()}
        logger.info(f"Loaded normalization stats and type weights: {type_weights}")

    # ── Tokenizer ──
    tokenizer = load_esm2_tokenizer(model_config.esm_model_name)

    # ── Datasets ──
    train_dataset = AffinityDataset(
        train_df, tokenizer, data_config, is_training=True, norm_stats=norm_stats
    )
    val_dataset = AffinityDataset(
        val_df, tokenizer, data_config, is_training=False, norm_stats=norm_stats
    )

    # ── Balanced sampler ──
    sampler = create_balanced_sampler(train_df, data_config)

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size * 2,  # larger batch for eval
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )

    # ── Model ──
    model = AffinityModel(model_config)

    # ── Train ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_config,
        eval_config=eval_config,
        model_config=model_config,
        type_weights=type_weights,
        norm_stats=norm_stats,
    )

    best_metrics = trainer.train()
    logger.info("\n" + "=" * 60)
    logger.info("Training complete! Best validation sequence (overall):")
    logger.info("=" * 60)
    for k, v in sorted(best_metrics.items()):
        # Filter to only show core overall metrics, ignore per-head splits in CLI
        if k.startswith("overall/") and isinstance(v, float):
            metric_name = k.replace("overall/", "").replace("_", " ").title()
            logger.info(f"  {metric_name:<20}: {v:.4f}")
    logger.info("=" * 60)
    logger.info(f"Detailed per-head metrics saved to {LOGS_DIR}/training_history.json")
if __name__ == "__main__":
    main()
