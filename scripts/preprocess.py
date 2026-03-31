"""
Data preprocessing pipeline entry point.

Loads the raw CSV, preprocesses, clusters antigens, creates splits,
and saves everything to disk for use by the training script.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DataConfig, OUTPUT_DIR
from src.data.preprocessing import preprocess_data, compute_per_type_stats
from src.data.clustering import cluster_antigens
from src.data.splitting import create_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    config = DataConfig()
    output_dir = OUTPUT_DIR / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Preprocess ──
    logger.info("=" * 60)
    logger.info("Step 1: Preprocessing data")
    logger.info("=" * 60)
    df = preprocess_data(config)

    # ── Step 2: Cluster antigens ──
    logger.info("=" * 60)
    logger.info("Step 2: Clustering antigens into families")
    logger.info("=" * 60)
    df, family_info = cluster_antigens(
        df, identity_threshold=config.cluster_identity_threshold
    )

    # Save family info
    family_info_serializable = {
        str(k): [s[:50] + "..." for s in v] for k, v in family_info.items()
    }
    with open(output_dir / "antigen_families.json", "w") as f:
        json.dump(family_info_serializable, f, indent=2)
    logger.info(f"Saved antigen family info to {output_dir / 'antigen_families.json'}")

    # ── Step 3: Create splits ──
    logger.info("=" * 60)
    logger.info("Step 3: Creating data splits")
    logger.info("=" * 60)
    splits = create_splits(df, config)

    # Save stratified splits
    train_df, val_df, test_df = splits["stratified"]
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    logger.info(
        f"Saved stratified splits: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Save LOAFO splits info
    loafo_info = {}
    for i, (train_fold, test_fold) in enumerate(splits["loafo"]):
        fold_dir = output_dir / f"loafo_fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_fold.to_csv(fold_dir / "train.csv", index=False)
        test_fold.to_csv(fold_dir / "test.csv", index=False)
        loafo_info[f"fold_{i}"] = {
            "train_size": len(train_fold),
            "test_size": len(test_fold),
            "test_families": test_fold["antigen_family"].unique().tolist(),
        }

    with open(output_dir / "loafo_info.json", "w") as f:
        json.dump(loafo_info, f, indent=2)
    logger.info(f"Saved {len(splits['loafo'])} LOAFO folds")

    # ── Step 4: Compute normalization stats ──
    logger.info("=" * 60)
    logger.info("Step 4: Computing normalization statistics")
    logger.info("=" * 60)
    stats = compute_per_type_stats(train_df)
    with open(output_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Normalization stats: {json.dumps(stats, indent=2)}")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1024 / 1024
            logger.info(f"  {p.relative_to(output_dir)} ({size_mb:.1f} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
