"""
Evaluation entry point.

Loads a trained model and evaluates on test set or LOAFO folds.
Saves metrics to JSON and generates plots.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AFFINITY_TYPES,
    DataConfig,
    EvalConfig,
    ModelConfig,
    OUTPUT_DIR,
    LOGS_DIR,
)
from src.data.dataset import AffinityDataset
from src.model.model import AffinityModel
from src.model.encoder import load_esm2_tokenizer
from src.training.metrics import compute_all_metrics, calibration_curve

import pandas as pd



try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, model_config: ModelConfig) -> AffinityModel:
    """Load model from checkpoint."""
    model = AffinityModel(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded model from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )
    return model


@torch.no_grad()
def predict(
    model: AffinityModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference and return predictions, targets, type indices."""
    model.eval()
    all_preds, all_targets, all_types = [], [], []

    for batch in tqdm(loader, desc="Predicting"):
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                heavy_input_ids=batch["heavy_input_ids"],
                heavy_attention_mask=batch["heavy_attention_mask"],
                light_input_ids=batch["light_input_ids"],
                light_attention_mask=batch["light_attention_mask"],
                antigen_input_ids=batch["antigen_input_ids"],
                antigen_attention_mask=batch["antigen_attention_mask"],
                affinity_type_idx=batch["affinity_type_idx"],
            )

        all_preds.append(outputs["predictions"].float().cpu().numpy())
        all_targets.append(batch["target_value"].float().cpu().numpy())
        all_types.append(batch["affinity_type_idx"].cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_types),
    )


def denormalize_predictions(
    preds: np.ndarray, targets: np.ndarray, types: np.ndarray, norm_stats: Optional[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """Denormalize predictions and targets back to original units."""
    if not norm_stats:
        return preds, targets
    
    for i, atype in enumerate(AFFINITY_TYPES):
        mask = types == i
        if mask.any() and atype in norm_stats:
            mean = norm_stats[atype].get("mean", 0.0)
            std = norm_stats[atype].get("std", 1.0)
            std = std if std > 1e-6 else 1.0
            preds[mask] = preds[mask] * std + mean
            targets[mask] = targets[mask] * std + mean
            
    return preds, targets


def save_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    type_idx: np.ndarray,
    plot_dir: Path,
    prefix: str = "",
):
    """Generate and save evaluation plots."""
    if not HAS_PLOT:
        logger.warning("matplotlib/seaborn not available — skipping plots")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Overall scatter plot ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hexbin(y_true, y_pred, gridsize=50, cmap="viridis", mincnt=1)
    ax.set_xlabel("True Affinity")
    ax.set_ylabel("Predicted Affinity")
    ax.set_title(f"{prefix}Predicted vs True Affinity")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect")
    ax.legend()
    plt.colorbar(ax.collections[0], ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}scatter_overall.png", dpi=150)
    plt.close(fig)

    # ── Per-type scatter plots ──
    for i, atype in enumerate(AFFINITY_TYPES):
        mask = type_idx == i
        if mask.sum() < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.3, s=10)
        ax.set_xlabel("True Affinity")
        ax.set_ylabel("Predicted Affinity")
        ax.set_title(f"{prefix}{atype} (n={mask.sum()})")
        lims = [
            min(y_true[mask].min(), y_pred[mask].min()),
            max(y_true[mask].max(), y_pred[mask].max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.7)
        fig.tight_layout()
        safe_name = atype.replace(" ", "_").replace("-", "_")
        fig.savefig(plot_dir / f"{prefix}scatter_{safe_name}.png", dpi=150)
        plt.close(fig)

    # ── Residual histogram ──
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=100, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (Predicted - True)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix}Residual Distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}residuals.png", dpi=150)
    plt.close(fig)

    # ── Calibration curve ──
    bins, mean_true, mean_pred = calibration_curve(y_true, y_pred, n_bins=20)
    valid = ~np.isnan(mean_true)
    if valid.sum() > 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(mean_pred[valid], mean_true[valid], s=50, zorder=3)
        ax.plot(
            [mean_pred[valid].min(), mean_pred[valid].max()],
            [mean_pred[valid].min(), mean_pred[valid].max()],
            "r--",
            alpha=0.7,
        )
        ax.set_xlabel("Mean Predicted (binned)")
        ax.set_ylabel("Mean True (binned)")
        ax.set_title(f"{prefix}Calibration Curve")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{prefix}calibration.png", dpi=150)
        plt.close(fig)

    logger.info(f"Saved plots to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate binding affinity model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=str(OUTPUT_DIR / "preprocessed"),
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--mode",
        choices=["test", "loafo", "both"],
        default="both",
        help="Evaluation mode",
    )

    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()
    eval_config = EvalConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load family info
    preprocessed_dir = Path(args.preprocessed_dir)
    families_path = preprocessed_dir / "antigen_families.json"
    if families_path.exists():
        with open(families_path) as f:
            model_config.num_antigen_families = len(json.load(f))

    # Load model
    model = load_model(args.checkpoint, model_config)
    model.to(device)

    # Tokenizer
    tokenizer = load_esm2_tokenizer(model_config.esm_model_name)

    # Load norm stats
    norm_stats = None
    stats_path = preprocessed_dir / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            norm_stats = json.load(f)

    plot_dir = Path(eval_config.plot_dir)

    all_results = {}

    # ── Test set evaluation ──
    if args.mode in ("test", "both"):
        logger.info("=" * 60)
        logger.info("Evaluating on test set")
        logger.info("=" * 60)

        test_df = pd.read_csv(preprocessed_dir / "test.csv")
        test_dataset = AffinityDataset(
            test_df, tokenizer, data_config, is_training=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=data_config.num_workers,
        )

        preds, targets, types = predict(model, test_loader, device)
        preds, targets = denormalize_predictions(preds, targets, types, norm_stats)
        
        metrics = compute_all_metrics(
            y_true=targets,
            y_pred=preds,
            affinity_type_idx=types,
            affinity_type_names=AFFINITY_TYPES,
            ndcg_k_values=eval_config.ndcg_k_values,
            precision_k_percentiles=eval_config.precision_k_percentiles,
            enrichment_thresholds=eval_config.enrichment_thresholds,
        )

        logger.info("Test set metrics:")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        all_results["test"] = metrics

        if eval_config.save_plots:
            save_plots(targets, preds, types, plot_dir / "test", prefix="test_")



    # ── LOAFO evaluation ──
    if args.mode in ("loafo", "both"):
        logger.info("=" * 60)
        logger.info("Evaluating LOAFO folds")
        logger.info("=" * 60)

        loafo_info_path = preprocessed_dir / "loafo_info.json"
        if not loafo_info_path.exists():
            logger.warning("No LOAFO folds found — skipping")
        else:
            with open(loafo_info_path) as f:
                loafo_info = json.load(f)

            loafo_results = {}
            for fold_name, fold_info in loafo_info.items():
                fold_dir = preprocessed_dir / fold_name.replace("fold", "loafo_fold")
                test_path = fold_dir / "test.csv"
                if not test_path.exists():
                    continue

                test_df = pd.read_csv(test_path)
                if len(test_df) < 2:
                    continue

                logger.info(f"  {fold_name}: {len(test_df)} test samples")
                test_dataset = AffinityDataset(
                    test_df, tokenizer, data_config, is_training=False
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=data_config.num_workers,
                )

                preds, targets, types = predict(model, test_loader, device)
                preds, targets = denormalize_predictions(preds, targets, types, norm_stats)
                
                metrics = compute_all_metrics(
                    y_true=targets,
                    y_pred=preds,
                    affinity_type_idx=types,
                    affinity_type_names=AFFINITY_TYPES,
                )

                loafo_results[fold_name] = metrics

                pearson = metrics.get("overall/pearson_r", float("nan"))
                spearman = metrics.get("overall/spearman_rho", float("nan"))
                logger.info(
                    f"    Pearson: {pearson:.4f}, Spearman: {spearman:.4f}"
                )

                if eval_config.save_plots:
                    save_plots(
                        targets, preds, types,
                        plot_dir / "loafo" / fold_name,
                        prefix=f"{fold_name}_",
                    )

            all_results["loafo"] = loafo_results

            # ── Print LOAFO Summary Table ──
            if loafo_results:
                logger.info("\n" + "=" * 60)
                logger.info("LOAFO Cross-Validation Summary")
                logger.info("=" * 60)
                logger.info(f"{'Fold':<10} | {'Family':<15} | {'N Test':<8} | {'Pearson':<8} | {'Spearman':<8} | {'RMSE':<8}")
                logger.info("-" * 69)
                
                table_data = []
                for idx, (fold, metrics) in enumerate(loafo_results.items()):
                    n_test = metrics.get('overall/count', 0)
                    pearson = metrics.get('overall/pearson_r', float("nan"))
                    spearman = metrics.get('overall/spearman_rho', float("nan"))
                    rmse = metrics.get('overall/rmse', float("nan"))
                    
                    # Try to get family name from info if available
                    family_name = f"Fam_?"
                    if 'loafo_info' in locals() and fold in loafo_info:
                        families = loafo_info[fold].get("test_families", [])
                        if families:
                            family_name = f"Fam_{families[0]}"
                    
                    logger.info(f"{fold:<10} | {family_name:<15} | {n_test:<8} | {pearson:<8.4f} | {spearman:<8.4f} | {rmse:<8.4f}")
                    table_data.append([fold, family_name, n_test, pearson, spearman, rmse])
                
                logger.info("=" * 69)
                
                # Save table
                out_csv = LOGS_DIR / "loafo_results_table.csv"
                df_table = pd.DataFrame(table_data, columns=["Fold", "Family", "N_Test", "Pearson", "Spearman", "RMSE"])
                df_table.to_csv(out_csv, index=False)
                logger.info(f"Saved LOAFO table to {out_csv}")



    # Save results
    results_path = LOGS_DIR / "evaluation_results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"Saved results to {results_path}")




if __name__ == "__main__":
    main()
