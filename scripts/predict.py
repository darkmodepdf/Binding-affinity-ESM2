"""
Inference entry point.

Given a CSV with heavy_sequence, light_sequence, antigen_sequence columns,
predicts binding affinity for each row.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DataConfig, ModelConfig, OUTPUT_DIR, AFFINITY_TYPES
from src.data.dataset import AffinityDataset
from src.model.model import AffinityModel
from src.model.encoder import load_esm2_tokenizer


def denormalize_predictions(
    preds: np.ndarray, types: np.ndarray, norm_stats: dict
) -> np.ndarray:
    """Denormalize predictions back to original units."""
    if not norm_stats:
        return preds
    
    for i, atype in enumerate(AFFINITY_TYPES):
        mask = types == i
        if mask.any() and atype in norm_stats:
            mean = norm_stats[atype].get("mean", 0.0)
            std = norm_stats[atype].get("std", 1.0)
            std = std if std > 1e-6 else 1.0
            preds[mask] = preds[mask] * std + mean
            
    return preds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict binding affinity")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV with heavy_sequence, light_sequence, antigen_sequence, affinity_type columns",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load family info for model config
    families_path = Path(OUTPUT_DIR) / "preprocessed" / "antigen_families.json"
    if families_path.exists():
        with open(families_path) as f:
            model_config.num_antigen_families = len(json.load(f))

    # Load model
    model = AffinityModel(model_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint}")

    # Load input data
    df = pd.read_csv(args.input)
    required_cols = ["heavy_sequence", "light_sequence", "antigen_sequence", "affinity_type"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add dummy columns as needed
    from src.config import AFFINITY_TYPE_TO_IDX
    df["affinity_type_idx"] = df["affinity_type"].map(AFFINITY_TYPE_TO_IDX)
    df["target_value"] = 0.0  # dummy
    df["antigen_family"] = 0  # dummy
    df["confidence"] = "high"
    df["dataset"] = "prediction"
    df["target_name"] = "unknown"

    # Tokenizer and dataset
    tokenizer = load_esm2_tokenizer(model_config.esm_model_name)
    
    # Load norm stats
    norm_stats = None
    stats_path = Path(OUTPUT_DIR) / "preprocessed" / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            norm_stats = json.load(f)

    dataset = AffinityDataset(df, tokenizer, data_config, is_training=False, norm_stats=norm_stats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    # Predict
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
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

    predictions = np.concatenate(all_preds)
    types = df["affinity_type_idx"].values
    predictions = denormalize_predictions(predictions, types, norm_stats)
    
    df["predicted_affinity"] = predictions

    # Save
    output_cols = [
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "affinity_type",
        "predicted_affinity",
    ]
    df[output_cols].to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
