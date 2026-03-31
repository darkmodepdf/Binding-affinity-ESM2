"""
Data preprocessing pipeline.

Handles:
- Loading the regression-ready dataset
- Dropping non-numeric affinity rows
- Log-transforming kd and ic_50 to put them on a comparable log scale
- Extracting target names from metadata JSON
- Cleaning sequences
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import AFFINITY_TYPE_TO_IDX, LOG_TRANSFORM_TYPES, DataConfig

logger = logging.getLogger(__name__)


def _extract_target_name(metadata_str: str) -> str:
    """Extract target_name from the JSON metadata column."""
    try:
        meta = json.loads(metadata_str)
        return meta.get("target_name", "unknown")
    except (json.JSONDecodeError, TypeError):
        return "unknown"


def _clean_sequence(seq: str) -> str:
    """Remove whitespace and ensure uppercase amino acid sequence."""
    if not isinstance(seq, str):
        return ""
    return seq.strip().upper()


def preprocess_data(config: DataConfig) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Returns a DataFrame with columns:
        heavy_sequence, light_sequence, antigen_sequence,
        affinity_type, affinity_type_idx, target_value,
        target_name, dataset, confidence
    """
    logger.info(f"Loading data from {config.data_file}")
    df = pd.read_csv(config.data_file, low_memory=False)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Convert processed_measurement to numeric, drop failures ──
    df["target_value"] = pd.to_numeric(df["processed_measurement"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["target_value"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} rows with non-numeric affinity values")

    # ── Log-transform kd and ic_50 ──
    for atype in LOG_TRANSFORM_TYPES:
        mask = df["affinity_type"] == atype
        n_type = mask.sum()
        if n_type > 0:
            vals = df.loc[mask, "target_value"]
            # Clamp to avoid log(0)
            vals = vals.clip(lower=1e-12)
            df.loc[mask, "target_value"] = -np.log10(vals)
            logger.info(
                f"Log-transformed {n_type} '{atype}' values → -log10 scale "
                f"(range: {df.loc[mask, 'target_value'].min():.3f} to "
                f"{df.loc[mask, 'target_value'].max():.3f})"
            )

    # ── Map affinity type to index ──
    df["affinity_type_idx"] = df["affinity_type"].map(AFFINITY_TYPE_TO_IDX)
    unmapped = df["affinity_type_idx"].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows have unknown affinity_type — dropping")
        df = df.dropna(subset=["affinity_type_idx"]).copy()
    df["affinity_type_idx"] = df["affinity_type_idx"].astype(int)

    # ── Clean sequences ──
    tqdm.pandas(desc="Cleaning heavy chains")
    df["heavy_sequence"] = df["heavy_sequence"].progress_apply(_clean_sequence)
    tqdm.pandas(desc="Cleaning light chains")
    df["light_sequence"] = df["light_sequence"].progress_apply(_clean_sequence)
    tqdm.pandas(desc="Cleaning antigen sequences")
    df["antigen_sequence"] = df["antigen_sequence"].progress_apply(_clean_sequence)

    # Drop rows with empty sequences
    for col in ["heavy_sequence", "antigen_sequence"]:
        empty = (df[col] == "").sum()
        if empty > 0:
            logger.warning(f"Dropping {empty} rows with empty {col}")
            df = df[df[col] != ""].copy()

    # ── Extract target name from metadata ──
    tqdm.pandas(desc="Extracting target names")
    df["target_name"] = df["metadata"].progress_apply(_extract_target_name)

    # ── Select and order final columns ──
    output_cols = [
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "affinity_type",
        "affinity_type_idx",
        "target_value",
        "target_name",
        "dataset",
        "confidence",
    ]
    df = df[output_cols].reset_index(drop=True)

    # ── Summary stats ──
    logger.info(f"Preprocessed dataset: {len(df)} rows")
    logger.info(f"Affinity type distribution:\n{df['affinity_type'].value_counts()}")
    logger.info(f"Target value stats:\n{df['target_value'].describe()}")
    logger.info(f"Unique antigens: {df['antigen_sequence'].nunique()}")
    logger.info(f"Unique targets: {df['target_name'].nunique()}")

    return df


def compute_per_type_stats(df: pd.DataFrame) -> dict:
    """
    Compute per-affinity-type normalization statistics (mean, std).
    Used for z-score normalization during training.
    """
    stats = {}
    valid_types = [t for t in df["affinity_type"].unique() if len(df[df["affinity_type"] == t]) > 0]
    num_types = len(valid_types)
    total_count = len(df)

    for atype in valid_types:
        subset = df[df["affinity_type"] == atype]["target_value"]
        count = len(subset)
        stats[atype] = {
            "mean": float(subset.mean()),
            "std": float(subset.std()) if count > 1 else 1.0,
            "min": float(subset.min()),
            "max": float(subset.max()),
            "count": int(count),
            "loss_weight": float(total_count / (num_types * count)) if count > 0 else 1.0
        }
    return stats
