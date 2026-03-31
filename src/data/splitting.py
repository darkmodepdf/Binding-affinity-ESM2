"""
Data splitting strategies.

Provides:
1. Leave-One-Antigen-Family-Out (LOAFO) cross-validation splits
2. Stratified random splits within families for in-distribution evaluation
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import DataConfig

logger = logging.getLogger(__name__)


def create_loafo_splits(
    df: pd.DataFrame,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create Leave-One-Antigen-Family-Out cross-validation splits.

    Each fold holds out one antigen family for testing and trains
    on all remaining families.

    Args:
        df: DataFrame with 'antigen_family' column.

    Returns:
        List of (train_df, test_df) tuples, one per family.
    """
    families = sorted(df["antigen_family"].unique())
    splits = []

    for held_out_family in families:
        test_mask = df["antigen_family"] == held_out_family
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        if len(test_df) == 0:
            logger.warning(f"Family {held_out_family} has 0 test samples — skipping")
            continue

        logger.info(
            f"LOAFO fold {held_out_family}: "
            f"train={len(train_df)}, test={len(test_df)} "
            f"(test antigens: {test_df['antigen_sequence'].nunique()})"
        )
        splits.append((train_df, test_df))

    return splits


def create_stratified_split(
    df: pd.DataFrame,
    config: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test split.

    Stratification is based on (antigen_family, affinity_type) to ensure
    each split has proportional representation of all families and
    measurement types.

    Args:
        df: DataFrame with 'antigen_family' and 'affinity_type' columns.
        config: DataConfig with val_ratio, test_ratio, random_seed.

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    # Create stratification key
    df = df.copy()
    df["_strat_key"] = (
        df["antigen_family"].astype(str) + "||" + df["affinity_type"]
    )

    # Filter out groups with too few samples for splitting
    group_counts = df["_strat_key"].value_counts()
    small_groups = group_counts[group_counts < 3].index
    if len(small_groups) > 0:
        # Merge small groups into a catch-all
        df.loc[df["_strat_key"].isin(small_groups), "_strat_key"] = "__rare__"
        logger.info(
            f"Merged {len(small_groups)} rare strat groups into '__rare__'"
        )

    test_ratio = config.test_ratio
    val_ratio = config.val_ratio

    # First split: separate test set
    splitter1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=config.random_seed,
    )
    train_val_idx, test_idx = next(splitter1.split(df, df["_strat_key"]))
    test_df = df.iloc[test_idx].copy()
    train_val_df = df.iloc[train_val_idx].copy()

    # Second split: separate validation from training
    val_relative = val_ratio / (1 - test_ratio)
    # Rebuild strat key for remaining data
    train_val_df["_strat_key"] = (
        train_val_df["antigen_family"].astype(str)
        + "||"
        + train_val_df["affinity_type"]
    )
    group_counts2 = train_val_df["_strat_key"].value_counts()
    small_groups2 = group_counts2[group_counts2 < 3].index
    if len(small_groups2) > 0:
        train_val_df.loc[
            train_val_df["_strat_key"].isin(small_groups2), "_strat_key"
        ] = "__rare__"

    splitter2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_relative,
        random_state=config.random_seed,
    )
    train_idx, val_idx = next(
        splitter2.split(train_val_df, train_val_df["_strat_key"])
    )
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Clean up helper column
    for d in [train_df, val_df, test_df]:
        d.drop(columns=["_strat_key"], inplace=True, errors="ignore")

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        f"Stratified split: train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df


def create_splits(
    df: pd.DataFrame,
    config: DataConfig,
) -> dict:
    """
    Create all required data splits.

    Returns:
        Dictionary with keys:
            'stratified': (train_df, val_df, test_df)
            'loafo': list of (train_df, test_df) per family
    """
    stratified = create_stratified_split(df, config)
    loafo = create_loafo_splits(df)

    return {
        "stratified": stratified,
        "loafo": loafo,
    }
