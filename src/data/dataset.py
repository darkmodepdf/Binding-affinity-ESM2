"""
PyTorch Dataset for antibody-antigen binding affinity prediction.

Features:
- Tokenizes sequences using the ESM-2 tokenizer
- Balanced sampling via antigen-family-weighted sampler
- Light chain masking (anti-bias augmentation)
- Antigen subsequence masking (augmentation)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from transformers import AutoTokenizer

from src.config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class AffinityDataset(Dataset):
    """
    PyTorch Dataset that yields tokenized (heavy, light, antigen) triplets
    along with affinity type index and target value.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        config: DataConfig,
        is_training: bool = True,
        norm_stats: Optional[Dict] = None,
    ):
        """
        Args:
            df: Preprocessed DataFrame with required columns.
            tokenizer: ESM-2 tokenizer.
            config: DataConfig instance.
            is_training: If True, apply augmentations (masking).
            norm_stats: Dictionary of per-type mean/std for target z-scoring.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training

        # Pre-compute some things
        self.heavy_seqs = self.df["heavy_sequence"].tolist()
        self.light_seqs = self.df["light_sequence"].tolist()
        self.antigen_seqs = self.df["antigen_sequence"].tolist()
        self.affinity_type_idx = torch.tensor(
            self.df["affinity_type_idx"].values, dtype=torch.long
        )
        self.target_values = torch.tensor(
            self.df["target_value"].values, dtype=torch.float32
        )

        # Apply z-score normalization
        if norm_stats is not None:
            norm_targets = np.zeros_like(self.df["target_value"].values)
            for atype, stats in norm_stats.items():
                mask = self.df["affinity_type"] == atype
                if mask.any():
                    mean = stats.get("mean", 0.0)
                    std = stats.get("std", 1.0)
                    std = std if std > 1e-6 else 1.0  # prevent div by zero
                    norm_targets[mask] = (self.df.loc[mask, "target_value"] - mean) / std
            self.target_values = torch.tensor(norm_targets, dtype=torch.float32)
            logger.info(f"Applied Z-score normalization using stats for {len(norm_stats)} types")

        # Antigen family for gradient reversal
        self.antigen_families = torch.tensor(
            self.df["antigen_family"].values, dtype=torch.long
        )

        # Mask token for augmentation
        self.mask_token = tokenizer.mask_token or "<mask>"
        self.mask_token_id = getattr(tokenizer, "mask_token_id", 32)
        
        # ── Pre-tokenize Unique Sequences for Extreme Speedup ──
        logger.info("Pre-tokenizing unique sequences to bypass Python DataLoader GIL lockups...")
        unique_heavy = df["heavy_sequence"].unique().tolist()
        unique_light = df["light_sequence"].unique().tolist()
        unique_antigen = df["antigen_sequence"].unique().tolist()
        
        from tqdm import tqdm
        self.heavy_cache = self._batch_tokenize(unique_heavy, config.max_heavy_len)
        self.light_cache = self._batch_tokenize(unique_light, config.max_light_len)
        self.antigen_cache = self._batch_tokenize(unique_antigen, config.max_antigen_len)

        logger.info(
            f"Created {'train' if is_training else 'eval'} dataset with "
            f"{len(self)} samples"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _batch_tokenize(
        self, sequences: List[str], max_len: int
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Batch tokenize sequences in chunks using ultra-fast Rust backend."""
        cache = {}
        chunk_size = 10000
        for i in range(0, len(sequences), chunk_size):
            chunk = sequences[i:i + chunk_size]
            spaced_seqs = [" ".join(list(seq.replace(self.mask_token, "<mask>"))) for seq in chunk]

            encoding = self.tokenizer(
                spaced_seqs,
                max_length=max_len + 2,  # +2 for BOS/EOS
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for j, seq in enumerate(chunk):
                cache[seq] = {
                    "input_ids": encoding["input_ids"][j].clone(),
                    "attention_mask": encoding["attention_mask"][j].clone(),
                }
        return cache

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        heavy_seq = self.heavy_seqs[idx]
        light_seq = self.light_seqs[idx]
        antigen_seq = self.antigen_seqs[idx]

        if self.is_training:
            # Heavy is never augmented — no clone needed
            heavy_enc = self.heavy_cache[heavy_seq]
            # Light and antigen are mutated by augmentation — must clone
            light_enc = {k: v.clone() for k, v in self.light_cache[light_seq].items()}
            antigen_enc = {k: v.clone() for k, v in self.antigen_cache[antigen_seq].items()}

            # ── Ultra-fast Tensor Augmentations ──
            # 1. Light chain masking
            if np.random.random() < self.config.light_chain_mask_prob:
                valid_mask = (light_enc["attention_mask"] == 1)
                valid_mask[0] = False # Don't mask BOS
                nz = valid_mask.nonzero()
                if len(nz) > 0:
                    valid_mask[nz[-1].item()] = False # Don't mask EOS
                light_enc["input_ids"][valid_mask] = self.mask_token_id

            # 2. Antigen subsequence masking
            if self.config.antigen_mask_prob > 0:
                valid_mask = (antigen_enc["attention_mask"] == 1)
                valid_mask[0] = False
                nz = valid_mask.nonzero()
                if len(nz) > 0:
                    valid_mask[nz[-1].item()] = False
                
                # Apply random mask probability
                rand_mask = torch.rand(antigen_enc["input_ids"].shape) < self.config.antigen_mask_prob
                final_mask = valid_mask & rand_mask
                antigen_enc["input_ids"][final_mask] = self.mask_token_id
        else:
            # Eval: no augmentation, no mutation — zero-copy from cache
            heavy_enc = self.heavy_cache[heavy_seq]
            light_enc = self.light_cache[light_seq]
            antigen_enc = self.antigen_cache[antigen_seq]

        return {
            "heavy_input_ids": heavy_enc["input_ids"],
            "heavy_attention_mask": heavy_enc["attention_mask"],
            "light_input_ids": light_enc["input_ids"],
            "light_attention_mask": light_enc["attention_mask"],
            "antigen_input_ids": antigen_enc["input_ids"],
            "antigen_attention_mask": antigen_enc["attention_mask"],
            "affinity_type_idx": self.affinity_type_idx[idx],
            "target_value": self.target_values[idx],
            "antigen_family": self.antigen_families[idx],
        }


def create_balanced_sampler(
    df: pd.DataFrame,
    config: DataConfig,
) -> WeightedRandomSampler:
    """
    Create a sampler that balances training across antigen families.

    Weight per sample = 1 / (family_count ^ sampling_power)
    where sampling_power = 0.5 (sqrt dampening) prevents over-weighting
    extremely rare families.
    """
    family_counts = df["antigen_family"].value_counts().to_dict()

    weights = []
    for family_id in df["antigen_family"]:
        count = family_counts[family_id]
        weight = 1.0 / (count ** config.sampling_power)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float64)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    logger.info(
        f"Created balanced sampler: {len(family_counts)} families, "
        f"weight range [{weights.min():.6f}, {weights.max():.6f}]"
    )
    return sampler
