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

        logger.info(
            f"Created {'train' if is_training else 'eval'} dataset with "
            f"{len(self)} samples"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _mask_sequence(self, seq: str, mask_prob: float) -> str:
        """Randomly mask amino acids with <mask> token."""
        if mask_prob <= 0:
            return seq
        residues = list(seq)
        for i in range(len(residues)):
            if np.random.random() < mask_prob:
                residues[i] = self.mask_token
        return "".join(residues)

    def _tokenize(
        self, sequence: str, max_len: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a single sequence with padding/truncation."""
        # Insert spaces between amino acids for ESM tokenizer
        spaced_seq = " ".join(list(sequence.replace(self.mask_token, "<mask>")))

        encoding = self.tokenizer(
            spaced_seq,
            max_length=max_len + 2,  # +2 for BOS/EOS
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        heavy_seq = self.heavy_seqs[idx]
        light_seq = self.light_seqs[idx]
        antigen_seq = self.antigen_seqs[idx]

        # ── Augmentations (training only) ──
        if self.is_training:
            # Light chain masking: replace entire light chain with mask tokens
            if np.random.random() < self.config.light_chain_mask_prob:
                light_seq = self.mask_token * min(len(light_seq), 10)

            # Antigen subsequence masking
            antigen_seq = self._mask_sequence(
                antigen_seq, self.config.antigen_mask_prob
            )

        # ── Tokenize ──
        heavy_enc = self._tokenize(heavy_seq, self.config.max_heavy_len)
        light_enc = self._tokenize(light_seq, self.config.max_light_len)
        antigen_enc = self._tokenize(antigen_seq, self.config.max_antigen_len)

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
