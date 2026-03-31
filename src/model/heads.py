"""
Multi-task affinity prediction heads.

One MLP head per affinity measurement type. Only the relevant head
fires per sample (determined by affinity_type_idx).
"""

import torch
import torch.nn as nn
from typing import Dict, List

from src.config import AFFINITY_TYPES


class AffinityHead(nn.Module):
    """Single affinity prediction head (2-layer MLP)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B_sub, pool_dim) pooled interaction features for this affinity type.
        Returns:
            pred: (B_sub, 1) predicted affinity values.
        """
        return self.mlp(x)


class MultiTaskHeads(nn.Module):
    """
    Collection of affinity prediction heads, one per measurement type.

    During forward pass, samples are routed to the correct head based
    on their affinity_type_idx.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        affinity_types: List[str] = AFFINITY_TYPES,
    ):
        super().__init__()
        self.affinity_types = affinity_types
        self.heads = nn.ModuleDict({
            self._sanitize_key(at): AffinityHead(input_dim, hidden_dim, dropout)
            for at in affinity_types
        })
        self._key_map = {
            i: self._sanitize_key(at) for i, at in enumerate(affinity_types)
        }

    @staticmethod
    def _sanitize_key(name: str) -> str:
        """ModuleDict keys can't have spaces or special chars."""
        return name.replace(" ", "_").replace("-", "_")

    def forward(
        self,
        features: torch.Tensor,
        affinity_type_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route each sample to its corresponding head.

        Args:
            features: (B, pool_dim) pooled interaction features.
            affinity_type_idx: (B,) integer index of affinity type per sample.

        Returns:
            predictions: (B,) predicted affinity values.
        """
        predictions = torch.zeros(
            features.size(0), device=features.device, dtype=features.dtype
        )

        for type_idx, key in self._key_map.items():
            mask = affinity_type_idx == type_idx
            if mask.any():
                sub_features = features[mask]
                sub_preds = self.heads[key](sub_features).squeeze(-1)
                predictions[mask] = sub_preds.to(predictions.dtype)

        return predictions
