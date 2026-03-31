"""
Multi-task loss computation.

Handles:
- Per-affinity-type MSE loss (only active head contributes)
- Adversarial antigen family classification loss (gradient reversal)
- Combined weighted loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.config import AFFINITY_TYPES


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining per-type regression losses
    and optional adversarial family classification loss.
    """

    def __init__(
        self,
        affinity_types: list = AFFINITY_TYPES,
        grl_weight: float = 0.1,
    ):
        super().__init__()
        self.affinity_types = affinity_types
        self.grl_weight = grl_weight
        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        affinity_type_idx: torch.Tensor,
        family_logits: Optional[torch.Tensor] = None,
        family_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: (B,) predicted values
            targets: (B,) target values
            affinity_type_idx: (B,) affinity type index per sample
            family_logits: (B, num_families) adversarial classifier output
            family_labels: (B,) true antigen family labels

        Returns:
            Dict with 'total_loss', per-type losses, and 'grl_loss'.
        """
        result = {}

        # ── Per-type regression losses ──
        per_sample_loss = self.mse(predictions, targets)  # (B,)

        total_regression_loss = per_sample_loss.mean()
        result["regression_loss"] = total_regression_loss

        # Per-type breakdowns (for logging)
        for i, atype in enumerate(self.affinity_types):
            mask = affinity_type_idx == i
            if mask.any():
                type_loss = per_sample_loss[mask].mean()
                key = f"loss_{atype.replace(' ', '_').replace('-', '_')}"
                result[key] = type_loss

        # ── Adversarial family classification loss ──
        total_loss = total_regression_loss

        if family_logits is not None and family_labels is not None:
            grl_loss = self.ce(family_logits, family_labels)
            result["grl_loss"] = grl_loss
            total_loss = total_loss + self.grl_weight * grl_loss

        result["total_loss"] = total_loss
        return result
