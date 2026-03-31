"""
Attentive pooling module.

Replaces mean pooling with a learned attention-weighted aggregation,
which is better suited for binding affinity prediction since binding
interactions are localized to specific residues (paratope/epitope).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentivePooling(nn.Module):
    """
    Attention-based pooling over token embeddings.

    Uses a learnable query vector to attend over the sequence,
    producing a fixed-size interaction representation.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        nn.init.xavier_normal_(self.query)

        self.attn_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, L, D) token embeddings
            mask: (B, L) attention mask (1=attend, 0=ignore)

        Returns:
            pooled: (B, output_dim) pooled representation
        """
        # Compute attention scores
        attn_scores = self.attn_proj(embeddings).squeeze(-1)  # (B, L)

        # Mask padded positions
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.bool(), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, L)
        attn_weights = attn_weights.unsqueeze(-1)  # (B, L, 1)

        # Weighted sum
        pooled = (embeddings * attn_weights).sum(dim=1)  # (B, D)

        # Project to output dim
        pooled = self.output_proj(pooled)  # (B, output_dim)

        return pooled
