"""
Cross-attention module for modeling antibody-antigen interactions.

Implements bidirectional cross-attention between antibody (heavy+light)
token embeddings and antigen token embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttentionLayer(nn.Module):
    """
    Single bidirectional cross-attention layer.

    Given antibody embeddings Q_ab and antigen embeddings Q_ag:
    1. Ab attends to Ag: CrossAttn(Q=ab, KV=ag) → ab'
    2. Ag attends to Ab: CrossAttn(Q=ag, KV=ab) → ag'
    3. FFN on both with residual connections and LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Ab → Ag cross-attention
        self.cross_attn_ab = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ab1 = nn.LayerNorm(d_model)

        # Ag → Ab cross-attention
        self.cross_attn_ag = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ag1 = nn.LayerNorm(d_model)

        # FFN for antibody
        self.ffn_ab = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ab2 = nn.LayerNorm(d_model)

        # FFN for antigen
        self.ffn_ag = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ag2 = nn.LayerNorm(d_model)

    def forward(
        self,
        ab_emb: torch.Tensor,
        ag_emb: torch.Tensor,
        ab_mask: Optional[torch.Tensor] = None,
        ag_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            ab_emb: (B, L_ab, D) antibody embeddings
            ag_emb: (B, L_ag, D) antigen embeddings
            ab_mask: (B, L_ab) attention mask for antibody (1=attend, 0=ignore)
            ag_mask: (B, L_ag) attention mask for antigen

        Returns:
            ab_out: (B, L_ab, D) updated antibody embeddings
            ag_out: (B, L_ag, D) updated antigen embeddings
        """
        # Convert masks to key_padding_mask format: True = ignore
        ab_key_mask = ~ab_mask.bool() if ab_mask is not None else None
        ag_key_mask = ~ag_mask.bool() if ag_mask is not None else None

        # Antibody attends to antigen
        ab_cross, _ = self.cross_attn_ab(
            query=ab_emb,
            key=ag_emb,
            value=ag_emb,
            key_padding_mask=ag_key_mask,
        )
        ab_emb = self.norm_ab1(ab_emb + ab_cross)
        ab_emb = self.norm_ab2(ab_emb + self.ffn_ab(ab_emb))

        # Antigen attends to antibody
        ag_cross, _ = self.cross_attn_ag(
            query=ag_emb,
            key=ab_emb,
            value=ab_emb,
            key_padding_mask=ab_key_mask,
        )
        ag_emb = self.norm_ag1(ag_emb + ag_cross)
        ag_emb = self.norm_ag2(ag_emb + self.ffn_ag(ag_emb))

        return ab_emb, ag_emb


class CrossAttentionStack(nn.Module):
    """Stack of bidirectional cross-attention layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        ab_emb: torch.Tensor,
        ag_emb: torch.Tensor,
        ab_mask: Optional[torch.Tensor] = None,
        ag_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Apply all cross-attention layers sequentially.

        Returns:
            ab_emb: (B, L_ab, D) final antibody embeddings
            ag_emb: (B, L_ag, D) final antigen embeddings
        """
        for layer in self.layers:
            ab_emb, ag_emb = layer(ab_emb, ag_emb, ab_mask, ag_mask)
        return ab_emb, ag_emb
