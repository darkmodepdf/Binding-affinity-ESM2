"""
Perceiver Latent Interaction module for antibody-antigen binding.

Uses a bounded set of trainable latent tokens (e.g. 64) to cross-attend 
into the lengthy sequence representations. This compresses a massive O(N^2)
interaction bottleneck into O(K * N), making training and inference lightning fast.
"""

import torch
import torch.nn as nn
from typing import Optional


class PerceiverInteractionLayer(nn.Module):
    """
    A single interaction step.
    The Latent Array queries the Paratope and the Epitope, mixes the 
    information via a multi-layer FFN, and returns the updated Latents.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Latent queries Antibody
        self.cross_attn_ab = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ab = nn.LayerNorm(d_model)

        # Latent queries Antigen
        self.cross_attn_ag = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ag = nn.LayerNorm(d_model)

        # Latent Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_self1 = nn.LayerNorm(d_model)
        self.norm_self2 = nn.LayerNorm(d_model)

        # Mixing FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        ab_emb: torch.Tensor,
        ag_emb: torch.Tensor,
        ab_key_mask: Optional[torch.Tensor] = None,
        ag_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: (B, K, D)
            ab_emb: (B, L_ab, D)
            ag_emb: (B, L_ag, D)
        """
        # 1. Latent attends to Antibody
        ab_cross, _ = self.cross_attn_ab(query=latents, key=ab_emb, value=ab_emb, key_padding_mask=ab_key_mask)
        latents = self.norm_ab(latents + ab_cross)

        # 2. Latent attends to Antigen
        ag_cross, _ = self.cross_attn_ag(query=latents, key=ag_emb, value=ag_emb, key_padding_mask=ag_key_mask)
        latents = self.norm_ag(latents + ag_cross)

        # 3. Latent Self-Attention (Mixing)
        self_mixed, _ = self.self_attn(query=latents, key=latents, value=latents)
        latents = self.norm_self1(latents + self_mixed)

        # 4. FFN
        latents = self.norm_self2(latents + self.ffn(latents))

        return latents


class PerceiverInteractionStack(nn.Module):
    """
    Maintains the Latent token array and flows it through repeated layers.
    """

    def __init__(
        self,
        num_latents: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # The learnable bottleneck tokens
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, d_model))
        nn.init.xavier_normal_(self.latent_tokens)
        
        self.layers = nn.ModuleList([
            PerceiverInteractionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        ab_emb: torch.Tensor,
        ag_emb: torch.Tensor,
        ab_mask: Optional[torch.Tensor] = None,
        ag_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            ab_emb: (B, L_ab, D)
            ag_emb: (B, L_ag, D)
            ab_mask: (B, L_ab) 1=valid, 0=ignore
            ag_mask: (B, L_ag) 1=valid, 0=ignore
            
        Returns:
            latent_out: (B, K, D) The final distilled representation of the interaction.
        """
        batch_size = ab_emb.size(0)
        
        # Broadcast learnable latents to batch size
        latents = self.latent_tokens.expand(batch_size, -1, -1)
        
        # Convert valid masks (1=valid) to key_padding masks (True=ignore)
        ab_key_mask = ~ab_mask.bool() if ab_mask is not None else None
        ag_key_mask = ~ag_mask.bool() if ag_mask is not None else None

        # Flow through stack
        for layer in self.layers:
            latents = layer(
                latents,
                ab_emb,
                ag_emb,
                ab_key_mask=ab_key_mask,
                ag_key_mask=ag_key_mask,
            )
            
        return latents
