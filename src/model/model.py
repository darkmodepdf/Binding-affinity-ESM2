"""
Full model assembly: AffinityModel.

Combines:
  ESM-2 encoder → cross-attention → attentive pooling → multi-task heads
  + optional gradient reversal adversarial branch.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.model.encoder import ESM2Encoder
from src.model.cross_attention import CrossAttentionStack
from src.model.pooling import AttentivePooling
from src.model.heads import MultiTaskHeads
from src.model.gradient_reversal import AntigenFamilyClassifier

logger = logging.getLogger(__name__)


class AffinityModel(nn.Module):
    """
    End-to-end antibody-antigen binding affinity prediction model.

    Architecture:
        1. Shared ESM-2 encoder (with LoRA) encodes heavy, light, antigen
        2. Heavy + light embeddings concatenated → antibody representation
        3. Bidirectional cross-attention models ab-ag interaction
        4. Attentive pooling → fixed-dim interaction vector
        5. Multi-task heads predict affinity per measurement type
        6. (Optional) Gradient reversal adversarial branch
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── ESM-2 encoder ──
        self.encoder = ESM2Encoder(config)

        # ── Embedding dropout (anti-bias) ──
        self.antibody_dropout = nn.Dropout(config.antibody_embedding_dropout)
        self.antigen_dropout = nn.Dropout(config.antigen_embedding_dropout)

        # ── Cross-attention ──
        self.cross_attention = CrossAttentionStack(
            d_model=config.esm_hidden_dim,
            n_heads=config.cross_attn_heads,
            n_layers=config.cross_attn_layers,
            dropout=config.cross_attn_dropout,
        )

        # ── Attentive pooling (pool both ab and ag, then combine) ──
        self.ab_pooling = AttentivePooling(
            input_dim=config.esm_hidden_dim,
            output_dim=config.pool_dim,
        )
        self.ag_pooling = AttentivePooling(
            input_dim=config.esm_hidden_dim,
            output_dim=config.pool_dim,
        )

        # Combine pooled ab + ag representations
        self.interaction_proj = nn.Sequential(
            nn.Linear(config.pool_dim * 2, config.pool_dim),
            nn.LayerNorm(config.pool_dim),
            nn.GELU(),
            nn.Dropout(config.cross_attn_dropout),
        )

        # ── Multi-task prediction heads ──
        self.heads = MultiTaskHeads(
            input_dim=config.pool_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )

        # ── Gradient reversal (optional) ──
        self.family_classifier = None
        if config.use_gradient_reversal:
            self.family_classifier = AntigenFamilyClassifier(
                input_dim=config.pool_dim,
                num_families=config.num_antigen_families,
                grl_lambda=0.0, # Will be ramped up to config.grl_lambda_max by trainer
            )

        # Count total params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"AffinityModel: {trainable:,} trainable / {total:,} total params"
        )

    def encode_sequences(
        self,
        heavy_input_ids: torch.Tensor,
        heavy_attention_mask: torch.Tensor,
        light_input_ids: torch.Tensor,
        light_attention_mask: torch.Tensor,
        antigen_input_ids: torch.Tensor,
        antigen_attention_mask: torch.Tensor,
    ) -> tuple:
        """
        Encode all sequences through the shared ESM-2 backbone.

        Returns:
            ab_emb: (B, L_h+L_l, D) concatenated antibody embeddings
            ab_mask: (B, L_h+L_l) concatenated antibody mask
            ag_emb: (B, L_ag, D) antigen embeddings
            ag_mask: (B, L_ag) antigen mask
        """
        # Encode each sequence
        heavy_emb = self.encoder(heavy_input_ids, heavy_attention_mask)
        light_emb = self.encoder(light_input_ids, light_attention_mask)
        antigen_emb = self.encoder(antigen_input_ids, antigen_attention_mask)

        # Concatenate heavy + light → antibody
        ab_emb = torch.cat([heavy_emb, light_emb], dim=1)  # (B, L_h+L_l, D)
        ab_mask = torch.cat(
            [heavy_attention_mask, light_attention_mask], dim=1
        )  # (B, L_h+L_l)

        # Apply dropout
        ab_emb = self.antibody_dropout(ab_emb)
        antigen_emb = self.antigen_dropout(antigen_emb)

        return ab_emb, ab_mask, antigen_emb, antigen_attention_mask

    def forward(
        self,
        heavy_input_ids: torch.Tensor,
        heavy_attention_mask: torch.Tensor,
        light_input_ids: torch.Tensor,
        light_attention_mask: torch.Tensor,
        antigen_input_ids: torch.Tensor,
        antigen_attention_mask: torch.Tensor,
        affinity_type_idx: torch.Tensor,
        antigen_family: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with keys:
            'predictions': (B,) predicted affinity values
            'family_logits': (B, num_families) if gradient reversal is active
        """
        # ── Encode ──
        ab_emb, ab_mask, ag_emb, ag_mask = self.encode_sequences(
            heavy_input_ids,
            heavy_attention_mask,
            light_input_ids,
            light_attention_mask,
            antigen_input_ids,
            antigen_attention_mask,
        )

        # ── Cross-attention ──
        ab_emb, ag_emb = self.cross_attention(ab_emb, ag_emb, ab_mask, ag_mask)

        # ── Attentive pooling ──
        ab_pooled = self.ab_pooling(ab_emb, ab_mask)  # (B, pool_dim)
        ag_pooled = self.ag_pooling(ag_emb, ag_mask)  # (B, pool_dim)

        # ── Combine ──
        interaction = torch.cat([ab_pooled, ag_pooled], dim=-1)  # (B, pool_dim*2)
        interaction = self.interaction_proj(interaction)  # (B, pool_dim)

        # ── Predict ──
        predictions = self.heads(interaction, affinity_type_idx)

        result = {"predictions": predictions}

        # ── Adversarial family classification ──
        if self.family_classifier is not None and antigen_family is not None:
            family_logits = self.family_classifier(interaction)
            result["family_logits"] = family_logits

        return result
