"""
Full model assembly: AffinityModel.

Combines:
  ESM-2 encoder → cross-attention → attentive pooling → multi-task heads
  + optional gradient reversal adversarial branch.
"""

import logging
from typing import Dict, Optional

import torch.nn.functional as F

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.model.encoder import ESM2Encoder
from src.model.cross_attention import PerceiverInteractionStack
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

        # ── Perceiver Interaction Bottleneck ──
        self.cross_attention = PerceiverInteractionStack(
            num_latents=config.num_latent_tokens,
            d_model=config.esm_hidden_dim,
            n_heads=config.interaction_heads,
            n_layers=config.interaction_layers,
            dropout=config.interaction_dropout,
        )

        # ── Latent Pooling ──
        # Since the Perceiver produces K interacting latents, we just mean-pool 
        # them and project into the expected fixed-size head representation.
        self.interaction_proj = nn.Sequential(
            nn.Linear(config.esm_hidden_dim, config.pool_dim),
            nn.LayerNorm(config.pool_dim),
            nn.GELU(),
            nn.Dropout(config.interaction_dropout),
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
        Batches heavy+light into a single ESM-2 call (3→2 encoder passes).

        Returns:
            ab_emb: (B, L_h+L_l, D) concatenated antibody embeddings
            ab_mask: (B, L_h+L_l) concatenated antibody mask
            ag_emb: (B, L_ag, D) antigen embeddings
            ag_mask: (B, L_ag) antigen mask
        """
        # Save original sequence lengths
        h_len = heavy_input_ids.size(1)
        l_len = light_input_ids.size(1)
        max_ab_len = max(h_len, l_len)

        # Save original masks before any padding
        orig_heavy_mask = heavy_attention_mask
        orig_light_mask = light_attention_mask

        # Pad shorter antibody chain to match longer for batched encoding
        if h_len < max_ab_len:
            pad_len = max_ab_len - h_len
            heavy_input_ids = F.pad(heavy_input_ids, (0, pad_len), value=1)  # ESM-2 pad_token_id=1
            heavy_attention_mask = F.pad(heavy_attention_mask, (0, pad_len), value=0)
        elif l_len < max_ab_len:
            pad_len = max_ab_len - l_len
            light_input_ids = F.pad(light_input_ids, (0, pad_len), value=1)
            light_attention_mask = F.pad(light_attention_mask, (0, pad_len), value=0)

        # Single batched ESM-2 pass for both antibody chains (2B, max_ab_len)
        ab_ids = torch.cat([heavy_input_ids, light_input_ids], dim=0)
        ab_masks = torch.cat([heavy_attention_mask, light_attention_mask], dim=0)
        ab_emb_all = self.encoder(ab_ids, ab_masks)
        heavy_emb, light_emb = ab_emb_all.chunk(2, dim=0)

        # Trim back to original lengths
        heavy_emb = heavy_emb[:, :h_len, :]
        light_emb = light_emb[:, :l_len, :]

        # Antigen: separate pass (very different sequence length)
        antigen_emb = self.encoder(antigen_input_ids, antigen_attention_mask)

        # Concatenate heavy + light → antibody with original masks
        ab_emb = torch.cat([heavy_emb, light_emb], dim=1)  # (B, L_h+L_l, D)
        ab_mask = torch.cat([orig_heavy_mask, orig_light_mask], dim=1)  # (B, L_h+L_l)

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

        # ── Perceiver Bottleneck Interaction ──
        # Extracts dense bidirectional context into the K trainable latents
        latents = self.cross_attention(ab_emb, ag_emb, ab_mask, ag_mask)  # (B, K, D)

        # ── Pool and Project ──
        # Latents already fully represent the paratope/epitope union
        latent_pooled = latents.mean(dim=1)  # (B, D)
        interaction = self.interaction_proj(latent_pooled)  # (B, pool_dim)

        # ── Predict ──
        predictions = self.heads(interaction, affinity_type_idx)

        result = {"predictions": predictions}

        # ── Adversarial family classification ──
        if self.family_classifier is not None and antigen_family is not None:
            family_logits = self.family_classifier(interaction)
            result["family_logits"] = family_logits

        return result
