"""
ESM-2 backbone encoder with LoRA fine-tuning.

Wraps the ESM-2 650M model with Parameter-Efficient Fine-Tuning (PEFT)
using LoRA adapters on the attention layers.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModel, AutoTokenizer

from src.config import ModelConfig

logger = logging.getLogger(__name__)


class ESM2Encoder(nn.Module):
    """
    ESM-2 protein language model encoder with LoRA adapters.

    Encodes amino acid sequences into per-residue embeddings.
    A single encoder instance is shared for heavy, light, and antigen sequences.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        logger.info(f"Loading ESM-2 model: {config.esm_model_name}")
        self.esm_model = AutoModel.from_pretrained(
            config.esm_model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )

        # Enable gradient checkpointing to save memory
        self.esm_model.gradient_checkpointing_enable()
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        self.esm_model = get_peft_model(self.esm_model, lora_config)

        trainable, total = self._count_params()
        logger.info(
            f"ESM-2 LoRA: {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )

    def _count_params(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of sequences.

        Args:
            input_ids: (B, L) token IDs.
            attention_mask: (B, L) attention mask.

        Returns:
            embeddings: (B, L, D) per-residue embeddings, D=1280 for ESM-2 650M.
        """
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return outputs.last_hidden_state


def load_esm2_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the ESM-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")
    return tokenizer
