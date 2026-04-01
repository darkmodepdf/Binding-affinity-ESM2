"""
Central configuration for the Binding Affinity Prediction project.
All hyperparameters, paths, and constants in one place.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


# ── Project paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create dirs on import
for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Affinity type definitions ────────────────────────────────────
AFFINITY_TYPES = [
    "-log KD",
    "kd",
    "log_enrichment",
    "ddg",
    "elisa_mut_to_wt_ratio",
    "ic_50",
]

AFFINITY_TYPE_TO_IDX = {at: i for i, at in enumerate(AFFINITY_TYPES)}
NUM_AFFINITY_TYPES = len(AFFINITY_TYPES)

# Affinity types that need log-transformation during preprocessing
LOG_TRANSFORM_TYPES = {"kd", "ic_50"}


@dataclass
class ModelConfig:
    """ESM-2 backbone and architecture hyperparameters."""

    # Backbone (Option D: Fast 150M)
    esm_model_name: str = "facebook/esm2_t30_150M_UR50D"
    esm_hidden_dim: int = 640
    freeze_backbone: bool = False  # we use LoRA instead

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value"]
    )

    # Perceiver Latent Interaction
    num_latent_tokens: int = 64
    interaction_layers: int = 4
    interaction_heads: int = 10  # 640 / 10 = 64 dim per head
    interaction_dropout: float = 0.1

    # Pooling
    pool_dim: int = 512

    # Multi-task heads
    head_hidden_dim: int = 256
    head_dropout: float = 0.2
    num_affinity_types: int = NUM_AFFINITY_TYPES

    # Gradient reversal
    use_gradient_reversal: bool = True
    grl_lambda_max: float = 0.1  # max weight of adversarial loss
    grl_gamma: float = 10.0  # schedule steepness for ramping
    num_antigen_families: int = 8  # set dynamically after clustering

    # Antigen dropout (anti-bias)
    antigen_embedding_dropout: float = 0.3
    antibody_embedding_dropout: float = 0.1


@dataclass
class DataConfig:
    """Data loading and preprocessing parameters."""

    data_file: str = str(DATA_DIR / "asd_regression_ready_hla.csv")

    # Sequence length limits (truncation)
    max_heavy_len: int = 150
    max_light_len: int = 230
    max_antigen_len: int = 600

    # Antigen clustering
    cluster_identity_threshold: float = 0.4  # 40% for family grouping

    # Splitting
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # Balanced sampling
    sampling_power: float = 0.5  # sqrt dampening: weight = 1/count^power

    # Light chain masking probability (anti-bias)
    light_chain_mask_prob: float = 0.5

    # Antigen subsequence masking probability (augmentation)
    antigen_mask_prob: float = 0.15

    # Number of dataloader workers
    # Set to 0 because the entire dataset is pre-tokenized in RAM.
    # PyTorch IPC multiprocessing queues actually SLOW down in-memory fetches.
    num_workers: int = 0


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimization
    backbone_lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 128  # batch_size * grad_accum

    # Schedule
    num_epochs: int = 30
    warmup_ratio: float = 0.05
    scheduler_type: str = "cosine"

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4

    # Precision
    use_bf16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_dir: str = str(LOGS_DIR)

    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: str = str(CHECKPOINT_DIR)

    # Reproducibility
    seed: int = 42

    # A100 optimizations
    use_torch_compile: bool = False  # torch.compile — requires C compiler (Triton/Inductor); enable with --compile flag


@dataclass
class EvalConfig:
    """Evaluation parameters."""

    # Ranking metrics top-k values
    ndcg_k_values: List[int] = field(default_factory=lambda: [10, 50])
    precision_k_percentiles: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10]
    )

    # Enrichment factor thresholds
    enrichment_thresholds: List[float] = field(
        default_factory=lambda: [0.01, 0.05]
    )

    # Calibration bins
    calibration_bins: int = 20

    # Plots
    save_plots: bool = True
    plot_dir: str = str(LOGS_DIR / "plots")
