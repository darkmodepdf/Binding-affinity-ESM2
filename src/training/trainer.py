"""
Training loop with tqdm progress tracking and JSON metric logging.

Handles:
- Multi-task training with balanced sampling
- bf16 mixed precision on H100
- Gradient accumulation
- Early stopping
- Checkpointing
- Comprehensive metric logging to JSON files
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import AFFINITY_TYPES, EvalConfig, ModelConfig, TrainConfig, LOGS_DIR
from src.model.model import AffinityModel
from src.training.losses import MultiTaskLoss
from src.training.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for the AffinityModel.
    """

    def __init__(
        self,
        model: AffinityModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_config: TrainConfig,
        eval_config: EvalConfig,
        model_config: ModelConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_config = train_config
        self.eval_config = eval_config
        self.model_config = model_config

        self.device = torch.device(train_config.device)
        self.model.to(self.device)

        # ── Optimizer (differential learning rates) ──
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": train_config.backbone_lr},
                {"params": head_params, "lr": train_config.head_lr},
            ],
            weight_decay=train_config.weight_decay,
            betas=(train_config.adam_beta1, train_config.adam_beta2),
            eps=train_config.adam_eps,
        )

        # ── Scheduler (linear warmup → cosine decay) ──
        total_steps = len(train_loader) * train_config.num_epochs
        warmup_steps = int(total_steps * train_config.warmup_ratio)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # ── Loss ──
        self.criterion = MultiTaskLoss(
            grl_weight=model_config.grl_lambda if model_config.use_gradient_reversal else 0.0,
        )

        # ── Early stopping ──
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # ── Training history (JSON logging) ──
        self.history = {"train": [], "val": []}
        self.history_path = LOGS_DIR / "training_history.json"

        self.global_step = 0

    def _save_history(self):
        """Save training history to JSON."""

        def make_serializable(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        with open(self.history_path, "w") as f:
            json.dump(make_serializable(self.history), f, indent=2)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        epoch_losses = []
        all_loss_components = {}

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.train_config.num_epochs}",
            leave=True,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    heavy_input_ids=batch["heavy_input_ids"],
                    heavy_attention_mask=batch["heavy_attention_mask"],
                    light_input_ids=batch["light_input_ids"],
                    light_attention_mask=batch["light_attention_mask"],
                    antigen_input_ids=batch["antigen_input_ids"],
                    antigen_attention_mask=batch["antigen_attention_mask"],
                    affinity_type_idx=batch["affinity_type_idx"],
                    antigen_family=batch.get("antigen_family"),
                )

                losses = self.criterion(
                    predictions=outputs["predictions"],
                    targets=batch["target_value"],
                    affinity_type_idx=batch["affinity_type_idx"],
                    family_logits=outputs.get("family_logits"),
                    family_labels=batch.get("antigen_family"),
                )

                loss = losses["total_loss"] / self.train_config.gradient_accumulation_steps

            # Backward
            loss.backward()

            # Gradient accumulation step
            if (step + 1) % self.train_config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track losses
            epoch_losses.append(losses["total_loss"].item())
            for k, v in losses.items():
                if k not in all_loss_components:
                    all_loss_components[k] = []
                all_loss_components[k].append(v.item())

            # Update tqdm with detailed info
            avg_loss = np.mean(epoch_losses[-50:])  # rolling 50-step average
            progress.set_postfix(
                loss=f"{losses['total_loss'].item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                step=self.global_step,
            )

        # Epoch averages
        avg_losses = {k: np.mean(v) for k, v in all_loss_components.items()}
        avg_losses["epoch"] = epoch

        return avg_losses

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """Run evaluation and compute all metrics."""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_type_idx = []
        all_losses = []

        for batch in tqdm(loader, desc=f"  Evaluating ({prefix})", leave=False, bar_format="{l_bar}{bar:30}{r_bar}"):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    heavy_input_ids=batch["heavy_input_ids"],
                    heavy_attention_mask=batch["heavy_attention_mask"],
                    light_input_ids=batch["light_input_ids"],
                    light_attention_mask=batch["light_attention_mask"],
                    antigen_input_ids=batch["antigen_input_ids"],
                    antigen_attention_mask=batch["antigen_attention_mask"],
                    affinity_type_idx=batch["affinity_type_idx"],
                    antigen_family=batch.get("antigen_family"),
                )

                losses = self.criterion(
                    predictions=outputs["predictions"],
                    targets=batch["target_value"],
                    affinity_type_idx=batch["affinity_type_idx"],
                    family_logits=outputs.get("family_logits"),
                    family_labels=batch.get("antigen_family"),
                )

            all_preds.append(outputs["predictions"].float().cpu().numpy())
            all_targets.append(batch["target_value"].float().cpu().numpy())
            all_type_idx.append(batch["affinity_type_idx"].cpu().numpy())
            all_losses.append(losses["total_loss"].item())

        # Concatenate
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        type_idx = np.concatenate(all_type_idx)

        # Compute metrics
        metrics = compute_all_metrics(
            y_true=targets,
            y_pred=preds,
            affinity_type_idx=type_idx,
            affinity_type_names=AFFINITY_TYPES,
            ndcg_k_values=self.eval_config.ndcg_k_values,
            precision_k_percentiles=self.eval_config.precision_k_percentiles,
            enrichment_thresholds=self.eval_config.enrichment_thresholds,
        )

        metrics[f"{prefix}/loss"] = float(np.mean(all_losses))

        # Prefix all keys
        prefixed = {}
        for k, v in metrics.items():
            if not k.startswith(prefix):
                prefixed[f"{prefix}/{k}"] = v
            else:
                prefixed[k] = v

        return prefixed

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        ckpt_dir = Path(self.train_config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
        }

        # Save periodic checkpoint
        path = ckpt_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        # Save best model
        if is_best:
            best_path = ckpt_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")

    def train(self) -> Dict[str, float]:
        """
        Full training loop with early stopping.

        Returns the best validation metrics.
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        logger.info(f"  Epochs: {self.train_config.num_epochs}")
        logger.info(f"  Batch size: {self.train_config.batch_size}")
        logger.info(f"  Effective batch size: {self.train_config.effective_batch_size}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Training history → {self.history_path}")

        best_metrics = {}

        # Outer epoch progress bar
        epoch_bar = tqdm(
            range(self.train_config.num_epochs),
            desc="Training",
            position=0,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

        for epoch in epoch_bar:
            t0 = time.time()

            # ── Train ──
            train_losses = self.train_epoch(epoch)
            epoch_time = time.time() - t0

            # ── Evaluate ──
            val_metrics = self.evaluate(self.val_loader, prefix="val")
            val_loss = val_metrics["val/loss"]
            pearson = val_metrics.get("val/overall/pearson_r", float("nan"))
            spearman = val_metrics.get("val/overall/spearman_rho", float("nan"))

            # Update epoch bar
            epoch_bar.set_postfix(
                train_loss=f"{train_losses['total_loss']:.4f}",
                val_loss=f"{val_loss:.4f}",
                pearson=f"{pearson:.4f}",
                time=f"{epoch_time:.0f}s",
            )

            logger.info(
                f"Epoch {epoch+1}/{self.train_config.num_epochs} — "
                f"Train: {train_losses['total_loss']:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"Pearson: {pearson:.4f} | "
                f"Spearman: {spearman:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # ── Save history ──
            self.history["train"].append({
                "epoch": epoch + 1,
                "time_seconds": epoch_time,
                **{k: v for k, v in train_losses.items() if k != "epoch"},
            })
            self.history["val"].append({
                "epoch": epoch + 1,
                **val_metrics,
            })
            self._save_history()

            # ── Checkpointing ──
            is_best = val_loss < self.best_val_loss - self.train_config.min_delta
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_metrics = val_metrics
                logger.info(f"  ★ New best model! Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement ({self.patience_counter}/{self.train_config.patience})"
                )

            if (epoch + 1) % self.train_config.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # ── Early stopping ──
            if self.patience_counter >= self.train_config.patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(patience={self.train_config.patience})"
                )
                break

        # ── Final summary ──
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Training history saved to: {self.history_path}")
        logger.info("=" * 60)

        return best_metrics
