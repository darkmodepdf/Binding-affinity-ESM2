"""
Gradient Reversal Layer for domain-adversarial training.

Used to discourage the shared representation from encoding
antigen-family-specific features, promoting generalization
to unseen antigen families.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016)
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    """Gradient reversal: forward pass is identity, backward negates gradients."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps the gradient reversal function as an nn.Module."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


class AntigenFamilyClassifier(nn.Module):
    """
    Adversarial classifier that predicts antigen family from shared features.

    Connected via a Gradient Reversal Layer so that the backbone
    is trained to produce features that are NOT predictive of the
    antigen family, while the classifier itself learns to predict it
    (providing a training signal for the reversal).
    """

    def __init__(
        self,
        input_dim: int,
        num_families: int,
        grl_lambda: float = 0.1,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_families),
        )

    def set_lambda(self, lambda_: float):
        """Update the GRL lambda (can be scheduled during training)."""
        self.grl.set_lambda(lambda_)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) shared interaction features.

        Returns:
            logits: (B, num_families) antigen family classification logits.
        """
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)
