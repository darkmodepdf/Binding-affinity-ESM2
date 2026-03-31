"""
Comprehensive evaluation metrics for binding affinity prediction.

Provides:
- Regression metrics: Pearson, Spearman, RMSE, MAE, R², CI, MAPE
- Ranking metrics: NDCG@k, Precision@k, Enrichment Factor
- Classification-derived: AUROC, AUPRC (binarized at median)
- Calibration: binned predicted vs actual
- Per-affinity-type and overall aggregation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    ndcg_score,
)

logger = logging.getLogger(__name__)


# ── Core regression metrics ─────────────────────────────────────

def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if len(y_true) < 2:
        return float("nan")
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    if len(y_true) < 2:
        return float("nan")
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance Index (CI): fraction of pairs where predicted ordering
    agrees with true ordering. Vectorized with numpy for ~100x speedup.

    CI = (# concordant pairs) / (# comparable pairs)
    """
    n = len(y_true)
    if n < 2:
        return float("nan")

    # Subsample for very large datasets (O(n²) memory for outer product)
    if n > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=5000, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    # Vectorized pairwise comparison via broadcasting
    diff_true = y_true[:, None] - y_true[None, :]  # (n, n)
    diff_pred = y_pred[:, None] - y_pred[None, :]  # (n, n)

    comparable = diff_true != 0
    concordant = ((diff_true * diff_pred) > 0) & comparable

    total = comparable.sum()
    if total == 0:
        return float("nan")
    return float(concordant.sum() / total)


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """MAPE — only computed where |y_true| > epsilon to avoid division by zero."""
    mask = np.abs(y_true) > 1e-8
    if mask.sum() < 2:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Ranking metrics ─────────────────────────────────────────────

def precision_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, k_pct: float
) -> float:
    """
    Precision at top-k%: fraction of true top-k% binders that appear
    in the predicted top-k%.
    """
    n = len(y_true)
    k = max(1, int(n * k_pct))

    true_top_k = set(np.argsort(y_true)[-k:])
    pred_top_k = set(np.argsort(y_pred)[-k:])

    if len(true_top_k) == 0:
        return float("nan")
    return len(true_top_k & pred_top_k) / len(true_top_k)


def enrichment_factor(
    y_true: np.ndarray, y_pred: np.ndarray, threshold_pct: float
) -> float:
    """
    Enrichment Factor at threshold%: how many times better than random
    the model is at identifying top binders.

    EF = (hits in top threshold%) / (expected hits by random)
    """
    n = len(y_true)
    k = max(1, int(n * threshold_pct))

    # True top binders
    true_threshold = np.percentile(y_true, 100 * (1 - threshold_pct))
    true_positives = y_true >= true_threshold

    # Model's top-k predictions
    pred_top_k_idx = np.argsort(y_pred)[-k:]
    hits = true_positives[pred_top_k_idx].sum()

    expected = true_positives.sum() * threshold_pct
    if expected == 0:
        return float("nan")
    return float(hits / expected)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """NDCG@k using sklearn."""
    if len(y_true) < k or len(y_true) < 2:
        return float("nan")
    try:
        # sklearn expects 2D arrays
        return float(
            ndcg_score(
                y_true.reshape(1, -1),
                y_pred.reshape(1, -1),
                k=k,
            )
        )
    except Exception:
        return float("nan")


# ── Classification-derived metrics ──────────────────────────────

def binarized_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUROC after binarizing true values at median."""
    if len(y_true) < 2:
        return float("nan")
    median = np.median(y_true)
    binary_true = (y_true >= median).astype(int)
    if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
        return float("nan")
    try:
        return float(roc_auc_score(binary_true, y_pred))
    except Exception:
        return float("nan")


def binarized_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUPRC after binarizing true values at median."""
    if len(y_true) < 2:
        return float("nan")
    median = np.median(y_true)
    binary_true = (y_true >= median).astype(int)
    if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
        return float("nan")
    try:
        return float(average_precision_score(binary_true, y_pred))
    except Exception:
        return float("nan")


# ── Calibration ─────────────────────────────────────────────────

def calibration_curve(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve: bin predictions, compute mean true vs mean pred.

    Returns:
        bin_centers, mean_true_per_bin, mean_pred_per_bin
    """
    bin_edges = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_true = np.full(n_bins, np.nan)
    mean_pred = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_pred == bin_edges[i + 1])
        if mask.sum() > 0:
            mean_true[i] = y_true[mask].mean()
            mean_pred[i] = y_pred[mask].mean()

    return bin_centers, mean_true, mean_pred


# ── Aggregate metric computation ────────────────────────────────

def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute all regression metrics for a single group."""
    metrics = {}

    metrics["pearson_r"] = pearson_correlation(y_true, y_pred)
    metrics["spearman_rho"] = spearman_correlation(y_true, y_pred)
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    metrics["concordance_index"] = concordance_index(y_true, y_pred)
    metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)

    return metrics


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ndcg_k_values: List[int] = [10, 50],
    precision_k_percentiles: List[float] = [0.01, 0.05, 0.10],
    enrichment_thresholds: List[float] = [0.01, 0.05],
) -> Dict[str, float]:
    """Compute all ranking and enrichment metrics."""
    metrics = {}

    for k in ndcg_k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k)

    for pct in precision_k_percentiles:
        pct_label = f"{int(pct * 100)}pct"
        metrics[f"precision@top_{pct_label}"] = precision_at_k(y_true, y_pred, pct)

    for thresh in enrichment_thresholds:
        pct_label = f"{int(thresh * 100)}pct"
        metrics[f"enrichment_factor@{pct_label}"] = enrichment_factor(
            y_true, y_pred, thresh
        )

    metrics["auroc_binarized"] = binarized_auroc(y_true, y_pred)
    metrics["auprc_binarized"] = binarized_auprc(y_true, y_pred)

    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    affinity_type_idx: Optional[np.ndarray] = None,
    affinity_type_names: Optional[List[str]] = None,
    ndcg_k_values: List[int] = [10, 50],
    precision_k_percentiles: List[float] = [0.01, 0.05, 0.10],
    enrichment_thresholds: List[float] = [0.01, 0.05],
) -> Dict[str, float]:
    """
    Compute all metrics, both overall and per-affinity-type.

    Returns a flat dictionary suitable for logging.
    """
    all_metrics = {}

    # ── Overall metrics ──
    reg = compute_regression_metrics(y_true, y_pred)
    for k, v in reg.items():
        all_metrics[f"overall/{k}"] = v

    rank = compute_ranking_metrics(
        y_true, y_pred, ndcg_k_values, precision_k_percentiles, enrichment_thresholds
    )
    for k, v in rank.items():
        all_metrics[f"overall/{k}"] = v

    # ── Per-type metrics ──
    if affinity_type_idx is not None and affinity_type_names is not None:
        for i, atype in enumerate(affinity_type_names):
            mask = affinity_type_idx == i
            if mask.sum() < 2:
                continue

            type_key = atype.replace(" ", "_").replace("-", "_")

            type_reg = compute_regression_metrics(y_true[mask], y_pred[mask])
            for k, v in type_reg.items():
                all_metrics[f"per_type/{type_key}/{k}"] = v

            if mask.sum() >= 10:  # need enough samples for ranking metrics
                type_rank = compute_ranking_metrics(
                    y_true[mask],
                    y_pred[mask],
                    ndcg_k_values,
                    precision_k_percentiles,
                    enrichment_thresholds,
                )
                for k, v in type_rank.items():
                    all_metrics[f"per_type/{type_key}/{k}"] = v

    return all_metrics
