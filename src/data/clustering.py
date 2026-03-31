"""
Antigen family clustering using pairwise sequence identity.

Since MMseqs2 may not be available on all systems, this module provides
a pure-Python fallback using Biopython's pairwise alignment to compute
a rough sequence identity matrix, then applies agglomerative clustering.

For production, consider pre-computing clusters with MMseqs2/CD-HIT.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute approximate sequence identity between two sequences.
    Uses k-mer overlap as a fast proxy for full alignment identity.
    """
    k = 3
    if len(seq1) < k or len(seq2) < k:
        return 0.0

    kmers1 = set(seq1[i : i + k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i : i + k] for i in range(len(seq2) - k + 1))

    if not kmers1 or not kmers2:
        return 0.0

    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return intersection / union  # Jaccard similarity of k-mers


def cluster_antigens(
    df: pd.DataFrame,
    identity_threshold: float = 0.4,
) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    """
    Cluster unique antigen sequences into families based on sequence similarity.

    Args:
        df: DataFrame with 'antigen_sequence' column.
        identity_threshold: Similarity threshold for family grouping.
            Lower values → fewer, larger families.

    Returns:
        df: DataFrame with added 'antigen_family' column (int family ID).
        family_info: Dict mapping family_id → list of antigen sequences.
    """
    unique_antigens = df["antigen_sequence"].unique()
    n = len(unique_antigens)
    logger.info(f"Clustering {n} unique antigen sequences")

    if n <= 1:
        df["antigen_family"] = 0
        return df, {0: list(unique_antigens)}

    # ── Compute pairwise distance matrix ──
    # Distance = 1 - identity
    n_pairs = n * (n - 1) // 2
    dist_matrix = np.zeros((n, n))
    with tqdm(total=n_pairs, desc="Computing pairwise distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                identity = _sequence_identity(unique_antigens[i], unique_antigens[j])
                dist = 1.0 - identity
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                pbar.update(1)

    # ── Hierarchical clustering ──
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method="average")

    # Convert threshold: if identity >= threshold → same family
    #   means distance <= (1 - threshold) → same cluster
    distance_cutoff = 1.0 - identity_threshold
    cluster_labels = fcluster(linkage_matrix, t=distance_cutoff, criterion="distance")

    # ── Build mapping ──
    antigen_to_family = {}
    family_info = {}

    for seq, label in zip(unique_antigens, cluster_labels):
        family_id = int(label) - 1  # 0-indexed
        antigen_to_family[seq] = family_id
        if family_id not in family_info:
            family_info[family_id] = []
        family_info[family_id].append(seq)

    df["antigen_family"] = df["antigen_sequence"].map(antigen_to_family)

    # ── Log summary ──
    n_families = len(family_info)
    logger.info(f"Formed {n_families} antigen families")
    for fid, seqs in sorted(family_info.items()):
        member_count = df[df["antigen_family"] == fid].shape[0]
        logger.info(
            f"  Family {fid}: {len(seqs)} sequences, "
            f"{member_count} rows, "
            f"seq lengths: {[len(s) for s in seqs]}"
        )

    return df, family_info
