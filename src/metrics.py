"""
Ranking metrics (all computed per graph, then averaged over graphs).

    mAP@k   base paper's main metric: are the true top-k edges found, and how early?
    NDCG@k  like mAP@k but each pick earns its true gain, so near-best edges get partial credit
"""
import numpy as np


def relevant_set(true_values, k):
    """Ground-truth top-k edges. Edges tied with the k-th best also count as relevant."""
    kth = np.sort(true_values)[::-1][k - 1]
    return true_values >= kth


def average_precision_at_k(pred_order, true_values, k):
    """AP@k = (1/k) * sum_{j=1..k} P@j * Rel(j)
    P@j    = fraction of the model's top-j edges that are relevant
    Rel(j) = 1 if the model's j-th edge is in the ground-truth top-k."""
    rel = relevant_set(true_values, k)[pred_order[:k]].astype(float)
    precision_at_j = np.cumsum(rel) / np.arange(1, k + 1)
    return float((precision_at_j * rel).sum() / k)


def ndcg_at_k(pred_order, true_values, k):
    """NDCG@k = sum_j gain(j-th pick) / log2(j + 1), divided by the same sum for the perfect order."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (true_values[pred_order[:k]] * disc).sum()
    idcg = (np.sort(true_values)[::-1][:k] * disc).sum()
    return float(dcg / idcg)
