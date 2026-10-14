"""
Step 4a: evaluation metrics (paper Section 4.3).
"""
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def relevant_set(true_values, k):
    """Ground-truth top-k edges. Edges tied with the k-th best are also counted as relevant
    (average-betweenness reductions tie often)."""
    kth = np.sort(true_values)[::-1][k - 1]
    return true_values >= kth


def average_precision_at_k(pred_order, true_values, k):
    """Paper: AP@k = (1/k) * sum_{j=1..k} P@j * Rel(j)
    P@j    = fraction of the model's top-j edges that are relevant
    Rel(j) = 1 if the model's j-th edge is in the ground-truth top-k."""
    rel = relevant_set(true_values, k)[pred_order[:k]].astype(float)
    precision_at_j = np.cumsum(rel) / np.arange(1, k + 1)
    return float((precision_at_j * rel).sum() / k)


def precision_at_k(pred_order, true_values, k):
    return float(relevant_set(true_values, k)[pred_order[:k]].mean())


def correlations_at_k(pred_scores, true_values, k):
    """Pearson / Spearman / Kendall between predicted scores and true reductions,
    computed over the ground-truth top-k edges (how well the model orders the best edges)."""
    top = np.argsort(-true_values, kind="stable")[:k]
    p, t = pred_scores[top], true_values[top]
    if np.std(p) == 0 or np.std(t) == 0:
        return {"pearson": np.nan, "spearman": np.nan, "kendall": np.nan}
    return {"pearson": float(pearsonr(p, t)[0]),
            "spearman": float(spearmanr(p, t)[0]),
            "kendall": float(kendalltau(p, t)[0])}


def evaluate_ranking(pred_scores, true_values, k_values):
    """All metrics for one graph. `pred_scores`: higher = better edge."""
    order = np.argsort(-pred_scores, kind="stable")
    out = {}
    for k in k_values:
        out[k] = {"map": average_precision_at_k(order, true_values, k),
                  "precision": precision_at_k(order, true_values, k)}
        if k > 2:
            out[k].update(correlations_at_k(pred_scores, true_values, k))
    return out


def average_over_graphs(per_graph):
    """per_graph: list of evaluate_ranking outputs -> {k: {metric: mean}}  (paper: mAP = mean AP)."""
    ks = per_graph[0].keys()
    return {k: {m: float(np.nanmean([g[k][m] for g in per_graph])) for m in per_graph[0][k]}
            for k in ks}
