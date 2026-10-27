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


def summary_metrics(pred_scores, true_values):
    """Headline metrics for one graph (TA, 23 Oct + optimality ratio).
    opt_ratio      : reduction of the model's #1 edge / reduction of the true best edge
    best10_ratio   : best true reduction among the model's top 10 / true best
    top50_overlap  : share of the model's top 50 that are in the true top 50 (ties included)
    top1_in_top50  : 1 if the model's #1 edge is in the true top 50
    within_2.5pct / within_10pct : share of the model's top 50 whose reduction is within
                     2.5% / 10% of the best reduction
    in_top2.5pct   : share of the model's top 50 that lie in the top 2.5% of all edges
    """
    order = np.argsort(-pred_scores, kind="stable")
    t = true_values
    best = t.max()
    top50 = order[:50]
    rel50 = relevant_set(t, 50)
    rank = np.argsort(np.argsort(-t, kind="stable"), kind="stable")   # 0 = best
    out = {
        "map1": average_precision_at_k(order, t, 1),
        "map10": average_precision_at_k(order, t, 10),
        "map50": average_precision_at_k(order, t, 50),
        "map100": average_precision_at_k(order, t, 100),
        "p10": precision_at_k(order, t, 10),
        "top50_overlap": float(rel50[top50].mean()),
        "opt_ratio": float(t[order[0]] / best),
        "best10_ratio": float(t[order[:10]].max() / best),
        "top1_in_top50": float(rel50[order[0]]),
        "within_2.5pct": float((t[top50] >= 0.975 * best).mean()),
        "within_10pct": float((t[top50] >= 0.90 * best).mean()),
        "in_top2.5pct": float((rank[top50] < 0.025 * len(t)).mean()),
    }
    c = correlations_at_k(pred_scores, t, 50)
    out["spearman50"] = c["spearman"]
    return out
