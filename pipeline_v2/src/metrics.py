import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def relevant_set(true_scores, k):
    """
    Returns boolean array indicating which edges belong to the ground-truth top-k.
    Edges tied with the k-th score are also treated as relevant.
    """
    kth_score = np.sort(true_scores)[::-1][k - 1]
    return true_scores >= kth_score


def compute_ap_at_k(pred_scores, true_scores, k=10):
    """
    Computes Average Precision at k (AP@k).
    AP@k = (1/k) * sum_{j=1..k} (Precision@j * Rel(j))
    """
    pred_scores = np.asarray(pred_scores)
    true_scores = np.asarray(true_scores)
    k = min(k, len(pred_scores))
    
    # Relevant ground truth indicator
    is_rel = relevant_set(true_scores, k)
    
    # Model's top-k predicted edge indices (descending)
    pred_order = np.argsort(-pred_scores, kind="stable")[:k]
    
    rel_in_pred = is_rel[pred_order].astype(float)
    cumulative_hits = np.cumsum(rel_in_pred)
    precisions_at_j = cumulative_hits / np.arange(1, k + 1)
    
    return float(np.sum(precisions_at_j * rel_in_pred) / k)


def compute_precision_at_k(pred_scores, true_scores, k=10):
    """
    Computes Precision@k (fraction of model's top-k predictions that are in true top-k).
    """
    pred_scores = np.asarray(pred_scores)
    true_scores = np.asarray(true_scores)
    k = min(k, len(pred_scores))
    
    is_rel = relevant_set(true_scores, k)
    pred_order = np.argsort(-pred_scores, kind="stable")[:k]
    
    return float(np.mean(is_rel[pred_order]))


def compute_threshold_metrics(pred_scores, true_scores):
    """
    Computes threshold metrics as discussed in the meeting notes:
    - Best possible reduction (max true delta)
    - Predicted top-1 edge reduction
    - Reduction ratio: pred_top1_delta / best_possible_delta
    - Within 2.5% threshold: reduction_ratio >= 0.975
    - Within 10.0% threshold: reduction_ratio >= 0.900
    - Top-1 in true top 50
    - Top-1 in true top 10
    """
    pred_order = np.argsort(-pred_scores, kind="stable")
    top_pred_idx = pred_order[0]
    
    best_possible_idx = np.argmax(true_scores)
    best_possible_val = float(true_scores[best_possible_idx])
    pred_top1_val = float(true_scores[top_pred_idx])
    
    ratio = (pred_top1_val / best_possible_val) if best_possible_val > 0 else 0.0
    
    # Ground truth top 50 and top 10 sets
    true_top50_indices = set(np.argsort(-true_scores, kind="stable")[:50])
    true_top10_indices = set(np.argsort(-true_scores, kind="stable")[:10])
    
    return {
        "Pred_Top1_Reduction": pred_top1_val,
        "Best_Possible_Reduction": best_possible_val,
        "Reduction_Ratio": ratio,
        "Within_2.5%_Threshold": 1.0 if ratio >= 0.975 else 0.0,
        "Within_10%_Threshold": 1.0 if ratio >= 0.900 else 0.0,
        "Top1_in_Top50": 1.0 if top_pred_idx in true_top50_indices else 0.0,
        "Top1_in_Top10": 1.0 if top_pred_idx in true_top10_indices else 0.0,
    }


def compute_all_metrics(pred_scores, true_scores, k_values=(1, 5, 10, 20, 50, 100)):
    """
    Computes comprehensive evaluation metrics for a single graph:
    - mAP@k and Precision@k for all specified k
    - Threshold performance (2.5% and 10% reductions)
    - Full rank correlations (Pearson, Spearman, Kendall-Tau)
    - Top-50 rank correlations
    """
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    true_scores = np.asarray(true_scores, dtype=np.float64)
    
    results = {}
    for k in k_values:
        results[f"mAP@{k}"] = compute_ap_at_k(pred_scores, true_scores, k=k)
        results[f"P@{k}"] = compute_precision_at_k(pred_scores, true_scores, k=k)
        
    threshold_metrics = compute_threshold_metrics(pred_scores, true_scores)
    results.update(threshold_metrics)
    
    # Correlations on full set
    if np.std(pred_scores) > 1e-8 and np.std(true_scores) > 1e-8:
        pc, _ = pearsonr(pred_scores, true_scores)
        sc, _ = spearmanr(pred_scores, true_scores)
        kc, _ = kendalltau(pred_scores, true_scores)
    else:
        pc, sc, kc = 0.0, 0.0, 0.0
    results["Pearson_All"] = float(pc)
    results["Spearman_All"] = float(sc)
    results["Kendall_All"] = float(kc)
    
    # Correlations on true top-50
    top50_idx = np.argsort(-true_scores, kind="stable")[:50]
    p_top50 = pred_scores[top50_idx]
    t_top50 = true_scores[top50_idx]
    if np.std(p_top50) > 1e-8 and np.std(t_top50) > 1e-8:
        sc50, _ = spearmanr(p_top50, t_top50)
    else:
        sc50 = 0.0
    results["Spearman_Top50"] = float(sc50)
    
    return results

