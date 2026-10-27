import torch
import torch.nn.functional as F


def compute_classification_loss(logits, targets, num_classes=1):
    """
    Computes BCE loss for binary classification or CE loss for multi-class.
    """
    if num_classes == 1:
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    else:
        return F.cross_entropy(logits, targets.long())


def compute_triplet_loss(pred_scores, triplets, margin=0.5):
    """
    Computes Triplet Margin Ranking Loss:
    Loss = mean( ReLU(-I_mn*(s_m - s_n) + M) + ReLU(-I_no*(s_n - s_o) + M) + ReLU(-I_om*(s_o - s_m) + M) ) / 3
    """
    m = triplets["m"]
    n = triplets["n"]
    o = triplets["o"]
    I_mn = triplets["I_mn"]
    I_no = triplets["I_no"]
    I_om = triplets["I_om"]
    
    s_m = pred_scores[m]
    s_n = pred_scores[n]
    s_o = pred_scores[o]
    
    term1 = F.relu(-I_mn * (s_m - s_n) + margin)
    term2 = F.relu(-I_no * (s_n - s_o) + margin)
    term3 = F.relu(-I_om * (s_o - s_m) + margin)
    
    return (term1 + term2 + term3).mean()


def compute_ranking_loss(pred_scores, target_norm_scores, triplets=None, margin=0.5, mse_weight=1.0):
    """
    Combined ranking loss: Triplet Margin Loss + Target-standardized MSE regression loss.
    Standardized MSE stabilizes gradients when target differences are small,
    while Triplet loss sharpens relative ordering.
    """
    loss_mse = F.mse_loss(pred_scores, target_norm_scores)
    if triplets is not None:
        loss_triplet = compute_triplet_loss(pred_scores, triplets, margin=margin)
        return loss_triplet + mse_weight * loss_mse, loss_triplet, loss_mse
    return mse_weight * loss_mse, torch.tensor(0.0, device=pred_scores.device), loss_mse

