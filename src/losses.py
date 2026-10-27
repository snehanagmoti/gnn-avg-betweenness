"""
Step 3a: loss functions.

classification_loss       : cross-entropy (paper eq. 3); also works for 3+ classes
triplet_margin_ranking_loss: paper eq. 4, tied pairs ignored
pairwise_loss             : RankNet - logistic loss on random pairs with different true values
regression_loss           : MSE between the score and the per-graph standardized reduction
listnet_loss              : cross-entropy between softmax(scores) and softmax(standardized reduction)
"""
import torch
import torch.nn.functional as F


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def triplet_margin_ranking_loss(scores, targets, triplets, margin=1.0):
    """Paper eq. 4: for a triplet of edges (m, n, o)
        Loss = max(0, -I_mn (s_m - s_n) + M, -I_no (s_n - s_o) + M, -I_om (s_o - s_m) + M)
    I_ab = sign(true(a) - true(b)); ties (I = 0) are ignored."""
    m, n, o = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    terms = []
    for a, b in ((m, n), (n, o), (o, m)):
        I = torch.sign(targets[a] - targets[b])
        term = F.relu(-I * (scores[a] - scores[b]) + margin)
        terms.append(term * (I != 0))
    return torch.stack(terms, dim=1).max(dim=1).values.mean()


def pairwise_loss(scores, targets, pairs):
    """RankNet: -log sigmoid(s_better - s_worse) over pairs with different true values."""
    a, b = pairs[:, 0], pairs[:, 1]
    I = torch.sign(targets[a] - targets[b])
    keep = I != 0
    if keep.sum() == 0:
        return scores.sum() * 0.0
    return F.softplus(-I[keep] * (scores[a] - scores[b])[keep]).mean()


def regression_loss(scores, z_targets):
    return F.mse_loss(scores, z_targets)


def listnet_loss(scores, z_targets, temperature=1.0):
    p_true = torch.softmax(z_targets / temperature, dim=0)
    return -(p_true * torch.log_softmax(scores, dim=0)).sum()
