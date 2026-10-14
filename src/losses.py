"""
Step 3a: loss functions (paper Section 3.3).
"""
import torch
import torch.nn.functional as F


def classification_loss(logits, labels):
    """Paper eq. 3: binary cross-entropy. With a 2-class softmax output this is cross-entropy."""
    return F.cross_entropy(logits, labels)


def triplet_margin_ranking_loss(scores, targets, triplets, margin=1.0):
    """Paper eq. 4: for a triplet of edges (m, n, o)
        Loss = max(0, -I_mn (s_m - s_n) + M, -I_no (s_n - s_o) + M, -I_om (s_o - s_m) + M)
    where s = predicted score, I_ab = +1 if true(a) > true(b) else -1.

    Change for our problem: average-betweenness reductions often TIE exactly. The paper sets
    I = -1 for ties, which would push tied edges apart arbitrarily; we set I = 0 and ignore
    that pair instead.
    """
    m, n, o = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    terms = []
    for a, b in ((m, n), (n, o), (o, m)):
        I = torch.sign(targets[a] - targets[b])                 # +1, -1 or 0 (tie)
        term = F.relu(-I * (scores[a] - scores[b]) + margin)
        terms.append(term * (I != 0))                           # ignore tied pairs
    return torch.stack(terms, dim=1).max(dim=1).values.mean()
