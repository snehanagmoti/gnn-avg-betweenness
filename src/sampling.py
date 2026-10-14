"""
Step 3b: which edges and triplets the model trains on (paper Section 4.2).

The paper samples edges with a y = x^4 shape so both extremes (big decrease / big increase of
MAX betweenness) are picked often and the flat middle rarely. For AVERAGE betweenness there is
no "increase" side, so we put the x^4 weight on the top of the ranking only:
    weight(rank) = (1 - r)^power,  r = rank / num_edges in [0, 1), r = 0 is the best edge
With power 4 and the top 10% labelled positive, ~40% of sampled edges are positives.
"""
import numpy as np


def sample_edges(reductions, sample_size, power, rng):
    """Return indices of `sample_size` edges, biased towards the best ones."""
    E = len(reductions)
    order = np.argsort(-reductions, kind="stable")      # best first
    r = np.arange(E) / E
    w = (1.0 - r) ** power
    w /= w.sum()
    picked = rng.choice(E, size=min(sample_size, E), replace=False, p=w)
    return order[picked]


def sample_triplets(candidate_positions, num_triplets, rng):
    """Random triplets (m, n, o) of distinct positions from `candidate_positions`."""
    c = np.asarray(candidate_positions)
    if len(c) < 3:
        return np.empty((0, 3), dtype=np.int64)
    t = np.stack([rng.choice(c, size=num_triplets) for _ in range(3)], axis=1)
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return t[keep]
