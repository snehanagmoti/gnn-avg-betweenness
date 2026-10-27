"""
Step 3b: which edges the model trains on each epoch, and which triplets / pairs the ranking loss uses.

  top_biased : the paper's x^4 idea applied to the top of the ranking only (original pipeline)
               weight(rank) = (1 - r)^power,  r = rank / num_edges,  r = 0 is the best edge
  stratified : TA (23 Oct) - a fixed mix of good, near-good, middle and bad edges.
               Bands are top fractions, e.g. [0.02, 0.10, 0.50] -> top 2% | 2-10% | 10-50% | bottom 50%,
               and `ratios` says which share of the sample comes from each band.
  uniform    : every edge equally likely
"""
import numpy as np


def sample_edges(reductions, sample_size, rng, method="top_biased", power=4.0,
                 bands=(0.02, 0.10, 0.50), ratios=(0.3, 0.3, 0.2, 0.2)):
    E = len(reductions)
    size = min(sample_size, E)
    order = np.argsort(-reductions, kind="stable")      # best first
    if method == "top_biased":
        r = np.arange(E) / E
        w = (1.0 - r) ** power
        w /= w.sum()
        return order[rng.choice(E, size=size, replace=False, p=w)]
    if method == "uniform":
        return rng.choice(E, size=size, replace=False)
    if method == "stratified":
        cuts = [0] + [int(round(b * E)) for b in bands] + [E]
        picked = []
        for (lo, hi), share in zip(zip(cuts[:-1], cuts[1:]), ratios):
            k = min(int(round(share * size)), hi - lo)
            picked.append(order[lo + rng.choice(hi - lo, size=k, replace=False)])
        return np.concatenate(picked)
    raise ValueError(f"Unknown sampling method: {method}")


def sample_triplets(candidate_positions, num_triplets, rng):
    """Random triplets (m, n, o) of distinct positions from `candidate_positions`."""
    c = np.asarray(candidate_positions)
    if len(c) < 3:
        return np.empty((0, 3), dtype=np.int64)
    t = np.stack([rng.choice(c, size=num_triplets) for _ in range(3)], axis=1)
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return t[keep]
