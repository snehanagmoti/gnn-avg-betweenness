"""
Step 1b: ground truth = how much adding each missing edge reduces AVERAGE betweenness.

Fast exact method (connected, unweighted, undirected graphs):
    total betweenness = sum over node pairs {s,t} of (d(s,t) - 1)
so the reduction from adding edge (u,v) only depends on how shortest-path distances change:
    d'(s,t) = min( d(s,t), d(s,u) + 1 + d(v,t), d(s,v) + 1 + d(u,t) )
This gives exactly the same values as recomputing betweenness (checked by `brute_force_check`),
but ~400x faster (~5 s instead of ~30 min for a 200-node graph).
"""
import random

import networkx as nx
import numpy as np
from scipy.stats import rankdata


def normalization_scale(n: int, normalize: bool) -> float:
    """Factor that converts raw (undirected) betweenness to NetworkX's normalized=True scale."""
    return 2.0 / ((n - 1) * (n - 2)) if normalize else 1.0


def average_betweenness(G: nx.Graph, normalize: bool = True) -> float:
    bc = nx.betweenness_centrality(G, normalized=normalize)
    return sum(bc.values()) / G.number_of_nodes()


def compute_reductions(G: nx.Graph, normalize: bool = True, chunk: int = 64):
    """Return (edges, reductions):
    edges      -> int array (E_missing, 2), every missing edge (u < v)
    reductions -> float array (E_missing,), drop in average betweenness if that edge is added
    Vectorised: for a fixed u, all candidate partners v are updated together (in chunks).
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    assert nodes == list(range(n)), "Nodes must be labelled 0..n-1"
    D = nx.floyd_warshall_numpy(G, nodelist=nodes)
    scale = normalization_scale(n, normalize)
    A = nx.to_numpy_array(G, nodelist=nodes) > 0
    total = D.sum()

    edges, reductions = [], []
    for u in range(n):
        vs = np.array([v for v in range(u + 1, n) if not A[u, v]], dtype=np.int64)
        for i in range(0, len(vs), chunk):
            vc = vs[i:i + chunk]
            du = D[:, u][None, :, None]                  # d(s, u)
            dv = D[vc][:, :, None]                       # d(s, v)   (k, n, 1)
            newD = np.minimum(D[None], np.minimum(du + 1 + D[vc][:, None, :],
                                                  dv + 1 + D[u][None, None, :]))
            # each unordered pair appears twice -> /2 ; average over n nodes -> /n
            reductions.extend((total - newD.sum(axis=(1, 2))) / 2.0 / n * scale)
            edges.extend((u, int(v)) for v in vc)
    return np.array(edges, dtype=np.int64), np.array(reductions, dtype=np.float64)


def competition_rank(reductions: np.ndarray) -> np.ndarray:
    """Rank 1 = best edge; tied edges share the best rank (1, 1, 3, ...) - TA, 23 Oct."""
    return rankdata(-reductions, method="min")


def make_labels(reductions: np.ndarray, bands) -> np.ndarray:
    """Class label = number of top-fraction bands the edge falls into (higher = better).
    bands=[0.10] -> binary top-10%; bands=[0.10, 0.02] -> 3 classes. Ties are kept together."""
    if np.isscalar(bands):
        bands = [bands]
    rank = competition_rank(reductions)
    E = len(reductions)
    return sum((rank <= max(1, int(round(b * E)))).astype(np.int64) for b in bands)


def brute_force_check(G: nx.Graph, edges, reductions, normalize=True, num_checks=3, seed=0):
    """Sanity check: recompute betweenness from scratch for a few random edges."""
    rng = random.Random(seed)
    base = average_betweenness(G, normalize)
    for idx in rng.sample(range(len(edges)), num_checks):
        u, v = edges[idx]
        H = G.copy()
        H.add_edge(int(u), int(v))
        true_red = base - average_betweenness(H, normalize)
        if not np.isclose(true_red, reductions[idx], rtol=1e-6, atol=1e-12):
            raise AssertionError(f"Mismatch on edge {(u, v)}: fast={reductions[idx]}, brute={true_red}")
    return True


def reductions_for_edges(G: nx.Graph, edges, normalize: bool = True, D=None):
    """Exact reduction for a SHORT list of candidate edges (used to re-rank the model's shortlist).
    Cost: O(len(edges) * n^2) instead of O(n^4) for all candidates."""
    n = G.number_of_nodes()
    if D is None:
        D = nx.floyd_warshall_numpy(G, nodelist=range(n))
    scale = normalization_scale(n, normalize)
    total = D.sum()
    out = np.empty(len(edges))
    for i, (u, v) in enumerate(edges):
        newD = np.minimum(D, np.minimum(D[:, [u]] + 1 + D[[v], :], D[:, [v]] + 1 + D[[u], :]))
        out[i] = (total - newD.sum()) / 2.0 / n * scale
    return out
