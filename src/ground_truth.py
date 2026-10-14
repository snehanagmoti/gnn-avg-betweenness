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


def normalization_scale(n: int, normalize: bool) -> float:
    """Factor that converts raw (undirected) betweenness to NetworkX's normalized=True scale."""
    return 2.0 / ((n - 1) * (n - 2)) if normalize else 1.0


def average_betweenness(G: nx.Graph, normalize: bool = True) -> float:
    bc = nx.betweenness_centrality(G, normalized=normalize)
    return sum(bc.values()) / G.number_of_nodes()


def compute_reductions(G: nx.Graph, normalize: bool = True):
    """Return (edges, reductions):
    edges      -> int array (E_missing, 2), every missing edge (u < v)
    reductions -> float array (E_missing,), drop in average betweenness if that edge is added
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    assert nodes == list(range(n)), "Nodes must be labelled 0..n-1"
    D = nx.floyd_warshall_numpy(G, nodelist=nodes)
    scale = normalization_scale(n, normalize)

    edges, reductions = [], []
    for u, v in nx.non_edges(G):
        u, v = min(u, v), max(u, v)
        newD = np.minimum(D, np.minimum(D[:, [u]] + 1 + D[[v], :],
                                        D[:, [v]] + 1 + D[[u], :]))
        # each unordered pair appears twice in the matrix -> /2 ; average over n nodes -> /n
        reductions.append((D - newD).sum() / 2.0 / n * scale)
        edges.append((u, v))
    return np.array(edges, dtype=np.int64), np.array(reductions, dtype=np.float64)


def make_labels(reductions: np.ndarray, positive_fraction: float) -> np.ndarray:
    """Binary labels for the classifier: 1 if the edge is in the top `positive_fraction` of its graph.
    Ties at the threshold are all labelled positive."""
    threshold = np.quantile(reductions, 1.0 - positive_fraction)
    return (reductions >= threshold).astype(np.int64)


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
