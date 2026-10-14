"""
Step 4b: simple baselines to compare the GNN against.
If the GNN does not beat these, it has not learned anything useful yet.
"""
import networkx as nx
import numpy as np


def random_scores(missing_edges, rng):
    return rng.random(len(missing_edges))


def distance_scores(G, missing_edges, rng):
    """Longer current distance between u and v = bigger shortcut. Ties broken randomly."""
    D = nx.floyd_warshall_numpy(G, nodelist=range(G.number_of_nodes()))
    d = D[missing_edges[:, 0], missing_edges[:, 1]]
    return d + 1e-3 * rng.random(len(d))


def degree_product_scores(G, missing_edges, rng):
    """Connect high-degree nodes (hubs). Ties broken randomly."""
    deg = np.array([G.degree(v) for v in range(G.number_of_nodes())], dtype=float)
    s = deg[missing_edges[:, 0]] * deg[missing_edges[:, 1]]
    return s + 1e-3 * rng.random(len(s))


def closeness_gain_scores(G, missing_edges, rng):
    """Heuristic: link a very central node to a very peripheral one
    (closeness(u) - closeness(v) in absolute value, times their distance)."""
    c = nx.closeness_centrality(G)
    c = np.array([c[v] for v in range(G.number_of_nodes())])
    D = nx.floyd_warshall_numpy(G, nodelist=range(G.number_of_nodes()))
    u, v = missing_edges[:, 0], missing_edges[:, 1]
    s = np.abs(c[u] - c[v]) * D[u, v]
    return s + 1e-9 * rng.random(len(s))


BASELINES = {
    "random": lambda G, e, rng: random_scores(e, rng),
    "distance": distance_scores,
    "degree_product": degree_product_scores,
    "closeness_gap_x_distance": closeness_gain_scores,
}
