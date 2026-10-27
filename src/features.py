"""
Node and edge input features.

Node features (per node, standardized within each graph so they transfer to unseen graphs):
    degree (log), clustering, closeness, core_number, pagerank, eigenvector, eccentricity,
    betweenness, hub_dist (distance to the 5 highest-degree nodes), random (paper setting)
Edge features (per candidate edge, standardized within each graph):
    dist, common_neighbors, jaccard, adamic_adar, closeness_gap, closeness_minmax, degree_minmax
"""
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def distance_matrix(G: nx.Graph) -> np.ndarray:
    A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()))
    return shortest_path(A, unweighted=True, directed=False)


def node_features(G: nx.Graph, feature_set, seed: int = 0, random_dim: int = 16, D=None) -> np.ndarray:
    """Return an (n, d) float32 matrix of node features."""
    n = G.number_of_nodes()
    nodes = range(n)
    if D is None:
        D = distance_matrix(G)
    cols = []
    for name in feature_set:
        if name == "degree":
            cols.append(np.log1p([G.degree(v) for v in nodes]))
        elif name == "clustering":
            c = nx.clustering(G)
            cols.append([c[v] for v in nodes])
        elif name == "closeness":
            cols.append((n - 1) / D.sum(1))
        elif name == "core_number":
            c = nx.core_number(G)
            cols.append([c[v] for v in nodes])
        elif name == "pagerank":
            c = nx.pagerank(G)
            cols.append([c[v] for v in nodes])
        elif name == "eigenvector":
            c = nx.eigenvector_centrality_numpy(G)
            cols.append([abs(c[v]) for v in nodes])
        elif name == "eccentricity":
            cols.append(D.max(1))
        elif name == "betweenness":
            c = nx.betweenness_centrality(G)
            cols.append([c[v] for v in nodes])
        elif name == "hub_dist":
            hubs = np.argsort([-G.degree(v) for v in nodes], kind="stable")[:5]
            cols.extend(D[:, h] for h in hubs)
        elif name == "random":
            rng = np.random.default_rng(seed)
            cols.extend(rng.standard_normal((random_dim, n)))
        else:
            raise ValueError(f"Unknown node feature: {name}")
    X = np.stack([_standardize(c) for c in cols], axis=1)
    return X.astype(np.float32)


def edge_features(G: nx.Graph, missing_edges: np.ndarray, feature_set, D=None) -> np.ndarray:
    """Return an (E, d) float32 matrix of hand-made features for the candidate edges (d may be 0)."""
    E = len(missing_edges)
    if not feature_set:
        return np.zeros((E, 0), dtype=np.float32)
    n = G.number_of_nodes()
    if D is None:
        D = distance_matrix(G)
    u, v = missing_edges[:, 0], missing_edges[:, 1]
    A = nx.to_numpy_array(G, nodelist=range(n))
    deg = A.sum(1)
    clo = (n - 1) / D.sum(1)
    cols = []
    for name in feature_set:
        if name == "dist":
            cols.append(D[u, v])
        elif name == "common_neighbors":
            cols.append((A @ A)[u, v])
        elif name == "jaccard":
            inter = (A @ A)[u, v]
            cols.append(inter / np.maximum(deg[u] + deg[v] - inter, 1))
        elif name == "adamic_adar":
            w = 1.0 / np.log(np.maximum(deg, 2))
            cols.append(((A * w) @ A)[u, v])
        elif name == "closeness_gap":
            cols.append(np.abs(clo[u] - clo[v]))
        elif name == "closeness_minmax":
            cols += [np.minimum(clo[u], clo[v]), np.maximum(clo[u], clo[v])]
        elif name == "degree_minmax":
            cols += [np.log1p(np.minimum(deg[u], deg[v])), np.log1p(np.maximum(deg[u], deg[v]))]
        else:
            raise ValueError(f"Unknown edge feature: {name}")
    X = np.stack([_standardize(c) for c in cols], axis=1)
    return X.astype(np.float32)
