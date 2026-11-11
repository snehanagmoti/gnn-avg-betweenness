"""
Node and edge input features.

Node features (per node, standardized within each graph so they transfer to unseen graphs):
    log-degree, clustering, closeness, core number
The 9 "old" edge clues (per candidate edge, standardized within each graph):
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


def node_features(G: nx.Graph, D=None) -> np.ndarray:
    """(n, 4) float32: log-degree, clustering, closeness, core number, each standardized in the graph."""
    n = G.number_of_nodes()
    nodes = range(n)
    if D is None:
        D = distance_matrix(G)
    clus, core = nx.clustering(G), nx.core_number(G)
    cols = [np.log1p([G.degree(v) for v in nodes]), [clus[v] for v in nodes],
            (n - 1) / D.sum(1), [core[v] for v in nodes]]
    return np.stack([_standardize(c) for c in cols], axis=1).astype(np.float32)


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
