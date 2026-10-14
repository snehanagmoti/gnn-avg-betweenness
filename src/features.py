"""
Node input features for the GNN.

The paper initialises node vectors randomly. Random vectors carry no information that transfers
to a new, unseen graph, so by default we use cheap structural features instead.
Use feature set ["random"] to reproduce the paper's setting.
"""
import networkx as nx
import numpy as np


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def node_features(G: nx.Graph, feature_set, seed: int = 0, random_dim: int = 16) -> np.ndarray:
    """Return an (n, d) float32 matrix of node features, standardized per graph."""
    n = G.number_of_nodes()
    nodes = range(n)
    cols = []
    for name in feature_set:
        if name == "degree":
            cols.append(np.log1p([G.degree(v) for v in nodes]))
        elif name == "clustering":
            c = nx.clustering(G)
            cols.append([c[v] for v in nodes])
        elif name == "closeness":
            c = nx.closeness_centrality(G)
            cols.append([c[v] for v in nodes])
        elif name == "core_number":
            c = nx.core_number(G)
            cols.append([c[v] for v in nodes])
        elif name == "betweenness":
            c = nx.betweenness_centrality(G)
            cols.append([c[v] for v in nodes])
        elif name == "random":
            rng = np.random.default_rng(seed)
            return rng.standard_normal((n, random_dim)).astype(np.float32)
        else:
            raise ValueError(f"Unknown feature: {name}")
    X = np.stack([_standardize(np.asarray(c, dtype=np.float64)) for c in cols], axis=1)
    return X.astype(np.float32)
