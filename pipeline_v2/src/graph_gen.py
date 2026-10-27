import networkx as nx
import numpy as np
import torch


def generate_connected_er_graph(n=200, p=0.04, seed=None):
    """
    Generates a connected Erdős-Rényi (ER) graph G(n, p).
    If the sampled graph is disconnected, retries with incremented seeds.
    """
    curr_seed = seed if seed is not None else np.random.randint(0, 100000)
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=curr_seed)
        if nx.is_connected(G):
            return G, curr_seed
        curr_seed += 1


def generate_connected_ba_graph(n=200, m=3, seed=None):
    """
    Generates a connected Barabási-Albert (BA) graph G(n, m).
    BA graphs are connected by construction when m >= 1.
    """
    curr_seed = seed if seed is not None else np.random.randint(0, 100000)
    G = nx.barabasi_albert_graph(n, m, seed=curr_seed)
    return G, curr_seed


def get_missing_edges(G):
    """
    Returns an array of candidate missing edges (u, v) where u < v and (u, v) not in E.
    """
    n = G.number_of_nodes()
    existing_edges = set(tuple(sorted(e)) for e in G.edges())
    missing_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing_edges:
                missing_edges.append((u, v))
    return np.array(missing_edges, dtype=np.int64)


def build_normalized_adj(G):
    """
    Constructs row-normalized adjacency matrix (A + I) / D_tilde
    for neighborhood mean aggregation in GraphSAGE:
    A_norm @ X = mean_{u in N_tilde(v)} X_u
    """
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G, dtype=np.float32) + np.eye(n, dtype=np.float32)
    row_sums = adj.sum(axis=1, keepdims=True)
    adj_norm = adj / np.maximum(row_sums, 1.0)
    return torch.tensor(adj_norm, dtype=torch.float32)

