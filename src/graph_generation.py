"""
Step 1a: generate synthetic connected networks.

  ba          : Barabasi-Albert, each new node attaches to m nodes (hubs + periphery)
  er          : Erdos-Renyi G(n, p), regenerated with the next seed until connected
  random_tree : uniformly random labelled tree
  ktree       : k-tree - start from a (k+1)-clique, each new node joins a random existing k-clique
  path        : path graph (node count varied so the graphs differ)
The ground-truth trick needs connected graphs, so every generator guarantees connectivity.
"""
import networkx as nx
import numpy as np


def generate_ba_graph(num_nodes: int, m: int, seed: int) -> nx.Graph:
    G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    assert nx.is_connected(G), "Graph must be connected"
    return G


def generate_er_graph(num_nodes: int, p: float, seed: int, max_tries: int = 1000):
    for k in range(max_tries):
        G = nx.erdos_renyi_graph(num_nodes, p, seed=seed * 1000 + k)
        if nx.is_connected(G):
            return G
    raise RuntimeError("could not generate a connected ER graph; increase p")


def generate_ktree(num_nodes: int, k: int, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.complete_graph(k + 1)
    cliques = [tuple(c) for c in __import__("itertools").combinations(range(k + 1), k)]
    for v in range(k + 1, num_nodes):
        base = cliques[rng.integers(len(cliques))]
        G.add_edges_from((v, u) for u in base)
        for drop in range(k):            # new k-cliques containing v
            cliques.append(tuple(sorted(base[:drop] + base[drop + 1:] + (v,))))
    return G


def generate_graph(graph_type, num_nodes, seed, m=3, p=0.04, k=2):
    if graph_type == "ba":
        return generate_ba_graph(num_nodes, m, seed)
    if graph_type == "er":
        return generate_er_graph(num_nodes, p, seed)
    if graph_type == "random_tree":
        return nx.random_labeled_tree(num_nodes, seed=seed)
    if graph_type == "ktree":
        return generate_ktree(num_nodes, k, seed)
    if graph_type == "path":
        return nx.path_graph(num_nodes)
    raise ValueError(f"Unknown graph type: {graph_type}")


def generate_graphs(num_graphs, num_nodes, m_values=(3,), base_seed=0, graph_type="ba",
                    p_values=(0.04,), k=2):
    """Return a list of (graph, metadata) pairs."""
    graphs = []
    for i in range(num_graphs):
        seed = base_seed + i
        m = m_values[i % len(m_values)]
        p = p_values[i % len(p_values)]
        n = num_nodes if graph_type != "path" else num_nodes - 10 * (num_graphs // 2) + 10 * i
        G = nx.convert_node_labels_to_integers(generate_graph(graph_type, n, seed, m, p, k))
        assert nx.is_connected(G)
        meta = {"graph_id": i, "type": graph_type, "n": G.number_of_nodes(), "seed": seed}
        meta.update({"ba": {"m": m}, "er": {"p": p}, "ktree": {"k": k}}.get(graph_type, {}))
        graphs.append((G, meta))
    return graphs
