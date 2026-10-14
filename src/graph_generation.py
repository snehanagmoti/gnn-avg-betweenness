"""
Step 1a: generate synthetic networks.
"""
import networkx as nx


def generate_ba_graph(num_nodes: int, m: int, seed: int) -> nx.Graph:
    """Barabasi-Albert graph: each new node attaches to m existing nodes (preferential attachment).
    BA graphs are always connected, which the ground-truth computation relies on."""
    G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    assert nx.is_connected(G), "Graph must be connected"
    return G


def generate_graphs(num_graphs: int, num_nodes: int, m_values, base_seed: int):
    """Generate `num_graphs` BA graphs, cycling through the given m values.
    Returns a list of (graph, metadata) pairs."""
    graphs = []
    for i in range(num_graphs):
        m = m_values[i % len(m_values)]
        seed = base_seed + i
        G = generate_ba_graph(num_nodes, m, seed)
        graphs.append((G, {"graph_id": i, "n": num_nodes, "m": m, "seed": seed}))
    return graphs
