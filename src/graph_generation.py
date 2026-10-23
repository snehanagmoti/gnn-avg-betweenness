"""
Step 1a: generate synthetic networks (Erdos-Renyi & Barabasi-Albert).
"""
import networkx as nx


def generate_ba_graph(num_nodes: int, m: int, seed: int) -> nx.Graph:
    """Barabasi-Albert graph: each new node attaches to m existing nodes (preferential attachment).
    BA graphs are always connected, which the ground-truth computation relies on."""
    G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    assert nx.is_connected(G), "Graph must be connected"
    return G


def generate_er_graph(num_nodes: int, p: float, seed: int, max_retries: int = 100) -> nx.Graph:
    """Erdos-Renyi G(n, p) graph guaranteed to be connected.
    If the initial seed yields a disconnected graph, retries with incremented seeds."""
    current_seed = seed
    for _ in range(max_retries):
        G = nx.erdos_renyi_graph(num_nodes, p, seed=current_seed)
        if nx.is_connected(G):
            G = nx.convert_node_labels_to_integers(G)
            return G
        current_seed += 1
    raise RuntimeError(f"Could not generate a connected ER graph after {max_retries} attempts.")


def generate_graphs(num_graphs: int, num_nodes: int, m_values=None, p_values=None, base_seed: int = 0, graph_type: str = "er"):
    """Generate `num_graphs` synthetic graphs (ER or BA).
    Returns a list of (graph, metadata) pairs."""
    graphs = []
    for i in range(num_graphs):
        seed = base_seed + i
        if graph_type == "er":
            p = p_values[i % len(p_values)] if p_values else 0.04
            G = generate_er_graph(num_nodes, p, seed)
            graphs.append((G, {"graph_id": i, "n": num_nodes, "p": p, "seed": seed, "type": "er"}))
        else:
            m = m_values[i % len(m_values)] if m_values else 3
            G = generate_ba_graph(num_nodes, m, seed)
            graphs.append((G, {"graph_id": i, "n": num_nodes, "m": m, "seed": seed, "type": "ba"}))
    return graphs
