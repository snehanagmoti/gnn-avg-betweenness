import networkx as nx
import numpy as np


def compute_baseline_scores(G, missing_edges, baseline_type="closeness_gap_x_distance", seed=42):
    """
    Computes candidate edge scores using heuristic baselines.
    Handles ties robustly by adding a deterministic small perturbation where necessary.
    """
    rng = np.random.RandomState(seed)
    n = G.number_of_nodes()
    num_edges = len(missing_edges)
    u_nodes = missing_edges[:, 0]
    v_nodes = missing_edges[:, 1]
    
    if baseline_type == "random":
        return rng.uniform(0.0, 1.0, size=num_edges)
        
    elif baseline_type == "distance":
        # Shortest path distance between candidate nodes
        # In small diameter graphs, many edges tie in distance (e.g., dist=2, 3).
        # We add a small tie-breaker so ranking is well-defined.
        dist_dict = dict(nx.all_pairs_shortest_path_length(G))
        dists = np.array([dist_dict[u][v] for u, v in missing_edges], dtype=np.float32)
        tie_breaker = rng.uniform(0.0, 1e-4, size=num_edges)
        return dists + tie_breaker
        
    elif baseline_type == "degree_product":
        degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
        scores = degrees[u_nodes] * degrees[v_nodes]
        tie_breaker = rng.uniform(0.0, 1e-4, size=num_edges)
        return scores + tie_breaker

    elif baseline_type == "degree_sum":
        degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
        scores = degrees[u_nodes] + degrees[v_nodes]
        tie_breaker = rng.uniform(0.0, 1e-4, size=num_edges)
        return scores + tie_breaker

    elif baseline_type == "closeness_gap_x_distance":
        # Connect a central node with a peripheral node across current distance
        closeness = np.array(list(nx.closeness_centrality(G).values()), dtype=np.float32)
        dist_dict = dict(nx.all_pairs_shortest_path_length(G))
        dists = np.array([dist_dict[u][v] for u, v in missing_edges], dtype=np.float32)
        gap = np.abs(closeness[u_nodes] - closeness[v_nodes])
        scores = gap * dists
        tie_breaker = rng.uniform(0.0, 1e-6, size=num_edges)
        return scores + tie_breaker

    elif baseline_type == "common_neighbors":
        scores = np.zeros(num_edges, dtype=np.float32)
        adj_sets = [set(G.neighbors(node)) for node in range(n)]
        for i in range(num_edges):
            u, v = u_nodes[i], v_nodes[i]
            scores[i] = len(adj_sets[u] & adj_sets[v])
        tie_breaker = rng.uniform(0.0, 1e-4, size=num_edges)
        return scores + tie_breaker

    elif baseline_type == "jaccard":
        scores = np.zeros(num_edges, dtype=np.float32)
        adj_sets = [set(G.neighbors(node)) for node in range(n)]
        for i in range(num_edges):
            u, v = u_nodes[i], v_nodes[i]
            union_len = len(adj_sets[u] | adj_sets[v])
            scores[i] = (len(adj_sets[u] & adj_sets[v]) / union_len) if union_len > 0 else 0.0
        tie_breaker = rng.uniform(0.0, 1e-4, size=num_edges)
        return scores + tie_breaker

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


ALL_BASELINES = [
    "random",
    "distance",
    "degree_product",
    "degree_sum",
    "closeness_gap_x_distance",
    "common_neighbors",
    "jaccard"
]

