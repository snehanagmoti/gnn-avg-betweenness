import networkx as nx
import numpy as np
from scipy.stats import rankdata


def compute_avg_betweenness_brandes(G, normalized=True):
    """
    Computes exact average betweenness centrality using Brandes algorithm.
    Used for sanity checks and baseline validation.
    """
    bc = nx.betweenness_centrality(G, normalized=normalized)
    return float(np.mean(list(bc.values())))


def compute_all_pairs_shortest_paths(G):
    """
    Returns an (n, n) numpy matrix of all-pairs shortest path distances.
    """
    n = G.number_of_nodes()
    dist_matrix = np.full((n, n), 999999, dtype=np.int32)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for u in range(n):
        for v, d in lengths[u].items():
            dist_matrix[u, v] = d
    return dist_matrix


def compute_exact_delta_avg_bc_fast(dist_matrix, u, v, n, normalized=True):
    """
    Computes exact reduction in average betweenness when edge (u, v) is added,
    using the mathematical identity:
    AvgBC(G) - AvgBC(G + (u,v)) = 2 / (n*(n-1)*(n-2)) * sum_{s < t} (d(s,t) - d'(s,t))
    where d'(s,t) = min(d(s,t), d(s,u) + 1 + d(v,t), d(s,v) + 1 + d(u,t)).
    """
    d_u = dist_matrix[:, u][:, None]  # (n, 1)
    d_v = dist_matrix[v, :][None, :]  # (1, n)
    d_via_1 = d_u + 1 + d_v           # s -> u -> v -> t

    d_v_src = dist_matrix[:, v][:, None]
    d_u_dst = dist_matrix[u, :][None, :]
    d_via_2 = d_v_src + 1 + d_u_dst   # s -> v -> u -> t

    d_via = np.minimum(d_via_1, d_via_2)
    diff = np.maximum(0, dist_matrix - d_via)

    triu_indices = np.triu_indices(n, k=1)
    total_dist_decrease = np.sum(diff[triu_indices])

    if normalized:
        scale = 2.0 / (n * (n - 1) * (n - 2))
        return float(total_dist_decrease * scale)
    else:
        return float(total_dist_decrease)


def compute_all_ground_truth(G, missing_edges, normalized=True):
    """
    Computes exact delta average betweenness for all candidate missing edges.
    """
    n = G.number_of_nodes()
    dist_matrix = compute_all_pairs_shortest_paths(G)
    
    deltas = np.zeros(len(missing_edges), dtype=np.float32)
    for idx, (u, v) in enumerate(missing_edges):
        deltas[idx] = compute_exact_delta_avg_bc_fast(dist_matrix, u, v, n, normalized=normalized)
    
    return deltas


def assign_ranks_with_ties(deltas, method="dense", descending=True):
    """
    Handles ties consistently as discussed in meeting notes:
    - method='dense': tied values receive the same rank, next distinct value receives next consecutive integer.
    - method='min': tied values receive the minimum rank in their tie group.
    - method='average': tied values receive the average rank.
    """
    scores = -deltas if descending else deltas
    ranks = rankdata(scores, method=method)
    return ranks


def partition_edge_slots(deltas, top_pct=25.0, bottom_pct=25.0):
    """
    Partitions edges into three distribution slots based on reduction percentiles:
    1. Top slot: top_pct% highest reduction edges
    2. Middle slot: edges between top and bottom percentiles
    3. Bottom slot: bottom_pct% lowest reduction edges
    
    Returns:
    - slot_indices: dict mapping 'top', 'middle', 'bottom' to numpy arrays of edge indices
    - binary_labels: 1 if in top slot, 0 otherwise
    - multi_class_labels: 2 for top slot, 1 for middle slot, 0 for bottom slot
    """
    num_edges = len(deltas)
    
    # Cutoff thresholds
    top_threshold = np.percentile(deltas, 100.0 - top_pct)
    bottom_threshold = np.percentile(deltas, bottom_pct)
    
    top_mask = deltas >= top_threshold
    bottom_mask = deltas <= bottom_threshold
    middle_mask = ~(top_mask | bottom_mask)
    
    # In case of discrete ties causing empty middle, fallback cleanly
    if not np.any(middle_mask):
        sorted_indices = np.argsort(-deltas)
        n_top = int(num_edges * (top_pct / 100.0))
        n_bot = int(num_edges * (bottom_pct / 100.0))
        top_idx = sorted_indices[:n_top]
        bottom_idx = sorted_indices[-n_bot:]
        middle_idx = sorted_indices[n_top:-n_bot]
    else:
        top_idx = np.where(top_mask)[0]
        middle_idx = np.where(middle_mask)[0]
        bottom_idx = np.where(bottom_mask)[0]
        
    slot_indices = {
        "top": top_idx,
        "middle": middle_idx,
        "bottom": bottom_idx
    }
    
    # Binary labels (1 = high impact / top slot, 0 = non-top)
    binary_labels = np.zeros(num_edges, dtype=np.int64)
    binary_labels[top_idx] = 1
    
    # Multi-class labels (2 = top, 1 = middle, 0 = bottom)
    multi_labels = np.zeros(num_edges, dtype=np.int64)
    multi_labels[middle_idx] = 1
    multi_labels[top_idx] = 2
    
    return slot_indices, binary_labels, multi_labels

