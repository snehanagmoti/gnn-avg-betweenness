import os
import sys
import time
import pickle
import numpy as np
import networkx as nx

# Add parent directory to path so pipeline_v2/src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_gen import generate_connected_er_graph, generate_connected_ba_graph, get_missing_edges, build_normalized_adj
from src.ground_truth import compute_all_ground_truth, partition_edge_slots, assign_ranks_with_ties, compute_avg_betweenness_brandes
from src.features import compute_node_features


def generate_and_save_dataset(
    graph_type="er",
    num_graphs=20,
    n=200,
    cache_path=None,
    force_regenerate=False,
    seed_base=100
):
    """
    Generates and caches 20 graphs (either ER or BA) with exact ground-truth
    reductions, structural + random node features, and multi-slot partitions.
    """
    if cache_path is None:
        cache_path = f"pipeline_v2/data/{graph_type}_{num_graphs}_graphs.pkl"
        
    if os.path.exists(cache_path) and not force_regenerate:
        print(f"[{graph_type.upper()}] Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
            
    print(f"\n{'='*70}")
    print(f"GENERATING {num_graphs} {graph_type.upper()} GRAPHS (n={n})")
    print(f"{'='*70}")
    
    start_total = time.time()
    graphs_data = []
    
    # ER probabilities or BA attachment parameters
    er_ps = np.linspace(0.035, 0.045, num_graphs)
    ba_ms = [3] * num_graphs  # m=3 as discussed in meeting
    
    for i in range(num_graphs):
        t0 = time.time()
        seed = seed_base + i * 17
        
        if graph_type == "er":
            p = float(er_ps[i])
            G, used_seed = generate_connected_er_graph(n=n, p=p, seed=seed)
            param_str = f"p={p:.4f}"
        elif graph_type == "ba":
            m = int(ba_ms[i])
            G, used_seed = generate_connected_ba_graph(n=n, m=m, seed=seed)
            param_str = f"m={m}"
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
            
        missing_edges = get_missing_edges(G)
        num_missing = len(missing_edges)
        
        # Exact ground truth reductions
        true_deltas = compute_all_ground_truth(G, missing_edges, normalized=True)
        
        # Standardize for regression / ranking stability
        d_mean = float(np.mean(true_deltas))
        d_std = float(np.std(true_deltas)) + 1e-8
        norm_deltas = (true_deltas - d_mean) / d_std
        
        # Distribution slot partitioning (Top 25%, Middle 50%, Bottom 25%)
        slot_indices, binary_labels, multi_labels = partition_edge_slots(
            true_deltas, top_pct=25.0, bottom_pct=25.0
        )
        
        # Dense ranks handling ties
        ranks = assign_ranks_with_ties(true_deltas, method="dense", descending=True)
        
        # Precompute structural and random node features (in_dim=16)
        x_structural = compute_node_features(G, mode="structural", in_dim=16, seed=used_seed)
        x_random = compute_node_features(G, mode="random", in_dim=16, seed=used_seed)
        adj_norm = build_normalized_adj(G)
        
        graph_item = {
            "graph_idx": i,
            "graph_type": graph_type,
            "G": G,
            "n": n,
            "seed": used_seed,
            "num_edges": G.number_of_edges(),
            "missing_edges": missing_edges,
            "true_deltas": true_deltas,
            "norm_deltas": norm_deltas,
            "ranks": ranks,
            "slot_indices": slot_indices,
            "binary_labels": binary_labels,
            "multi_labels": multi_labels,
            "x_structural": x_structural,
            "x_random": x_random,
            "adj_norm": adj_norm,
        }
        graphs_data.append(graph_item)
        
        dt = time.time() - t0
        print(f"  Graph {i+1:2d}/{num_graphs}: {param_str}, Seed={used_seed:5d} | Edges: {G.number_of_edges()} | Missing: {num_missing} | Max Delta: {true_deltas.max():.7f} | Time: {dt:.2f}s")
        
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(graphs_data, f)
        
    print(f"\nAll {num_graphs} {graph_type.upper()} graphs generated and saved to {cache_path} in {time.time() - start_total:.2f}s!")
    return graphs_data


if __name__ == "__main__":
    # Generate 20 ER graphs
    er_graphs = generate_and_save_dataset(
        graph_type="er", num_graphs=20, cache_path="pipeline_v2/data/er_20_graphs.pkl"
    )
    # Generate 20 BA graphs
    ba_graphs = generate_and_save_dataset(
        graph_type="ba", num_graphs=20, cache_path="pipeline_v2/data/ba_20_graphs.pkl"
    )

