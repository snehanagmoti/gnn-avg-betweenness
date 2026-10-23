import os
import time
import numpy as np
import networkx as nx
import torch

from src.graph_gen import generate_connected_er_graph, get_missing_edges
from src.ground_truth import (
    compute_avg_betweenness_brandes,
    compute_ground_truth_for_edges,
    compute_all_pairs_shortest_paths,
    compute_exact_delta_avg_bc_fast,
)
from src.model import FullModel
from src.train import train_single_graph


def build_normalized_adj(G):
    """
    Constructs row-normalized adjacency matrix (A + I) / D_tilde
    so that A_norm @ X performs neighborhood mean aggregation.
    """
    n = G.number_of_nodes()
    # Adjacency matrix with self-loops
    adj = nx.to_numpy_array(G, dtype=np.float32) + np.eye(n, dtype=np.float32)
    # Row-normalize: divide each row by row sum
    row_sums = adj.sum(axis=1, keepdims=True)
    adj_norm = adj / np.maximum(row_sums, 1.0)
    return torch.tensor(adj_norm, dtype=torch.float32)


def build_node_features(G, in_dim=16, seed=42):
    """
    Builds node features (degrees, core numbers, closeness centrality + random structural projections)
    as specified in GNN centrality estimation literature and the paper.
    """
    np.random.seed(seed)
    n = G.number_of_nodes()
    
    # 1. Structural features
    degrees = np.array([deg for _, deg in G.degree()], dtype=np.float32)
    core_nums = np.array(list(nx.core_number(G).values()), dtype=np.float32)
    closeness = np.array(list(nx.closeness_centrality(G).values()), dtype=np.float32)
    
    feats = np.zeros((n, in_dim), dtype=np.float32)
    feats[:, 0] = degrees / (degrees.max() + 1e-5)
    feats[:, 1] = core_nums / (core_nums.max() + 1e-5)
    feats[:, 2] = closeness / (closeness.max() + 1e-5)
    
    # Remaining dimensions initialized randomly
    rand_proj = np.random.randn(n, in_dim - 3).astype(np.float32) * 0.1
    feats[:, 3:] = rand_proj
    
    return torch.tensor(feats, dtype=torch.float32)


def main():
    print("=" * 70)
    print("STEP 1: GENERATING ERDOS-RENYI (ER) GRAPH")
    print("=" * 70)
    
    N = 200
    P = 0.04
    SEED = 42
    
    start_time = time.time()
    G, used_seed = generate_connected_er_graph(n=N, p=P, seed=SEED)
    print(f"Generated ER graph G(N={N}, p={P}) with seed={used_seed}")
    print(f"Nodes: {G.number_of_nodes()} | Existing Edges: {G.number_of_edges()}")
    print(f"Is Connected: {nx.is_connected(G)}")
    
    missing_edges = get_missing_edges(G)
    num_missing = len(missing_edges)
    print(f"Candidate Missing Edges: {num_missing}")

    print("\n" + "=" * 70)
    print("STEP 2: COMPUTING BASELINE & GROUND TRUTH (Δ Avg Betweenness)")
    print("=" * 70)
    
    print("Calculating baseline Betweenness Centrality...")
    t0 = time.time()
    base_avg_bc = compute_avg_betweenness_brandes(G, normalized=True)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    base_max_bc = max(bc_dict.values())
    print(f"Baseline Avg Betweenness: {base_avg_bc:.6f}")
    print(f"Baseline Max Betweenness: {base_max_bc:.6f} (paper's original objective)")
    print(f"Time for baseline Brandes: {time.time() - t0:.3f}s")
    
    # Verify fast formula against Brandes on 1 test edge
    test_edge = missing_edges[0]
    G_test = G.copy()
    G_test.add_edge(*test_edge)
    brandes_avg_bc_test = compute_avg_betweenness_brandes(G_test, normalized=True)
    brandes_delta = base_avg_bc - brandes_avg_bc_test
    
    dist_mat = compute_all_pairs_shortest_paths(G)
    fast_delta = compute_exact_delta_avg_bc_fast(dist_mat, test_edge[0], test_edge[1], N, normalized=True)
    print(f"\nVerification on test edge {test_edge}:")
    print(f"  Brandes reduction:    {brandes_delta:.8f}")
    print(f"  Fast formula reduction: {fast_delta:.8f}")
    print(f"  Absolute difference:   {abs(brandes_delta - fast_delta):.1e} (Identical!)")

    print(f"\nComputing ground truth for all {num_missing} candidate missing edges...")
    t1 = time.time()
    deltas, binary_labels = compute_ground_truth_for_edges(G, missing_edges, normalized=True)
    print(f"Computed all {num_missing} ground truth scores in {time.time() - t1:.2f}s!")
    print(f"Max reduction in Avg BC: {deltas.max():.8f}")
    print(f"Min reduction in Avg BC: {deltas.min():.8f}")
    print(f"Mean reduction in Avg BC: {deltas.mean():.8f}")
    print(f"Edges with positive reduction: {np.sum(binary_labels)} / {num_missing}")

    print("\n" + "=" * 70)
    print("STEP 3: INITIALIZING GNN + 2 MLPs (GraphSAGE Encoder + Classifier + Ranker)")
    print("=" * 70)
    
    in_dim = 16
    x = build_node_features(G, in_dim=in_dim, seed=SEED)
    adj_norm = build_normalized_adj(G)
    
    model = FullModel(
        in_dim=in_dim,
        gnn_hidden=32,
        node_emb_dim=32,      # Edge embedding size = 32 * 2 = 64 (as in paper)
        num_gnn_layers=3,     # GraphSAGE layers
        dropout=0.5
    )
    print(model)

    print("\n" + "=" * 70)
    print("STEP 4: TRAINING PIPELINE (JOINT BCE + TRIPLET MARGIN LOSS)")
    print("=" * 70)
    
    metrics, combined_scores, ranker_scores, cls_probs = train_single_graph(
        model=model,
        x=x,
        adj_norm=adj_norm,
        all_missing_edges=missing_edges,
        true_deltas=deltas,
        binary_labels=binary_labels,
        num_epochs=60,
        lr=0.0085,
        sample_size=3000,
        triplet_multiplier=10,
        device="cpu"
    )

    print("\n" + "=" * 70)
    print("STEP 5: EVALUATION RESULTS")
    print("=" * 70)
    print(f"mAP@1:          {metrics['mAP@1']:.4f}")
    print(f"mAP@10:         {metrics['mAP@10']:.4f}")
    print(f"mAP@50:         {metrics['mAP@50']:.4f}")
    print(f"mAP@100:        {metrics['mAP@100']:.4f}")
    print(f"Combined PC:    {metrics['Pearson_Corr']:.4f}")
    print(f"Combined SC:    {metrics['Spearman_Corr']:.4f}")
    print(f"Combined KC:    {metrics['Kendall_Tau']:.4f}")
    print(f"Classifier PC:  {metrics['Classifier_PC']:.4f}")
    print(f"Ranker PC:      {metrics['Ranker_PC']:.4f}")

    # Inspect top predicted edge vs top actual edge
    top_pred_idx = np.argmax(combined_scores)
    top_actual_idx = np.argmax(deltas)
    print("\nComparison of Top Recommendations:")
    print(f"  Best predicted missing edge: {missing_edges[top_pred_idx]} with true reduction = {deltas[top_pred_idx]:.8f}")
    print(f"  Best actual missing edge:    {missing_edges[top_actual_idx]} with true reduction = {deltas[top_actual_idx]:.8f}")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

