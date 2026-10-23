import os
import time
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from src.graph_gen import generate_connected_er_graph, get_missing_edges
from src.ground_truth import compute_ground_truth_for_edges, compute_avg_betweenness_brandes
from src.dataset import sample_triplets_from_indices, sample_edges_x4
from src.model import FullModel
from src.metrics import compute_all_metrics
from run_single_graph import build_node_features, build_normalized_adj


def compute_triplet_loss(pred_scores, triplets, margin=0.5):
    """
    Triplet Margin Ranking Loss:
    Loss = mean( ReLU(-I_mn*(s_m - s_n) + M) + ReLU(-I_no*(s_n - s_o) + M) + ReLU(-I_om*(s_o - s_m) + M) ) / 3
    """
    m = triplets["m"]
    n = triplets["n"]
    o = triplets["o"]
    I_mn = triplets["I_mn"]
    I_no = triplets["I_no"]
    I_om = triplets["I_om"]
    
    s_m = pred_scores[m]
    s_n = pred_scores[n]
    s_o = pred_scores[o]
    
    term1 = F.relu(-I_mn * (s_m - s_n) + margin)
    term2 = F.relu(-I_no * (s_n - s_o) + margin)
    term3 = F.relu(-I_om * (s_o - s_m) + margin)
    
    return (term1 + term2 + term3).mean()


def generate_or_load_10_graphs(cache_path="data/er_10_graphs.pkl", n=200, force_regenerate=False):
    """
    Generates 10 connected ER graphs with varying probabilities and seeds as in paper,
    or loads from cache if already computed.
    """
    if os.path.exists(cache_path) and not force_regenerate:
        print(f"Loading cached 10-graph dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    print(f"Generating 10 ER graphs (N={n}) and computing ground truth...")
    start_t = time.time()
    
    # Table 1 in paper: ER p in [0.03, 0.0648], seeds in [0, 1000]
    p_values = [0.038, 0.040, 0.042, 0.044, 0.040, 0.042, 0.045, 0.040, 0.042, 0.044]
    seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    
    graphs_data = []
    
    for i in range(10):
        t0 = time.time()
        p = p_values[i]
        seed = seeds[i]
        
        G, used_seed = generate_connected_er_graph(n=n, p=p, seed=seed)
        missing_edges = get_missing_edges(G)
        deltas, binary_labels = compute_ground_truth_for_edges(G, missing_edges, threshold_percentile=75.0)
        
        # Build tensor features
        x = build_node_features(G, in_dim=16, seed=used_seed)
        adj_norm = build_normalized_adj(G)
        
        # Mean and std for standardizing scores
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas) + 1e-8)
        norm_deltas = ((deltas - mean_d) / std_d).astype(np.float32)
        
        graph_item = {
            "graph_id": i,
            "G": G,
            "p": p,
            "seed": used_seed,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "missing_edges": np.array(missing_edges),
            "true_deltas": deltas,
            "norm_deltas": norm_deltas,
            "binary_labels": binary_labels,
            "x": x,
            "adj_norm": adj_norm,
        }
        graphs_data.append(graph_item)
        print(f"  Graph {i+1}/10: p={p:.3f}, Seed={used_seed:3d} | Edges: {G.number_of_edges()} | Missing: {len(missing_edges)} | Time: {time.time() - t0:.2f}s")
        
    print(f"All 10 graphs generated & processed in {time.time() - start_t:.2f}s!")
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(graphs_data, f)
    print(f"Saved dataset to {cache_path}")
    
    return graphs_data


def train_multi_graphs(
    model,
    train_graphs,
    val_graphs,
    num_epochs=40,
    lr=0.005,
    sample_size_per_graph=2000,
    triplet_multiplier=8,
    device="cpu"
):
    """
    Trains GNN across multiple graphs (6 train graphs) with validation tracking (2 val graphs).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Starting Multi-Graph Training (Epochs: {num_epochs}, LR: {lr}) ---")
    print(f"Training on {len(train_graphs)} graphs | Validating on {len(val_graphs)} graphs")
    
    best_val_pc = -1.0
    best_weights = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_bce = 0.0
        total_triplet = 0.0
        total_reg = 0.0
        
        # Loop over each training graph
        for g_data in train_graphs:
            x = g_data["x"].to(device)
            adj_norm = g_data["adj_norm"].to(device)
            missing_edges_np = g_data["missing_edges"]
            binary_labels = g_data["binary_labels"]
            norm_deltas = g_data["norm_deltas"]
            
            optimizer.zero_grad()
            
            # Sample edges using x^4 distribution
            sample_size = min(sample_size_per_graph, len(missing_edges_np))
            sampled_idx = sample_edges_x4(norm_deltas, sample_size)
            
            sampled_edges = torch.tensor(missing_edges_np[sampled_idx], dtype=torch.long, device=device)
            sampled_labels = torch.tensor(binary_labels[sampled_idx], dtype=torch.float32, device=device)
            sampled_norm_deltas = torch.tensor(norm_deltas[sampled_idx], dtype=torch.float32, device=device)
            
            # Forward pass
            cls_logits, cls_probs, ranking_scores = model(x, adj_norm, sampled_edges)
            
            # Classification Loss
            loss_c = F.binary_cross_entropy(cls_probs, sampled_labels)
            
            # Triplet Ranking Loss
            num_triplets = int(sample_size * triplet_multiplier)
            triplets = sample_triplets_from_indices(np.arange(sample_size), sampled_norm_deltas.cpu().numpy(), num_triplets)
            for k in triplets:
                triplets[k] = triplets[k].to(device)
                
            loss_r = compute_triplet_loss(ranking_scores, triplets, margin=0.5)
            loss_reg = F.mse_loss(ranking_scores, sampled_norm_deltas)
            
            loss = loss_c + loss_r + loss_reg
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            total_bce += loss_c.item()
            total_triplet += loss_r.item()
            total_reg += loss_reg.item()
            
        n_train = len(train_graphs)
        avg_loss = total_epoch_loss / n_train
        
        # Evaluate on validation graphs every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate_graphs(model, val_graphs, device=device)
            val_pc = val_metrics["Pearson_Corr"]
            val_sc = val_metrics["Spearman_Corr"]
            
            print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {avg_loss:.4f} (BCE: {total_bce/n_train:.3f}, Trip: {total_triplet/n_train:.3f}) | Val PC: {val_pc:.4f} | Val SC: {val_sc:.4f}")
            
            if val_pc > best_val_pc:
                best_val_pc = val_pc
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Loaded best checkpoint with Validation PC = {best_val_pc:.4f}")
        
    return model


def evaluate_graphs(model, graphs, device="cpu"):
    """
    Evaluates model inductively across a set of graphs (averages metrics across graphs).
    """
    model.eval()
    all_graph_metrics = []
    
    with torch.no_grad():
        for g_data in graphs:
            x = g_data["x"].to(device)
            adj_norm = g_data["adj_norm"].to(device)
            missing_edges_np = g_data["missing_edges"]
            true_deltas = g_data["true_deltas"]
            
            all_edges_tensor = torch.tensor(missing_edges_np, dtype=torch.long, device=device)
            cls_logits, cls_probs, ranking_scores = model(x, adj_norm, all_edges_tensor)
            
            cls_probs_np = cls_probs.cpu().numpy()
            ranking_scores_np = ranking_scores.cpu().numpy()
            
            # Combine prediction as in paper
            combined_scores = np.where(cls_probs_np >= 0.5, ranking_scores_np, ranking_scores_np - 10.0)
            
            metrics = compute_all_metrics(combined_scores, true_deltas, k_values=(1, 10, 50, 100))
            metrics["Ranker_PC"] = float(np.corrcoef(ranking_scores_np, true_deltas)[0, 1])
            metrics["Classifier_PC"] = float(np.corrcoef(cls_probs_np, true_deltas)[0, 1])
            
            # Also store top prediction vs top actual
            top_pred_idx = np.argmax(combined_scores)
            top_actual_idx = np.argmax(true_deltas)
            metrics["top_pred_edge"] = tuple(missing_edges_np[top_pred_idx])
            metrics["top_pred_reduction"] = float(true_deltas[top_pred_idx])
            metrics["top_actual_edge"] = tuple(missing_edges_np[top_actual_idx])
            metrics["top_actual_reduction"] = float(true_deltas[top_actual_idx])
            
            all_graph_metrics.append(metrics)

    # Compute mean across graphs
    keys_to_average = ["mAP@1", "mAP@10", "mAP@50", "mAP@100", "Pearson_Corr", "Spearman_Corr", "Kendall_Tau", "Ranker_PC", "Classifier_PC"]
    avg_metrics = {}
    for k in keys_to_average:
        avg_metrics[k] = float(np.mean([m[k] for m in all_graph_metrics]))
        
    avg_metrics["per_graph"] = all_graph_metrics
    return avg_metrics


def main():
    print("=" * 75)
    print("PHASE 2: 10 ER GRAPH PIPELINE (6 TRAIN / 2 VAL / 2 TEST)")
    print("=" * 75)
    
    total_start = time.time()
    
    # 1. Generate or load 10 graphs
    dataset = generate_or_load_10_graphs(cache_path="data/er_10_graphs.pkl", n=200)
    
    # Split 6 : 2 : 2
    train_graphs = dataset[0:6]
    val_graphs = dataset[6:8]
    test_graphs = dataset[8:10]
    
    print(f"\nDataset Split:")
    print(f"  Training graphs (IDs 0-5):   {len(train_graphs)} graphs")
    print(f"  Validation graphs (IDs 6-7): {len(val_graphs)} graphs")
    print(f"  Test graphs (IDs 8-9):       {len(test_graphs)} graphs (Completely Unseen)")
    
    # 2. Instantiate Model
    model = FullModel(
        in_dim=16,
        gnn_hidden=32,
        node_emb_dim=32,      # Edge embedding size = 32 * 2 = 64 (as in paper)
        num_gnn_layers=3,     # GraphSAGE layers
        dropout=0.3
    )
    
    # 3. Train across training graphs
    trained_model = train_multi_graphs(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        num_epochs=35,
        lr=0.005,
        sample_size_per_graph=2000,
        triplet_multiplier=8,
        device="cpu"
    )
    
    # 4. Inductive Evaluation on Unseen Test Graphs
    print("\n" + "=" * 75)
    print("FINAL INDUCTIVE EVALUATION ON UNSEEN TEST GRAPHS")
    print("=" * 75)
    
    test_results = evaluate_graphs(trained_model, test_graphs, device="cpu")
    
    print("\nAverage Metrics across Unseen Test Graphs:")
    print(f"  mAP@1:          {test_results['mAP@1']:.4f}")
    print(f"  mAP@10:         {test_results['mAP@10']:.4f}")
    print(f"  mAP@50:         {test_results['mAP@50']:.4f}")
    print(f"  mAP@100:        {test_results['mAP@100']:.4f}")
    print(f"  Pearson Corr:   {test_results['Pearson_Corr']:.4f}")
    print(f"  Spearman Corr:  {test_results['Spearman_Corr']:.4f}")
    print(f"  Kendall's Tau:  {test_results['Kendall_Tau']:.4f}")
    print(f"  Ranker PC:      {test_results['Ranker_PC']:.4f}")
    print(f"  Classifier PC:  {test_results['Classifier_PC']:.4f}")
    
    print("\nPer-Test-Graph Detailed Results:")
    for i, g_res in enumerate(test_results["per_graph"]):
        gid = test_graphs[i]["graph_id"]
        print(f"  Test Graph ID {gid} (Seed {test_graphs[i]['seed']}):")
        print(f"    Pearson: {g_res['Pearson_Corr']:.4f} | Spearman: {g_res['Spearman_Corr']:.4f} | Kendall: {g_res['Kendall_Tau']:.4f}")
        print(f"    mAP@10:  {g_res['mAP@10']:.4f} | mAP@100: {g_res['mAP@100']:.4f}")
        print(f"    Best Predicted Edge: {g_res['top_pred_edge']} (True Reduction = {g_res['top_pred_reduction']:.8f})")
        print(f"    Actual Best Edge:    {g_res['top_actual_edge']} (True Reduction = {g_res['top_actual_reduction']:.8f})")

    # Save trained model weights
    save_path = "data/gnn_model_10graphs.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"\nTrained model weights saved to {save_path}")
    print(f"Total pipeline time for 10 graphs: {time.time() - total_start:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()

