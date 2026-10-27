import os
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import sample_edges_stratified_slots, sample_edges_x4, sample_triplets_from_indices
from src.losses import compute_classification_loss, compute_ranking_loss
from src.metrics import compute_all_metrics


def pretrain_classifier(
    model,
    train_graphs,
    val_graphs,
    num_epochs=30,
    lr=0.005,
    sample_size_per_graph=2000,
    sampling_method="slots",
    slot_ratios=(0.34, 0.33, 0.33),
    device="cpu",
    verbose=True
):
    """
    Stage 1: Pre-trains GNN Encoder + MLP Classifier on classification loss.
    Edge sampling draws from Top, Middle, and Bottom slots so the encoder learns
    general structural patterns distinguishing high vs low impact regions.
    """
    model.to(device)
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=lr
    )
    
    best_val_loss = float("inf")
    best_state = None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STAGE 1: PRE-TRAINING CLASSIFIER ({num_epochs} Epochs, LR: {lr})")
        print(f"Sampling: {sampling_method} | Train Graphs: {len(train_graphs)} | Val: {len(val_graphs)}")
        print(f"{'='*70}")
        
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        
        for g_data in train_graphs:
            x = g_data["x"].to(device)
            adj_norm = g_data["adj_norm"].to(device)
            missing_edges_np = g_data["missing_edges"]
            slot_indices = g_data["slot_indices"]
            binary_labels = g_data["binary_labels"]
            norm_deltas = g_data["norm_deltas"]
            
            optimizer.zero_grad()
            
            # Sampling strategy
            if sampling_method == "slots":
                sampled_idx = sample_edges_stratified_slots(
                    slot_indices, sample_size=sample_size_per_graph, ratios=slot_ratios
                )
            else:
                sampled_idx = sample_edges_x4(norm_deltas, sample_size_per_graph)
                
            sampled_edges = torch.tensor(missing_edges_np[sampled_idx], dtype=torch.long, device=device)
            sampled_targets = torch.tensor(binary_labels[sampled_idx], dtype=torch.float32, device=device)
            
            logits, probs = model.forward_classifier(x, adj_norm, sampled_edges)
            loss_c = compute_classification_loss(logits, sampled_targets, num_classes=1)
            
            loss_c.backward()
            optimizer.step()
            
            preds = (probs >= 0.5).float()
            acc = (preds == sampled_targets).float().mean()
            
            total_train_loss += loss_c.item()
            total_train_acc += acc.item()
            
        n_train = len(train_graphs)
        avg_train_loss = total_train_loss / n_train
        avg_train_acc = total_train_acc / n_train
        
        # Validation
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            val_loss, val_acc = evaluate_classifier(model, val_graphs, device=device)
            if verbose:
                print(f"Pre-train Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Loaded best pre-trained checkpoint with Validation Loss = {best_val_loss:.4f}")
            
    return model


def evaluate_classifier(model, graphs, device="cpu"):
    """Evaluates classifier loss and accuracy on a set of graphs."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for g_data in graphs:
            x = g_data["x"].to(device)
            adj_norm = g_data["adj_norm"].to(device)
            missing_edges_np = g_data["missing_edges"]
            binary_labels = g_data["binary_labels"]
            
            all_edges = torch.tensor(missing_edges_np, dtype=torch.long, device=device)
            targets = torch.tensor(binary_labels, dtype=torch.float32, device=device)
            
            logits, probs = model.forward_classifier(x, adj_norm, all_edges)
            loss_c = compute_classification_loss(logits, targets, num_classes=1)
            preds = (probs >= 0.5).float()
            acc = (preds == targets).float().mean()
            
            total_loss += loss_c.item()
            total_acc += acc.item()
            
    n_g = len(graphs)
    return total_loss / n_g, total_acc / n_g


def finetune_ranker(
    model,
    train_graphs,
    val_graphs,
    num_epochs=30,
    lr_ranker=0.005,
    lr_encoder=0.0005,
    freeze_encoder=False,
    sample_size_per_graph=2000,
    triplet_multiplier=8,
    sampling_method="slots",
    slot_ratios=(0.34, 0.33, 0.33),
    device="cpu",
    verbose=True
):
    """
    Stage 2: Fine-tunes MLP Ranker (with pre-trained encoder weights).
    Uses Triplet Margin Ranking Loss + Standardized MSE regression loss.
    """
    model.to(device)
    
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.ranker.parameters(), lr=lr_ranker)
    else:
        # Encoder trained with smaller learning rate to preserve pre-trained features
        optimizer = torch.optim.Adam([
            {"params": model.encoder.parameters(), "lr": lr_encoder},
            {"params": model.ranker.parameters(), "lr": lr_ranker}
        ])
        
    best_val_metric = -1.0
    best_state = None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STAGE 2: FINE-TUNING RANKER ({num_epochs} Epochs, Freeze Encoder: {freeze_encoder})")
        print(f"Sampling: {sampling_method} | LR Ranker: {lr_ranker}, LR Encoder: {lr_encoder}")
        print(f"{'='*70}")
        
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_trip = 0.0
        total_mse = 0.0
        
        for g_data in train_graphs:
            x = g_data["x"].to(device)
            adj_norm = g_data["adj_norm"].to(device)
            missing_edges_np = g_data["missing_edges"]
            slot_indices = g_data["slot_indices"]
            norm_deltas = g_data["norm_deltas"]
            
            optimizer.zero_grad()
            
            # Sampling strategy
            if sampling_method == "slots":
                sampled_idx = sample_edges_stratified_slots(
                    slot_indices, sample_size=sample_size_per_graph, ratios=slot_ratios
                )
            else:
                sampled_idx = sample_edges_x4(norm_deltas, sample_size_per_graph)
                
            sampled_edges = torch.tensor(missing_edges_np[sampled_idx], dtype=torch.long, device=device)
            sampled_norm_deltas = torch.tensor(norm_deltas[sampled_idx], dtype=torch.float32, device=device)
            
            # Sample triplets among the sampled edges
            num_triplets = int(len(sampled_idx) * triplet_multiplier)
            triplets = sample_triplets_from_indices(
                np.arange(len(sampled_idx)),
                sampled_norm_deltas.cpu().numpy(),
                num_triplets
            )
            for k in triplets:
                triplets[k] = triplets[k].to(device)
                
            pred_scores = model.forward_ranker(x, adj_norm, sampled_edges)
            loss_r, loss_trip_val, loss_mse_val = compute_ranking_loss(
                pred_scores, sampled_norm_deltas, triplets=triplets, margin=0.5, mse_weight=1.0
            )
            
            loss_r.backward()
            optimizer.step()
            
            total_loss += loss_r.item()
            total_trip += loss_trip_val.item()
            total_mse += loss_mse_val.item()
            
        n_train = len(train_graphs)
        avg_loss = total_loss / n_train
        
        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            val_results = evaluate_model(model, val_graphs, device=device)
            val_map10 = val_results["mAP@10"]
            val_sc = val_results["Spearman_All"]
            
            if verbose:
                print(f"Fine-tune Epoch {epoch:2d}/{num_epochs:2d} | Loss: {avg_loss:.4f} (Trip: {total_trip/n_train:.3f}, MSE: {total_mse/n_train:.3f}) | Val mAP@10: {val_map10:.4f} | Val SC: {val_sc:.4f}")
                
            if val_map10 > best_val_metric:
                best_val_metric = val_map10
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Loaded best ranker checkpoint with Validation mAP@10 = {best_val_metric:.4f}")
            
    return model


def train_simultaneous(
    model,
    train_graphs,
    val_graphs,
    num_epochs=40,
    lr=0.005,
    sample_size_per_graph=2000,
    triplet_multiplier=8,
    sampling_method="x4",
    slot_ratios=(0.34, 0.33, 0.33),
    device="cpu",
    verbose=True
):
    """
    Simultaneous training baseline: trains Encoder + Classifier + Ranker together
    with a single joint loss function: Total Loss = Loss_c + Loss_r + Loss_mse.
    This serves as the exact ablation baseline to compare against two-stage training.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_metric = -1.0
    best_state = None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SIMULTANEOUS JOINT TRAINING ({num_epochs} Epochs, LR: {lr})")
        print(f"{'='*70}")
        
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for g_data in train_graphs:
            x = g_data["x"].to(device)
            adj_norm = g_data["adj_norm"].to(device)
            missing_edges_np = g_data["missing_edges"]
            slot_indices = g_data["slot_indices"]
            binary_labels = g_data["binary_labels"]
            norm_deltas = g_data["norm_deltas"]
            
            optimizer.zero_grad()
            
            if sampling_method == "slots":
                sampled_idx = sample_edges_stratified_slots(
                    slot_indices, sample_size=sample_size_per_graph, ratios=slot_ratios
                )
            else:
                sampled_idx = sample_edges_x4(norm_deltas, sample_size_per_graph)
                
            sampled_edges = torch.tensor(missing_edges_np[sampled_idx], dtype=torch.long, device=device)
            sampled_targets = torch.tensor(binary_labels[sampled_idx], dtype=torch.float32, device=device)
            sampled_norm_deltas = torch.tensor(norm_deltas[sampled_idx], dtype=torch.float32, device=device)
            
            # Forward full
            logits, probs, pred_scores = model.forward_full(x, adj_norm, sampled_edges)
            loss_c = compute_classification_loss(logits, sampled_targets, num_classes=1)
            
            num_triplets = int(len(sampled_idx) * triplet_multiplier)
            triplets = sample_triplets_from_indices(
                np.arange(len(sampled_idx)),
                sampled_norm_deltas.cpu().numpy(),
                num_triplets
            )
            for k in triplets:
                triplets[k] = triplets[k].to(device)
                
            loss_r, _, _ = compute_ranking_loss(
                pred_scores, sampled_norm_deltas, triplets=triplets, margin=0.5, mse_weight=1.0
            )
            
            loss = loss_c + loss_r
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        n_train = len(train_graphs)
        avg_loss = total_loss / n_train
        
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            val_results = evaluate_model(model, val_graphs, device=device)
            val_map10 = val_results["mAP@10"]
            if verbose:
                print(f"Simultaneous Epoch {epoch:2d}/{num_epochs:2d} | Joint Loss: {avg_loss:.4f} | Val mAP@10: {val_map10:.4f}")
                
            if val_map10 > best_val_metric:
                best_val_metric = val_map10
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model


def evaluate_model(model, graphs, score_combination="combined", penalty=10.0, device="cpu"):
    """
    Evaluates model across a list of test graphs.
    Computes all evaluation metrics for each graph and returns the mean.
    
    score_combination:
    - 'combined': ranker score if classifier >= 0.5 else (ranker score - penalty)
    - 'ranker_only': pure ranker score
    - 'classifier_only': pure classifier probability
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for g_data in graphs:
            x = g_data["x"].to(device)
            adj_norm = g_data["adj_norm"].to(device)
            missing_edges_np = g_data["missing_edges"]
            true_deltas = g_data["true_deltas"]
            
            all_edges = torch.tensor(missing_edges_np, dtype=torch.long, device=device)
            logits, probs, ranker_scores = model.forward_full(x, adj_norm, all_edges)
            
            probs_np = probs.cpu().numpy()
            ranker_np = ranker_scores.cpu().numpy()
            
            if score_combination == "combined":
                pred_scores = np.where(probs_np >= 0.5, ranker_np, ranker_np - penalty)
            elif score_combination == "ranker_only":
                pred_scores = ranker_np
            elif score_combination == "classifier_only":
                pred_scores = probs_np
            else:
                pred_scores = ranker_np
                
            metrics = compute_all_metrics(pred_scores, true_deltas)
            all_metrics.append(metrics)
            
    # Compute mean for each metric across graphs
    metric_keys = all_metrics[0].keys()
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in metric_keys}
    return avg_metrics

