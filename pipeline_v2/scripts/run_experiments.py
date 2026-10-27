import os
import sys
import time
import json
import pickle
import numpy as np
import torch

# Ensure pipeline_v2 root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import ModularTwoStageModel
from src.trainer import pretrain_classifier, finetune_ranker, train_simultaneous, evaluate_model
from src.baselines import compute_baseline_scores, ALL_BASELINES
from src.metrics import compute_all_metrics


def load_dataset(dataset_path):
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def evaluate_baselines_on_test(test_graphs):
    """Evaluates all heuristic baselines across the test graphs."""
    baseline_results = {}
    for b_name in ALL_BASELINES:
        graph_metrics = []
        for g_data in test_graphs:
            G = g_data["G"]
            missing_edges_np = g_data["missing_edges"]
            true_deltas = g_data["true_deltas"]
            
            b_scores = compute_baseline_scores(G, missing_edges_np, baseline_type=b_name, seed=g_data["seed"])
            metrics = compute_all_metrics(b_scores, true_deltas)
            graph_metrics.append(metrics)
            
        # Average across test graphs
        metric_keys = graph_metrics[0].keys()
        avg_m = {k: float(np.mean([m[k] for m in graph_metrics])) for k in metric_keys}
        baseline_results[f"Baseline: {b_name}"] = avg_m
    return baseline_results


def run_single_pipeline_configuration(
    train_graphs,
    val_graphs,
    test_graphs,
    feature_mode="structural",
    training_mode="two_stage",     # 'two_stage' vs 'simultaneous'
    sampling_mode="slots",         # 'slots' (top/middle/bottom) vs 'x4'
    edge_method="sum_diff",        # 'concat', 'sum_diff', 'rich_symmetric'
    in_dim=16,
    num_epochs_pretrain=25,
    num_epochs_finetune=25,
    num_epochs_simul=35,
    lr=0.005,
    freeze_encoder=False,
    device="cpu",
    config_name="Model"
):
    """
    Executes a specific pipeline configuration end-to-end and returns test metrics.
    """
    print(f"\n{'#'*75}")
    print(f"RUNNING CONFIGURATION: {config_name}")
    print(f"  Feature Mode: {feature_mode} | Edge Representation: {edge_method}")
    print(f"  Training Mode: {training_mode} | Sampling: {sampling_mode}")
    print(f"{'#'*75}")
    
    # Set the appropriate node feature tensor in each graph dict
    feat_key = "x_structural" if feature_mode == "structural" else "x_random"
    for g in train_graphs + val_graphs + test_graphs:
        g["x"] = g[feat_key]
        
    model = ModularTwoStageModel(
        in_dim=in_dim,
        gnn_hidden=32,
        node_emb_dim=32,
        num_gnn_layers=3,
        edge_method=edge_method,
        num_classes=1,
        dropout=0.5
    )
    
    start_t = time.time()
    
    if training_mode == "two_stage":
        # 1. Pre-train Classifier
        model = pretrain_classifier(
            model=model,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            num_epochs=num_epochs_pretrain,
            lr=lr,
            sample_size_per_graph=2000,
            sampling_method=sampling_mode,
            slot_ratios=(0.34, 0.33, 0.33),
            device=device,
            verbose=False
        )
        
        # 2. Fine-tune Ranker
        model = finetune_ranker(
            model=model,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            num_epochs=num_epochs_finetune,
            lr_ranker=lr,
            lr_encoder=lr * 0.1,
            freeze_encoder=freeze_encoder,
            sample_size_per_graph=2000,
            triplet_multiplier=8,
            sampling_method=sampling_mode,
            slot_ratios=(0.34, 0.33, 0.33),
            device=device,
            verbose=False
        )
    elif training_mode == "simultaneous":
        model = train_simultaneous(
            model=model,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            num_epochs=num_epochs_simul,
            lr=lr,
            sample_size_per_graph=2000,
            triplet_multiplier=8,
            sampling_method=sampling_mode,
            device=device,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
        
    train_time = time.time() - start_t
    print(f"  Training completed in {train_time:.1f}s. Evaluating on {len(test_graphs)} unseen test graphs...")
    
    test_metrics = evaluate_model(model, test_graphs, score_combination="combined", device=device)
    test_metrics["Train_Time_s"] = train_time
    
    print(f"  -> mAP@1: {test_metrics['mAP@1']:.4f} | mAP@10: {test_metrics['mAP@10']:.4f} | mAP@50: {test_metrics['mAP@50']:.4f} | mAP@100: {test_metrics['mAP@100']:.4f}")
    print(f"  -> Reduction Ratio: {test_metrics['Reduction_Ratio']:.4f} | Within 2.5% Optimal: {test_metrics['Within_2.5%_Threshold']*100:.1f}% | Within 10% Optimal: {test_metrics['Within_10%_Threshold']*100:.1f}%")
    print(f"  -> Top-1 in True Top-50: {test_metrics['Top1_in_Top50']*100:.1f}% | Spearman Corr: {test_metrics['Spearman_All']:.4f}")
    
    return test_metrics, model


def format_results_table_and_csv(results_dict, csv_path):
    """Formats metrics dictionary into a readable formatted table and writes CSV."""
    metric_cols = [
        ("mAP@1", "mAP@1"),
        ("mAP@10", "mAP@10"),
        ("mAP@50", "mAP@50"),
        ("mAP@100", "mAP@100"),
        ("Reduction_Ratio", "RedRatio"),
        ("Within_2.5%_Threshold", "Opt<=2.5%"),
        ("Within_10%_Threshold", "Opt<=10%"),
        ("Top1_in_Top50", "InTop50"),
        ("Spearman_All", "Spearman")
    ]
    
    headers = ["Method"] + [label for _, label in metric_cols]
    col_widths = [max(len(h), 10) for h in headers]
    col_widths[0] = max(max(len(k) for k in results_dict.keys()), 35)
    
    # Format rows
    table_lines = []
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_str = "-+-".join("-" * w for w in col_widths)
    table_lines.append(header_str)
    table_lines.append(sep_str)
    
    csv_rows = [",".join(headers)]
    
    for name, metrics in results_dict.items():
        row_vals = [f"{name:<{col_widths[0]}}"]
        csv_vals = [f'"{name}"']
        for key, _ in metric_cols:
            val = metrics.get(key, 0.0)
            val_str = f"{val:.4f}"
            row_vals.append(val_str)
            csv_vals.append(val_str)
        table_lines.append(" | ".join(f"{v:<{w}}" for v, w in zip(row_vals, col_widths)))
        csv_rows.append(",".join(csv_vals))
        
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows) + "\n")
        
    return "\n".join(table_lines)


def main():
    device = "cpu"
    print("=" * 80)
    print("NETWORKS PROJECT PIPELINE V2: COMPREHENSIVE EXPERIMENT BENCHMARK")
    print("=" * 80)
    
    er_data_path = "pipeline_v2/data/er_20_graphs.pkl"
    ba_data_path = "pipeline_v2/data/ba_20_graphs.pkl"
    
    if not os.path.exists(er_data_path) or not os.path.exists(ba_data_path):
        print("Data files not found. Please wait for generate_datasets.py to finish.")
        return

    er_graphs = load_dataset(er_data_path)
    ba_graphs = load_dataset(ba_data_path)
    
    # Split 20 graphs: 14 Train (70%), 2 Validation (10%), 4 Test (20%)
    er_train, er_val, er_test = er_graphs[:14], er_graphs[14:16], er_graphs[16:]
    ba_train, ba_val, ba_test = ba_graphs[:14], ba_graphs[14:16], ba_graphs[16:]
    
    print(f"Loaded {len(er_graphs)} ER graphs and {len(ba_graphs)} BA graphs.")
    print(f"Split: Train={len(er_train)}, Val={len(er_val)}, Test={len(er_test)}")
    
    # -------------------------------------------------------------
    # 1. ER Graphs: Baselines & Pipeline Component Ablations
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 1: ERDOS-RENYI (ER) GRAPHS BENCHMARK")
    print("="*80)
    
    er_results = {}
    
    # Baselines on ER
    print("\nEvaluating Heuristic Baselines on ER Test Graphs...")
    er_baselines = evaluate_baselines_on_test(er_test)
    er_results.update(er_baselines)
    
    # Model Configuration 1: Full Proposed Pipeline
    # Two-Stage (Pre-train + Fine-tune) + Stratified Slot Sampling (Top/Mid/Bot) + Structural Features + Rich Symmetric Edges
    res_proposed, model_proposed = run_single_pipeline_configuration(
        train_graphs=er_train,
        val_graphs=er_val,
        test_graphs=er_test,
        feature_mode="structural",
        training_mode="two_stage",
        sampling_mode="slots",
        edge_method="rich_symmetric",
        num_epochs_pretrain=20,
        num_epochs_finetune=20,
        device=device,
        config_name="GNN: Two-Stage + SlotSampling + StructFeats + RichSymEdge (Proposed)"
    )
    er_results["GNN: Proposed Pipeline"] = res_proposed
    
    # Model Configuration 2: Ablation on Training Process
    # Simultaneous Training (Old simultaneous loss) vs Two-Stage
    res_simul, _ = run_single_pipeline_configuration(
        train_graphs=er_train,
        val_graphs=er_val,
        test_graphs=er_test,
        feature_mode="structural",
        training_mode="simultaneous",
        sampling_mode="slots",
        edge_method="rich_symmetric",
        num_epochs_simul=30,
        device=device,
        config_name="Ablation 1: Simultaneous Joint Training (Old method)"
    )
    er_results["GNN: Simultaneous Training"] = res_simul
    
    # Model Configuration 3: Ablation on Sampling Strategy
    # Old x^4 Sampling vs Stratified Top/Middle/Bottom Slot Sampling
    res_x4, _ = run_single_pipeline_configuration(
        train_graphs=er_train,
        val_graphs=er_val,
        test_graphs=er_test,
        feature_mode="structural",
        training_mode="two_stage",
        sampling_mode="x4",
        edge_method="rich_symmetric",
        num_epochs_pretrain=20,
        num_epochs_finetune=20,
        device=device,
        config_name="Ablation 2: x^4 / Top-biased Sampling"
    )
    er_results["GNN: x^4 Sampling"] = res_x4

    # Model Configuration 4: Ablation on Node Features
    # Pure Random Features vs Structural Features
    res_random_feat, _ = run_single_pipeline_configuration(
        train_graphs=er_train,
        val_graphs=er_val,
        test_graphs=er_test,
        feature_mode="random",
        training_mode="two_stage",
        sampling_mode="slots",
        edge_method="rich_symmetric",
        num_epochs_pretrain=20,
        num_epochs_finetune=20,
        device=device,
        config_name="Ablation 3: Pure Random Node Features"
    )
    er_results["GNN: Pure Random Features"] = res_random_feat

    # Model Configuration 5: Ablation on Edge Representation
    # Standard Concat vs Rich Symmetric
    res_concat_edge, _ = run_single_pipeline_configuration(
        train_graphs=er_train,
        val_graphs=er_val,
        test_graphs=er_test,
        feature_mode="structural",
        training_mode="two_stage",
        sampling_mode="slots",
        edge_method="concat",
        num_epochs_pretrain=20,
        num_epochs_finetune=20,
        device=device,
        config_name="Ablation 4: Direct Node Concat [h_u, h_v] Edge Vector"
    )
    er_results["GNN: Concat Edge Vector"] = res_concat_edge
    
    table_er = format_results_table_and_csv(er_results, "pipeline_v2/outputs/er_experiments_summary.csv")
    print("\n" + "="*80)
    print("ERDOS-RENYI (ER) GRAPHS - SUMMARY COMPARISON TABLE")
    print("="*80)
    print(table_er)
    
    with open("pipeline_v2/outputs/er_experiments_full.json", "w") as f:
        json.dump(er_results, f, indent=2)

    # -------------------------------------------------------------
    # 2. Barabasi-Albert (BA) Graphs: Evaluation & Comparison
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 2: BARABASI-ALBERT (BA) GRAPHS BENCHMARK")
    print("="*80)
    
    ba_results = {}
    
    # Baselines on BA
    print("\nEvaluating Heuristic Baselines on BA Test Graphs...")
    ba_baselines = evaluate_baselines_on_test(ba_test)
    ba_results.update(ba_baselines)
    
    # Proposed GNN Pipeline on BA
    res_ba_proposed, model_ba_proposed = run_single_pipeline_configuration(
        train_graphs=ba_train,
        val_graphs=ba_val,
        test_graphs=ba_test,
        feature_mode="structural",
        training_mode="two_stage",
        sampling_mode="slots",
        edge_method="rich_symmetric",
        num_epochs_pretrain=20,
        num_epochs_finetune=20,
        device=device,
        config_name="BA GNN: Two-Stage + SlotSampling + StructFeats + RichSymEdge"
    )
    ba_results["GNN: Proposed Pipeline"] = res_ba_proposed
    
    # Pure Random Features on BA
    res_ba_random, _ = run_single_pipeline_configuration(
        train_graphs=ba_train,
        val_graphs=ba_val,
        test_graphs=ba_test,
        feature_mode="random",
        training_mode="two_stage",
        sampling_mode="slots",
        edge_method="rich_symmetric",
        num_epochs_pretrain=20,
        num_epochs_finetune=20,
        device=device,
        config_name="BA GNN: Pure Random Node Features"
    )
    ba_results["GNN: Pure Random Features"] = res_ba_random
    
    table_ba = format_results_table_and_csv(ba_results, "pipeline_v2/outputs/ba_experiments_summary.csv")
    print("\n" + "="*80)
    print("BARABASI-ALBERT (BA) GRAPHS - SUMMARY COMPARISON TABLE")
    print("="*80)
    print(table_ba)
    
    with open("pipeline_v2/outputs/ba_experiments_full.json", "w") as f:
        json.dump(ba_results, f, indent=2)

    # Save models
    torch.save(model_proposed.state_dict(), "pipeline_v2/outputs/gnn_er_model.pt")
    torch.save(model_ba_proposed.state_dict(), "pipeline_v2/outputs/gnn_ba_model.pt")
    print("\nSaved trained model weights to pipeline_v2/outputs/")
    print("All experiments completed successfully!")


if __name__ == "__main__":
    main()


