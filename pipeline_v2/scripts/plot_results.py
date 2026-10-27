import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_benchmark_bars(json_path, title, save_path):
    """Plots a horizontal bar chart comparing mAP@10, Reduction Ratio, and Top1 in Top50."""
    with open(json_path, "r") as f:
        data = json.load(f)
        
    methods = list(data.keys())
    # Sort by mAP@10 ascending
    methods = sorted(methods, key=lambda m: data[m].get("mAP@10", 0.0))
    
    map10 = [data[m].get("mAP@10", 0.0) for m in methods]
    reduct_ratio = [data[m].get("Reduction_Ratio", 0.0) for m in methods]
    in_top50 = [data[m].get("Top1_in_Top50", 0.0) for m in methods]
    
    plt.figure(figsize=(10, 6))
    y = np.arange(len(methods))
    bar_width = 0.25
    
    plt.barh(y - bar_width, map10, height=bar_width, label="mAP@10", color="#4285F4")
    plt.barh(y, reduct_ratio, height=bar_width, label="Reduction Ratio", color="#34A853")
    plt.barh(y + bar_width, in_top50, height=bar_width, label="Top-1 in Top-50", color="#FBBC05")
    
    # Shorten method labels if too long
    clean_labels = [m.replace("Baseline: ", "").replace("GNN: ", "") for m in methods]
    plt.yticks(y, clean_labels, fontsize=9)
    plt.xlabel("Score (0.0 to 1.0)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved bar plot to {save_path}")


def plot_map_vs_k(full_json_path, title, save_path):
    """Plots mAP vs k curves for key models and baselines."""
    with open(full_json_path, "r") as f:
        data = json.load(f)
        
    k_vals = [1, 5, 10, 20, 50, 100]
    
    plt.figure(figsize=(8, 5))
    
    selected_methods = [
        "GNN: Proposed Pipeline",
        "GNN: Simultaneous Training",
        "GNN: Pure Random Features",
        "Baseline: closeness_gap_x_distance",
        "Baseline: distance",
        "Baseline: degree_product",
        "Baseline: random"
    ]
    
    colors = ["#1A73E8", "#D93025", "#9334E6", "#1E8E3E", "#F9AB00", "#E37400", "#70757A"]
    markers = ["o", "s", "^", "D", "v", "x", "*"]
    
    for i, method in enumerate(selected_methods):
        if method in data:
            map_scores = [data[method].get(f"mAP@{k}", 0.0) for k in k_vals]
            plt.plot(k_vals, map_scores, label=method, color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=2)
            
    plt.xlabel("k (Top-k Cutoff)", fontsize=11)
    plt.ylabel("mAP@k", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved mAP vs k plot to {save_path}")


def main():
    os.makedirs("pipeline_v2/plots", exist_ok=True)
    
    er_json = "pipeline_v2/outputs/er_experiments_full.json"
    if os.path.exists(er_json):
        plot_benchmark_bars(er_json, "Erdős–Rényi (ER) Graphs: GNN vs Baselines & Ablations", "pipeline_v2/plots/er_benchmark_bars.png")
        plot_map_vs_k(er_json, "Erdős–Rényi (ER) Graphs: mAP@k across Methods", "pipeline_v2/plots/er_map_vs_k.png")
        
    ba_json = "pipeline_v2/outputs/ba_experiments_full.json"
    if os.path.exists(ba_json):
        plot_benchmark_bars(ba_json, "Barabási–Albert (BA) Graphs: GNN vs Baselines", "pipeline_v2/plots/ba_benchmark_bars.png")
        plot_map_vs_k(ba_json, "Barabási–Albert (BA) Graphs: mAP@k across Methods", "pipeline_v2/plots/ba_map_vs_k.png")


if __name__ == "__main__":
    main()

