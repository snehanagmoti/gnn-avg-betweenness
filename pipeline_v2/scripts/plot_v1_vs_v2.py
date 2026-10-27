import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def create_v1_vs_v2_comparison_chart(output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Architecture & Metric Comparison (ER Graphs)
    metrics = ["mAP@10", "Spearman", "Opt <= 10%", "Top-1 in Top-50"]
    v1_scores = [0.4339, 0.2847, 0.5000, 0.5000] # From V1 test_metrics.csv & report
    v2_scores = [0.2038, 0.4863, 0.5000, 0.7500] # V2 ER proposed
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, v1_scores, width, label="Pipeline V1 (Simultaneous + 10 Graphs)", color="#EA4335", alpha=0.85)
    rects2 = axes[0].bar(x + width/2, v2_scores, width, label="Pipeline V2 (Two-Stage + 20 Graphs)", color="#34A853", alpha=0.85)
    
    axes[0].set_ylabel("Score / Rate", fontsize=11)
    axes[0].set_title("ER Graphs: V1 vs. V2 Metrics", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=10)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    # 2. Generalization to Barabási-Albert (BA) Graphs in V2
    ba_metrics = ["mAP@1", "mAP@10", "mAP@50", "Reduction Ratio", "Top-1 in Top-50", "Spearman"]
    ba_v2_scores = [0.7500, 0.3782, 0.4678, 0.9639, 1.0000, 0.7655]
    
    x2 = np.arange(len(ba_metrics))
    rects_ba = axes[1].bar(x2, ba_v2_scores, 0.5, label="Pipeline V2 on BA Graphs", color="#4285F4", alpha=0.85)
    
    axes[1].set_ylabel("Score / Rate", fontsize=11)
    axes[1].set_title("Pipeline V2 Performance on Scale-Free BA Graphs", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(ba_metrics, rotation=25, ha="right", fontsize=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    
    for rect in rects_ba:
        h = rect.get_height()
        axes[1].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
                         
    plt.suptitle("Pipeline Evolution: V1 (Joint Loss, 10 Graphs) vs. V2 (Two-Stage Pre-training, 20 Graphs)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved V1 vs V2 plot to {output_path}")

if __name__ == "__main__":
    os.makedirs("pipeline_v2/plots", exist_ok=True)
    create_v1_vs_v2_comparison_chart("pipeline_v2/plots/v1_vs_v2_comparison.png")

