"""
All plots.
"""
import os

import matplotlib
matplotlib.use("Agg")                 # save to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from src.data_io import load_all


def plot_reduction_curves(data_dir, output_dir):
    """Fig. 2-style plot: every missing edge sorted by reduction, one line per graph,
    plus the average curve (rank-wise mean over graphs)."""
    os.makedirs(output_dir, exist_ok=True)
    graphs = load_all(data_dir)
    curves = [np.sort(g["reductions"])[::-1] for g in graphs]

    plt.figure(figsize=(8, 4.5))
    for g, c in zip(graphs, curves):
        plt.plot(c, color="tab:blue", alpha=0.25, lw=1)
    min_len = min(len(c) for c in curves)
    plt.plot(np.mean([c[:min_len] for c in curves], axis=0), color="tab:red", lw=2,
             label=f"average of {len(curves)} graphs")
    plt.plot([], [], color="tab:blue", alpha=0.5, label="individual graphs")
    plt.xlabel("Missing edges (sorted by reduction)")
    plt.ylabel("Reduction in avg betweenness")
    plt.title("Contribution of each missing edge (BA graphs)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "reduction_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_training_history(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="train total loss")
    ax[0].plot(epochs, history["train_cls_loss"], label="classification loss", alpha=0.7)
    ax[0].plot(epochs, history["train_rank_loss"], label="ranking loss", alpha=0.7)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Training loss")
    ax[1].plot(epochs, history["val_map10"], label="val mAP@10", color="tab:green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mAP@10"); ax[1].set_ylim(0, 1)
    ax[1].legend(); ax[1].set_title("Validation mAP@10")
    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_metric_vs_k(results, metric, output_dir, ylabel=None):
    """results: {method_name: {k: value}}"""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6.5, 4))
    for name, vals in results.items():
        ks = sorted(vals)
        plt.plot(ks, [vals[k] for k in ks], marker="o", label=name)
    plt.xlabel("k"); plt.ylabel(ylabel or metric)
    plt.title(f"{ylabel or metric} on test graphs")
    plt.ylim(0, 1.02) if metric == "map" else None
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"test_{metric}_vs_k.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
