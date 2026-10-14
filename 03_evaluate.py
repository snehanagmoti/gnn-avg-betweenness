"""
STEP 4 - Evaluation
Rank every missing edge of each TEST graph with the trained model and with simple baselines,
then compute mAP@k, Precision@k and Pearson/Spearman/Kendall (paper Section 4.3).
Results go to outputs/test_metrics.json, outputs/test_metrics.csv and plots.

Run:  python 03_evaluate.py     (after 02_train.py)
"""
import csv
import json
import os

import numpy as np
import torch

from config import CFG
from src.baselines import BASELINES
from src.data_io import load_all
from src.metrics import average_over_graphs, evaluate_ranking
from src.plots import plot_metric_vs_k
from src.trainer import load_checkpoint, predict, to_tensors


def main(cfg=CFG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = load_all(cfg.data_dir)
    with open(os.path.join(cfg.output_dir, "training_log.json")) as f:
        split = json.load(f)["split"]
    test = [graphs[i] for i in split["test"]]
    model = load_checkpoint(cfg, cfg.output_dir, device)
    rng = np.random.default_rng(cfg.seed)

    per_method = {"GNN (ours)": []}
    per_method.update({name: [] for name in BASELINES})
    top10_rows = []
    for g in test:
        final, _, _ = predict(model, to_tensors(g, device), cfg.ranking_mode)
        per_method["GNN (ours)"].append(evaluate_ranking(final, g["reductions"], cfg.k_values))
        for name, fn in BASELINES.items():
            s = fn(g["G"], g["missing_edges"], rng)
            per_method[name].append(evaluate_ranking(s, g["reductions"], cfg.k_values))

        # side-by-side top 10 for inspection
        pred_top = np.argsort(-final, kind="stable")[:10]
        true_top = np.argsort(-g["reductions"], kind="stable")[:10]
        for r in range(10):
            top10_rows.append([g["graph_id"], r + 1,
                               tuple(g["missing_edges"][true_top[r]]), f"{g['reductions'][true_top[r]]:.4e}",
                               tuple(g["missing_edges"][pred_top[r]]), f"{g['reductions'][pred_top[r]]:.4e}"])

    results = {name: average_over_graphs(v) for name, v in per_method.items()}

    # ---- save ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "test_metrics.json"), "w") as f:
        json.dump({"test_graphs": split["test"], "results": results}, f, indent=2)
    metrics = ["map", "precision", "pearson", "spearman", "kendall"]
    with open(os.path.join(cfg.output_dir, "test_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k"] + metrics)
        for name, res in results.items():
            for k, vals in res.items():
                w.writerow([name, k] + [f"{vals.get(m, float('nan')):.4f}" for m in metrics])
    with open(os.path.join(cfg.output_dir, "test_top10_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "rank", "true_edge", "true_reduction", "predicted_edge", "its_true_reduction"])
        w.writerows(top10_rows)

    plot_metric_vs_k({n: {k: v[k]["map"] for k in v} for n, v in results.items()}, "map",
                     cfg.output_dir, "mAP@k")
    plot_metric_vs_k({n: {k: v[k]["spearman"] for k in v if "spearman" in v[k]} for n, v in results.items()},
                     "spearman", cfg.output_dir, "Spearman (true top-k)")

    # ---- print summary ----
    print(f"Test graphs: {split['test']}")
    print(f"{'method':28s} {'mAP@1':>7s} {'mAP@10':>7s} {'mAP@100':>8s} {'P@10':>6s} {'Spearman@100':>13s}")
    for name, res in results.items():
        print(f"{name:28s} {res[1]['map']:7.3f} {res[10]['map']:7.3f} {res[100]['map']:8.3f} "
              f"{res[10]['precision']:6.2f} {res[100].get('spearman', float('nan')):13.3f}")


if __name__ == "__main__":
    main()
