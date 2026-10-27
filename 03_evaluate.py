"""
STEP 4 - Evaluation
Rank every missing edge of each TEST graph with the trained model and with simple baselines,
then compute mAP@k, Precision@k, Pearson/Spearman/Kendall (paper Section 4.3) and the summary
metrics (top-50 overlap, optimality ratio, within 2.5% / 10% of the best edge).
"GNN + exact top-K" re-scores the model's top-K edges with the exact method (cfg.exact_rerank_k).
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
from src.features import distance_matrix
from src.ground_truth import reductions_for_edges
from src.metrics import average_over_graphs, evaluate_ranking, summary_metrics
from src.plots import plot_metric_vs_k
from src.trainer import load_checkpoint, predict, to_tensors


def main(cfg=CFG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = load_all(cfg.data_dir)
    with open(os.path.join(cfg.output_dir, "training_log.json")) as f:
        split = json.load(f)["split"]
    test = [graphs[i] for i in split["test"]]
    tensors = [to_tensors(g, cfg, device) for g in test]
    model = load_checkpoint(cfg, cfg.output_dir, tensors[0], device)
    rng = np.random.default_rng(cfg.seed)

    K = cfg.exact_rerank_k
    names = ["GNN (ours)"] + ([f"GNN + exact top-{K}"] if K else []) + list(BASELINES)
    per_method = {name: [] for name in names}
    summary = {name: [] for name in names}
    top10_rows = []
    for g, t in zip(test, tensors):
        final, _, _ = predict(model, t, cfg)
        scored = {"GNN (ours)": final}
        if K:
            short = np.argsort(-final, kind="stable")[:K]
            exact = reductions_for_edges(g["G"], g["missing_edges"][short], cfg.normalize_betweenness,
                                         D=distance_matrix(g["G"]))
            hybrid = np.argsort(np.argsort(final, kind="stable"), kind="stable").astype(float)
            hybrid[short] = len(final) + exact / exact.max()
            scored[f"GNN + exact top-{K}"] = hybrid
        for name, fn in BASELINES.items():
            scored[name] = fn(g["G"], g["missing_edges"], rng)
        for name, s in scored.items():
            per_method[name].append(evaluate_ranking(s, g["reductions"], cfg.k_values))
            summary[name].append(summary_metrics(s, g["reductions"]))

        # side-by-side top 10 for inspection
        pred_top = np.argsort(-final, kind="stable")[:10]
        true_top = np.argsort(-g["reductions"], kind="stable")[:10]
        for r in range(10):
            top10_rows.append([g["graph_id"], r + 1,
                               tuple(g["missing_edges"][true_top[r]]), f"{g['reductions'][true_top[r]]:.4e}",
                               tuple(g["missing_edges"][pred_top[r]]), f"{g['reductions'][pred_top[r]]:.4e}"])

    results = {name: average_over_graphs(v) for name, v in per_method.items()}
    summ = {name: {k: float(np.nanmean([d[k] for d in v])) for k in v[0]} for name, v in summary.items()}

    # ---- save ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "test_metrics.json"), "w") as f:
        json.dump({"test_graphs": split["test"], "results": results, "summary": summ}, f, indent=2)
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
    print(f"{'method':28s} {'mAP@10':>7s} {'mAP@50':>7s} {'top-50':>7s} {'opt.ratio':>9s} {'<=10% of best':>13s}")
    for name, v in summ.items():
        print(f"{name:28s} {v['map10']:7.3f} {v['map50']:7.3f} {v['top50_overlap']:7.3f} "
              f"{v['opt_ratio']:9.3f} {v['within_10pct']:13.3f}")


if __name__ == "__main__":
    main()
