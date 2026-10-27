"""
Final evaluation of a trained configuration (models saved by lab.py):
  * single models (mean over seeds)
  * ensemble  : average the rank of every edge over the seed models
  * shortlist + exact (hybrid): the model's top-K edges are re-scored with the exact distance
    method (K x O(n^2) work instead of O(n^4) for all candidates), the rest keep the model order
  * the same hybrid with the best heuristic as the shortlist, for a fair comparison
    python experiments/final_eval.py <name> '<json overrides>' seeds K1,K2,...
"""
import dataclasses, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from config import Config, legacy_config
from src.baselines import BASELINES
from src.data_io import load_all
from src.features import distance_matrix
from src.ground_truth import reductions_for_edges
from src.metrics import summary_metrics
from src.trainer import build_model, predict, to_tensors

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))


def rank_score(x):
    """Higher = better; rank-based so models on different scales can be averaged."""
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def hybrid(scores, g, K, D):
    order = np.argsort(-scores, kind="stable")
    short = order[:K]
    t0 = time.time()
    exact = reductions_for_edges(g["G"], g["missing_edges"][short], D=D)
    secs = time.time() - t0
    new = rank_score(scores)
    new[short] = len(scores) + exact / exact.max()       # shortlist first, by exact value
    return new, secs


def mean(ds):
    return {k: float(np.nanmean([d[k] for d in ds])) for k in ds[0]}


def main(name, overrides, seeds, Ks, data_dir="data"):
    cfg = legacy_config(data_dir=data_dir, **overrides)
    graphs = load_all(data_dir)
    a, b, _ = cfg.split
    test = graphs[a + b:]
    tt = [to_tensors(g, cfg) for g in test]
    models = []
    for s in seeds:
        m = build_model(cfg, tt[0])
        m.load_state_dict(torch.load(os.path.join(HERE, "models", f"{name}__s{s}.pt")))
        models.append(m)
    out = {}
    single = [[], []]
    ens_scores = []
    t_inf = []
    for g, t in zip(test, tt):
        per_model = []
        for m in models:
            t0 = time.time()
            f, _, _ = predict(m, t, cfg)
            t_inf.append(time.time() - t0)
            per_model.append(f)
            single[0].append(summary_metrics(f, g["reductions"]))
        ens_scores.append(np.mean([rank_score(f) for f in per_model], axis=0))
    out["single model (mean of seeds)"] = mean(single[0])
    out["ensemble"] = mean([summary_metrics(s, g["reductions"]) for s, g in zip(ens_scores, test)])
    rng = np.random.default_rng(0)
    heur = [BASELINES["closeness_gap_x_distance"](g["G"], g["missing_edges"], rng) for g in test]
    out["heuristic (closeness gap x distance)"] = mean([summary_metrics(h, g["reductions"]) for h, g in zip(heur, test)])
    times = {"gnn_inference_s": float(np.mean(t_inf))}
    for K in Ks:
        for label, src in [("ensemble", ens_scores), ("heuristic", heur)]:
            res, secs = [], []
            for s, g in zip(src, test):
                D = distance_matrix(g["G"])
                h, sec = hybrid(s, g, K, D)
                res.append(summary_metrics(h, g["reductions"]))
                secs.append(sec)
            out[f"{label} shortlist K={K} + exact"] = mean(res)
            times[f"exact_on_K={K}_s"] = float(np.mean(secs))
    with open(os.path.join(HERE, f"final_{name}.json"), "w") as f:
        json.dump({"results": out, "times": times}, f, indent=1)
    print(f"{'method':42s} {'mAP@10':>7s} {'mAP@50':>7s} {'top50':>6s} {'opt':>6s} {'sp50':>6s}")
    for k, v in out.items():
        print(f"{k:42s} {v['map10']:7.3f} {v['map50']:7.3f} {v['top50_overlap']:6.3f} {v['opt_ratio']:6.3f} {v['spearman50']:6.3f}")
    print(times)


if __name__ == "__main__":
    main(sys.argv[1], json.loads(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")],
         [int(x) for x in sys.argv[4].split(",")])
