"""Train-on-200, test-on-500 nodes: GNN alone and GNN shortlist + exact, with timings.
    python experiments/size_eval.py <name> '<json overrides>' seeds"""
import dataclasses, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from config import Config, legacy_config
from src.baselines import BASELINES
from src.data_io import load_all, load_meta
from src.features import distance_matrix
from src.metrics import summary_metrics
from src.trainer import build_model, predict, to_tensors
from final_eval import hybrid, mean, rank_score

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
name, ov, seeds = sys.argv[1], json.loads(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")]
cfg = legacy_config(**ov)
graphs = load_all("data/ba500_test")
gt_secs = [m.get("seconds", np.nan) for m in load_meta("data/ba500_test")["graphs"]]
t0 = time.time(); tt = [to_tensors(g, cfg) for g in graphs]; prep = (time.time() - t0) / len(graphs)
models = []
for s in seeds:
    m = build_model(cfg, tt[0]); m.load_state_dict(torch.load(os.path.join(HERE, "models", f"{name}__s{s}.pt"))); models.append(m)
out, times = {}, {"features_s": prep}
singles, ens, tinf = [], [], []
for g, t in zip(graphs, tt):
    fs = []
    for m in models:
        a = time.time(); f, _, _ = predict(m, t, cfg); tinf.append(time.time() - a)
        fs.append(f); singles.append(summary_metrics(f, g["reductions"]))
    ens.append(np.mean([rank_score(f) for f in fs], 0))
times["gnn_inference_s"] = float(np.mean(tinf))
out["GNN single model (mean of seeds)"] = mean(singles)
out["GNN ensemble"] = mean([summary_metrics(e, g["reductions"]) for e, g in zip(ens, graphs)])
rng = np.random.default_rng(0)
heur = [BASELINES["closeness_gap_x_distance"](g["G"], g["missing_edges"], rng) for g in graphs]
out["heuristic (closeness gap x distance)"] = mean([summary_metrics(h, g["reductions"]) for h, g in zip(heur, graphs)])
for K in [100, 200, 500, 1000]:
    for label, src in [("GNN", ens), ("heuristic", heur)]:
        res, secs = [], []
        for s, g in zip(src, graphs):
            h, sec = hybrid(s, g, K, distance_matrix(g["G"])); res.append(summary_metrics(h, g["reductions"])); secs.append(sec)
        out[f"{label} shortlist K={K} + exact"] = mean(res); times[f"exact_K={K}_s"] = float(np.mean(secs))
json.dump({"results": out, "times": times}, open(os.path.join(HERE, f"size500_{name}.json"), "w"), indent=1)
print(f"{'method':40s} {'mAP@10':>7s} {'mAP@50':>7s} {'top50':>6s} {'opt':>6s}")
for k, v in out.items():
    print(f"{k:40s} {v['map10']:7.3f} {v['map50']:7.3f} {v['top50_overlap']:6.3f} {v['opt_ratio']:6.3f}")
print(times)
