import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict, build_model
from src.metrics import relevant_set, average_precision_at_k, correlations_at_k
from scipy.stats import rankdata
torch.set_num_threads(1)
cfg = Config(data_dir=os.path.join(ROOT, "data")); gs = load_all(cfg.data_dir); g = gs[16]
t = to_tensors(g, cfg)
model = build_model(cfg, t); model.load_state_dict(torch.load(os.path.join(ROOT, "experiments/models/E31_batch500__s0.pt")))
final, raw, probs = predict(model, t, cfg)
red = g["reductions"]; E = g["missing_edges"]
order = np.argsort(-final, kind="stable"); rank = rankdata(-red, method="min").astype(int)
rel10 = relevant_set(red, 10); rel50 = relevant_set(red, 50)
print("distinct values", len(np.unique(red)), "| relevant set sizes: top10", rel10.sum(), "top50", rel50.sum())
print("10th best value", np.sort(red)[::-1][9], "11th", np.sort(red)[::-1][10])
hits = 0
for j, i in enumerate(order[:10], 1):
    r = rel10[i]; hits += r
    print(f"pos {j:2d} edge {tuple(E[i])} true rank {rank[i]:4d} rel {int(r)} P@{j}={hits/j:.3f} class {probs[i].argmax()}")
print("AP@10", average_precision_at_k(order, red, 10), "AP@50", average_precision_at_k(order, red, 50))
print("top50 overlap", rel50[order[:50]].mean(), "opt", red[order[0]]/red.max(), "best10", red[order[:10]].max()/red.max())
best = red.max(); print("within10%:", (red >= 0.9*best).sum(), "edges in graph;", "within2.5%:", (red >= 0.975*best).sum())
print("spearman50", correlations_at_k(final, red, 50)["spearman"], "on raw ranker score", correlations_at_k(raw, red, 50)["spearman"])
pos = np.empty(len(order), int); pos[order] = np.arange(len(order))
print("positions of true top10 in model list:", sorted((pos[rel10] + 1).tolist()))
for k in (5, 10, 20, 30, 50):
    print("AP@%d" % k, round(average_precision_at_k(order, red, k), 3), "relevant set", relevant_set(red, k).sum())
