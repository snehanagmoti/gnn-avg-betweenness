"""Classifier-style accuracy: version 2's classifier head vs version 3's scores cut into the same 3 classes."""
import json, os, sys
import numpy as np, torch, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict as predict2, build_model
from src.ground_truth import make_labels
from gnn3 import Net, tensors, predict as predict3
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2)
OUT = os.path.join(ROOT, "experiments", "v3", "results")
cfg = Config()
def cls_metrics(pred, true):
    bal = np.mean([(pred[true == c] == c).mean() for c in range(3)])
    out = {"bal_acc": bal, "acc": (pred == true).mean()}
    for c, nm in ((2, "top2"), (1, "2-10")):
        tp = ((pred == c) & (true == c)).sum()
        out[f"{nm}_recall"] = tp / max((true == c).sum(), 1)
        out[f"{nm}_precision"] = tp / max((pred == c).sum(), 1)
    good_t, good_p = true >= 1, pred >= 1
    out["good_recall"] = (good_t & good_p).sum() / good_t.sum()
    out["good_precision"] = (good_t & good_p).sum() / good_p.sum()
    return out
def by_rank(score, E):
    r = np.argsort(np.argsort(-score, kind="stable"), kind="stable")
    return (r < round(0.10 * E)).astype(int) + (r < round(0.02 * E)).astype(int)
sets = {"4 test graphs": base_records()[16:20], "50 new test graphs": extra_records(50, start_seed=5000, tag="test_ba200")}
opt3 = json.load(open(os.path.join(OUT, "N13_final_5seeds.json")))["opt"]
for sname, recs in sets.items():
    gl = []
    for i, r in enumerate(recs):
        G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
        gl.append({"G": G, "missing_edges": r["E"], "reductions": r["red"], "edges": r["edges"], "graph_id": i})
    t2 = [to_tensors(g, cfg) for g in gl]
    t3 = [tensors(r, opt3["edge_blocks"]) for r in recs]
    res = {"v2 classifier head": [], "v2 final score cut by rank": [], "v3 score cut by rank": []}
    auc = {"v2": [], "v3": []}
    for s in range(5):
        m2 = build_model(cfg, t2[0]); m2.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/E31_batch500__s{s}.pt")))
        m3 = Net(t3[0]["x"].shape[1], t3[0]["ef"].shape[1], hidden=64); m3.load_state_dict(torch.load(os.path.join(OUT, f"N13_final_5seeds__s{s}.pt")))
        for g, a, b, r in zip(gl, t2, t3, recs):
            true = make_labels(r["red"], [0.10, 0.02])
            final, raw, probs = predict2(m2, a, cfg)
            sc3 = predict3(m3, b)
            res["v2 classifier head"].append(cls_metrics(probs.argmax(1), true))
            res["v2 final score cut by rank"].append(cls_metrics(by_rank(final, len(true)), true))
            res["v3 score cut by rank"].append(cls_metrics(by_rank(sc3, len(true)), true))
            auc["v2"].append(roc_auc_score(true == 2, probs[:, 2])); auc["v3"].append(roc_auc_score(true == 2, sc3))
    print(f"== {sname}")
    for k, v in res.items():
        m = {kk: np.mean([x[kk] for x in v]) for kk in v[0]}
        print(f"{k:28s} " + " ".join(f"{kk} {vv:.3f}" for kk, vv in m.items()))
    print(f"AUC top-2% vs rest: v2 {np.mean(auc['v2']):.4f}  v3 {np.mean(auc['v3']):.4f}")
