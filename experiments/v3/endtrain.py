"""Accuracy at the end of training: training graphs, validation graphs and test graphs, v2 vs v3."""
import json, os, sys
import numpy as np, torch, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict as predict2, build_model
from src.ground_truth import make_labels
from gnn3 import Net, tensors, predict as predict3
from cls_compare import by_rank, cls_metrics
torch.set_num_threads(2)
OUT = os.path.join(ROOT, "experiments", "v3", "results")
cfg = Config(); opt3 = json.load(open(os.path.join(OUT, "N13_final_5seeds.json")))["opt"]
base = base_records()
gs = load_all(os.path.join(ROOT, "data"))
for nm, ids in (("training graphs 0-13", range(14)), ("validation graphs 14-15", range(14, 16)), ("test graphs 16-19", range(16, 20))):
    recs = [base[i] for i in ids]
    t2 = [to_tensors(gs[i], cfg) for i in ids]; t3 = [tensors(r, opt3["edge_blocks"]) for r in recs]
    out = {"v2": [], "v3": []}
    for s in range(5):
        m2 = build_model(cfg, t2[0]); m2.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/E31_batch500__s{s}.pt")))
        m3 = Net(t3[0]["x"].shape[1], t3[0]["ef"].shape[1], hidden=64); m3.load_state_dict(torch.load(os.path.join(OUT, f"N13_final_5seeds__s{s}.pt")))
        for a, b, r in zip(t2, t3, recs):
            true = make_labels(r["red"], [0.10, 0.02])
            f2, _, p2 = predict2(m2, a, cfg); f3 = predict3(m3, b)
            e2, e3 = eval_one(f2, r["red"]), eval_one(f3, r["red"])
            out["v2"].append([e2["map10"], e2["map50"], cls_metrics(p2.argmax(1), true)["bal_acc"], cls_metrics(by_rank(f2, len(true)), true)["bal_acc"]])
            out["v3"].append([e3["map10"], e3["map50"], np.nan, cls_metrics(by_rank(f3, len(true)), true)["bal_acc"]])
    for k, v in out.items():
        v = np.nanmean(np.array(v, dtype=float), 0)
        print(f"{nm:26s} {k}: mAP@10 {v[0]:.3f} mAP@50 {v[1]:.3f} | bal.acc classifier head {v[2]:.3f} | bal.acc score cut {v[3]:.3f}", flush=True)
