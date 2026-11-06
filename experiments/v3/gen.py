"""
Generalisation of a trained version-3 GNN (saved by gnn3.py) without retraining:
bigger BA graphs and other graph types; plus graph-level out-of-distribution (anomaly) scores.
    python gen.py <trained name>
"""
import json, os, sys, time
import numpy as np, torch
from sklearn.ensemble import IsolationForest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from gnn3 import Net, tensors, predict

OUT = os.path.join(ROOT, "experiments", "v3", "results")
name = sys.argv[1]
info = json.load(open(os.path.join(OUT, f"{name}.json")))
opt = info["opt"]
SETS = {
    "BA-200 (50 new)": lambda: extra_records(50, start_seed=5000, tag="test_ba200"),
    "BA-500 (6)": lambda: extra_records(6, n=500, start_seed=7000, tag="test_ba500"),
    "ER-200 p=0.04 (10)": lambda: extra_records(10, gtype="er", start_seed=7100, tag="test_er200"),
    "2-tree-200 (10)": lambda: extra_records(10, gtype="ktree", start_seed=7200, tag="test_ktree200"),
    "random tree-200 (10)": lambda: extra_records(10, gtype="random_tree", start_seed=7300, tag="test_tree200"),
}
blocks = ["xn", "xe", "xp"]
base = base_records()
rng = np.random.default_rng(0)
Xtr = np.concatenate([features(r, blocks)[rng.choice(len(r["E"]), 2000, replace=False)] for r in base[:14]])
iso = IsolationForest(n_estimators=200, max_samples=512, random_state=0, n_jobs=2).fit(Xtr)
ref = -iso.score_samples(Xtr).mean()
rows = {}
for set_name, fn in SETS.items():
    t0 = time.time()
    recs = fn()
    ts = [tensors(r, opt["edge_blocks"]) for r in recs]
    runs, times = [], []
    for s in opt["seeds"]:
        model = Net(ts[0]["x"].shape[1], ts[0]["ef"].shape[1], hidden=opt.get("hidden", 32), no_gnn=opt["no_gnn"])
        model.load_state_dict(torch.load(os.path.join(OUT, f"{name}__s{s}.pt")))
        t1 = time.time(); sc = [predict(model, t) for t in ts]; times.append((time.time() - t1) / len(ts))
        runs.append(eval_many(sc, recs))
    m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    m["ood"] = float(np.mean([-iso.score_samples(features(r, blocks)).mean() for r in recs]) - ref)
    m["sec_per_graph"] = float(np.mean(times))
    rows[set_name] = m
    print(fmt(set_name, m), f"| OOD {m['ood']:+.3f} | {m['sec_per_graph']:.2f}s/graph | {time.time()-t0:.0f}s", flush=True)
json.dump(rows, open(os.path.join(OUT, f"gen_{name}.json"), "w"), indent=1)
