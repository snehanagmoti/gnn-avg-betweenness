"""mAP@5 (and @10, @50 for reference) for version 2 and version 3 on the 50 new BA-200 test graphs."""
import json, os, sys
import numpy as np, torch, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict as v2_predict, build_model
import gnn3
from ndcg import rescored

def maps(scores, recs):
    out = {}
    for k in (5, 10, 50):
        out[f"map{k}"] = float(np.mean([average_precision_at_k(np.argsort(-s, kind="stable"), r["red"], k) for s, r in zip(scores, recs)]))
        out[f"ex100_map{k}"] = float(np.mean([average_precision_at_k(np.argsort(-rescored(s, r["red"]), kind="stable"), r["red"], k) for s, r in zip(scores, recs)]))
    return out

big = extra_records(50, start_seed=5000, tag="test_ba200")
cfg = Config(); gl = []
for i, r in enumerate(big):
    G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
    gl.append({"G": G, "missing_edges": r["E"], "reductions": r["red"], "edges": r["edges"], "graph_id": 5000 + i})
tt = [to_tensors(g, cfg) for g in gl]
res = {}
runs = []
for s in range(5):
    m = build_model(cfg, tt[0]); m.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/E31_batch500__s{s}.pt")))
    runs.append(maps([v2_predict(m, t, cfg)[0] for t in tt], big))
res["v2"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
T = [gnn3.tensors(r, ["xe", "xp", "xh"]) for r in big]
runs = []
for s in range(5):
    m = gnn3.Net(T[0]["x"].shape[1], T[0]["ef"].shape[1], hidden=64)
    m.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/v3/results/N13_final_5seeds__s{s}.pt")))
    runs.append(maps([gnn3.predict(m, t) for t in T], big))
res["v3"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
res["v3_sd_map5"] = float(np.std([r["map5"] for r in runs]))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "experiments/v3/results/map5_50.json"), "w"), indent=1)
