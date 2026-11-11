"""NDCG@k for version 2 and version 3 on the 50 new BA-200 test graphs (and with exact top-100 check)."""
import json, os, sys
import numpy as np, torch, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict as v2_predict, build_model
import gnn3

def ndcg(score, red, k):
    order = np.argsort(-score, kind="stable")[:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (red[order] * disc).sum()
    idcg = (np.sort(red)[::-1][:k] * disc).sum()
    return dcg / idcg

def rescored(score, red, K=100):
    order = np.argsort(-score, kind="stable")
    s2 = np.empty(len(score)); s2[order] = -np.arange(len(score), dtype=float)
    top = order[:K]; s2[top] = 1e6 + red[top] / red.max()
    return s2

def summarise(score_lists, recs):
    out = {}
    for k in (10, 50):
        out[f"ndcg{k}"] = float(np.mean([ndcg(s, r["red"], k) for s, r in zip(score_lists, recs)]))
        out[f"ex100_ndcg{k}"] = float(np.mean([ndcg(rescored(s, r["red"]), r["red"], k) for s, r in zip(score_lists, recs)]))
    return out

if __name__ == "__main__":
    big = extra_records(50, start_seed=5000, tag="test_ba200")
    res = {}
    cfg = Config(); gl = []
    for i, r in enumerate(big):
        G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
        gl.append({"G": G, "missing_edges": r["E"], "reductions": r["red"], "edges": r["edges"], "graph_id": 5000 + i})
    tt = [to_tensors(g, cfg) for g in gl]
    runs = []
    for s in range(5):
        m = build_model(cfg, tt[0]); m.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/E31_batch500__s{s}.pt")))
        runs.append(summarise([v2_predict(m, t, cfg)[0] for t in tt], big))
    res["v2"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    T = [gnn3.tensors(r, ["xe", "xp", "xh"]) for r in big]
    runs = []
    for s in range(5):
        m = gnn3.Net(T[0]["x"].shape[1], T[0]["ef"].shape[1], hidden=64)
        m.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/v3/results/N13_final_5seeds__s{s}.pt")))
        runs.append(summarise([gnn3.predict(m, t) for t in T], big))
    res["v3"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    res["rule_one_sided"] = summarise([r["xp"][:, PAIR_FEATS.index("one_sided")] for r in big], big)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "experiments/v3/results/ndcg50.json"), "w"), indent=1)
