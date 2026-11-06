"""Wall-clock time per graph for each part of the pipeline (BA graphs, m = 3)."""
import json, os, sys, time
import numpy as np, torch, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from common import level_hist_block
from gnn3 import Net, tensors, predict
from src.fastgain import exact_gains_fast, pair_features
torch.set_num_threads(1)
OUT = os.path.join(ROOT, "experiments", "v3", "results")
opt = json.load(open(os.path.join(OUT, "N13_final_5seeds.json")))["opt"]
rows = {}
for n in (200, 500, 1000):
    G = nx.barabasi_albert_graph(n, 3, seed=99)
    T = {}
    t = time.time(); D = distance_matrix(G); T["distances"] = time.time() - t
    E = missing_edges_of(G); Di = D.astype(np.int64)
    exact_gains_fast(Di, E[:5])
    t = time.time(); g = exact_gains_fast(Di, E); T["exact, ALL edges (compiled)"] = time.time() - t
    t = time.time(); nf = node_features(G, NODE_FEATS, D=D); xe = edge_features(G, E, EDGE_FEATS, D=D); T["v2 features"] = time.time() - t
    t = time.time(); pf = pair_features(Di, E) if n <= 500 else None; T["affected-set features (numpy)"] = time.time() - t if n <= 500 else float("nan")
    r = build_record(G) if n <= 500 else None
    if r is not None:
        t = time.time(); r["xh"] = level_hist_block(r); T["level-histogram features"] = time.time() - t
        ts = tensors(r, opt["edge_blocks"])
        model = Net(ts["x"].shape[1], ts["ef"].shape[1], hidden=64); model.load_state_dict(torch.load(os.path.join(OUT, "N13_final_5seeds__s0.pt")))
        t = time.time(); sc = predict(model, ts); T["GNN forward, all edges"] = time.time() - t
        top = np.argsort(-sc)[:100]
        t = time.time(); exact_gains_fast(Di, E[top]); T["exact on GNN top 100"] = time.time() - t
    rows[n] = T
    print(n, {k: round(v, 3) for k, v in T.items()}, flush=True)
json.dump({str(k): v for k, v in rows.items()}, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
