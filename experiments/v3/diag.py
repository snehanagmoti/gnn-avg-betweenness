"""Which single clue still finds the best edges on big / real graphs? Each feature column used alone as a ranking,
for sampled (R landmarks) and exact (all nodes) affected-set features."""
import os, sys, time
import numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scale import GraphPrep
from real_data import load
from src.fastgain import all_gains, pair_features

COLS = ["dist", "cn", "jacc", "aa", "clo_gap", "clo_min", "clo_max", "deg_min", "deg_max",
        "aff_min", "aff_max", "aff_prod", "lsum_min", "lsum_max", "one_sided", "ub",
        "hmin1", "hmin2", "hmin3", "hmin4", "hmax1", "hmax2", "hmax3", "hmax4"]
for gname in sys.argv[1:]:
    if gname.startswith("BA-"):
        n = int(gname[3:]); G = nx.barabasi_albert_graph(n, 3, seed=9000)
    else:
        G = load(gname); n = G.number_of_nodes()
    gtp = os.path.join(ROOT, "data_real", f"gt_{gname}.npz")
    out = {}
    for R in (128, n):
        P = GraphPrep(G, R=R)
        if os.path.exists(gtp):
            gain = np.load(gtp)["gain"]
        else:
            gain, _ = all_gains(P.D, P.A)
        eu, ev = P.candidates(0, n)
        X = P.raw(eu, ev)
        for j, c in enumerate(COLS):
            best = max((eval_one(s * X[:, j], gain) for s in (1, -1)), key=lambda m: m["rec10@500"])
            out[(c, R)] = best
    top = np.argsort(-gain)[:10]
    print(f"\n== {gname} n={n}  | true top-10 edges: dist {P.D[eu[top], ev[top]].tolist()}  "
          f"deg-min {np.minimum(P.deg[eu[top]], P.deg[ev[top]]).astype(int).tolist()} deg-max {np.maximum(P.deg[eu[top]], P.deg[ev[top]]).astype(int).tolist()}")
    print(f"{'clue':10s} | R=128: mAP@10 rec10@500 kneed10 | exact(all nodes): mAP@10 rec10@500 kneed10")
    for c in COLS:
        a, b = out[(c, 128)], out[(c, n)]
        print(f"{c:10s} | {a['map10']:.3f} {a['rec10@500']:.2f} {a['kneed10']:8d} | {b['map10']:.3f} {b['rec10@500']:.2f} {b['kneed10']:8d}", flush=True)
