"""Version 2 GNN (trained on BA-200) applied without retraining to the generalisation test sets."""
import json, os, sys
import numpy as np, torch, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict, build_model
from src.baselines import BASELINES
torch.set_num_threads(1)
SETS = {"BA-500 (6)": extra_records(6, n=500, start_seed=7000, tag="test_ba500"),
        "ER-200 p=0.04 (10)": extra_records(10, gtype="er", start_seed=7100, tag="test_er200"),
        "2-tree-200 (10)": extra_records(10, gtype="ktree", start_seed=7200, tag="test_ktree200"),
        "random tree-200 (10)": extra_records(10, gtype="random_tree", start_seed=7300, tag="test_tree200")}
cfg = Config(); rows = {}
for nm, recs in SETS.items():
    gl = []
    for i, r in enumerate(recs):
        G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
        gl.append({"G": G, "missing_edges": r["E"], "reductions": r["red"], "edges": r["edges"], "graph_id": i})
    tt = [to_tensors(g, cfg) for g in gl]
    runs = []
    for s in range(5):
        model = build_model(cfg, tt[0]); model.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/E31_batch500__s{s}.pt")))
        runs.append(eval_many([predict(model, t, cfg)[0] for t in tt], recs))
    rows["V2 " + nm] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    rng = np.random.default_rng(0)
    rows["rule " + nm] = eval_many([BASELINES["closeness_gap_x_distance"](g["G"], g["missing_edges"], rng) for g in gl], recs)
    for k in ("V2 ", "rule "):
        print(fmt(k + nm, rows[k + nm]), flush=True)
json.dump(rows, open(os.path.join(ROOT, "experiments/v3/results/ref_gen.json"), "w"), indent=1)
