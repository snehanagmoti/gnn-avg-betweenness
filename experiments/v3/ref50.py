"""Version 2 GNN and simple rules on 50 NEW test graphs (seeds 5000-5049)."""
import json, os, sys
import numpy as np, torch, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict, build_model
from src.baselines import BASELINES
big = extra_records(50, start_seed=5000, tag="test_ba200")
cfg = Config()
gl = []
for i, r in enumerate(big):
    G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
    gl.append({"G": G, "missing_edges": r["E"], "reductions": r["red"], "edges": r["edges"], "graph_id": 5000 + i})
tt = [to_tensors(g, cfg) for g in gl]
rows = {}
runs = []
for s in range(5):
    model = build_model(cfg, tt[0]); model.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/E31_batch500__s{s}.pt")))
    runs.append(eval_many([predict(model, t, cfg)[0] for t in tt], big))
rows["V2 GNN (E31, 5 seeds)"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
rng = np.random.default_rng(0)
rows["rule: closeness_gap_x_distance"] = eval_many([BASELINES["closeness_gap_x_distance"](g["G"], g["missing_edges"], rng) for g in gl], big)
rows["rule: one_sided"] = eval_many([r["xp"][:, PAIR_FEATS.index("one_sided")] for r in big], big)
print(header())
for k, m in rows.items():
    print(fmt(k, m))
json.dump(rows, open(os.path.join(ROOT, "experiments/v3/results/ref50.json"), "w"), indent=1)
