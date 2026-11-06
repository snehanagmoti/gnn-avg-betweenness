"""Ensembles of saved version-3 GNN runs: average of per-graph standardized scores.
    python ens.py <name1> [<name2> ...]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from gnn3 import Net, tensors, predict

OUT = os.path.join(ROOT, "experiments", "v3", "results")
names = sys.argv[1:]
base = base_records()
sets = {"4 test graphs": base[16:20], "50 new test graphs": extra_records(50, start_seed=5000, tag="test_ba200")}
z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
for set_name, recs in sets.items():
    total = [np.zeros(len(r["E"])) for r in recs]
    k = 0
    for nm in names:
        opt = json.load(open(os.path.join(OUT, f"{nm}.json")))["opt"]
        ts = [tensors(r, opt["edge_blocks"]) for r in recs]
        for s in opt["seeds"]:
            model = Net(ts[0]["x"].shape[1], ts[0]["ef"].shape[1], hidden=opt.get("hidden", 32), no_gnn=opt["no_gnn"])
            model.load_state_dict(torch.load(os.path.join(OUT, f"{nm}__s{s}.pt")))
            for i, t in enumerate(ts):
                total[i] += z(predict(model, t))
            k += 1
    m = eval_many(total, recs)
    print(fmt(f"ENS[{'+'.join(names)}] x{k} {set_name}"[:60], m), flush=True)
