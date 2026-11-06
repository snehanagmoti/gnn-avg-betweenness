"""Reference numbers with the new filter metrics: version 2 GNN (5 saved models) and simple rules."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from config import Config
from src.trainer import to_tensors, predict, build_model
from src.baselines import BASELINES

t0 = time.time()
recs = base_records(); test = recs[16:20]
print("records ready %.1fs" % (time.time() - t0), flush=True)
rows = {}
cfg = Config(data_dir=os.path.join(ROOT, "data"))
gs = load_all(cfg.data_dir)
tt = [to_tensors(gs[i], cfg) for i in range(16, 20)]
for i, r in zip(range(16, 20), test):          # same edge order?
    assert (gs[i]["missing_edges"] == r["E"]).all()
runs = []
for s in range(5):
    model = build_model(cfg, tt[0]); model.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/E31_batch500__s{s}.pt")))
    sc = [predict(model, t, cfg)[0] for t in tt]
    runs.append(eval_many(sc, test))
rows["V2 GNN (E31, 5 seeds)"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
rng = np.random.default_rng(0)
for b in ["closeness_gap_x_distance", "degree_product", "random"]:
    rows[f"rule: {b}"] = eval_many([BASELINES[b](gs[i]["G"], gs[i]["missing_edges"], rng) for i in range(16, 20)], test)
# single cheap pair features as rules (higher = better)
names = PAIR_FEATS
for j, nm in enumerate(names):
    rows[f"rule: {nm}"] = eval_many([r["xp"][:, j] for r in test], test)
rows["rule: UB raw"] = eval_many([r["ub"] for r in test], test)
print(header())
for k, m in rows.items():
    print(fmt(k, m))
json.dump(rows, open(os.path.join(ROOT, "experiments/v3/results/ref.json"), "w"), indent=1)
