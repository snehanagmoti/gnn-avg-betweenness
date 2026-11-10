"""Version 2 (models trained by lab.py on data_paper) + simple rule, per graph type, with the paper's metrics."""
import dataclasses, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
from gnn_paper import paper_metrics
from config import legacy_config
from src.data_io import load_all, load_meta
from src.trainer import to_tensors, predict, build_model
from src.baselines import BASELINES
torch.set_num_threads(1)
d = os.path.join(ROOT, "data_paper")
ov = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
cfg = dataclasses.replace(legacy_config(data_dir=d), **ov)
graphs = load_all(d); meta = load_meta(d)["graphs"]
a, b, c = json.load(open(os.path.join(d, "split.json")))["split"]
test = list(range(a + b, a + b + c))
tt = [to_tensors(graphs[i], cfg) for i in test]
types = [meta[i]["type"] for i in test]
name = os.environ.get("V2NAME", "P_v2")
rows = {"v2": [], "rule": []}
for s in [int(x) for x in os.environ.get("V2SEEDS", "0,1,2").split(",")]:
    m = build_model(cfg, tt[0]); m.load_state_dict(torch.load(os.path.join(ROOT, f"experiments/models/{name}__s{s}.pt")))
    rows["v2"].append([paper_metrics(predict(m, t, cfg)[0], graphs[i]["reductions"]) for t, i in zip(tt, test)])
rng = np.random.default_rng(0)
rows["rule"].append([paper_metrics(BASELINES["closeness_gap_x_distance"](graphs[i]["G"], graphs[i]["missing_edges"], rng), graphs[i]["reductions"]) for i in test])
out = {}
for k, runs in rows.items():
    out[k] = {}
    for t in ("ba", "er", "tree", "path", "all"):
        idx = [j for j, ty in enumerate(types) if t == "all" or ty == t]
        vals = [run[j] for run in runs for j in idx]
        m = {kk: float(np.nanmean([v[kk] for v in vals])) for kk in vals[0]}
        out[k][t] = m
        print(f"{k:5s} {t:5s} n={len(idx):3d} mAP@1 {m['map1']:.3f} @10 {m['map10']:.3f} @50 {m['map50']:.3f} @100 {m['map100']:.3f} "
              f"| PC {m['pc100']:.2f} SC {m['sc100']:.2f} KC {m['kc100']:.2f} | kneed10 {m['kneed10']:.0f} ex100 {m['ex100_map10']:.3f}/{m['ex100_map50']:.3f}", flush=True)
json.dump(out, open(os.path.join(ROOT, f"experiments/v3/results/{name}_pertype.json"), "w"), indent=1)
