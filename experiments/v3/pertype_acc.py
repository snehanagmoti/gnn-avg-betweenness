import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from gnn3 import Net, tensors, predict
from cls_compare import by_rank, cls_metrics
from src.ground_truth import make_labels
import paper_data
torch.set_num_threads(2)
OUT = os.path.join(ROOT, "experiments", "v3", "results")
name = sys.argv[1]
info = json.load(open(os.path.join(OUT, f"{name}.json"))); opt = info["opt"]
recs = [r for r in paper_data.load(1) if r["split"] == "test"]
ts = [tensors(r, opt["edge_blocks"]) for r in recs]
acc = {}
for s in opt["seeds"]:
    m = Net(ts[0]["x"].shape[1], ts[0]["ef"].shape[1], hidden=opt["hidden"]); m.load_state_dict(torch.load(os.path.join(OUT, f"{name}__s{s}.pt")))
    for t, r in zip(ts, recs):
        true = make_labels(r["red"], [0.10, 0.02]); sc = predict(m, t)
        c = cls_metrics(by_rank(sc, len(true)), true)
        acc.setdefault(r["type"], []).append(c)
for ty, v in acc.items():
    print(ty, {k: round(float(np.mean([x[k] for x in v])), 3) for k in ("bal_acc", "acc", "top2_recall", "top2_precision")})
print(json.dumps({t: {k: round(v[k], 3) for k in ("map1","map10","map50","map100","top50","pc100","sc100","kc100","rec10@100","kneed10","ex100_map10","ex100_map50","graphs")} for t, v in info["mean"].items()}, indent=0))
