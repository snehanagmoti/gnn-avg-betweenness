"""
Evaluate the saved final version 3 models (no training).

    python v3/evaluate.py ba       # v3/results/final_ba__s0..4.pt on 4 original + 50 new BA-200 test graphs
    python v3/evaluate.py paper    # v3/results/final_paper__s0..2.pt on the 48 paper-style test graphs, per type

Metrics: mAP@k, NDCG@k, top-50 overlap, rec10@100, kneed10 and mAP after the exact check of the
top 100 (ex100). Each is averaged over test graphs, then over the saved models (seeds).
Writes v3/results/final_<dataset>.json.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, base_records, eval_many, extra_records, fmt     # noqa: E402
from train import Net, by_type, mean_of, predict, tensors                   # noqa: E402


def models(name):
    paths = sorted(glob.glob(os.path.join(RESULTS, f"{name}__s*.pt")))
    assert paths, f"no saved models {name}__s*.pt in {RESULTS}"
    out = []
    for p in paths:
        m = Net()
        m.load_state_dict(torch.load(p))
        out.append(m)
    return out


def evaluate_ba():
    te = [tensors(r) for r in base_records()[16:20]]
    new = [tensors(r) for r in extra_records(50, start_seed=5000, tag="test_ba200")]
    runs = [{"4 original test graphs": eval_many([predict(m, t) for t in te], [t["red"] for t in te]),
             "50 new BA-200 graphs": eval_many([predict(m, t) for t in new], [t["red"] for t in new])}
            for m in models("final_ba")]
    return mean_of(runs)


def evaluate_paper():
    import paper_data
    test_r = [r for r in paper_data.load() if r["split"] == "test"]
    te = [tensors(r) for r in test_r]
    return mean_of([by_type([predict(m, t) for t in te], test_r) for m in models("final_paper")])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ba"
    res = evaluate_ba() if which == "ba" else evaluate_paper()
    json.dump({"models": f"final_{which}__s*.pt", "mean": res},
              open(os.path.join(RESULTS, f"final_{which}.json"), "w"), indent=1)
    for g, m in res.items():
        print(fmt(g + (f" ({m['graphs']:.0f})" if "graphs" in m else ""), m))
