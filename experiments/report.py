"""Aggregate experiments/runs/*.json into experiments/results.csv and print a table per step."""
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["map10", "map50", "top50_overlap", "opt_ratio", "best10_ratio", "within_10pct", "spearman50"]


def load():
    runs = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*.json"))):
        r = json.load(open(p))
        runs[r["name"]].append(r)
    return runs


def agg(rs):
    out = {"n": len(rs), "step": rs[0]["step"], "overrides": json.dumps(rs[0]["overrides"])}
    for c in COLS:
        v = [r["test"][c] for r in rs]
        out[c], out[c + "_sd"] = float(np.mean(v)), float(np.std(v))
    out["cls_bal_acc"] = float(np.mean([r["classifier"].get("cls_bal_acc", np.nan) for r in rs]))
    out["val_map10"] = float(np.mean([r["best_val_map10"] for r in rs]))
    out["epochs"] = float(np.mean([r["epochs_run"] for r in rs]))
    out["secs"] = float(np.mean([r["train_seconds"] for r in rs]))
    return out


if __name__ == "__main__":
    runs = load()
    table = {n: agg(rs) for n, rs in runs.items()}
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["n", "step", "overrides"] + [k for c in COLS for k in (c, c + "_sd")] + ["cls_bal_acc", "val_map10", "epochs", "secs"]
        w.writerow(["name"] + keys)
        for n, a in table.items():
            w.writerow([n] + [a[k] for k in keys])
    steps = sys.argv[1:] or sorted({a["step"] for a in table.values()})
    for st in steps:
        print(f"\n== {st} ==")
        print(f"{'name':34s} {'n':>2s} {'mAP@10':>12s} {'mAP@50':>12s} {'top50':>6s} {'opt':>6s} {'best10':>6s} {'sp50':>6s} {'clsBA':>6s} {'ep':>5s}")
        for n, a in table.items():
            if a["step"] != st:
                continue
            print(f"{n:34s} {a['n']:2d} {a['map10']:.3f}±{a['map10_sd']:.3f} {a['map50']:.3f}±{a['map50_sd']:.3f} "
                  f"{a['top50_overlap']:.3f} {a['opt_ratio']:.3f} {a['best10_ratio']:.3f} {a['spearman50']:.3f} "
                  f"{a['cls_bal_acc']:.3f} {a['epochs']:5.0f}")
