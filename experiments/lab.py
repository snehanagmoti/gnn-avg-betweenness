"""
Experiment harness: run one named configuration with several seeds, evaluate on the test graphs
and save one JSON per run in experiments/runs/. `python experiments/report.py` builds the tables.

    python experiments/lab.py <name> '<json overrides>' [seeds]
"""
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from config import Config, legacy_config
from src.baselines import BASELINES
from src.data_io import load_all
from src.metrics import summary_metrics
from src.trainer import predict, to_tensors, train

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


def split_ids(n, split):
    a, b, _ = split
    ids = list(range(n))
    return ids[:a], ids[a:a + b], ids[a + b:]


def classifier_metrics(probs, labels):
    pred = probs.argmax(1)
    good_true, good_pred = labels >= 1, pred >= 1
    tp = (good_true & good_pred).sum()
    prec = tp / max(good_pred.sum(), 1)
    rec = tp / max(good_true.sum(), 1)
    bal = np.mean([(pred[labels == c] == c).mean() for c in np.unique(labels)])
    return {"cls_acc": float((pred == labels).mean()), "cls_bal_acc": float(bal),
            "cls_f1_good": float(2 * prec * rec / max(prec + rec, 1e-12))}


def mean_dict(ds):
    return {k: float(np.nanmean([d[k] for d in ds])) for k in ds[0]}


def run(name, overrides, seeds=(0, 1, 2), data_dir="data", step=""):
    os.makedirs(RUNS, exist_ok=True)
    base = legacy_config(data_dir=data_dir)
    graphs = load_all(data_dir)
    cfg0 = dataclasses.replace(base, **overrides)
    tr, va, te = split_ids(len(graphs), cfg0.split)
    train_t = [to_tensors(graphs[i], cfg0) for i in tr]
    val_t = [to_tensors(graphs[i], cfg0) for i in va]
    test_t = [to_tensors(graphs[i], cfg0) for i in te]
    for s in seeds:
        out_path = os.path.join(RUNS, f"{name}__s{s}.json")
        if os.path.exists(out_path):
            continue
        cfg = dataclasses.replace(cfg0, seed=s)
        t0 = time.time()
        model, history, best = train(cfg, None, None, verbose=False, train_t=train_t, val_t=val_t)
        secs = time.time() - t0
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ROOT, "models", f"{name}__s{s}.pt"))
        per, cls = [], []
        for t in test_t:
            final, _, probs = predict(model, t, cfg)
            per.append(summary_metrics(final, t["reductions"]))
            if cfg.train_mode != "rank_only":
                cls.append(classifier_metrics(probs, t["labels_np"]))
        res = {"name": name, "step": step, "seed": s, "overrides": overrides,
               "test": mean_dict(per), "test_per_graph": per,
               "classifier": mean_dict(cls) if cls else {},
               "best_epoch": best["epoch"], "best_val_map10": best["val_map10"],
               "stage1_epoch": best.get("stage1_epoch"), "train_seconds": secs,
               "epochs_run": sum(len(h["val_map10"]) for h in history.values())}
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        m = res["test"]
        print(f"{name} s{s}: mAP@10 {m['map10']:.3f} mAP@50 {m['map50']:.3f} top50 {m['top50_overlap']:.3f} "
              f"opt {m['opt_ratio']:.3f} | val {best['val_map10']:.3f} @ {best['epoch']} | {secs:.0f}s", flush=True)


def run_baselines(data_dir="data"):
    graphs = load_all(data_dir)
    _, _, te = split_ids(len(graphs), Config().split)
    rows = {}
    for bname, fn in BASELINES.items():
        per = []
        for rep in range(5):                       # random tie-breaking -> average over 5 draws
            rng = np.random.default_rng(rep)
            per += [summary_metrics(fn(graphs[i]["G"], graphs[i]["missing_edges"], rng),
                                    graphs[i]["reductions"]) for i in te]
        rows[bname] = mean_dict(per)
    with open(os.path.join(ROOT, "baselines.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for k, v in rows.items():
        print(f"{k:28s} mAP@10 {v['map10']:.3f} mAP@50 {v['map50']:.3f} top50 {v['top50_overlap']:.3f} opt {v['opt_ratio']:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "baselines":
        run_baselines()
    else:
        seeds = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (0, 1, 2)
        run(sys.argv[1], json.loads(sys.argv[2]), seeds, step=sys.argv[3] if len(sys.argv) > 3 else "")
