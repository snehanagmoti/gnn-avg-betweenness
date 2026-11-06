"""
Gradient-boosted trees (LightGBM) on hand-made edge features.
    python gbm.py <name> '<json options>'
options: blocks (feature blocks), obj (binary | multiclass | regression | lambdarank),
         n_extra (extra BA training graphs), seeds, neg_per_graph (sampled "rest" edges per graph),
         extra_type / extra_n (graph type and size of the extra training graphs)
Train = base graphs 0-13 (+ extras), early stopping on base graphs 14-15, test on 16-19.
"""
import json, os, sys, time
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = os.path.join(ROOT, "experiments", "v3", "results")


def labels_from(red):
    """Graded relevance by rank (ties share the best rank): 0 rest, 1 top10%, 2 top2%, 3 top0.5%, 4 top0.1%."""
    rank = rankdata(-red, method="min")
    E = len(red)
    return sum((rank <= max(1, round(f * E))).astype(int) for f in (0.10, 0.02, 0.005, 0.001))


def target(red, obj):
    lab = labels_from(red)
    if obj == "binary":
        return (lab >= 2).astype(int)                     # top 2%
    if obj == "multiclass":
        return np.minimum(lab, 2)                         # rest / 2-10% / top 2%
    if obj == "regression":                               # per-graph standardized log gain
        y = np.log(red / red.max() + 1e-6)
        return (y - y.mean()) / y.std()
    if obj == "regression_rank":                          # -log(rank / E): stretches the top
        return -np.log(rankdata(-red, method="min") / len(red))
    return lab                                            # lambdarank grades


def sample_rows(red, rng, neg):
    lab = labels_from(red)
    pos = np.nonzero(lab >= 1)[0]
    rest = np.nonzero(lab == 0)[0]
    return np.concatenate([pos, rng.choice(rest, min(neg, len(rest)), replace=False)])


def build(recs, opt, rng, sample=True):
    X, y, grp, W = [], [], [], []
    for r in recs:
        idx = sample_rows(r["red"], rng, opt["neg_per_graph"]) if sample else np.arange(len(r["red"]))
        X.append(features(r, opt["blocks"])[idx])
        y.append(target(r["red"], opt["obj"])[idx])
        W.append((1.0 + opt.get("weight_top", 0.0) * labels_from(r["red"]))[idx])
        grp.append(len(idx))
    return np.concatenate(X), np.concatenate(y), grp, np.concatenate(W)


def score(model, r, opt):
    p = model.predict(features(r, opt["blocks"]))
    if opt["obj"] == "multiclass":                        # expected class, ties broken by P(top)
        return p @ np.arange(p.shape[1]) + 1e-3 * p[:, -1]
    return p


def run(name, opt):
    opt = {**dict(blocks=["xn", "xe", "xp"], obj="binary", n_extra=0, seeds=[0, 1, 2],
                  neg_per_graph=4000, extra_type="ba", extra_n=200, lr=0.05, leaves=31, rounds=2000), **opt}
    base = base_records()
    train = base[:14] + (extra_records(opt["n_extra"], n=opt["extra_n"], gtype=opt["extra_type"])
                         if opt["n_extra"] else [])
    val, test = base[14:16], base[16:20]
    big = extra_records(50, start_seed=5000, tag="test_ba200") if opt.get("big_test", True) else []
    runs = []
    t0 = time.time()
    for s in opt["seeds"]:
        rng = np.random.default_rng(s)
        Xt, yt, gt, wt = build(train, opt, rng)
        Xv, yv, gv, wv = build(val, opt, rng)
        params = dict(learning_rate=opt["lr"], num_leaves=opt["leaves"], min_data_in_leaf=opt.get("min_leaf", 20),
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=s,
                      verbose=-1, num_threads=2)
        if opt["obj"] == "binary":
            params.update(objective="binary", scale_pos_weight=opt.get("pos_weight", 1.0))
        elif opt["obj"] == "multiclass":
            params.update(objective="multiclass", num_class=3)
        elif opt["obj"].startswith("regression"):
            params.update(objective=opt.get("loss", "regression"))
        else:
            params.update(objective="lambdarank", lambdarank_truncation_level=200,
                          label_gain=[0, 1, 3, 7, 15], eval_at=[10, 50])
        lr_ = opt["obj"] == "lambdarank"
        dt = lgb.Dataset(Xt, yt, group=gt if lr_ else None, weight=None if lr_ else wt)
        dv = lgb.Dataset(Xv, yv, group=gv if lr_ else None, weight=None if lr_ else wv, reference=dt)
        model = lgb.train(params, dt, opt["rounds"], valid_sets=[dv],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        runs.append(eval_many([score(model, r, opt) for r in test], test))
        runs[-1]["iters"] = model.best_iteration
        if big:
            mb = eval_many([score(model, r, opt) for r in big], big)
            runs[-1].update({"T50_" + k: v for k, v in mb.items()})
        if opt.get("save_model"):
            model.save_model(os.path.join(OUT, f"{name}__s{s}.txt"))
    m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    m.update({f"{k}_sd": float(np.std([r[k] for r in runs])) for k in ("map10", "map50")})
    m["secs"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump({"name": name, "opt": opt, "mean": m, "runs": runs}, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(fmt(name, m), f"| sd {m['map10_sd']:.3f} | it {m['iters']:.0f} | {m['secs']:.0f}s", flush=True)
    if big:
        print(fmt("   (50 new test graphs)", {k[4:]: v for k, v in m.items() if k.startswith("T50_")}), flush=True)
    return m


if __name__ == "__main__":
    run(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
