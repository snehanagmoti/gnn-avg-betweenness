"""
Version 3 GNN trained / evaluated on the paper-style dataset (paper_data.py), results per graph type.
    python gnn_paper.py <name> '<json options>'
Same options as gnn3.py, plus: scale (1 = paper sizes, k = k x more BA/ER graphs in the training split).
"""
import copy, json, os, sys, time
import numpy as np, torch
from scipy.stats import pearsonr, spearmanr, kendalltau
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from gnn3 import Net, tensors, predict, triplet
import paper_data

OUT = os.path.join(ROOT, "experiments", "v3", "results")
torch.set_num_threads(1)


def paper_metrics(score, red):
    """The paper's metrics: mAP@1/10/100 and Pearson / Spearman / Kendall over the true top 100."""
    m = eval_one(score, red)
    order = np.argsort(-score, kind="stable")
    m["map1"] = average_precision_at_k(order, red, 1)
    m["map100"] = average_precision_at_k(order, red, 100)
    top = np.argsort(-red, kind="stable")[:100]
    p, t = score[top], red[top]
    ok = p.std() > 0 and t.std() > 0
    m["pc100"] = float(pearsonr(p, t)[0]) if ok else np.nan
    m["sc100"] = float(spearmanr(p, t)[0]) if ok else np.nan
    m["kc100"] = float(kendalltau(p, t)[0]) if ok else np.nan
    return m


def by_type(scores, recs):
    out = {}
    for t in ("ba", "er", "tree", "path", "all"):
        idx = [i for i, r in enumerate(recs) if t == "all" or r["type"] == t]
        if not idx:
            continue
        per = [paper_metrics(scores[i], recs[i]["red"]) for i in idx]
        out[t] = {k: float(np.nanmean([p[k] for p in per])) for k in per[0]}
        out[t]["graphs"] = len(idx)
    return out


def run(name, opt):
    opt = {**dict(edge_blocks=["xe", "xp", "xh"], loss="reg", weight_top=2.0, graphs_per_epoch=40, hidden=64,
                  epochs=150, patience=25, lr=0.003, neg=4000, batch=500, seeds=[0, 1, 2], no_gnn=False, scale=1), **opt}
    recs = paper_data.load(1)
    train_r = [r for r in recs if r["split"] == "train"]
    if opt.get("extra"):                                   # extra TRAINING graphs only (no overlap with val/test)
        train_r = train_r + paper_data.extra_train(**opt["extra"])
    val_r = [r for r in recs if r["split"] == "val"]
    test_r = [r for r in recs if r["split"] == "test"]
    def T(rs):                                            # build tensors, then free the big record arrays
        out = []
        for r in rs:
            out.append(tensors(r, opt["edge_blocks"]))
            for k in ("xn", "xe", "xp", "xh", "xs", "ub"):
                r.pop(k, None)
        return out
    tr, va, te = T(train_r), T(val_r), T(test_r)
    runs = []
    t0 = time.time()
    for s in opt["seeds"]:
        torch.manual_seed(s); rng = np.random.default_rng(s)
        model = Net(tr[0]["x"].shape[1], tr[0]["ef"].shape[1], hidden=opt["hidden"], no_gnn=opt["no_gnn"])
        optim = torch.optim.Adam(model.parameters(), lr=opt["lr"])
        best, best_state, bad = -1, None, 0
        for ep in range(opt["epochs"]):
            model.train()
            for gi in rng.choice(len(tr), min(opt["graphs_per_epoch"], len(tr)), replace=False):
                t = tr[gi]
                pos = np.nonzero(t["grade"] >= 1)[0]
                rest = np.nonzero(t["grade"] == 0)[0]
                idx = rng.permutation(np.concatenate([pos, rng.choice(rest, min(opt["neg"], len(rest)), replace=False)]))
                for b0 in range(0, len(idx), opt["batch"]):
                    b = torch.tensor(idx[b0:b0 + opt["batch"]])
                    sc = model(t, b); yb = t["y"][b]
                    w = 1 + opt["weight_top"] * torch.tensor(t["grade"][b.numpy()], dtype=torch.float32)
                    loss = (w * (sc - yb) ** 2).mean()
                    optim.zero_grad(); loss.backward(); optim.step()
            v = np.mean([np.mean([eval_one(predict(model, t), t["red"])[k] for k in ("map10", "map50")]) for t in va])
            if v > best:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= opt["patience"]:
                    break
        model.load_state_dict(best_state)
        res = by_type([predict(model, t) for t in te], test_r)
        res["epochs"] = ep + 1; res["best_val"] = best
        runs.append(res)
        torch.save(model.state_dict(), os.path.join(OUT, f"{name}__s{s}.pt"))
        print(f"  seed {s}: " + " | ".join(f"{k} {v['map10']:.3f}/{v['map100']:.3f}" for k, v in res.items() if isinstance(v, dict)), flush=True)
    mean = {t: {k: float(np.nanmean([r[t][k] for r in runs])) for k in runs[0][t]} for t in runs[0] if isinstance(runs[0][t], dict)}
    json.dump({"name": name, "opt": opt, "mean": mean, "runs": runs, "secs": time.time() - t0},
              open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(f"{name} ({time.time()-t0:.0f}s)")
    for t, m in mean.items():
        print(f"  {t:5s} n={m['graphs']:3.0f} mAP@1 {m['map1']:.3f} @10 {m['map10']:.3f} @50 {m['map50']:.3f} @100 {m['map100']:.3f} "
              f"| PC {m['pc100']:.2f} SC {m['sc100']:.2f} KC {m['kc100']:.2f} | kneed10 {m['kneed10']:.0f} ex100 {m['ex100_map10']:.3f}/{m['ex100_map50']:.3f}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
