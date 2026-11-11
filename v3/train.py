"""
Train the version 3 model: GraphSAGE (3 layers, width 64) + MLP head -> one score per candidate edge.

    python v3/train.py ba       # 14 + 300 BA-200 graphs; test: 4 original + 50 new BA-200 graphs
    python v3/train.py paper    # paper-style dataset + extra trees/paths; test: 48 graphs, per type

Options: --seeds 0 1 2 3 4   --epochs 150   --name final_ba
Saves v3/results/<name>__s<seed>.pt and v3/results/<name>.json.
"""
import argparse
import copy
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, base_records, clues, eval_many, extra_records, fmt, grades,  # noqa: E402
                    target)
from src.features import node_features                      # noqa: E402
from src.model import GraphSAGE, mean_adjacency, mlp        # noqa: E402

torch.set_num_threads(1)
OPT = dict(hidden=64, weight_top=2.0, graphs_per_epoch=40, epochs=150, patience=25,
           lr=0.003, neg=4000, batch=500, n_extra=300)


def tensors(r):
    """Everything the model needs for one graph."""
    G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
    return dict(x=torch.tensor(node_features(G)), adj=mean_adjacency(r["edges"], int(r["n"])),
                E=torch.tensor(r["E"]), ef=torch.tensor(clues(r)),
                y=torch.tensor(target(r["red"]), dtype=torch.float32), grade=grades(r["red"]), red=r["red"])


class Net(nn.Module):
    """Edge (u, v) -> [h_u + h_v (64), |h_u - h_v| (64), 24 clues] = 152 numbers -> head -> score."""

    def __init__(self, in_dim=4, ef_dim=24, hidden=64):
        super().__init__()
        self.gnn = GraphSAGE(in_dim, hidden, 3, 0.2)
        self.head = mlp(2 * hidden + ef_dim, 64, 1, 3, 0.2)

    def forward(self, t, idx):
        e = t["E"][idx]
        h = self.gnn(t["x"], t["adj"])
        hu, hv = h[e[:, 0]], h[e[:, 1]]
        return self.head(torch.cat([hu + hv, (hu - hv).abs(), t["ef"][idx]], 1)).squeeze(-1)


@torch.no_grad()
def predict(model, t):
    model.eval()
    return model(t, torch.arange(len(t["E"]))).numpy()


def fit(tr, va, seed, opt):
    """Weighted-MSE training with early stopping on the validation mean of mAP@10 and mAP@50."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Net(tr[0]["x"].shape[1], tr[0]["ef"].shape[1], hidden=opt["hidden"])
    optim = torch.optim.Adam(model.parameters(), lr=opt["lr"])
    best, best_state, bad = -1, None, 0
    for ep in range(opt["epochs"]):
        model.train()
        for gi in rng.choice(len(tr), min(opt["graphs_per_epoch"], len(tr)), replace=False):
            t = tr[gi]
            pos = np.nonzero(t["grade"] >= 1)[0]                   # all top-10% edges ...
            rest = np.nonzero(t["grade"] == 0)[0]                  # ... + 4,000 random others
            idx = rng.permutation(np.concatenate([pos, rng.choice(rest, min(opt["neg"], len(rest)), replace=False)]))
            for b0 in range(0, len(idx), opt["batch"]):
                b = torch.tensor(idx[b0:b0 + opt["batch"]])
                w = 1 + opt["weight_top"] * torch.tensor(t["grade"][b.numpy()], dtype=torch.float32)
                loss = (w * (model(t, b) - t["y"][b]) ** 2).mean()
                optim.zero_grad(); loss.backward(); optim.step()
        m = eval_many([predict(model, t) for t in va], [t["red"] for t in va], ks=(10, 50))
        v = (m["map10"] + m["map50"]) / 2
        if v > best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= opt["patience"]:
                break
    model.load_state_dict(best_state)
    return model, ep + 1


def by_type(scores, recs):
    out = {}
    for t in ("ba", "er", "tree", "path", "all"):
        idx = [i for i, r in enumerate(recs) if t == "all" or r["type"] == t]
        out[t] = eval_many([scores[i] for i in idx], [recs[i]["red"] for i in idx], ks=(5, 10, 50, 100))
        out[t]["graphs"] = len(idx)
    return out


def run_ba(name, seeds, opt):
    base = base_records()
    tr = [tensors(r) for r in base[:14] + extra_records(opt["n_extra"])]
    va, te = [tensors(r) for r in base[14:16]], [tensors(r) for r in base[16:20]]
    new = [tensors(r) for r in extra_records(50, start_seed=5000, tag="test_ba200")]
    runs = []
    for s in seeds:
        model, ep = fit(tr, va, s, opt)
        torch.save(model.state_dict(), os.path.join(RESULTS, f"{name}__s{s}.pt"))
        runs.append({"test4": eval_many([predict(model, t) for t in te], [t["red"] for t in te]),
                     "new50": eval_many([predict(model, t) for t in new], [t["red"] for t in new]), "epochs": ep})
        print(fmt(f"seed {s} | 50 new graphs", runs[-1]["new50"]), flush=True)
    return runs


def run_paper(name, seeds, opt):
    import paper_data
    recs = paper_data.load()
    tr = [tensors(r) for r in [r for r in recs if r["split"] == "train"] + paper_data.extra_trees_paths()]
    va = [tensors(r) for r in recs if r["split"] == "val"]
    test_r = [r for r in recs if r["split"] == "test"]
    te = [tensors(r) for r in test_r]
    runs = []
    for s in seeds:
        model, ep = fit(tr, va, s, opt)
        torch.save(model.state_dict(), os.path.join(RESULTS, f"{name}__s{s}.pt"))
        runs.append({**by_type([predict(model, t) for t in te], test_r), "epochs": ep})
        print(fmt(f"seed {s} | all 48 test graphs", runs[-1]["all"]), flush=True)
    return runs


def mean_of(runs):
    keys = [k for k in runs[0] if isinstance(runs[0][k], dict)]
    return {g: {k: float(np.mean([r[g][k] for r in runs])) for k in runs[0][g]} for g in keys}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ba", "paper"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--epochs", type=int, default=OPT["epochs"])
    ap.add_argument("--name")
    a = ap.parse_args()
    opt = {**OPT, "epochs": a.epochs}
    seeds = a.seeds or ([0, 1, 2, 3, 4] if a.dataset == "ba" else [0, 1, 2])
    name = a.name or f"final_{a.dataset}"
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    runs = (run_ba if a.dataset == "ba" else run_paper)(name, seeds, opt)
    mean = mean_of(runs)
    json.dump({"name": name, "opt": opt, "seeds": seeds, "mean": mean, "runs": runs, "secs": time.time() - t0},
              open(os.path.join(RESULTS, f"{name}.json"), "w"), indent=1)
    for g, m in mean.items():
        print(fmt(f"mean | {g}", m))
