"""
GNN variants for version 3 (plain PyTorch GraphSAGE, as in version 2).
    python gnn3.py <name> '<json options>'
options:
  edge_blocks : hand-made edge feature blocks appended to the edge vector, e.g. ["xe","xp"]
  loss        : "reg" (MSE on per-graph standardized log gain) | "triplet" (version 2 ranking loss)
                | "reg+triplet"
  weight_top  : extra weight on better edges in the regression loss (1 + w * grade)
  n_extra     : extra BA training graphs; graphs_per_epoch: how many training graphs per epoch
  no_gnn      : true = MLP on the hand-made features only (control)
"""
import copy, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from gbm import labels_from, target
from src.features import node_features
from src.model import GraphSAGE, mean_adjacency, mlp

OUT = os.path.join(ROOT, "experiments", "v3", "results")
torch.set_num_threads(1)


def tensors(r, blocks):
    G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
    x = node_features(G, NODE_FEATS)
    y = target(r["red"], "regression")
    return dict(x=torch.tensor(x), adj=mean_adjacency(r["edges"], int(r["n"])), E=torch.tensor(r["E"]),
                ef=torch.tensor(features(r, blocks)) if blocks else torch.zeros(len(r["E"]), 0),
                y=torch.tensor(y, dtype=torch.float32), grade=labels_from(r["red"]), red=r["red"])


class Net(nn.Module):
    def __init__(self, in_dim, ef_dim, hidden=32, no_gnn=False):
        super().__init__()
        self.no_gnn = no_gnn
        self.gnn = GraphSAGE(in_dim, hidden, 3, 0.2)
        d = (0 if no_gnn else 2 * hidden) + ef_dim
        self.head = mlp(d, 64, 1, 3, 0.2)

    def forward(self, t, idx):
        e = t["E"][idx]
        parts = []
        if not self.no_gnn:
            h = self.gnn(t["x"], t["adj"])
            hu, hv = h[e[:, 0]], h[e[:, 1]]
            parts += [hu + hv, (hu - hv).abs()]
        parts.append(t["ef"][idx])
        return self.head(torch.cat(parts, 1)).squeeze(-1)


def triplet(scores, y, n, rng, margin=1.0):
    pos = np.arange(len(y))
    t = torch.tensor(np.stack([rng.choice(pos, n) for _ in range(3)], 1))
    loss = 0
    terms = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        I = torch.sign(y[t[:, a]] - y[t[:, b]])
        terms.append(F.relu(-I * (scores[t[:, a]] - scores[t[:, b]]) + margin) * (I != 0))
    return torch.stack(terms, 1).max(1).values.mean()


@torch.no_grad()
def predict(model, t):
    model.eval()
    return model(t, torch.arange(len(t["E"]))).numpy()


def run(name, opt):
    opt = {**dict(edge_blocks=["xe", "xp"], loss="reg", weight_top=0.0, n_extra=0, graphs_per_epoch=14,
                  epochs=150, patience=25, lr=0.003, neg=4000, batch=500, seeds=[0, 1, 2], no_gnn=False), **opt}
    base = base_records()
    train_r = base[:14] + (extra_records(opt["n_extra"]) if opt["n_extra"] else [])
    for gtype, num, n in opt.get("extra_sets", []):        # mixed graph types / sizes
        train_r += extra_records(num, n=n, gtype=gtype, start_seed=20000, tag=f"train_{gtype}{n}")
    big = extra_records(50, start_seed=5000, tag="test_ba200")
    T = lambda rs: [tensors(r, opt["edge_blocks"]) for r in rs]
    tr, va, te, bt = T(train_r), T(base[14:16]), T(base[16:20]), T(big)
    runs = []
    t0 = time.time()
    for s in opt["seeds"]:
        torch.manual_seed(s); rng = np.random.default_rng(s)
        model = Net(tr[0]["x"].shape[1], tr[0]["ef"].shape[1], hidden=opt.get("hidden", 32), no_gnn=opt["no_gnn"])
        optim = torch.optim.Adam(model.parameters(), lr=opt["lr"])
        best, best_state, bad = -1, None, 0
        for ep in range(opt["epochs"]):
            model.train()
            for gi in rng.choice(len(tr), min(opt["graphs_per_epoch"], len(tr)), replace=False):
                t = tr[gi]
                pos = np.nonzero(t["grade"] >= 1)[0]
                rest = np.nonzero(t["grade"] == 0)[0]
                idx = rng.permutation(np.concatenate([pos, rng.choice(rest, opt["neg"], replace=False)]))
                for b0 in range(0, len(idx), opt["batch"]):
                    b = torch.tensor(idx[b0:b0 + opt["batch"]])
                    sc = model(t, b)
                    yb = t["y"][b]
                    w = 1 + opt["weight_top"] * torch.tensor(t["grade"][b.numpy()], dtype=torch.float32)
                    loss = 0
                    if "reg" in opt["loss"]:
                        loss = loss + (w * (sc - yb) ** 2).mean()
                    if "triplet" in opt["loss"]:
                        good = torch.tensor(np.nonzero(t["grade"][b.numpy()] >= 1)[0])
                        if len(good) > 2:
                            loss = loss + triplet(sc[good], yb[good], 10 * len(b), rng)
                    if torch.is_tensor(loss):
                        optim.zero_grad(); loss.backward(); optim.step()
            v = np.mean([np.mean([eval_one(predict(model, t), t["red"])[k] for k in ("map10", "map50")]) for t in va])
            if v > best:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= opt["patience"]:
                    break
        model.load_state_dict(best_state)
        m = eval_many([predict(model, t) for t in te], [{"red": t["red"]} for t in te])
        mb = eval_many([predict(model, t) for t in bt], [{"red": t["red"]} for t in bt])
        m.update({"T50_" + k: v for k, v in mb.items()}); m["epochs"] = ep + 1
        runs.append(m)
        torch.save(model.state_dict(), os.path.join(OUT, f"{name}__s{s}.pt"))
    mm = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    mm.update({f"{k}_sd": float(np.std([r[k] for r in runs])) for k in ("map10", "map50")})
    mm["secs"] = time.time() - t0
    json.dump({"name": name, "opt": opt, "mean": mm, "runs": runs}, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(fmt(name, mm), f"| sd {mm['map10_sd']:.3f} | ep {mm['epochs']:.0f} | {mm['secs']:.0f}s", flush=True)
    print(fmt("   (50 new test graphs)", {k[4:]: v for k, v in mm.items() if k.startswith("T50_")}), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
