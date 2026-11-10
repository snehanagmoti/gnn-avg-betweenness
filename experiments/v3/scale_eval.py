"""
Evaluate a trained version-3 GNN (sampled "xs" features) on big graphs, block by block,
against the exact answer for every candidate edge; with timings.
    python scale_eval.py <model name> ba <n> <num graphs>
    python scale_eval.py <model name> real <network> [<network> ...]
"""
import json, os, sys, time
import numpy as np, torch, networkx as nx
from scipy.sparse.csgraph import shortest_path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import common
from scale import GraphPrep
from gnn3 import Net
from src.fastgain import all_gains, exact_gains_fast
from real_data import load

OUT = os.path.join(ROOT, "experiments", "v3", "results")
torch.set_num_threads(2)


def sparse_mean_adj(G, n):
    e = np.array(G.edges(), dtype=np.int64)
    r = np.concatenate([e[:, 0], e[:, 1]]); c = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(r, minlength=n).astype(np.float32)
    vals = 1.0 / np.maximum(deg[r], 1)
    return torch.sparse_coo_tensor(torch.tensor(np.stack([r, c])), torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()


@torch.no_grad()
def score_graph(models, G, R):
    n = G.number_of_nodes()
    T = {}
    t = time.time(); P = GraphPrep(G, R=R, seed=common.XS_SEED); P.fit_stats(); T["prep (distances, stats)"] = time.time() - t
    x = torch.tensor(node_features(G, NODE_FEATS, D=P.D.astype(np.float64)))
    adj = sparse_mean_adj(G, n)
    t = time.time()
    hs = []
    for m in models:
        m.eval(); hs.append(m.gnn(x, adj))
    T["GNN node embeddings"] = time.time() - t
    scores, tf, th = [], 0.0, 0.0
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    heur = []
    for eu, ev in P.blocks():
        t = time.time(); X = torch.tensor(P.features(eu, ev)); tf += time.time() - t
        heur.append(np.abs(P.clo[eu] - P.clo[ev]) * P.D[eu, ev])
        t = time.time()
        s = 0
        for m, h in zip(models, hs):
            hu, hv = h[torch.tensor(eu)], h[torch.tensor(ev)]
            s = s + m.head(torch.cat([hu + hv, (hu - hv).abs(), X], 1)).squeeze(-1).numpy()
        th += time.time() - t
        scores.append(s / len(models))
    T["edge features (sampled)"] = tf; T["MLP head"] = th
    return np.concatenate(scores), np.concatenate(heur), T, P


def evaluate(models, G, gain, name, R):
    t0 = time.time()
    sc, heur, T, P = score_graph(models, G, R)
    T["model total"] = time.time() - t0
    eu, ev = P.candidates(0, P.n) if len(sc) < 3e7 else (None, None)
    top = np.argsort(-sc)[:100]
    t = time.time()
    if eu is not None:
        exact_gains_fast(P.D.astype(np.int64), np.stack([eu[top], ev[top]], 1))
    T["exact on top 100"] = time.time() - t
    m = eval_one(sc, gain); mh = eval_one(heur, gain)
    return m, mh, T


def load_models(name):
    opt = json.load(open(os.path.join(OUT, f"{name}.json")))["opt"]
    ms = []
    for s in opt["seeds"]:
        net = Net(4, 24, hidden=opt.get("hidden", 32)); net.load_state_dict(torch.load(os.path.join(OUT, f"{name}__s{s}.pt"))); ms.append(net)
    return ms


if __name__ == "__main__":
    name, kind = sys.argv[1], sys.argv[2]
    models = load_models(name)
    R = int(os.environ.get("XS_R", 128)); common.XS_R = R
    rows = {}
    if kind == "ba":
        n, k = int(sys.argv[3]), int(sys.argv[4])
        graphs = [(f"BA-{n} #{i}", nx.barabasi_albert_graph(n, 3, seed=9000 + i)) for i in range(k)]
    else:
        graphs = [(nm, load(nm)) for nm in sys.argv[3:]]
    for gname, G in graphs:
        n = G.number_of_nodes()
        gt_path = os.path.join(ROOT, "data_real", f"gt_{gname}.npz")
        if os.path.exists(gt_path):
            z = np.load(gt_path); gain, t_exact = z["gain"], float(z["secs"])
        else:
            t = time.time()
            Asp = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
            D = shortest_path(Asp, unweighted=True, directed=False).astype(np.int16)
            gain, _ = all_gains(D, Asp.toarray() > 0); t_exact = time.time() - t
        m, mh, T = evaluate(models, G, gain, gname, R)
        T["EXACT all edges"] = t_exact
        rows[gname] = {"model": m, "heuristic": mh, "time": T, "n": n, "candidates": int(len(gain))}
        print(f"{gname:18s} n={n:5d} cand={len(gain):>10,d} | MODEL mAP@10 {m['map10']:.3f} mAP@50 {m['map50']:.3f} "
              f"kneed10 {m['kneed10']:6d} ex100 {m['ex100_map10']:.3f}/{m['ex100_map50']:.3f} | RULE mAP@10 {mh['map10']:.3f} "
              f"ex100 {mh['ex100_map10']:.3f} | time model {T['model total']:.1f}s vs exact {t_exact:.1f}s", flush=True)
        print("     ", {k: round(v, 2) for k, v in T.items()}, flush=True)
    tag = f"{kind}_{'_'.join(sys.argv[3:])}_{name}_R{R}"
    json.dump(rows, open(os.path.join(OUT, f"scale_{tag}.json"), "w"), indent=1)
