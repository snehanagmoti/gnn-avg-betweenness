"""
Diverse training / validation graphs (several structures and sizes) with exact labels and the scalable
"xs" features, stored as compact records: all node data + a SUBSET of candidate edges
(all top-2% edges up to `top_cap`, plus `rest` random others) so big graphs stay cheap to train on.
"""
import os, sys, time
import numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, NODE_FEATS
import common
from scale import GraphPrep
from src.features import node_features
from src.fastgain import all_gains

CACHE = os.path.join(ROOT, "experiments", "v3", "cache_diverse")


def largest_cc(G):
    return nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())


def make(kind, n, seed):
    rng = np.random.default_rng(seed)
    if kind == "ba":
        G = nx.barabasi_albert_graph(n, int(rng.integers(1, 6)), seed=seed)
    elif kind == "plc":                                   # clustered scale-free (Holme-Kim)
        G = nx.powerlaw_cluster_graph(n, int(rng.integers(2, 5)), float(rng.uniform(0.2, 0.8)), seed=seed)
    elif kind == "ws":                                    # small world, long paths when p is small
        G = nx.connected_watts_strogatz_graph(n, int(rng.choice([4, 6])), float(rng.uniform(0.01, 0.15)), seed=seed)
    elif kind == "geo":                                   # random geometric graph ~ road / power networks
        r = float(rng.uniform(1.6, 2.4)) / np.sqrt(np.pi * n)
        G = largest_cc(nx.random_geometric_graph(int(n * 1.3), r * 1.0, seed=seed))
    elif kind == "grid":                                  # perturbed grid
        s = int(np.sqrt(n)); G = nx.grid_2d_graph(s, s)
        G = nx.convert_node_labels_to_integers(G)
        drop = rng.choice(G.number_of_edges(), int(0.15 * G.number_of_edges()), replace=False)
        ed = list(G.edges()); G.remove_edges_from([ed[i] for i in drop]); G = largest_cc(G)
    elif kind == "tree":
        G = nx.random_labeled_tree(n, seed=seed)
    elif kind == "er":
        G = largest_cc(nx.gnp_random_graph(n, float(rng.uniform(2.5, 6)) / n, seed=seed))
    return nx.convert_node_labels_to_integers(G)


def record(kind, n, seed, top_cap=3000, rest=3000):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{kind}_{n}_{seed}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    G = make(kind, n, seed)
    nn_ = G.number_of_nodes()
    P = GraphPrep(G, R=common.XS_R, seed=common.XS_SEED)
    gain, _ = all_gains(P.D, P.A)
    eu, ev = P.candidates(0, nn_)
    rng = np.random.default_rng(seed)
    order = np.argsort(-gain, kind="stable")
    E = len(gain)
    top = order[: max(1, round(0.02 * E))]
    top = top if len(top) <= top_cap else rng.choice(top, top_cap, replace=False)
    others = np.setdiff1d(np.arange(E), top)
    idx = np.sort(np.concatenate([top, rng.choice(others, min(rest, len(others)), replace=False)]))
    P.fit_stats()
    from scipy.stats import rankdata
    rank = rankdata(-gain, method="min")
    grade = sum((rank <= max(1, round(f * E))).astype(np.int64) for f in (0.10, 0.02, 0.005, 0.001))
    y = np.log(gain / gain.max() + 1e-6); y = (y - y.mean()) / y.std()
    r = dict(kind=kind, n=nn_, edges=np.array(G.edges(), dtype=np.int64), x=node_features(G, NODE_FEATS, D=P.D.astype(np.float64)),
             E=np.stack([eu[idx], ev[idx]], 1), xs=P.features(eu[idx], ev[idx]), y=y[idx].astype(np.float32),
             grade=grade[idx], gain=gain[idx], ncand=E)
    np.savez_compressed(path, **r)
    return r


PLAN = {"ba": [200, 500, 1000], "plc": [300, 700], "ws": [300, 500], "geo": [300, 500], "grid": [400, 625],
        "tree": [300], "er": [300, 600]}


def build(num_per_size=6, start=100):
    recs = []
    for kind, sizes in PLAN.items():
        for n in sizes:
            for s in range(start, start + num_per_size):
                t = time.time(); r = record(kind, n, s); recs.append(r)
                print(f"{kind:5s} n={r['n']:5d} cand={r['ncand']:>8,d} {time.time()-t:5.1f}s", flush=True)
    return recs


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 6, int(sys.argv[2]) if len(sys.argv) > 2 else 100)
