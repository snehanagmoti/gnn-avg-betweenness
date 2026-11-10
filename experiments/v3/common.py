"""
Shared code for the version 3 experiments: graph records with exact gains and all features,
caching, and the evaluation of a scoring function as a ranking AND as a shortlist filter.
"""
import os
import sys

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data_io import load_all                                   # noqa: E402
from src.fastgain import exact_gains_fast, pair_features, scale_factor  # noqa: E402
from src.features import distance_matrix, edge_features, node_features  # noqa: E402
from src.graph_generation import generate_graph                   # noqa: E402
from src.metrics import average_precision_at_k, relevant_set       # noqa: E402

XS_R, XS_SEED = 128, 0
CACHE = os.path.join(ROOT, "experiments", "v3", "cache")
NODE_FEATS = ["degree", "clustering", "closeness", "core_number"]
EDGE_FEATS = ["dist", "common_neighbors", "jaccard", "adamic_adar", "closeness_gap",
              "closeness_minmax", "degree_minmax"]
PAIR_FEATS = ["aff_min", "aff_max", "aff_prod", "lvl_sum_min", "lvl_sum_max", "one_sided", "ub"]


def _std(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def missing_edges_of(G):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n)) > 0
    iu = np.triu_indices(n, 1)
    m = ~A[iu]
    return np.stack([iu[0][m], iu[1][m]], 1).astype(np.int64)


def build_record(G):
    """Exact reductions + three feature blocks for every missing edge of G."""
    n = G.number_of_nodes()
    D = distance_matrix(G)
    Di = D.astype(np.int64)
    E = missing_edges_of(G)
    gain = exact_gains_fast(Di, E)
    red = gain * scale_factor(n)
    nf = node_features(G, NODE_FEATS, D=D)                        # (n, 4), standardized
    u, v = E[:, 0], E[:, 1]
    xn = np.concatenate([np.minimum(nf[u], nf[v]), np.maximum(nf[u], nf[v])], 1)
    xe = edge_features(G, E, EDGE_FEATS, D=D)                     # (E, 9), standardized
    pf = pair_features(Di, E)
    xp = np.stack([_std(np.log1p(pf[k])) for k in PAIR_FEATS], 1)
    raw_ub = pf["ub"]
    return dict(E=E, red=red, xn=xn.astype(np.float32), xe=xe.astype(np.float32),
                xp=xp.astype(np.float32), ub=raw_ub, n=n, edges=np.array(G.edges(), dtype=np.int64))


def record(name, G):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    r = build_record(G)
    np.savez_compressed(path, **r)
    return r


def base_records():
    """The 20 version-2 BA graphs (0-13 train, 14-15 val, 16-19 test)."""
    gs = load_all(os.path.join(ROOT, "data"))
    recs = []
    for g in gs:
        r = record(f"base_{g['graph_id']:03d}", g["G"])
        assert np.allclose(np.sort(r["red"]), np.sort(g["reductions"]), rtol=1e-9, atol=1e-18)
        recs.append(r)
    return recs


def extra_records(num, n=200, gtype="ba", start_seed=1000, m=3, p=0.04, k=2, tag=None):
    tag = tag or f"{gtype}{n}"
    recs = []
    for s in range(start_seed, start_seed + num):
        G = nx.convert_node_labels_to_integers(generate_graph(gtype, n, s, m, p, k))
        recs.append(record(f"{tag}_{s}", G))
    return recs


def features(r, blocks=("xn", "xe", "xp")):
    if "xa" in blocks and "xa" not in r:            # anomaly score (Isolation Forest, per graph)
        from sklearn.ensemble import IsolationForest
        X = np.concatenate([r[b] for b in ("xn", "xe", "xp")], 1)
        f = IsolationForest(n_estimators=100, max_samples=512, random_state=0, n_jobs=2).fit(X)
        r["xa"] = _std(-f.score_samples(X))[:, None].astype(np.float32)
    if "xs" in blocks and "xs" not in r:            # scalable features with R sampled landmarks
        from scale import GraphPrep
        G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
        P = GraphPrep(G, R=XS_R, seed=XS_SEED)
        eu, ev = P.candidates(0, P.n)
        assert (eu == r["E"][:, 0]).all() and (ev == r["E"][:, 1]).all()
        r["xs"] = P.features(eu, ev)
    if "xh" in blocks and "xh" not in r:            # affected-set level histograms (levels 1..4+)
        r["xh"] = level_hist_block(r)
    return np.concatenate([r[b] for b in blocks], 1)


def level_hist_block(r, levels=4, chunk=20000):
    from src.fastgain import level_histograms
    G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
    D = distance_matrix(G).astype(np.int64)
    E = r["E"]
    L = int(D.max())
    cols = []
    for i in range(0, len(E), chunk):
        cA, cB = level_histograms(D, E[i:i + chunk], L)
        A = np.concatenate([cA[:, 1:levels], cA[:, levels:].sum(1, keepdims=True)], 1)
        B = np.concatenate([cB[:, 1:levels], cB[:, levels:].sum(1, keepdims=True)], 1)
        cols.append(np.concatenate([np.minimum(A, B), np.maximum(A, B)], 1))
    X = np.concatenate(cols)
    return np.stack([_std(np.log1p(X[:, j])) for j in range(X.shape[1])], 1).astype(np.float32)


# ------------------------------------------------------------------ evaluation
KS = (50, 100, 200, 500)


def eval_one(score, red):
    order = np.argsort(-score, kind="stable")
    pos = np.empty(len(order), np.int64)
    pos[order] = np.arange(len(order))
    out = {"map10": average_precision_at_k(order, red, 10),
           "map50": average_precision_at_k(order, red, 50)}
    rel50 = relevant_set(red, 50)
    out["top50"] = float(rel50[order[:50]].mean())
    out["opt"] = float(red[order[0]] / red.max())
    for k in (10, 50):
        rel = relevant_set(red, k)
        p = pos[rel]
        out[f"kneed{k}"] = int(p.max()) + 1                # shortlist size that catches all of them
        for K in KS:
            out[f"rec{k}@{K}"] = float((p < K).mean())
    for K in (50, 100, 200):                               # shortlist of K, re-scored exactly
        s2 = -pos.astype(np.float64)                       # model order for everything else
        top = order[:K]
        s2[top] = 1e6 + red[top] / red.max()
        o2 = np.argsort(-s2, kind="stable")
        out[f"ex{K}_map10"] = average_precision_at_k(o2, red, 10)
        out[f"ex{K}_map50"] = average_precision_at_k(o2, red, 50)
    return out


def eval_many(scores, recs):
    per = [eval_one(s, r["red"]) for s, r in zip(scores, recs)]
    out = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    for k in (10, 50):
        out[f"kneed{k}_max"] = max(p[f"kneed{k}"] for p in per)
    return out


SHOW = ["map10", "map50", "top50", "rec10@50", "rec10@200", "rec50@200", "kneed10_max",
        "kneed50_max", "ex100_map10", "ex100_map50", "ex200_map50"]


def fmt(name, m, cols=SHOW):
    return f"{name:34s} " + " ".join(f"{m[c]:.3f}" if isinstance(m[c], float) else f"{m[c]:5d}"
                                       for c in cols)


def header(cols=SHOW):
    return f"{'method':34s} " + " ".join(f"{c[:11]:>5s}" for c in cols)
