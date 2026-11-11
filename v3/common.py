"""
Shared code for version 3:
  * graph records: every candidate edge with its exact gain and its 24 clues (cached in v3/cache/)
  * the training target and the edge grades
  * evaluation of a scoring function (mAP@k, NDCG@k, shortlist metrics, exact check of the top 100)
"""
import os
import sys

import networkx as nx
import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_io import load_all                                              # noqa: E402
from src.fastgain import exact_gains_fast, level_histograms, pair_features, scale_factor  # noqa: E402
from src.features import distance_matrix, edge_features, node_features        # noqa: E402
from src.graph_generation import generate_graph                              # noqa: E402
from src.metrics import average_precision_at_k, ndcg_at_k, relevant_set       # noqa: E402

CACHE = os.path.join(ROOT, "v3", "cache")
RESULTS = os.path.join(ROOT, "v3", "results")
EDGE_FEATS = ["dist", "common_neighbors", "jaccard", "adamic_adar", "closeness_gap",
              "closeness_minmax", "degree_minmax"]                            # 9 old clues
PAIR_FEATS = ["aff_min", "aff_max", "aff_prod", "lvl_sum_min", "lvl_sum_max", "one_sided", "ub"]  # 7 new


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


# ------------------------------------------------------------------ graph records
def build_record(G):
    """Exact reduction (in average betweenness) + clue blocks for every missing edge of G."""
    n = G.number_of_nodes()
    D = distance_matrix(G)
    Di = D.astype(np.int64)
    E = missing_edges_of(G)
    red = exact_gains_fast(Di, E) * scale_factor(n)
    xe = edge_features(G, E, EDGE_FEATS, D=D)                     # 9 old clues, standardized
    pf = pair_features(Di, E)
    xp = np.stack([_std(np.log1p(pf[k])) for k in PAIR_FEATS], 1)  # 7 affected-set clues
    return dict(E=E, red=red, xe=xe.astype(np.float32), xp=xp.astype(np.float32), n=n,
                edges=np.array(G.edges(), dtype=np.int64))


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
    """The 20 original BA-200 graphs in data/ (0-13 train, 14-15 val, 16-19 test)."""
    recs = []
    for g in load_all(os.path.join(ROOT, "data")):
        r = record(f"base_{g['graph_id']:03d}", g["G"])
        assert np.allclose(np.sort(r["red"]), np.sort(g["reductions"]), rtol=1e-9, atol=1e-18)
        recs.append(r)
    return recs


def extra_records(num, n=200, gtype="ba", start_seed=1000, tag=None):
    """Freshly generated graphs: seeds 1000+ for extra training, 5000-5049 for the 50 new test graphs."""
    tag = tag or f"{gtype}{n}"
    return [record(f"{tag}_{s}", nx.convert_node_labels_to_integers(generate_graph(gtype, n, s)))
            for s in range(start_seed, start_seed + num)]


def level_hist_block(r, levels=4, chunk=20000):
    """8 histogram clues: nodes saving 1, 2, 3, 4+ hops on each side, as (smaller, bigger) count."""
    G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
    D = distance_matrix(G).astype(np.int64)
    E, L = r["E"], int(D.max())
    cols = []
    for i in range(0, len(E), chunk):
        cA, cB = level_histograms(D, E[i:i + chunk], L)
        A = np.concatenate([cA[:, 1:levels], cA[:, levels:].sum(1, keepdims=True)], 1)
        B = np.concatenate([cB[:, 1:levels], cB[:, levels:].sum(1, keepdims=True)], 1)
        cols.append(np.concatenate([np.minimum(A, B), np.maximum(A, B)], 1))
    X = np.concatenate(cols)
    return np.stack([_std(np.log1p(X[:, j])) for j in range(X.shape[1])], 1).astype(np.float32)


def clues(r):
    """The 24 clues per candidate edge: 9 old + 7 affected-set + 8 histogram."""
    if "xh" not in r:
        r["xh"] = level_hist_block(r)
    return np.concatenate([r["xe"], r["xp"], r["xh"]], 1)


# ------------------------------------------------------------------ target and grades
def grades(red):
    """0 rest, 1 top 10%, 2 top 2%, 3 top 0.5%, 4 top 0.1% (ties share the best rank)."""
    rank = rankdata(-red, method="min")
    E = len(red)
    return sum((rank <= max(1, round(f * E))).astype(int) for f in (0.10, 0.02, 0.005, 0.001))


def target(red):
    """What the model learns: log(reduction / best reduction), standardized within the graph."""
    y = np.log(red / red.max() + 1e-6)
    return (y - y.mean()) / y.std()


# ------------------------------------------------------------------ evaluation
def exact_check(score, red, K=100):
    """Re-score the model's top K edges with their exact reduction and put them first."""
    order = np.argsort(-score, kind="stable")
    s2 = np.empty(len(score)); s2[order] = -np.arange(len(score), dtype=float)
    s2[order[:K]] = 1e6 + red[order[:K]] / red.max()
    return s2


def eval_one(score, red, ks=(5, 10, 50)):
    order = np.argsort(-score, kind="stable")
    pos = np.empty(len(order), np.int64)
    pos[order] = np.arange(len(order))
    out = {f"map{k}": average_precision_at_k(order, red, k) for k in ks}
    out.update({f"ndcg{k}": ndcg_at_k(order, red, k) for k in (10, 50)})
    out["top50"] = float(relevant_set(red, 50)[order[:50]].mean())
    p10 = pos[relevant_set(red, 10)]
    out["rec10@100"] = float((p10 < 100).mean())               # share of true top 10 in the model's top 100
    out["kneed10"] = int(p10.max()) + 1                        # shortlist length that catches all of them
    o2 = np.argsort(-exact_check(score, red), kind="stable")
    out.update({f"ex100_map{k}": average_precision_at_k(o2, red, k) for k in ks})
    return out


def eval_many(scores, reds, ks=(5, 10, 50)):
    per = [eval_one(s, r, ks) for s, r in zip(scores, reds)]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def fmt(name, m):
    keys = [k for k in ("map5", "map10", "map50", "map100", "ndcg10", "ndcg50", "top50", "rec10@100",
                        "kneed10", "ex100_map10", "ex100_map50") if k in m]
    return f"{name:28s} " + "  ".join(f"{k} {m[k]:.3f}" if k != "kneed10" else f"{k} {m[k]:.0f}" for k in keys)
