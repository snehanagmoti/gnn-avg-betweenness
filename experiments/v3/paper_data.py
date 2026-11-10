"""
Dataset that mirrors the base paper (Aashish et al., Table 1), with OUR average-betweenness labels.
  BA   100 graphs: n in {165,180,190,200}, m in {3,4,5,6}, seeds in [0,1000]
  ER   100 graphs: n in {165,180,190,200}, p in [0.03, 0.0648]  (resampled until connected)
  Tree  30 graphs: full k-ary trees, n in {180,190,200,210,220} x k in {3,...,8}
  Path  10 graphs: n from 160 to 250
Split 64 / 16 / 20 (train / val / test) within each type.
Writes: version-3 records (experiments/v3/cache/paper_XXX.npz) and a version-2 data folder (data_paper/)
ordered train -> val -> test so version 2's lab.py can use split = [n_train, n_val, n_test].
    python paper_data.py [scale]     (scale = multiply the BA / ER counts, e.g. 5 for 500 + 500)
"""
import json, os, sys
import numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, record
from src.graph_generation import generate_er_graph
from src.ground_truth import make_labels
from src.data_io import save_graph, save_meta
from src.features import node_features, distance_matrix
from config import Config

SIZES = [165, 180, 190, 200]


def spec(scale=1):
    rng = np.random.default_rng(2026)
    out = []
    for i in range(100 * scale):
        out.append(dict(type="ba", n=int(rng.choice(SIZES)), m=int(rng.choice([3, 4, 5, 6])), seed=int(rng.integers(0, 1001))))
    for i in range(100 * scale):
        out.append(dict(type="er", n=int(rng.choice(SIZES)), p=float(rng.uniform(0.03, 0.0648)), seed=int(rng.integers(0, 100001))))
    for n in [180, 190, 200, 210, 220]:
        for k in range(3, 9):
            out.append(dict(type="tree", n=n, k=k))
    for n in np.linspace(160, 250, 10).astype(int):
        out.append(dict(type="path", n=int(n)))
    # 64/16/20 split inside each type (shuffled)
    for t in ("ba", "er", "tree", "path"):
        idx = [i for i, s in enumerate(out) if s["type"] == t]
        rng.shuffle(idx)
        a, b = round(0.64 * len(idx)), round(0.80 * len(idx))
        for j, i in enumerate(idx):
            out[i]["split"] = "train" if j < a else ("val" if j < b else "test")
    return out


def graph_of(s):
    if s["type"] == "ba":
        return nx.barabasi_albert_graph(s["n"], s["m"], seed=s["seed"])
    if s["type"] == "er":
        return generate_er_graph(s["n"], s["p"], s["seed"])
    if s["type"] == "tree":
        return nx.full_rary_tree(s["k"], s["n"])
    return nx.path_graph(s["n"])


def build(scale=1, write_v2=True):
    sp = spec(scale)
    tag = "paper" if scale == 1 else f"paper{scale}x"
    recs = []
    for i, s in enumerate(sp):
        G = nx.convert_node_labels_to_integers(graph_of(s))
        assert nx.is_connected(G)
        r = record(f"{tag}_{i:04d}", G)
        r["type"], r["split"], r["spec"] = s["type"], s["split"], s
        recs.append(r)
    if write_v2:
        cfg = Config()
        d = os.path.join(ROOT, f"data_{tag}")
        order = [i for sp_ in ("train", "val", "test") for i, s in enumerate(sp) if s["split"] == sp_]
        metas = []
        for gid, i in enumerate(order):
            r = recs[i]; s = sp[i]
            G = nx.Graph(); G.add_nodes_from(range(int(r["n"]))); G.add_edges_from(r["edges"].tolist())
            D = distance_matrix(G); n = int(r["n"])
            avg_bc = float(((D.sum() / 2 - n * (n - 1) / 2)) * 2 / ((n - 1) * (n - 2)) / n)
            meta = {"graph_id": gid, "type": s["type"], "n": n, "seed": s.get("seed", 0), "src_index": i, "split": s["split"]}
            save_graph(d, meta, G, r["E"], r["red"], make_labels(r["red"], cfg.label_bands),
                       node_features(G, cfg.node_feature_set, seed=gid, D=D), avg_bc)
            metas.append(meta)
        save_meta(d, {"paper_like": True, "scale": scale}, metas)
        counts = {k: sum(1 for s in sp if s["split"] == k) for k in ("train", "val", "test")}
        json.dump({"split": [counts["train"], counts["val"], counts["test"]], "order": order},
                  open(os.path.join(d, "split.json"), "w"))
        print("v2 data folder:", d, counts)
    return recs


def load(scale=1):
    return build(scale, write_v2=False)


if __name__ == "__main__":
    recs = build(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    from collections import Counter
    print(Counter((r["type"], r["split"]) for r in recs))


def extra_train(n_ba=0, n_er=0, trees=False, paths=False):
    """Extra TRAINING graphs that never coincide with the paper-style val / test graphs."""
    rng = np.random.default_rng(777)
    sp = []
    for i in range(n_ba):
        sp.append(dict(type="ba", n=int(rng.choice(SIZES)), m=int(rng.choice([3, 4, 5, 6])), seed=int(rng.integers(2000, 10**6))))
    for i in range(n_er):
        sp.append(dict(type="er", n=int(rng.choice(SIZES)), p=float(rng.uniform(0.03, 0.0648)), seed=int(rng.integers(2000, 10**6))))
    if trees:      # full k-ary trees with sizes NOT used by the paper set (so no test tree is repeated)
        for n in range(150, 265, 5):
            if n in (180, 190, 200, 210, 220):
                continue
            for k in range(2, 10):
                sp.append(dict(type="tree", n=n, k=k))
    if paths:      # path lengths not used by the paper set
        base = set(np.linspace(160, 250, 10).astype(int).tolist())
        for n in range(100, 301, 3):
            if n not in base:
                sp.append(dict(type="path", n=n))
    recs = []
    for s in sp:
        G = nx.convert_node_labels_to_integers(graph_of(s))
        key = "_".join(str(s[k]) for k in sorted(s))
        r = record(f"pextra_{key}", G)
        r["type"], r["split"], r["spec"] = s["type"], "train", s
        recs.append(r)
    return recs
