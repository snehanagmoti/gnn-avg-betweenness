"""
Paper-style dataset (base paper, Table 1) with our exact average-betweenness labels.
  BA   100 graphs: n in {165,180,190,200}, m in {3,4,5,6}
  ER   100 graphs: n in {165,180,190,200}, p in [0.03, 0.0648]  (resampled until connected)
  Tree  30 graphs: full k-ary trees, n in {180,190,200,210,220} x k in {3,...,8}
  Path  10 graphs: n from 160 to 250
Split 64 / 16 / 20 (train / val / test) inside each type -> 153 / 39 / 48 graphs.
"""
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import record                                  # noqa: E402
from src.graph_generation import generate_er_graph         # noqa: E402

SIZES = [165, 180, 190, 200]


def spec():
    rng = np.random.default_rng(2026)
    out = []
    for _ in range(100):
        out.append(dict(type="ba", n=int(rng.choice(SIZES)), m=int(rng.choice([3, 4, 5, 6])), seed=int(rng.integers(0, 1001))))
    for _ in range(100):
        out.append(dict(type="er", n=int(rng.choice(SIZES)), p=float(rng.uniform(0.03, 0.0648)), seed=int(rng.integers(0, 100001))))
    for n in [180, 190, 200, 210, 220]:
        for k in range(3, 9):
            out.append(dict(type="tree", n=n, k=k))
    for n in np.linspace(160, 250, 10).astype(int):
        out.append(dict(type="path", n=int(n)))
    for t in ("ba", "er", "tree", "path"):                  # 64/16/20 split inside each type
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


def load():
    """All 240 graphs as records, each tagged with its type and split."""
    recs = []
    for i, s in enumerate(spec()):
        r = record(f"paper_{i:04d}", nx.convert_node_labels_to_integers(graph_of(s)))
        r["type"], r["split"] = s["type"], s["split"]
        recs.append(r)
    return recs


def extra_trees_paths():
    """Extra TRAINING trees and paths whose sizes are not used by the paper-style val/test graphs."""
    sp = []
    for n in range(150, 265, 5):
        if n not in (180, 190, 200, 210, 220):
            sp += [dict(type="tree", n=n, k=k) for k in range(2, 10)]
    base = set(np.linspace(160, 250, 10).astype(int).tolist())
    sp += [dict(type="path", n=n) for n in range(100, 301, 3) if n not in base]
    recs = []
    for s in sp:
        key = "_".join(str(s[k]) for k in sorted(s))
        r = record(f"pextra_{key}", nx.convert_node_labels_to_integers(graph_of(s)))
        r["type"], r["split"] = s["type"], "train"
        recs.append(r)
    return recs


if __name__ == "__main__":
    from collections import Counter
    print(Counter((r["type"], r["split"]) for r in load()))
