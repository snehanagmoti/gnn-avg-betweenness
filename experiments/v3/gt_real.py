"""Exact ground truth (gain of every candidate edge) for real networks; saved to data_real/gt_<name>.npz."""
import os, sys, time
import numpy as np, networkx as nx
from scipy.sparse.csgraph import shortest_path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
from real_data import load
from src.fastgain import all_gains
OUT = os.path.join(ROOT, "data_real")
for name in sys.argv[1:]:
    path = os.path.join(OUT, f"gt_{name}.npz")
    if os.path.exists(path):
        continue
    G = load(name); n = G.number_of_nodes()
    t = time.time()
    Asp = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    D = shortest_path(Asp, unweighted=True, directed=False).astype(np.int16)
    A = Asp.toarray() > 0
    g, off = all_gains(D, A)
    secs = time.time() - t
    np.savez_compressed(path, gain=g, offset=off, edges=np.array(G.edges(), dtype=np.int32), n=n,
                        diameter=int(D.max()), secs=secs)
    print(f"{name:16s} n={n} candidates={len(g):,} diameter={D.max()} exact ALL in {secs:.1f}s", flush=True)
