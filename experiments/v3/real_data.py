"""Load real-world networks (largest connected component, undirected, simple) from data_real/raw.
Sources: R package 'networkdata' (KONECT copies, .rda igraph objects) and the karateclub datasets (.csv)."""
import os, warnings
import numpy as np, networkx as nx

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data_real", "raw")
RDA = ["maayan_pdzbase", "euroroad", "arenas_email", "maayan_faa", "usflights", "ucsocial", "petster",
       "powergrid", "chicagoroad"]
KC = ["wikipedia", "twitch", "lastfm", "deezer", "facebook", "github"]


def _edges_rda(name):
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.conversion.convert(rdata.parser.parse_file(os.path.join(RAW, f"{name}.rda")))
    g = obj[list(obj.keys())[0]]
    n = int(np.asarray(g[0])[0])
    src, dst = np.asarray(g[2], dtype=np.int64), np.asarray(g[3], dtype=np.int64)   # igraph: 0-based
    return n, np.stack([src, dst], 1)


def _edges_kc(name):
    e = np.loadtxt(os.path.join(RAW, f"kc_{name}.csv"), delimiter=",", skiprows=1, dtype=np.int64)
    return int(e.max()) + 1, e


def load(name):
    n, e = _edges_kc(name[3:]) if name.startswith("kc_") else _edges_rda(name)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(a), int(b)) for a, b in e if a != b)
    big = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(big).copy())


def all_names():
    return RDA + [f"kc_{k}" for k in KC]


if __name__ == "__main__":
    for nm in all_names():
        G = load(nm)
        n, m = G.number_of_nodes(), G.number_of_edges()
        print(f"{nm:16s} n={n:6d} m={m:7d} avg deg={2*m/n:5.2f} candidates={n*(n-1)//2 - m:>11,d}", flush=True)
