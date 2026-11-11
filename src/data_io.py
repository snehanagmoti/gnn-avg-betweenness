"""
Loading the 20 original BA-200 graphs in data/ (graph_XXX.npz + meta.json).
"""
import json
import os

import networkx as nx
import numpy as np


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def load_graph(data_dir, graph_id):
    """Return a dict with numpy arrays + a networkx graph."""
    z = np.load(os.path.join(data_dir, f"graph_{graph_id:03d}.npz"))
    G = nx.Graph()
    G.add_nodes_from(range(int(z["num_nodes"])))
    G.add_edges_from(map(tuple, z["edges"]))
    return {
        "graph_id": graph_id,
        "G": G,
        "edges": z["edges"],
        "missing_edges": z["missing_edges"],
        "reductions": z["reductions"],
        "labels": z["labels"],
        "features": z["features"],
        "avg_betweenness": float(z["avg_betweenness"]),
    }


def load_all(data_dir):
    meta = load_meta(data_dir)
    return [load_graph(data_dir, g["graph_id"]) for g in meta["graphs"]]
