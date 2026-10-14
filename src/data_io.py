"""
Saving and loading the dataset.

For every graph i the dataset folder contains:
    graph_XXX.npz          existing edges, missing edges, reductions, labels, node features
    graph_XXX_ranking.csv  human-readable ground-truth ranking (best edge first)
plus meta.json with the generation settings.
"""
import csv
import json
import os

import networkx as nx
import numpy as np


def save_graph(data_dir, meta, G, missing_edges, reductions, labels, features, avg_bc):
    os.makedirs(data_dir, exist_ok=True)
    gid = meta["graph_id"]
    np.savez_compressed(
        os.path.join(data_dir, f"graph_{gid:03d}.npz"),
        edges=np.array(G.edges(), dtype=np.int64),
        missing_edges=missing_edges,
        reductions=reductions,
        labels=labels,
        features=features,
        num_nodes=G.number_of_nodes(),
        avg_betweenness=avg_bc,
    )
    order = np.argsort(-reductions, kind="stable")
    with open(os.path.join(data_dir, f"graph_{gid:03d}_ranking.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "u", "v", "reduction", "label"])
        for r, i in enumerate(order, start=1):
            w.writerow([r, missing_edges[i, 0], missing_edges[i, 1], f"{reductions[i]:.10g}", labels[i]])


def save_meta(data_dir, cfg_dict, graph_metas):
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump({"config": cfg_dict, "graphs": graph_metas}, f, indent=2)


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
