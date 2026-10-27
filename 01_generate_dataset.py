"""
STEP 1 - Dataset
Generate graphs (BA by default), compute the ground-truth reduction in average betweenness for every
missing edge, make classifier labels and node features, and save everything to `data/`.

Run:  python 01_generate_dataset.py
"""
import dataclasses
import time

from config import CFG
from src.data_io import save_graph, save_meta
from src.features import node_features
from src.graph_generation import generate_graphs
from src.ground_truth import average_betweenness, brute_force_check, compute_reductions, make_labels
from src.plots import plot_reduction_curves


def main(cfg=CFG, verify=True):
    graphs = generate_graphs(cfg.num_graphs, cfg.num_nodes, cfg.m_values, cfg.base_seed,
                             cfg.graph_type, cfg.er_p_values, cfg.ktree_k)
    metas = []
    for G, meta in graphs:
        t = time.time()
        missing, red = compute_reductions(G, cfg.normalize_betweenness)
        if verify and meta["graph_id"] == 0:          # brute-force check on the first graph only
            brute_force_check(G, missing, red, cfg.normalize_betweenness)
        labels = make_labels(red, cfg.label_bands)
        feats = node_features(G, cfg.node_feature_set, seed=meta["graph_id"])
        avg_bc = average_betweenness(G, cfg.normalize_betweenness)
        save_graph(cfg.data_dir, meta, G, missing, red, labels, feats, avg_bc)

        meta.update(edges=G.number_of_edges(), missing_edges=len(missing),
                    avg_betweenness=avg_bc, best_reduction=float(red.max()),
                    positives=int((labels >= 1).sum()))
        metas.append(meta)
        print(f"graph {meta['graph_id']:3d} | {meta['type']} n={meta['n']} | edges={meta['edges']} | "
              f"missing={meta['missing_edges']} | positives={meta['positives']} | "
              f"best reduction={meta['best_reduction']:.3e} | {time.time() - t:.1f}s")

    save_meta(cfg.data_dir, dataclasses.asdict(cfg), metas)
    plot_reduction_curves(cfg.data_dir, cfg.output_dir)
    print(f"Saved {len(metas)} graphs to '{cfg.data_dir}/'")


if __name__ == "__main__":
    main()
