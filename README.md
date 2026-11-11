# Learning Which Edge to Add: GNN Link Prediction for Minimizing Average Betweenness (version 3)

Given a connected network, which **single missing edge** reduces the network's **average betweenness centrality** the most?
Version 3 ranks every candidate edge with a **GraphSAGE + MLP** model and then checks its **top 100 exactly** with a fast exact algorithm.

The approach follows *Aashish et al., "Deep Learning based Link Prediction for Minimizing Maximum Betweenness"*, adapted from maximum to average betweenness.

## Method

1. **Fast exact reduction** (`src/fastgain.py`). Total betweenness = Σ over node pairs of (d(s,t) − 1), so adding edge (u, v) lowers it by the total hops saved. A pair can only get shorter if s is at least 2 hops closer to u than to v and t at least 2 hops closer to v than to u ("affected sets", Parotsidis et al. 2015). Only those pairs are checked (≈ 0.2% of pairs on BA-200), compiled with numba:
   all candidate edges of a BA-200 graph in 0.006 s, BA-500 0.17 s, BA-1000 ≈ 1.5 s.
2. **Inputs.** 4 features per node (log-degree, clustering, closeness, core number) and **24 clues** per candidate edge: 9 classic ones (distance, common neighbours, Jaccard, Adamic–Adar, closeness, degree) + 7 affected-set clues + 8 level-histogram clues.
3. **Model** (`v3/train.py`, `src/model.py`, ≈ 35k weights). GraphSAGE (3 layers, width 64) gives 64 numbers per node. Edge vector = h_u + h_v, |h_u − h_v|, 24 clues = 152 numbers. MLP head 152 → 64 → 64 → 64 → 1 = **one score per edge**.
4. **Training.** Target = log(reduction / best reduction), standardized per graph. Weighted squared error (1 + 2 × grade; grades for the top 10% / 2% / 0.5% / 0.1%). Per graph all top-10% edges + 4,000 others, batches of 500, Adam (lr 0.003), early stopping on validation mAP.
5. **Prediction.** Score all candidates, keep the top 100, compute their exact reduction, re-sort.

## Results

50 unseen BA-200 graphs (5 trained models):

| | mAP@5 | mAP@10 | mAP@50 | NDCG@10 |
|---|---|---|---|---|
| Model alone | 0.843 | 0.845 | 0.884 | 0.991 |
| + exact check of the top 100 | 1.000 | 0.999 | 0.998 | 1.000 |

Paper-style dataset (240 graphs as in Aashish et al., Table 1; 48 test graphs; 3 trained models), mAP@10 / mAP@100:

| BA (20) | ER (20) | Tree (6) | Path (2) | All (48) |
|---|---|---|---|---|
| 0.88 / 0.91 | 0.80 / 0.84 | 1.00 / 0.93 | 0.04 / 0.30 | 0.83 / 0.86 |

Paths remain the weak spot. Metrics are explained in `v3/METRICS.md`.

## Repository

```
data/                 the 20 original BA-200 graphs (14 train / 2 val / 4 test) with exact reductions
src/
  fastgain.py         fast exact reduction + affected-set clues
  features.py         distances, node features, the 9 classic edge clues
  model.py            GraphSAGE, MLP, mean adjacency
  metrics.py          mAP@k, NDCG@k
  graph_generation.py BA / ER / k-tree / path generators
  data_io.py          loads data/
v3/
  common.py           graph records (cached in v3/cache/), clues, target, evaluation
  paper_data.py       paper-style dataset (240 graphs) + extra training trees / paths
  train.py            training (BA or paper-style dataset)
  evaluate.py         evaluates the saved final models
  METRICS.md          metric definitions with a worked example
  results/            final models (final_ba__s0-4.pt, final_paper__s0-2.pt) and their results
```

## Run

```bash
pip install -r requirements.txt
python v3/evaluate.py ba       # saved final BA model: 4 original + 50 new test graphs (~1 min)
python v3/evaluate.py paper    # saved final paper-style model, results per graph type (~2 min)
python v3/train.py ba          # retrain: 14 + 300 BA-200 graphs, 5 seeds
python v3/train.py paper       # retrain on the paper-style dataset, 3 seeds
```

The first run builds the graph records (exact reductions + clues) and caches them in `v3/cache/`.
