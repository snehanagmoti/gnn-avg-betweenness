# Version 3 experiment log

These experiments follow the three directions from the TA meeting after version 2:

1. Try anomaly detection.
2. Make the classifier a reliable filter, and find an efficient exact re-scoring algorithm.
3. Study the metrics in detail (see `METRICS.md`).

The method is the same as before: one change at a time, everything logged, and several seeds per setting.

## Setup

- **Data:** the version 2 data (20 BA graphs, n = 200, m = 3, split 14 / 2 / 4), so results are comparable with version 2.
- **New: 50 more test graphs** (BA-200, seeds 5000–5049).
  - With only 4 test graphs, run-to-run noise was about ±0.03.
  - Decisions are made mainly on the 50-graph set.
- **Extra training graphs:** now cheap thanks to the fast exact algorithm (BA-200, seeds 1000+).
- **New filter metrics** (all defined in `METRICS.md`):
  - `rec10@K`: share of the true top 10 inside the model's top K.
  - `kneed10`: the shortlist size needed to catch all of the true top 10.
  - `ex100_map10`: mAP@10 after exact re-scoring of the model's top 100.
- **Code** is in `experiments/v3/`:
  - `common.py`: records, features and metrics.
  - `gbm.py`: LightGBM.
  - `gnn3.py`: GNN.
  - `anomaly.py`, `ens.py`, `gen.py`, `timing.py`, `ref*.py`.
  - The algorithms are in `src/fastgain.py`.
  - JSON results are in `experiments/v3/results/`.

## 1. Efficient exact algorithm (`src/fastgain.py`)

**Total betweenness = Σ over pairs of (d(s,t) − 1)**, so adding (u, v) only matters through distance shortening.

**Which pairs can improve.** A pair {s, t} can only improve along s → u → v → t if both of these hold:
- s is at least 2 hops closer to u than to v (s ∈ S_u);
- t is at least 2 hops closer to v than to u (t ∈ T_v).

These are the "affected sets" of Parotsidis, Pitoura & Tsaparas (SDM 2015). So:

> gain(u,v) = Σ_{s∈S_u, t∈T_v} max(0, d(s,t) − d(s,u) − 1 − d(v,t))   (exact; each pair counted once)

**Why it is fast.** On BA-200 graphs the block S_u × T_v is on average **0.2% of all pairs**. Compiled with numba, the exact values match the old ground truth to 1e-20.

| Graph | Candidates | Old exact (all edges) | **New exact (all edges)** |
|---|---|---|---|
| BA-200 | 19,309 | ~3 s | **0.006 s** |
| BA-500 | 123,259 | ~7 min | **0.17 s** |
| BA-1000 | 496,509 | hours | **1.0–1.5 s** |
| BA-2000 | ~2 million | days | **13 s** |

**Upper bound.** From d(s,t) ≤ d(s,v) + d(v,t) and d(s,t) ≤ d(s,u) + d(u,t), each pair's gain is at most min(a_s, b_t). This gives a cheap bound:

> UB = Σ_{S_u × T_v} min(a_s, b_t)

UB can **certify** the exact top-k. Evaluating edges in UB order needed 25–35% of the edges at n = 200–1,000. That's correct, but not faster than the compiled exact for all edges.

**Consequence.** Up to ~2,000 nodes the exact answer for every edge takes seconds. Labels for training graphs are also almost free.

## 2. Simple rules and the version 2 reference (new metrics)

| Method | 4 test: mAP@10 / mAP@50 | 50 new: mAP@10 / mAP@50 / top-50 | 50 new: kneed10 (max) | 50 new: ex100 mAP@10 / mAP@50 |
|---|---|---|---|---|
| Version 2 GNN (E31, 5 seeds) | 0.538 / 0.662 | 0.608 / 0.681 / 0.735 | 571 | 0.974 / 0.921 |
| Closeness gap × distance | 0.271 / 0.385 | 0.370 / 0.409 / 0.495 | 18,683 | 0.840 / 0.653 |
| New feature "one-sided gain" alone | 0.575 / 0.628 | 0.628 / 0.704 / 0.731 | 5,276 | 0.928 / 0.860 |
| UB alone | 0.030 / 0.050 | – | – | – |

## 3. Classifier / ranker experiments (one change at a time)

### LightGBM on hand-made features (3 seeds)

Features:
- **xn**: node features of both ends (min/max).
- **xe**: version 2 edge features.
- **xp**: 7 affected-set features (O(n) per edge).
- **xh**: level histograms.

| Run | Change | 4 test: mAP@10 / mAP@50 | 50 new: mAP@10 / mAP@50 | kneed10 (4 test) |
|---|---|---|---|---|
| G01 | binary top-2%, xn+xe | 0.364 / 0.490 | – | 503 |
| G02 | binary top-2%, all | 0.128 / 0.388 | – | 279 |
| G03 | 3-class, all | 0.442 / 0.532 | – | 288 |
| G05 | LambdaMART (learning to rank) | 0.610 / 0.737 | – | 410 |
| **G04** | **regression on log(gain)** | **0.761 / 0.810** | 0.718 / 0.833 | **62** |
| G06 | regression, xn+xe only | 0.452 / 0.569 | – | 612 |
| G07 | regression, xp only | 0.576 / 0.709 | – | 2,521 |
| G08 | regression, xe+xp | 0.741 / 0.815 | – | 100 |
| G09 | all rows instead of sampling | 0.712 / 0.796 | – | 85 |
| G10–G12 | +50 / 150 / 300 extra graphs | 0.744 / 0.804 → 0.761 / 0.843 | – | 53 → 38 |
| G13 | top edges weighted ×3 | 0.777 / 0.800 | 0.729 / 0.834 | 75 |
| G14–G16 | rank target, 63 leaves, Huber | no gain | 0.709–0.724 / 0.823–0.835 | – |
| G17 | weighted + 300 extra | 0.756 / 0.840 | 0.774 / 0.847 | 36 |

**Finding:** predicting *how much* an edge helps (regression) beats any class label. A class throws away the order inside the top group. The new affected-set features are essential.

### GNN (GraphSAGE, edge vector = sum + |diff| of embeddings + hand-made edge features; 3 seeds)

| Run | Change | 4 test: mAP@10 / mAP@50 | 50 new: mAP@10 / mAP@50 / top-50 | 50 new: kneed10 |
|---|---|---|---|---|
| N05 | regression, version 2 edge features only | 0.555 / 0.647 | 0.599 / 0.658 / 0.724 | 310 |
| N02 | MLP on xe+xp, **no GNN** | 0.774 / 0.812 | 0.788 / 0.840 / 0.858 | 185 |
| N01 | GNN, regression, xe+xp | 0.817 / 0.885 | 0.810 / 0.850 / 0.874 | 127 |
| N03 | + top edges weighted ×3 | 0.827 / 0.883 | 0.808 / 0.843 / 0.871 | 91 |
| N04 | + triplet loss | 0.811 / 0.876 | 0.806 / 0.851 / 0.874 | 166 |
| N06 | triplet loss only, xe+xp | 0.773 / 0.861 | 0.816 / 0.856 / 0.876 | 164 |
| N07 | N03 + 300 extra graphs | 0.787 / 0.887 | 0.819 / 0.867 / 0.886 | 148 |
| N09 | N03 + level histograms (xh) | 0.833 / 0.900 | 0.803 / 0.849 / 0.873 | 98 |
| N10 | N03 + width 64 | 0.820 / 0.890 | 0.830 / 0.865 / 0.882 | 132 |
| N11 | width 64 + 300 extra | 0.825 / 0.896 | 0.840 / 0.882 / 0.896 | 124 |
| N12 | width 64 + 300 extra + xh | 0.808 / 0.914 | 0.847 / 0.882 / 0.898 | 88 |
| **N13** | **N12, 5 seeds (FINAL)** | **0.806 ± 0.033 / 0.905 ± 0.014** | **0.845 / 0.884 / 0.899** | 119 |
| ENS | average of N13's 5 runs | 0.808 / 0.909 | **0.854 / 0.896 / 0.908** | 101 |
| ENS | N10 + N07 + N09 (9 models) | 0.857 / 0.908 | 0.847 / 0.879 / 0.896 | 120 |
| N14 | trained on mixed types and sizes | 0.786 / 0.866 | 0.809 / 0.854 / 0.877 | 141 |

**Findings:**
- **The new pair features are the main source of the gain.** With version 2 features, regression ≈ version 2 (N05).
- **The GNN adds ~0.04–0.06 over the same features without message passing** (N02 vs N01).
- **Width 64, more training graphs, the top-weighted loss and the histograms** each add a little.

### Shortlist + exact re-scoring with the final model (N13, 50 new test graphs)

| Shortlist | mAP@10 | mAP@50 |
|---|---|---|
| model alone | 0.845 | 0.884 |
| top 50 → exact | 0.997 | 0.899 |
| top 100 → exact | **0.999** | **0.998** |
| top 200 → exact | 1.000 | 1.000 |

Share of the true top 10 inside the first 50: **99.5%**. For comparison, version 2 needed a top 200 shortlist to reach 0.98 / 0.98.

## 4. Anomaly detection (direction 1)

| Use | Result |
|---|---|
| Isolation Forest per graph, as a ranking | mAP@10 0.01; but 88–93% of the true top 10 are in its top 200 (best edges *are* unusual) |
| Local Outlier Factor per graph | ~0 |
| Isolation Forest fitted on ordinary edges (novelty) | mAP@10 0.01; recall10@200 0.49–0.72 |
| Isolation Forest score as an extra LightGBM feature (G18) | 0.729 / 0.835 vs 0.729 / 0.834 without: **no gain** |
| Graph-level "is this graph unfamiliar?" score | flagged ER most (+0.040), where the model works; flagged trees less (+0.018), where it fails; BA-500 not flagged although it drops → **not a reliable warning** |

**Conclusion:** anomaly detection does not help this problem in any of the four roles tested.

## 5. Generalisation without retraining (trained on BA-200 only)

| Test set | Version 2: mAP@10 / mAP@50 | Simple rule | **Version 3 (N13)** | Version 3 + exact top 100 (mAP@10) | Version 3 trained on mixed types (N14) |
|---|---|---|---|---|---|
| BA-500 (6 graphs) | 0.553 / 0.533 | 0.232 / 0.345 | **0.617 / 0.765** | 0.897 | 0.587 / 0.747 |
| ER-200 (10) | 0.441 / 0.430 | 0.433 / 0.358 | **0.720 / 0.754** | 0.978 | **0.793 / 0.806** |
| 2-tree-200 (10) | 0.072 / 0.248 | 0.001 / 0.113 | 0.503 / 0.561 | 0.976 | **0.620 / 0.679** |
| Random tree-200 (10) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.001 | 0.018 | 0.093 / 0.084 |

**Findings:**
- **Bigger graphs and ER transfer well.**
- **Training on mixed types** helps ER and 2-trees, but costs ~0.03–0.04 on BA.
- **Random trees remain unsolved** by learning, but the exact algorithm is instant there.

## 6. Timing per graph (single core except the compiled exact, which uses 2)

| Part | BA-200 | BA-500 | BA-1000 |
|---|---|---|---|
| distance matrix | 0.006 s | 0.03 s | 0.11 s |
| **exact gains, ALL edges (new)** | **0.006 s** | **0.17 s** | **1.5 s** |
| version 2 features | 0.009 s | 0.05 s | 0.21 s |
| affected-set + histogram features (numpy, not compiled) | 0.15 s | 2.6 s | – |
| GNN forward, all edges | 0.02 s | 0.22 s | – |
| exact on GNN top 100 | <0.001 s | 0.001 s | – |

**Honest conclusion on speed.** The new pair features cost the same order of work as the compiled exact algorithm, which is O(n) per edge plus a small block. So at these sizes the **exact algorithm is the fastest way to get the perfect answer**. The learned model is the best *predictor* we have, and the shortlist from it needs only ~100 exact evaluations. But it is not faster than computing everything exactly up to ~2,000 nodes.

## Final configuration (version 3)

- **Model:** GNN (GraphSAGE, 3 layers, width 64) plus an MLP that predicts the standardized log(gain) of every edge.
  - Loss: mean squared error, with edges weighted 1 + 2 × grade.
  - Grades: top 10% / 2% / 0.5% / 0.1% → 1–4.
- **Edge vector:** sum + |diff| of the two node embeddings, plus:
  - 9 version 2 edge features;
  - 7 affected-set features;
  - 8 level-histogram features.
- **Training:**
  - 14 version 2 graphs + 300 extra BA-200 graphs, 40 graphs per epoch;
  - all top-10% edges plus 4,000 others per graph;
  - mini-batches of 500, Adam lr 0.003;
  - early stopping on validation mean of mAP@10 and mAP@50.
- **Use:**
  - for best accuracy, re-score the top 100 with the compiled exact method;
  - for small graphs (≤ ~2,000 nodes), just run the exact method on all edges.
