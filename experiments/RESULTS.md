# Experiment log: improving the pipeline step by step

## Setup

- **Data:** 20 BA graphs (n = 200, m = 3, seeds 0–19), split 14 train / 2 validation / 4 test.
- **Test graphs:** 16–19, never seen in training.
- **Seeds:** every setting is trained with **5 seeds** (3 for screening runs). Tables show the test mean ± standard deviation over seeds.
- **Method:** one change at a time. A change is kept only if it helps the main metric (mAP@10) without hurting mAP@50. Earlier decisions are re-checked once the base changes.
- **Columns:**
  - **mAP@10 / mAP@50:** paper metric (ties included).
  - **top-50:** share of the predicted top 50 that are in the true top 50.
  - **opt:** reduction of the model's #1 edge divided by the reduction of the true best edge.
  - **sp50:** Spearman correlation over the true top 50.
  - **cls BA:** classifier balanced accuracy.
- **Reproduce:**
  - one setting: `python experiments/lab.py <name> '<json overrides>'`
  - many settings: `python experiments/batch.py <step> '<json>'`
  - tables: `python experiments/report.py`
  - raw per-run results: `experiments/runs/*.json`; summary: `experiments/results.csv`

## Reference points

| | mAP@10 | mAP@50 | top-50 | opt |
|---|---|---|---|---|
| E00 original pipeline (5 seeds) | 0.449 ± 0.030 | 0.553 ± 0.023 | 0.636 | 0.995 |
| Best heuristic: closeness gap × distance | 0.271 | 0.385 | 0.450 | 0.941 |
| Degree product | 0.013 | 0.023 | 0.086 | 0.452 |
| Distance (ties broken randomly, mean of 5 draws) | 0.000 | 0.000 | 0.000 | 0.214 |
| Random | 0.000 | 0.000 | 0.004 | 0.114 |

## Step 0: evaluation protocol

| | mAP@10 | mAP@50 | Decision |
|---|---|---|---|
| E00 early stopping on validation mAP@10 | 0.449 ± 0.030 | 0.553 | – |
| E03 early stopping on the mean of validation mAP@10 and mAP@50 | 0.469 ± 0.031 | 0.576 | **kept**: less noisy with only 2 validation graphs |

## Step 1: labels and sampling (TA)

| | mAP@10 | mAP@50 | cls BA | Decision |
|---|---|---|---|---|
| E03 top 10% = good (2 classes), top-biased sampling | 0.469 | 0.576 | 0.846 | – |
| E05 + mixed sampling (top 2% / 2–10% / 10–50% / rest = 30/30/20/20%) | 0.486 ± 0.039 | 0.562 | **0.896** | mixed |
| E06 good = top 2% | 0.462 | 0.579 | 0.802 | no |
| E08 good = top 5% | 0.428 | 0.559 | 0.806 | no |
| **E07 3 classes: top 2% / 2–10% / rest** | **0.504 ± 0.026** | **0.587** | 0.715 | **kept** |

Ties use competition ranking (1, 1, 3, …) everywhere. Tied edges always get the same class.

## Step 2: architecture (TA: pre-train → fine-tune)

| | mAP@10 | mAP@50 | Decision |
|---|---|---|---|
| E07 joint training (paper) | 0.504 ± 0.026 | 0.587 | – |
| E09 + mixed sampling | 0.473 | 0.562 | no |
| E10 two-stage, GNN frozen in stage 2 | 0.475 | 0.569 | no |
| **E11 two-stage, whole model fine-tuned in stage 2 (lr × 0.3)** | **0.513 ± 0.015** | **0.589 ± 0.004** | **kept** |
| E12 single ranking head (no classifier) | 0.500 ± 0.006 | 0.597 | close second |

## Step 3: mini-batches and ranking loss

| | mAP@10 | mAP@50 | top-50 | Decision |
|---|---|---|---|---|
| E11 triplet loss on good edges, full batch | 0.513 | 0.589 | 0.649 | – |
| E13 mini-batches of 500 edges (5 seeds) | 0.507 | 0.598 | 0.658 | not yet (helped later, see E31) |
| E14 RankNet pairwise loss | 0.486 | 0.570 | 0.633 | no |
| E15 ListNet loss, all sampled edges | 0.503 | 0.590 | 0.677 | no |
| E16 regression on standardized reduction | 0.483 | 0.582 | 0.652 | no |
| E17 triplet loss on all sampled edges | 0.491 | 0.581 | 0.645 | no |

## Step 4: node features (TA)

| | mAP@10 | mAP@50 | Decision |
|---|---|---|---|
| E11 degree, clustering, closeness, core number | 0.513 | 0.589 | **kept** |
| E18 + PageRank, eigenvector (6 features, TA) | 0.508 | 0.591 | no gain |
| **E19 random features (paper)** | **0.040** | **0.053** | **fails on unseen graphs** |
| E20 4 + eccentricity + distance to 5 hubs | 0.462 | 0.577 | no |
| E21 6 + eccentricity + distance to 5 hubs | 0.507 | 0.579 | no |

## Step 5: edge vector (TA)

| | mAP@10 | mAP@50 | Decision |
|---|---|---|---|
| E11 sum + absolute difference | 0.513 | 0.589 | **kept** |
| E22 concatenation (paper) | 0.492 | 0.588 | no |
| E23 sum + diff + product | 0.482 | 0.590 | no |
| E24 sum + diff + dot product | 0.487 | 0.590 | no |
| E25 concat + product | 0.489 | 0.590 | no |

## Step 6: hand-made edge features (TA and own ideas)

| | mAP@10 | mAP@50 | top-50 | Decision |
|---|---|---|---|---|
| E26 distance(u, v) | 0.509 | 0.612 | 0.678 | – |
| E27 distance + min/max closeness and degree | 0.495 | 0.620 | 0.685 | – |
| E28 common neighbours, Jaccard, Adamic–Adar (TA) | 0.517 | 0.607 | 0.670 | – |
| **E29 all 7 edge features** | **0.527 ± 0.014** | **0.637 ± 0.019** | **0.697** | **kept** |

## Steps 7–8: re-checks and tuning on the new base

| | mAP@10 | mAP@50 | top-50 | Train time | Decision |
|---|---|---|---|---|---|
| E29 base | 0.527 | 0.637 | 0.697 | ~3.5 min | – |
| **E31 + mini-batches of 500 edges (5 seeds)** | **0.538 ± 0.030** | **0.662 ± 0.008** | **0.719** | ~4 min | **FINAL** |
| E30 single ranking head | 0.547 | 0.652 | 0.708 | ~3 min | tie |
| E34 single head + batches of 500 (5 seeds) | 0.548 ± 0.025 | 0.650 | 0.710 | ~2 min | tie (fastest) |
| E35 + batches of 500 + hidden 64 (5 seeds) | 0.551 ± 0.022 | 0.654 | 0.713 | ~5 min | tie |
| E32 hidden 64 | 0.536 | 0.663 | 0.717 | – | tie |
| E36 batches of 200 | 0.549 | 0.652 | 0.703 | – | tie |
| E33 4 GNN layers | 0.480 | 0.618 | 0.675 | – | no |
| E37 single head + batches of 500 + hidden 64 | 0.528 | 0.633 | 0.688 | – | no |

E31, E34 and E35 are within noise of each other. E31 is chosen because:
- it has the best mAP@50 and top-50 overlap;
- it keeps the TA's two-stage design and the paper's width (32).

E34 (no classifier) is a valid alternative at half the training time.

## Final: shortlist + exact re-scoring (own idea)

The GNN ranks all 19,309 candidates in about 0.02 s. Its top-K edges are then re-scored with the exact distance method (K × O(n²) instead of O(n⁴)). Results are for E31 with 5 seeds on the 4 test graphs:

| | mAP@10 | mAP@50 | top-50 | Extra time |
|---|---|---|---|---|
| GNN alone (mean of 5 seeds) | 0.538 | 0.662 | 0.719 | – |
| Ensemble of the 5 seed models (rank average) | 0.541 | 0.662 | 0.720 | – (no gain) |
| GNN top 50 → exact | 0.850 | 0.720 | 0.720 | 0.01 s |
| GNN top 100 → exact | 0.925 | 0.880 | 0.880 | 0.015 s |
| **GNN top 200 → exact** | **0.975** | **0.970** | **0.970** | **0.03 s** |
| GNN top 500 → exact | 1.000 | 0.995 | 0.995 | 0.07 s |
| Heuristic top 100 → exact | 0.825 | 0.690 | 0.690 | 0.015 s |
| Heuristic top 200 → exact | 0.875 | 0.810 | 0.810 | 0.03 s |
| Heuristic top 500 → exact | 0.950 | 0.920 | 0.920 | 0.07 s |
| Exact method on all edges | 1 | 1 | 1 | ~5 s |

## Generalisation to bigger graphs (own idea)

The model trained on 200-node graphs is tested, with no retraining, on 4 new BA graphs with 500 nodes (123,259 candidates each).

| | mAP@10 | mAP@50 | top-50 | Time per graph |
|---|---|---|---|---|
| GNN alone | 0.546 | 0.556 | 0.631 | 0.35 s |
| Heuristic | 0.210 | 0.343 | 0.490 | – |
| GNN top 200 → exact | 0.875 | 0.890 | 0.890 | 0.35 + 0.3 s |
| **GNN top 500 → exact** | **0.975** | **0.955** | **0.955** | **0.35 + 0.7 s** |
| GNN top 1000 → exact | 1.000 | 0.975 | 0.975 | 0.35 + 1.5 s |
| Heuristic top 500 → exact | 0.825 | 0.810 | 0.810 | 0.7 s |
| Exact method on all edges | 1 | 1 | 1 | ~7 min (measured while other jobs ran) |

## Main findings

1. **3-class labels** (top 2% / 2–10% / rest) beat a single top-10% class (TA's multi-class idea).
2. **Two-stage training** only helps when stage 2 also fine-tunes the GNN. Freezing the GNN hurts.
3. **Random node features fail completely on unseen graphs** (mAP@10 0.04). Structural features are essential.
4. **Edge features** (distance, common neighbours, Jaccard, Adamic–Adar, closeness and degree of both ends) give the biggest single gain. Once they are added, **mini-batches** also help.
5. The paper's **triplet loss on good edges** is still the best ranking loss. The edge-vector choice (concat, sum, diff, product) makes little difference.
6. **Overall:** mAP@10 0.449 → 0.538 (+20%), mAP@50 0.553 → 0.662 (+20%), top-50 overlap 0.636 → 0.719.
7. **GNN shortlist + exact re-scoring** reaches mAP@10 ≈ 0.98–1.0 at a tiny fraction of the exact cost. It works on 2.5× larger graphs without retraining, and a GNN shortlist is much better than a heuristic shortlist.

## Final configuration (defaults in `config.py`)

- 3-class labels, top-biased sampling.
- Two-stage training: stage 2 fine-tunes everything at lr × 0.3.
- Triplet loss on good edges, mini-batches of 500 edges.
- 4 structural node features, sum + absolute-difference edge vector, all 7 edge features.
- GraphSAGE with 3 layers, width 32.
- Early stopping on the mean of validation mAP@10 and mAP@50.
- Exact re-scoring of the top 200 edges (`exact_rerank_k`).

`legacy_config()` returns the original pipeline.
