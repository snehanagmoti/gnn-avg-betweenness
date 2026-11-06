# Evaluation metrics: how each one is calculated

**Code:** `src/metrics.py` (version 2 metrics) and `experiments/v3/common.py` (filter metrics).

**How results are averaged.** Every metric is computed **per test graph**. We then average:

1. over the test graphs;
2. over the training seeds.

**Notation for one graph:**
- `red[e]` is the exact reduction of edge e.
- The model gives a score to every edge, and `order` is the edges sorted by score, best first. Equal scores keep their input order (stable sort).

**Worked example.** Test graph 16 with the version 2 model (seed 0) is used throughout. It has 19,309 candidate edges, but only **128 different reduction values**, so ties are everywhere.

## 1. Relevant set (used by mAP@k and top-50 overlap)

`relevant_set(red, k)` contains every edge whose reduction is **≥ the k-th best value**. So all ties with the k-th edge count as relevant, and the set can be larger than k.

On graph 16:
- the top-10 set has exactly 10 edges;
- the top-50 set has **52** edges (ties at the 50th value);
- the top-20 set has 21 edges, and the top-30 set has 32.

## 2. AP@k and mAP@k (base paper's main metric)

    AP@k = (1/k) · Σ_{j=1..k} P@j · rel(j)

- **rel(j)** = 1 if the model's j-th edge is in the relevant set.
- **P@j** = the share of relevant edges among the model's first j.
- **Note:** we always divide by **k**, not by the number of hits. A model that finds only 6 of the true top 10 can score at most 0.6, and less if the hits come late.

**Graph 16, model's top 10:**

| Position j | True rank of that edge | Relevant? | P@j |
|---|---|---|---|
| 1 | 2 | ✓ | 1/1 = 1.000 |
| 2 | 11 | – | 1/2 |
| 3 | 4 | ✓ | 2/3 = 0.667 |
| 4 | 5 | ✓ | 3/4 = 0.750 |
| 5 | 36 | – | 3/5 |
| 6 | 1 | ✓ | 4/6 = 0.667 |
| 7 | 22 | – | 4/7 |
| 8 | 18 | – | 4/8 |
| 9 | 3 | ✓ | 5/9 = 0.556 |
| 10 | 9 | ✓ | 6/10 = 0.600 |

AP@10 = (1.000 + 0.667 + 0.750 + 0.667 + 0.556 + 0.600) / 10 = **0.424**.

**mAP@k** = the mean of AP@k over the test graphs.

**Quirk:** AP@k is **not monotonic in k**, because the relevant set changes with k. On graph 16:
- AP@5 = 0.483, AP@10 = 0.424, AP@20 = 0.483, AP@30 = 0.534, AP@50 = 0.571.

This explains the dip around k = 10 in the mAP-vs-k curves.

## 3. Top-50 overlap (= precision@50)

The share of the model's top 50 that is in the relevant top-50 set. No weighting by position.

On graph 16: **0.64** (32 of 50).

## 4. Optimality ratio and "best in top 10"

- **Optimality** = red(model's #1 edge) ÷ red(true best edge).
  - On graph 16: **0.979**. The #1 pick is the true #2 edge.
- **Best-in-top-10** = the best true reduction among the model's top 10 ÷ the true best.
  - On graph 16: **1.0**, because the true #1 is at position 6.
- **Caveat:** both are almost always 0.95–1.0 for any reasonable model, so they separate models poorly.

## 5. "Within 2.5% / 10% of the best"

The share of the model's top 50 whose reduction is within 2.5% / 10% of the best.

On graph 16:
- only **2** edges in the whole graph are within 2.5% of the best;
- only **6** are within 10%.

So even a perfect model scores 6/50 = 0.12 at most. Every method sits near 0.06, so this metric is uninformative here. Suggestion: drop it, or measure it over the top 10 instead of the top 50.

## 6. Spearman@50

Take the **true** top-50 edges, and compute the Spearman rank correlation between the model's scores and the true reductions on those edges only.

- It measures how well the model **orders the good edges among themselves**.
- In version 2 it uses the final classify-then-rank score, so the class decision is part of it.
- On graph 16: 0.472 with the final score, 0.464 with the ranker's raw score.

## 7. Classifier balanced accuracy (version 2 only)

The accuracy on each class, averaged over classes. It is a health check only: a better classifier did not always give a better ranking.

## 8. New filter metrics (version 3)

These measure the model as a **shortlist for exact re-scoring**.

| Metric | Definition | Graph 16, version 2 model |
|---|---|---|
| `rec10@K` | share of the relevant top-10 set inside the model's first K | positions of the true top 10: 1, 3, 4, 6, 9, 10, 11, 56, 80, 232. rec10@50 = 0.7, rec10@100 = 0.9, rec10@200 = 0.9, rec10@500 = 1.0 |
| `rec50@K` | the same for the relevant top-50 set | – |
| `kneed10` / `kneed50` | the smallest shortlist that contains **all** of the relevant top-10 / top-50 set (= the largest position among them) | kneed10 = **232** |
| `kneed10_max` | the worst kneed10 over the test graphs | – |
| `exK_map10` / `exK_map50` | mAP after the model's top K are re-scored exactly and put first; the rest keep the model order | – |

**Why these matter for the TA's direction 2.** With a shortlist of K and exact re-scoring:
- the final top 10 is perfect exactly when `kneed10 ≤ K`;
- `ex100_map10` is the end-to-end accuracy of "model + exact".
