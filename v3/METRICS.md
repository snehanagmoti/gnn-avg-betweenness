# Evaluation metrics

**Code:** `src/metrics.py` (mAP@k, NDCG@k) and `v3/common.py` (`eval_one`: all metrics below).

Every metric is computed **per test graph**, then averaged over the test graphs, then over the saved models (seeds).

**Notation for one graph:** `red[e]` is the exact reduction of candidate edge e. The model scores every edge; `order` lists the edges by score, best first.

## Relevant set (true top k, with ties)

`relevant_set(red, k)` = every edge whose reduction is **≥ the k-th best value**. A 200-node graph has ~19,000 candidate edges but only ~128 different reduction values, so edges tied with the k-th one also count and the set can be larger than k (e.g. 52 edges for k = 50).

## mAP@k (base paper's main metric): which edges, and how early

    AP@k = (1/k) · Σ_{j=1..k} P@j · rel(j)

- **rel(j)** = 1 if the model's j-th edge is in the relevant set.
- **P@j** = share of relevant edges among the model's first j.
- We divide by **k**, so missing true top-k edges, or finding them late, lowers the score. 1.0 = perfect top k at the front.

**Example** (an earlier model on one test graph, k = 10): the true top 10 sit at the model's positions 1, 3, 4, 6, 9, 10:

| Position j | 1 | 3 | 4 | 6 | 9 | 10 |
|---|---|---|---|---|---|---|
| P@j | 1/1 | 2/3 | 3/4 | 4/6 | 5/9 | 6/10 |

AP@10 = (1.000 + 0.667 + 0.750 + 0.667 + 0.556 + 0.600) / 10 = **0.424**. mAP@k = mean of AP@k over the test graphs.

## NDCG@k: how good, and how early

    DCG@k  = Σ_{j=1..k} red(model's j-th edge) / log2(j + 1)
    NDCG@k = DCG@k of the model's order / DCG@k of the perfect order

Unlike mAP, a nearly-best edge outside the true top k still earns most of the credit. Because many edges are almost as good as the best one, every reasonable model scores ≈ 0.95+, so mAP separates models more clearly.

## Top-50 overlap

Share of the model's top 50 that is in the true top 50. Order inside the 50 does not matter.

## Shortlist metrics (for the exact check of the top 100)

| Metric | Meaning |
|---|---|
| `rec10@100` | share of the true top 10 inside the model's top 100 |
| `kneed10` | shortlist length needed to contain **all** of the true top 10 (position of the last one) |
| `ex100_mapk` | mAP@k after the model's top 100 are re-scored with their exact reduction and put first |

With a shortlist of 100 and exact re-scoring, the final top 10 is perfect exactly when `kneed10 ≤ 100`.
