# Pipeline V2: Decoupled Two-Stage GNN Link Prediction for Minimizing Average Betweenness Centrality

This folder contains the complete, self-contained implementation of **Pipeline V2**, developed based on advisory discussions with paper author **Aashish [NLN]** (October 2026).

---

## Key Improvements over Pipeline V1

1. **Two-Stage Architecture (Decoupled Pre-training & Fine-tuning):**
   - **Stage 1 (Pre-training):** Trains the GraphSAGE encoder and MLP Classifier on binary cross-entropy loss to learn topological boundaries separating high-reduction candidate regions from low-reduction regions.
   - **Stage 2 (Fine-tuning):** Initializes encoder weights from Stage 1 and fine-tunes the MLP Ranker using Triplet Margin Ranking Loss and standardized MSE regression loss (with the encoder trained at $0.1\times$ learning rate).
   - **Resolves:** Gradient conflicts and ranker loss stagnation (~0.90) observed during simultaneous joint training.

2. **Stratified Slot-Based Sampling (Top, Middle, Bottom):**
   - Splits the candidate edge distribution into **Top (25%)**, **Middle (50%)**, and **Bottom (25%)** reduction slots.
   - Samples proportionally (34% top, 33% middle, 33% bottom) during training to expose the model to the entire spectrum of candidate edges, eliminating false-positive over-scoring on unseen intermediate edges.

3. **Node & Edge Feature Representations:**
   - **Node Features:** Evaluated 6 key network structural properties (Degree, Clustering Coefficient, Closeness Centrality, Core Number, PageRank, Eigenvector Centrality) normalized to $[0, 1]$.
   - **Edge Vectors:** Rich symmetric edge representations $[h_u + h_v, |h_u - h_v|, h_u \odot h_v]$ ($96$-dim).
   - **Ablation:** Proved that pure random features fail ($mAP \le 0.0004$), whereas structural features provide critical signals for identifying bottleneck shortcuts.

4. **Expanded 20-Graph Datasets:**
   - Evaluated on **20 Erdős–Rényi (ER)** graphs ($n=200, p \in [0.035, 0.045]$) and **20 Barabási–Albert (BA)** graphs ($n=200, m=3$) using a 14 Train / 2 Val / 4 Test split.
   - Vectorized all-pairs shortest paths formula computes exact ground truth in ~4s per graph (~19,300 edges).

5. **Heuristic Baselines & Clinical Evaluation Metrics:**
   - Benchmarked against 7 baselines with robust tie-breaking: `random`, `distance`, `degree_product`, `degree_sum`, `closeness_gap_x_distance`, `common_neighbors`, `jaccard`.
   - New evaluation metrics: `mAP@k`, `Precision@k`, `Reduction_Ratio`, **Within 2.5% of Optimal**, **Within 10% of Optimal**, and **Top-1 in True Top-50**.

---

## Results Summary

### Barabási–Albert (BA) Graphs (Test Set: 4 Unseen Graphs, ~77,200 Candidates)

| Method | mAP@1 | mAP@10 | mAP@50 | ReductRatio | Within 2.5% Opt | Within 10% Opt | Top-1 in Top-50 | Spearman |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNN Proposed (Pipeline V2)** | **0.7500** | **0.3782** | **0.4678** | **0.9639** | **75.0%** | **75.0%** | **100.0%** | **0.7655** |
| Baseline: `closeness_gap_x_dist` | 0.2500 | 0.3385 | 0.3497 | 0.8573 | 25.0% | 50.0% | 100.0% | 0.5523 |
| Baseline: `degree_sum` | 0.0000 | 0.0161 | 0.0379 | 0.4661 | 0.0% | 0.0% | 25.0% | 0.3601 |
| Baseline: `distance` | 0.0000 | 0.0000 | 0.0000 | 0.2219 | 0.0% | 0.0% | 0.0% | 0.3134 |
| Baseline: `random` | 0.0000 | 0.0000 | 0.0000 | 0.1475 | 0.0% | 0.0% | 0.0% | -0.0031 |

### Erdős–Rényi (ER) Graphs (Test Set: 4 Unseen Graphs, ~76,500 Candidates)

| Method | mAP@1 | mAP@10 | mAP@50 | ReductRatio | Within 10% Opt | Top-1 in Top-50 | Spearman |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNN Proposed (Pipeline V2)** | 0.0000 | **0.2038** | **0.2024** | **0.8488** | **50.0%** | **75.0%** | **0.4863** |
| Ablation: Simultaneous Training | 0.2500 | 0.1495 | 0.2094 | 0.8063 | 50.0% | 50.0% | 0.5201 |
| Ablation: Pure Random Features | 0.0000 | 0.0000 | 0.0003 | 0.3037 | 0.0% | 0.0% | 0.0478 |
| Baseline: `distance` | 0.0000 | 0.0000 | 0.0014 | 0.4530 | 0.0% | 0.0% | 0.3436 |
| Baseline: `closeness_gap_x_dist` | 0.5000 | 0.4074 | 0.5293 | 0.9316 | 75.0% | 100.0% | 0.4447 |

---

## Directory Structure

```
pipeline_v2/
├── data/                    # Pre-generated 20 ER & 20 BA graph datasets (.pkl)
├── docs/                    # Detailed comparison and benchmark reports (.md)
├── outputs/                 # Metric summaries (.csv, .json) and trained model weights (.pt)
├── plots/                   # Benchmark bar charts and mAP@k curves (.png)
├── scripts/
│   ├── generate_datasets.py # Reproducible 20-graph dataset generator
│   ├── run_experiments.py   # Full benchmark and ablation experiment runner
│   ├── plot_results.py      # Generates evaluation figures
│   └── plot_v1_vs_v2.py     # Generates V1 vs V2 diagnostic chart
└── src/
    ├── baselines.py         # 7 heuristic baselines with robust tie-breakers
    ├── dataset.py           # Stratified slot-based edge and triplet sampler
    ├── features.py          # Structural (6-dim) and random node features + rich edge embeddings
    ├── graph_gen.py         # Connected ER and BA graph generator + normalized adj
    ├── ground_truth.py      # Vectorized delta avg BC + tie handling + slot partitions
    ├── losses.py            # Classification BCE/CE + Triplet loss + standardized MSE
    ├── metrics.py           # Comprehensive evaluation metrics suite
    ├── model.py             # GraphSAGE encoder + MLP Classifier + MLP Ranker
    └── trainer.py           # Stage 1 pre-training, Stage 2 fine-tuning, simultaneous
```

---

## How to Run

```bash
# 1. Generate 20 ER and 20 BA graphs (if not already cached)
python pipeline_v2/scripts/generate_datasets.py

# 2. Run the complete benchmark and ablation suite
python pipeline_v2/scripts/run_experiments.py

# 3. Generate comparison figures
python pipeline_v2/scripts/plot_results.py
```
