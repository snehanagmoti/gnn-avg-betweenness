"""
All settings for the pipeline live here. Change values here, not inside the modules.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---------------- Paths ----------------
    data_dir: str = "data"            # generated graphs + ground truth
    output_dir: str = "outputs"       # model checkpoints, metrics, plots

    # ---------------- Step 1: dataset ----------------
    num_graphs: int = 10              # TA: start with 10 graphs
    num_nodes: int = 200              # nodes per graph
    m_values: List[int] = field(default_factory=lambda: [3])   # BA m-value(s); e.g. [3, 4] to mix
    base_seed: int = 0                # graph i uses seed base_seed + i
    normalize_betweenness: bool = True  # same scale as NetworkX normalized=True (like the paper)

    # Positive-edge rule for the classifier (MLP1).
    # With AVERAGE betweenness every missing edge reduces it a bit, so "reduction > 0"
    # labels everything positive. Instead: an edge is positive if it is in the top
    # `positive_fraction` of edges of its graph. (Open question for the TA.)
    positive_fraction: float = 0.10

    # ---------------- Step 2: model ----------------
    node_feature_set: List[str] = field(
        default_factory=lambda: ["degree", "clustering", "closeness", "core_number"])
    hidden_dim: int = 32              # paper: hidden state 32
    edge_embedding_dim: int = 64      # paper: edge embedding 64 (= 2 x node embedding 32)
    num_gnn_layers: int = 3           # paper: diameter - 1 for BA (BA n=200, m=3 has diameter ~4-5)
    num_mlp_layers: int = 3           # paper: 3 hidden layers in each MLP
    gnn_dropout: float = 0.2          # paper: 0.5 (0.5 made the ranking MLP collapse on 7 training graphs)
    mlp_dropout: float = 0.2          # paper: 0.5 (see README "tuning notes")
    normalize_embeddings: bool = False  # L2-normalise node embeddings (original GraphSAGE); off = trains faster
    edge_combine: str = "symmetric"   # "concat" (paper, order-dependent) or "symmetric" (u,v order doesn't matter)

    # ---------------- Step 3: training ----------------
    split: List[int] = field(default_factory=lambda: [7, 1, 2])  # train / val / test graph counts
    epochs: int = 225                 # paper: max 225 with early stopping
    patience: int = 50                # early-stopping patience (epochs without val improvement)
    learning_rate: float = 0.003      # paper: 0.0085 (more stable here)
    edge_sample_size: int = 5000      # paper: 5000 edges sampled per graph per epoch
    sampling_power: float = 4.0       # paper: y = x^4; here applied to the top of the ranking
    triplets_per_edge: int = 10       # paper: 10/15/20 x edge sample size
    margin: float = 1.0               # paper: M = 1
    seed: int = 42

    # ---------------- Step 4: evaluation ----------------
    k_values: List[int] = field(default_factory=lambda: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # Ranking at test time:
    #   "classify_then_rank" (paper): edges MLP1 calls positive are ranked first by MLP2 score
    #   "score_only": rank all edges by MLP2 score
    ranking_mode: str = "classify_then_rank"


CFG = Config()
