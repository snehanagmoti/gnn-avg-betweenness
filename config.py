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
    graph_type: str = "er"            # "er" (Erdos-Renyi) or "ba" (Barabasi-Albert)
    p_values: List[float] = field(default_factory=lambda: [0.038, 0.040, 0.042, 0.044, 0.040, 0.042, 0.045, 0.040, 0.042, 0.044])
    m_values: List[int] = field(default_factory=lambda: [3])   # BA m-value(s); e.g. [3, 4] to mix
    base_seed: int = 101              # graph i uses seed base_seed + i
    normalize_betweenness: bool = True  # same scale as NetworkX normalized=True (like the paper)

    # Positive-edge rule for the classifier (MLP1).
    # With AVERAGE betweenness every missing edge reduces it a bit, so "reduction > 0"
    # labels everything positive. Instead: an edge is positive if it is in the top
    # `positive_fraction` of edges of its graph.
    positive_fraction: float = 0.25   # Top quartile (25%) as high-impact edges

    # ---------------- Step 2: model ----------------
    node_feature_set: List[str] = field(
        default_factory=lambda: ["degree", "clustering", "closeness", "core_number"])
    hidden_dim: int = 32              # paper: hidden state 32
    edge_embedding_dim: int = 64      # paper: edge embedding 64 (= 2 x node embedding 32)
    num_gnn_layers: int = 3           # paper: diameter - 1
    num_mlp_layers: int = 3           # paper: 3 hidden layers in each MLP
    gnn_dropout: float = 0.2          # paper: 0.5
    mlp_dropout: float = 0.2          # paper: 0.5
    normalize_embeddings: bool = False  # L2-normalise node embeddings
    edge_combine: str = "concat"      # "concat" (paper, Sec 3.1) or "symmetric"

    # ---------------- Step 3: training ----------------
    split: List[int] = field(default_factory=lambda: [6, 2, 2])  # train / val / test graph counts (6:2:2)
    epochs: int = 100                 # max epochs with early stopping
    patience: int = 30                # early-stopping patience
    learning_rate: float = 0.005      # paper: 0.0085
    edge_sample_size: int = 3000      # candidate edges sampled per graph per epoch
    sampling_power: float = 4.0       # paper: y = x^4
    triplets_per_edge: int = 8        # triplets sampled
    margin: float = 0.5               # margin for ranking loss
    seed: int = 42

    # ---------------- Step 4: evaluation ----------------
    k_values: List[int] = field(default_factory=lambda: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ranking_mode: str = "classify_then_rank"


CFG = Config()
