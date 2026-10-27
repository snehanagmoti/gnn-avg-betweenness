"""
All settings for the pipeline live here. Change values here, not inside the modules.

Every option added after the first TA review is a switch whose comment lists the choices.
The values below are the BEST combination found in the experiments
(see experiments/RESULTS.md); `legacy_config()` gives the original pipeline.
"""
from dataclasses import dataclass, field, replace
from typing import List


@dataclass
class Config:
    # ---------------- Paths ----------------
    data_dir: str = "data"            # generated graphs + ground truth
    output_dir: str = "outputs"       # model checkpoints, metrics, plots

    # ---------------- Step 1: dataset ----------------
    graph_type: str = "ba"            # "ba" | "er" | "random_tree" | "ktree" | "path"
    num_graphs: int = 20              # TA (23 Oct): 20 graphs while developing the pipeline
    num_nodes: int = 200              # nodes per graph
    m_values: List[int] = field(default_factory=lambda: [3])   # BA m-value(s); e.g. [3, 4] to mix
    er_p_values: List[float] = field(default_factory=lambda: [0.04])  # ER edge probability(ies)
    ktree_k: int = 2                  # k for k-trees
    base_seed: int = 0                # graph i uses seed base_seed + i
    normalize_betweenness: bool = True  # same scale as NetworkX normalized=True (like the paper)

    # ---------------- Labels (classifier) ----------------
    # Edges are ranked with competition ranking (ties share the best rank: 1, 1, 3, ...).
    # label_bands = top fractions; label = how many bands the edge falls into.
    #   [0.10]        -> 2 classes: top 10% vs rest         (original)
    #   [0.10, 0.02]  -> 3 classes: top 2% / 2-10% / rest
    label_bands: List[float] = field(default_factory=lambda: [0.10, 0.02])   # BEST: 3 classes

    # ---------------- Step 2: model ----------------
    # node features: degree, clustering, closeness, core_number, pagerank, eigenvector,
    #   eccentricity, betweenness, hub_dist (distance to the 5 biggest hubs), random
    node_feature_set: List[str] = field(
        default_factory=lambda: ["degree", "clustering", "closeness", "core_number"])
    # edge vector built from node embeddings h_u, h_v; parts joined with "+":
    #   concat | sum | absdiff | prod (element-wise) | dot (scalar);  "symmetric" = sum+absdiff
    edge_combine: str = "symmetric"
    # extra hand-made edge features appended to the edge vector:
    #   dist, common_neighbors, jaccard, adamic_adar, closeness_gap, closeness_minmax, degree_minmax
    edge_feature_set: List[str] = field(default_factory=lambda: [
        "dist", "common_neighbors", "jaccard", "adamic_adar", "closeness_gap", "closeness_minmax",
        "degree_minmax"])                                     # BEST: all edge features
    hidden_dim: int = 32              # paper: hidden state 32
    edge_embedding_dim: int = 64      # hidden width of the two MLPs
    num_gnn_layers: int = 3           # paper: diameter - 1
    num_mlp_layers: int = 3           # paper: 3 hidden layers in each MLP
    gnn_dropout: float = 0.2          # paper: 0.5
    mlp_dropout: float = 0.2          # paper: 0.5
    normalize_embeddings: bool = False

    # ---------------- Step 3: training ----------------
    split: List[int] = field(default_factory=lambda: [14, 2, 4])  # train / val / test graphs
    # "joint"     : classifier loss + ranking loss together (paper)
    # "two_stage" : pre-train GNN + classifier, then fine-tune the ranker (TA, 23 Oct)
    # "rank_only" : no classifier, one scoring head
    train_mode: str = "two_stage"     # BEST
    stage2_freeze: str = "none"       # two_stage: "gnn" (freeze GNN + classifier) | "none"
    stage2_lr_factor: float = 0.3     # two_stage: learning-rate multiplier in stage 2
    # ranking loss: "triplet" (paper) | "pairwise" (RankNet) | "regression" (MSE on standardized
    #   reduction) | "listnet" (softmax cross-entropy over the sampled list)
    rank_loss: str = "triplet"
    rank_on: str = "positives"        # which sampled edges the ranking loss uses: "positives" | "all"
    # edge sampling per graph per epoch:
    #   "top_biased" (x^4 weight on the top, original) | "stratified" (mix of bands) | "uniform"
    sampling: str = "top_biased"
    strat_bands: List[float] = field(default_factory=lambda: [0.02, 0.10, 0.50])
    strat_ratios: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.2])
    batch_size: int = 500              # edges per gradient step; 0 = all sampled edges of a graph
    epochs: int = 225                 # paper: max 225 with early stopping
    patience: int = 50                # early-stopping patience (epochs without val improvement)
    learning_rate: float = 0.003      # paper: 0.0085
    edge_sample_size: int = 5000      # paper: 5000 edges sampled per graph per epoch
    sampling_power: float = 4.0       # top_biased: weight (1 - r)^power
    triplets_per_edge: int = 10       # paper: 10/15/20 x edge sample size
    margin: float = 1.0               # paper: M = 1
    seed: int = 42
    select_metric: str = "map10+map50"     # early stopping on validation "map10" | "map10+map50"

    # ---------------- Step 4: evaluation ----------------
    k_values: List[int] = field(default_factory=lambda: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # "classify_then_rank" (paper): edges in a better predicted class come first, then by score
    # "score_only": rank by the ranking score only
    ranking_mode: str = "classify_then_rank"
    # Shortlist + exact re-scoring: the model's top-K edges are re-scored with the exact distance
    # method (K x O(n^2) work instead of O(n^4)); 0 = off. Reported next to the GNN-only ranking.
    exact_rerank_k: int = 200


def legacy_config(**kw):
    """The original pipeline (before the 23 Oct TA review), on the current dataset."""
    old = dict(label_bands=[0.10], edge_feature_set=[], train_mode="joint", stage2_freeze="gnn",
               stage2_lr_factor=1.0, batch_size=0, select_metric="map10")
    old.update(kw)
    return replace(Config(), **old)


CFG = Config()
