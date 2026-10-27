import torch
import torch.nn as nn
import torch.nn.functional as F
from src.features import build_edge_embeddings


class SAGEConvMean(nn.Module):
    """
    GraphSAGE layer with Mean Aggregator:
    h_v = sigma( W * CONCAT(h_v, mean_{u in N(v)} h_u) )
    """
    def __init__(self, in_channels, out_channels):
        super(SAGEConvMean, self).__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels, bias=True)

    def forward(self, x, adj_norm):
        neigh_mean = torch.sparse.mm(adj_norm, x) if adj_norm.is_sparse else torch.matmul(adj_norm, x)
        concat_feats = torch.cat([x, neigh_mean], dim=1)
        out = self.linear(concat_feats)
        return out


class GNNEncoder(nn.Module):
    """
    Stage 1: Multi-layer GraphSAGE encoder.
    Outputs node embeddings of dimension out_dim.
    """
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=32, num_layers=3, dropout=0.5):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConvMean(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConvMean(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.layers.append(SAGEConvMean(hidden_dim, out_dim))

    def forward(self, x, adj_norm):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_norm)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLPClassifier(nn.Module):
    """
    MLP Classifier head: 3-layer MLP predicting edge class probabilities.
    Can be used for binary classification (num_classes=1 with Sigmoid)
    or multi-class (e.g. 3 classes: Top, Middle, Bottom).
    """
    def __init__(self, in_dim=64, hidden_dims=(32, 16), num_classes=1, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, edge_emb):
        logits = self.net(edge_emb)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        return logits, probs


class MLPRanker(nn.Module):
    """
    MLP Ranker head: 3-layer MLP predicting continuous ranking scores for candidate edges.
    """
    def __init__(self, in_dim=64, hidden_dims=(32, 16), dropout=0.5):
        super(MLPRanker, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, edge_emb):
        score = self.net(edge_emb).squeeze(-1)
        return score


class ModularTwoStageModel(nn.Module):
    """
    Modular model supporting:
    1. Classifier pre-training (Encoder + Classifier)
    2. Ranker fine-tuning (Encoder + Ranker)
    3. Simultaneous joint training (for comparison/ablation)
    """
    def __init__(
        self,
        in_dim=16,
        gnn_hidden=32,
        node_emb_dim=32,
        num_gnn_layers=3,
        edge_method="sum_diff",
        num_classes=1,
        dropout=0.5
    ):
        super(ModularTwoStageModel, self).__init__()
        self.edge_method = edge_method
        self.encoder = GNNEncoder(
            in_dim=in_dim,
            hidden_dim=gnn_hidden,
            out_dim=node_emb_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # Calculate edge embedding dimension based on edge_method
        if edge_method in ["concat", "sum_diff"]:
            edge_dim = node_emb_dim * 2
        elif edge_method == "rich_symmetric":
            edge_dim = node_emb_dim * 3
        else:
            raise ValueError(f"Unknown edge method {edge_method}")
            
        self.classifier = MLPClassifier(in_dim=edge_dim, num_classes=num_classes, dropout=dropout)
        self.ranker = MLPRanker(in_dim=edge_dim, dropout=dropout)

    def encode_edges(self, x, adj_norm, edge_indices):
        """Encodes graph nodes and builds edge representations."""
        node_embs = self.encoder(x, adj_norm)
        edge_embs = build_edge_embeddings(node_embs, edge_indices, method=self.edge_method)
        return edge_embs

    def forward_classifier(self, x, adj_norm, edge_indices):
        """Used during pre-training: only passes through encoder and classifier."""
        edge_embs = self.encode_edges(x, adj_norm, edge_indices)
        logits, probs = self.classifier(edge_embs)
        return logits, probs

    def forward_ranker(self, x, adj_norm, edge_indices):
        """Used during fine-tuning: passes through encoder and ranker."""
        edge_embs = self.encode_edges(x, adj_norm, edge_indices)
        ranking_scores = self.ranker(edge_embs)
        return ranking_scores

    def forward_full(self, x, adj_norm, edge_indices):
        """Computes both classification and ranking scores."""
        edge_embs = self.encode_edges(x, adj_norm, edge_indices)
        logits, probs = self.classifier(edge_embs)
        ranking_scores = self.ranker(edge_embs)
        return logits, probs, ranking_scores

