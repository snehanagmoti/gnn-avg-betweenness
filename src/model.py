"""
Step 2: the model (follows Fig. 1 of the paper).

  GraphSAGE (mean aggregator) -> node embeddings h_v
  Edge vector: parts built from h_u, h_v (+ optional hand-made edge features)
  MLP1 (classifier): class of the edge (2 or more classes; higher class = better edge)
  MLP2 (scorer): ranking score

GraphSAGE is written in plain PyTorch (no torch_geometric needed).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGELayer(nn.Module):
    """h_v' = W . CONCAT(h_v, mean_{u in N(v)} h_u)   (paper eqs. 1-2)"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, h, adj_mean):
        return self.linear(torch.cat([h, adj_mean @ h], dim=1))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, normalize=False):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(SAGELayer(dims[i], dims[i + 1]) for i in range(num_layers))
        self.dropout = dropout
        self.normalize = normalize

    def forward(self, x, adj_mean):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj_mean)
            if i < len(self.layers) - 1:
                h = F.leaky_relu(h, 0.1)
                h = F.dropout(h, self.dropout, self.training)
        return F.normalize(h, dim=1) if self.normalize else h


def mlp(in_dim, hidden_dim, out_dim, num_layers, dropout):
    layers, d = [], in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(d, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


EDGE_PARTS = {"concat", "sum", "absdiff", "prod", "dot"}


def parse_combine(spec):
    parts = "sum+absdiff" if spec == "symmetric" else spec
    parts = parts.split("+")
    assert set(parts) <= EDGE_PARTS, f"unknown edge_combine part in {spec}"
    return parts


def edge_vector_dim(parts, h):
    return sum({"concat": 2 * h, "sum": h, "absdiff": h, "prod": h, "dot": 1}[p] for p in parts)


class EdgeRanker(nn.Module):
    def __init__(self, in_dim, cfg, edge_feat_dim=0, num_classes=2):
        super().__init__()
        self.gnn = GraphSAGE(in_dim, cfg.hidden_dim, cfg.num_gnn_layers, cfg.gnn_dropout,
                             cfg.normalize_embeddings)
        self.parts = parse_combine(cfg.edge_combine)
        edge_dim = edge_vector_dim(self.parts, cfg.hidden_dim) + edge_feat_dim
        self.classifier = mlp(edge_dim, cfg.edge_embedding_dim, num_classes, cfg.num_mlp_layers, cfg.mlp_dropout)
        self.scorer = mlp(edge_dim, cfg.edge_embedding_dim, 1, cfg.num_mlp_layers, cfg.mlp_dropout)

    def edge_embedding(self, h, edges, edge_feats=None):
        hu, hv = h[edges[:, 0]], h[edges[:, 1]]
        out = []
        for p in self.parts:
            if p == "concat":
                out.append(torch.cat([hu, hv], dim=1))
            elif p == "sum":
                out.append(hu + hv)
            elif p == "absdiff":
                out.append((hu - hv).abs())
            elif p == "prod":
                out.append(hu * hv)
            elif p == "dot":
                out.append((hu * hv).sum(1, keepdim=True))
        if edge_feats is not None and edge_feats.shape[1] > 0:
            out.append(edge_feats)
        return torch.cat(out, dim=1)

    def forward(self, x, adj_mean, edges, edge_feats=None):
        """Returns (class logits [E, C], scores [E])."""
        h = self.gnn(x, adj_mean)
        e = self.edge_embedding(h, edges, edge_feats)
        return self.classifier(e), self.scorer(e).squeeze(-1)


def mean_adjacency(edges, num_nodes):
    """Row-normalised adjacency matrix: (A_mean @ h)[v] = mean of v's neighbours' h."""
    A = torch.zeros(num_nodes, num_nodes)
    e = torch.as_tensor(edges, dtype=torch.long)
    A[e[:, 0], e[:, 1]] = 1.0
    A[e[:, 1], e[:, 0]] = 1.0
    deg = A.sum(1, keepdim=True).clamp(min=1)
    return A / deg
