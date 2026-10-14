"""
Step 2: the model (follows Fig. 1 of the paper).

  Stage 1: GraphSAGE (mean aggregator) -> node embeddings h_v
  Edge embedding: combine h_u and h_v
  Stage 2: MLP1 -> is the edge "positive"? (2-class softmax)
           MLP2 -> ranking score for the edge

GraphSAGE is written in plain PyTorch (no torch_geometric needed), so it installs
easily on Windows/Mac/Linux. For 200-node graphs a dense adjacency matrix is fine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGELayer(nn.Module):
    """One GraphSAGE layer with mean aggregation (paper eqs. 1-2):
        h_N(v) = mean_{u in N(v)} h_u
        h_v'   = sigma( W . CONCAT(h_v, h_N(v)) )
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, h, adj_mean):
        h_neigh = adj_mean @ h                      # mean of neighbours
        return self.linear(torch.cat([h, h_neigh], dim=1))


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
        # Optional L2 normalisation (original GraphSAGE). Off by default: it shrinks the
        # embeddings so much that the ranking MLP starts with near-constant scores and learns slowly.
        return F.normalize(h, dim=1) if self.normalize else h


def mlp(in_dim, hidden_dim, out_dim, num_layers, dropout):
    """LeakyReLU instead of ReLU: with plain ReLU the score MLP often 'dies' (all units output 0),
    giving every edge the same score and a ranking loss stuck at the margin."""
    layers, d = [], in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(d, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class EdgeRanker(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.gnn = GraphSAGE(in_dim, cfg.hidden_dim, cfg.num_gnn_layers, cfg.gnn_dropout,
                             cfg.normalize_embeddings)
        self.edge_combine = cfg.edge_combine
        edge_dim = 2 * cfg.hidden_dim               # 2 x 32 = 64 (paper's edge embedding size)
        self.classifier = mlp(edge_dim, cfg.edge_embedding_dim, 2, cfg.num_mlp_layers, cfg.mlp_dropout)  # MLP1
        self.scorer = mlp(edge_dim, cfg.edge_embedding_dim, 1, cfg.num_mlp_layers, cfg.mlp_dropout)      # MLP2

    def edge_embedding(self, h, edges):
        hu, hv = h[edges[:, 0]], h[edges[:, 1]]
        if self.edge_combine == "concat":           # paper: CONCAT(h_u, h_v)
            return torch.cat([hu, hv], dim=1)
        # symmetric: same result for (u, v) and (v, u) - natural for undirected graphs
        return torch.cat([hu + hv, (hu - hv).abs()], dim=1)

    def node_embeddings(self, x, adj_mean):
        return self.gnn(x, adj_mean)

    def forward(self, x, adj_mean, edges):
        """Returns (class logits [E, 2], scores [E])."""
        h = self.node_embeddings(x, adj_mean)
        e = self.edge_embedding(h, edges)
        return self.classifier(e), self.scorer(e).squeeze(-1)


def mean_adjacency(edges, num_nodes):
    """Row-normalised adjacency matrix: (A_mean @ h)[v] = mean of v's neighbours' h."""
    A = torch.zeros(num_nodes, num_nodes)
    e = torch.as_tensor(edges, dtype=torch.long)
    A[e[:, 0], e[:, 1]] = 1.0
    A[e[:, 1], e[:, 0]] = 1.0
    deg = A.sum(1, keepdim=True).clamp(min=1)
    return A / deg
