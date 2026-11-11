"""
Model building blocks for version 3 (plain PyTorch, no torch_geometric needed):
  GraphSAGE (mean aggregator, 3 layers) -> 64 numbers per node
  mlp       -> the head that turns the 152-number edge vector into one score
  mean_adjacency -> row-normalised adjacency, so (A_mean @ h)[v] = mean of v's neighbours
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


def mean_adjacency(edges, num_nodes):
    """Row-normalised adjacency matrix: (A_mean @ h)[v] = mean of v's neighbours' h."""
    A = torch.zeros(num_nodes, num_nodes)
    e = torch.as_tensor(edges, dtype=torch.long)
    A[e[:, 0], e[:, 1]] = 1.0
    A[e[:, 1], e[:, 0]] = 1.0
    deg = A.sum(1, keepdim=True).clamp(min=1)
    return A / deg
