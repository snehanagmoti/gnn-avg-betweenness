import networkx as nx
import numpy as np
import torch


def compute_node_features(G, mode="structural", in_dim=16, seed=42):
    """
    Computes node feature matrix X of shape (num_nodes, in_dim).
    
    Modes:
    - 'structural': Computes up to 6 key network structural properties:
        1. Degree
        2. Clustering Coefficient
        3. Closeness Centrality
        4. Core Number
        5. PageRank
        6. Eigenvector Centrality
      Remaining dimensions up to in_dim are initialized with random projections.
    - 'random': All in_dim features are initialized with standard Gaussian noise N(0, 1).
    - 'hybrid': 4 standard structural features + (in_dim - 4) random features.
    """
    np.random.seed(seed)
    n = G.number_of_nodes()
    feats = np.zeros((n, in_dim), dtype=np.float32)
    
    if mode == "random":
        # Pure random initialization as discussed in meeting notes
        feats = np.random.randn(n, in_dim).astype(np.float32) * 0.1
        return torch.tensor(feats, dtype=torch.float32)
        
    # Structural feature calculations
    # 1. Degree
    degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    feats[:, 0] = degrees / max_deg
    
    # 2. Clustering coefficient
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)
    feats[:, 1] = clustering
    
    # 3. Closeness centrality
    closeness = np.array(list(nx.closeness_centrality(G).values()), dtype=np.float32)
    max_c = closeness.max() if closeness.max() > 0 else 1.0
    feats[:, 2] = closeness / max_c
    
    # 4. Core numbers
    core_nums = np.array(list(nx.core_number(G).values()), dtype=np.float32)
    max_core = core_nums.max() if core_nums.max() > 0 else 1.0
    feats[:, 3] = core_nums / max_core
    
    num_struct = 4
    if mode == "structural" and in_dim >= 6:
        # 5. PageRank
        pr = np.array(list(nx.pagerank(G, max_iter=200).values()), dtype=np.float32)
        max_pr = pr.max() if pr.max() > 0 else 1.0
        feats[:, 4] = pr / max_pr
        
        # 6. Eigenvector centrality
        try:
            eig = np.array(list(nx.eigenvector_centrality(G, max_iter=300).values()), dtype=np.float32)
            max_eig = eig.max() if eig.max() > 0 else 1.0
            feats[:, 5] = eig / max_eig
        except Exception:
            feats[:, 5] = feats[:, 0]  # Fallback to degree if power iteration does not converge
            
        num_struct = 6
        
    # Remaining dimensions filled with random projections
    if in_dim > num_struct:
        rand_proj = np.random.randn(n, in_dim - num_struct).astype(np.float32) * 0.1
        feats[:, num_struct:] = rand_proj
        
    return torch.tensor(feats, dtype=torch.float32)


def build_edge_embeddings(node_embeddings, edge_indices, method="sum_diff"):
    """
    Constructs edge representations from node embeddings h_u and h_v.
    
    Methods:
    - 'concat': [h_u, h_v] -> dimension 2 * d
    - 'sum_diff': [h_u + h_v, |h_u - h_v|] -> dimension 2 * d (symmetric to edge orientation)
    - 'rich_symmetric': [h_u + h_v, |h_u - h_v|, h_u * h_v] -> dimension 3 * d
    """
    u = edge_indices[:, 0]
    v = edge_indices[:, 1]
    
    h_u = node_embeddings[u]
    h_v = node_embeddings[v]
    
    if method == "concat":
        return torch.cat([h_u, h_v], dim=-1)
    elif method == "sum_diff":
        sum_vec = h_u + h_v
        diff_vec = torch.abs(h_u - h_v)
        return torch.cat([sum_vec, diff_vec], dim=-1)
    elif method == "rich_symmetric":
        sum_vec = h_u + h_v
        diff_vec = torch.abs(h_u - h_v)
        prod_vec = h_u * h_v
        return torch.cat([sum_vec, diff_vec, prod_vec], dim=-1)
    else:
        raise ValueError(f"Unknown edge embedding method: {method}")

