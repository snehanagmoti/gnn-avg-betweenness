import numpy as np
import torch


def sample_edges_stratified_slots(slot_indices, sample_size, ratios=(0.34, 0.33, 0.33)):
    """
    Samples edges from Top, Middle, and Bottom slots according to the given ratios.
    This guarantees the model sees high-impact, intermediate, and low-impact candidate edges,
    preventing over-prediction on unseen middle/low edges.
    
    ratios: (top_ratio, middle_ratio, bottom_ratio) summing to 1.0
    """
    top_idx = slot_indices["top"]
    mid_idx = slot_indices["middle"]
    bot_idx = slot_indices["bottom"]
    
    n_top = int(round(sample_size * ratios[0]))
    n_mid = int(round(sample_size * ratios[1]))
    n_bot = sample_size - n_top - n_mid
    
    # Cap by available sizes
    n_top = min(n_top, len(top_idx))
    n_mid = min(n_mid, len(mid_idx))
    n_bot = min(n_bot, len(bot_idx))
    
    chosen_top = np.random.choice(top_idx, size=n_top, replace=False) if n_top > 0 else np.array([], dtype=int)
    chosen_mid = np.random.choice(mid_idx, size=n_mid, replace=False) if n_mid > 0 else np.array([], dtype=int)
    chosen_bot = np.random.choice(bot_idx, size=n_bot, replace=False) if n_bot > 0 else np.array([], dtype=int)
    
    sampled_indices = np.concatenate([chosen_top, chosen_mid, chosen_bot])
    np.random.shuffle(sampled_indices)
    return sampled_indices


def sample_edges_x4(scores, sample_size):
    """
    Samples edges according to y = x^4 distribution (as in original paper Sec 4.2),
    which prioritizes edges at extreme percentiles. Included for comparison/ablation.
    """
    num_edges = len(scores)
    sample_size = min(sample_size, num_edges)
    
    sorted_order = np.argsort(scores)
    x = np.linspace(-1, 1, num_edges)
    prob = (x ** 4) + 1e-4
    prob = prob / np.sum(prob)
    
    chosen_ranks = np.random.choice(num_edges, size=sample_size, replace=False, p=prob)
    return sorted_order[chosen_ranks]


def sample_triplets_from_indices(indices, ground_truth_scores, num_triplets):
    """
    Given an array of edge indices and their ground-truth continuous reduction scores,
    samples num_triplets triplets (m, n, o) sorted such that s_m* >= s_n* >= s_o*,
    and computes indicator values: I_ab = +1 if s_a* > s_b* else -1.
    """
    n_edges = len(indices)
    if n_edges < 3:
        raise ValueError("Need at least 3 edges to sample triplets.")

    cand_m = np.random.randint(0, n_edges, size=num_triplets)
    cand_n = np.random.randint(0, n_edges, size=num_triplets)
    cand_o = np.random.randint(0, n_edges, size=num_triplets)
    
    # Filter duplicate items in a triplet
    mask = (cand_m != cand_n) & (cand_n != cand_o) & (cand_m != cand_o)
    cand_m, cand_n, cand_o = cand_m[mask], cand_n[mask], cand_o[mask]
    
    if len(cand_m) == 0:
        cand_m = np.array([0])
        cand_n = np.array([1])
        cand_o = np.array([2])
        
    cand_triplets = np.stack([cand_m, cand_n, cand_o], axis=1)
    cand_scores = ground_truth_scores[cand_triplets]
    
    # Sort descending
    order = np.argsort(-cand_scores, axis=1)
    sorted_triplets = np.take_along_axis(cand_triplets, order, axis=1)
    
    m = sorted_triplets[:, 0]
    n = sorted_triplets[:, 1]
    o = sorted_triplets[:, 2]
    
    s_m = ground_truth_scores[m]
    s_n = ground_truth_scores[n]
    s_o = ground_truth_scores[o]
    
    I_mn = np.where(s_m > s_n, 1.0, -1.0).astype(np.float32)
    I_no = np.where(s_n > s_o, 1.0, -1.0).astype(np.float32)
    I_om = np.where(s_o > s_m, 1.0, -1.0).astype(np.float32)
    
    triplets = {
        "m": torch.tensor(m, dtype=torch.long),
        "n": torch.tensor(n, dtype=torch.long),
        "o": torch.tensor(o, dtype=torch.long),
        "I_mn": torch.tensor(I_mn, dtype=torch.float32),
        "I_no": torch.tensor(I_no, dtype=torch.float32),
        "I_om": torch.tensor(I_om, dtype=torch.float32),
    }
    return triplets

