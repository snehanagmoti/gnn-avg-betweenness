"""
Step 3c: training loop + prediction.

Training modes (cfg.train_mode):
  joint     : classification loss + ranking loss summed (paper)
  two_stage : stage 1 trains GNN + classifier only; stage 2 starts from those weights and trains
              the ranker only (TA, 23 Oct). cfg.stage2_freeze = "gnn" keeps the GNN fixed.
  rank_only : a single scoring head, no classifier
Early stopping keeps the epoch with the best validation mAP@10.
"""
import copy
import json
import os
import time

import numpy as np
import torch

from src.features import distance_matrix, edge_features, node_features
from src.ground_truth import make_labels
from src.losses import (classification_loss, listnet_loss, pairwise_loss, regression_loss,
                        triplet_margin_ranking_loss)
from src.metrics import average_precision_at_k
from src.model import EdgeRanker, mean_adjacency
from src.sampling import sample_edges, sample_triplets


def to_tensors(g, cfg, device="cpu"):
    """Build everything the model needs for one graph, from the raw dataset + the config."""
    G = g["G"]
    n = G.number_of_nodes()
    D = distance_matrix(G)
    red = g["reductions"]
    x = node_features(G, cfg.node_feature_set, seed=g["graph_id"], D=D)
    ef = edge_features(G, g["missing_edges"], cfg.edge_feature_set, D=D)
    labels = make_labels(red, cfg.label_bands)
    z = (red - red.mean()) / (red.std() + 1e-12)
    return {
        "x": torch.as_tensor(x, device=device),
        "adj": mean_adjacency(g["edges"], n).to(device),
        "missing": torch.as_tensor(g["missing_edges"], device=device),
        "efeat": torch.as_tensor(ef, device=device),
        "labels": torch.as_tensor(labels, device=device),
        "targets": torch.as_tensor(red, dtype=torch.float32, device=device),
        "z": torch.as_tensor(z, dtype=torch.float32, device=device),
        "reductions": red,
        "labels_np": labels,
    }


def num_classes(cfg):
    return len(cfg.label_bands) + 1


def build_model(cfg, t):
    return EdgeRanker(t["x"].shape[1], cfg, t["efeat"].shape[1], num_classes(cfg))


@torch.no_grad()
def predict(model, t, cfg, mode=None):
    """Score every missing edge. Returns (final_scores, raw_scores, class_probs) as numpy.
    mode: "classify_then_rank" | "score_only" | "classifier" (rank by expected class)."""
    mode = mode or ("score_only" if cfg.train_mode == "rank_only" else cfg.ranking_mode)
    model.eval()
    logits, scores = model(t["x"], t["adj"], t["missing"], t["efeat"])
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    scores = scores.cpu().numpy()
    if mode == "classifier":
        final = probs @ np.arange(probs.shape[1])
    elif mode == "classify_then_rank":
        offset = scores.max() - scores.min() + 1.0
        final = scores + offset * probs.argmax(1)
    else:
        final = scores
    return final, scores, probs


def val_map(model, val_t, cfg, mode=None):
    """Validation score used for early stopping: mAP@10, or the mean of mAP@10 and mAP@50
    (cfg.select_metric = "map10+map50", less noisy with few validation graphs)."""
    ks = [10, 50] if cfg.select_metric == "map10+map50" else [10]
    aps = []
    for t in val_t:
        final, _, _ = predict(model, t, cfg, mode)
        order = np.argsort(-final, kind="stable")
        aps.append(np.mean([average_precision_at_k(order, t["reductions"], k) for k in ks]))
    return float(np.mean(aps))


def rank_loss(cfg, scores, t, idx_t, labels_b, rng):
    if cfg.rank_on == "positives":
        cand = np.where(labels_b >= 1)[0]
    else:
        cand = np.arange(len(labels_b))
    if len(cand) < 3:
        return scores.sum() * 0.0
    n_pairs = cfg.triplets_per_edge * len(labels_b)
    if cfg.rank_loss == "triplet":
        trip = torch.as_tensor(sample_triplets(cand, n_pairs, rng))
        if len(trip) == 0:
            return scores.sum() * 0.0
        return triplet_margin_ranking_loss(scores, t["targets"][idx_t], trip, cfg.margin)
    if cfg.rank_loss == "pairwise":
        pairs = torch.as_tensor(np.stack([rng.choice(cand, n_pairs), rng.choice(cand, n_pairs)], 1))
        return pairwise_loss(scores, t["targets"][idx_t], pairs)
    c = torch.as_tensor(cand)
    if cfg.rank_loss == "regression":
        return regression_loss(scores[c], t["z"][idx_t][c])
    if cfg.rank_loss == "listnet":
        return listnet_loss(scores[c], t["z"][idx_t][c])
    raise ValueError(cfg.rank_loss)


def run_epoch(model, opt, train_t, cfg, rng, use_cls, use_rank):
    model.train()
    sums, steps = np.zeros(2), 0
    for i in rng.permutation(len(train_t)):
        t = train_t[i]
        idx = sample_edges(t["reductions"], cfg.edge_sample_size, rng, cfg.sampling,
                           cfg.sampling_power, cfg.strat_bands, cfg.strat_ratios)
        idx = rng.permutation(idx)
        bs = cfg.batch_size or len(idx)
        for start in range(0, len(idx), bs):
            b = idx[start:start + bs]
            idx_t = torch.as_tensor(b)
            logits, scores = model(t["x"], t["adj"], t["missing"][idx_t], t["efeat"][idx_t])
            loss_c = classification_loss(logits, t["labels"][idx_t]) if use_cls else scores.sum() * 0.0
            loss_r = rank_loss(cfg, scores, t, idx_t, t["labels_np"][b], rng) if use_rank else scores.sum() * 0.0
            loss = loss_c + loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += [loss_c.item(), loss_r.item()]
            steps += 1
    return sums / max(steps, 1)


def _fit(model, params, lr, train_t, val_t, cfg, rng, use_cls, use_rank, val_mode, tag, verbose):
    opt = torch.optim.Adam(params, lr=lr)
    hist = {"cls_loss": [], "rank_loss": [], "val_map10": []}
    best = {"val_map10": -1.0, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        lc, lr_ = run_epoch(model, opt, train_t, cfg, rng, use_cls, use_rank)
        vmap = val_map(model, val_t, cfg, val_mode)
        for k, v in zip(hist, [lc, lr_, vmap]):
            hist[k].append(float(v))
        if vmap > best["val_map10"]:
            best = {"val_map10": vmap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"[{tag}] epoch {epoch:3d} | cls {lc:.4f} rank {lr_:.4f} | val mAP@10 {vmap:.3f} "
                  f"| best {best['val_map10']:.3f} @ {best['epoch']} | {time.time() - start:.0f}s", flush=True)
        if epoch - best["epoch"] >= cfg.patience:
            break
    model.load_state_dict(best["state"])
    return hist, best


def train(cfg, train_graphs, val_graphs, device="cpu", verbose=True, train_t=None, val_t=None):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    train_t = train_t or [to_tensors(g, cfg, device) for g in train_graphs]
    val_t = val_t or [to_tensors(g, cfg, device) for g in val_graphs]
    model = build_model(cfg, train_t[0]).to(device)

    if cfg.train_mode == "joint":
        hist, best = _fit(model, model.parameters(), cfg.learning_rate, train_t, val_t, cfg, rng,
                          True, True, None, "joint", verbose)
        history = {"stage1": hist}
    elif cfg.train_mode == "rank_only":
        hist, best = _fit(model, model.parameters(), cfg.learning_rate, train_t, val_t, cfg, rng,
                          False, True, "score_only", "rank", verbose)
        history = {"stage1": hist}
    elif cfg.train_mode == "two_stage":
        h1, b1 = _fit(model, model.parameters(), cfg.learning_rate, train_t, val_t, cfg, rng,
                      True, False, "classifier", "pretrain", verbose)
        if cfg.stage2_freeze == "gnn":
            for p in list(model.gnn.parameters()) + list(model.classifier.parameters()):
                p.requires_grad_(False)
            params = model.scorer.parameters()
        else:
            params = model.parameters()
        h2, best = _fit(model, params, cfg.learning_rate * cfg.stage2_lr_factor, train_t, val_t, cfg,
                        rng, False, True, None, "finetune", verbose)
        best["stage1_epoch"], best["stage1_val_map10"] = b1["epoch"], b1["val_map10"]
        history = {"stage1": h1, "stage2": h2}
    else:
        raise ValueError(cfg.train_mode)
    return model, history, best


def save_checkpoint(model, cfg, history, best, split, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best["epoch"],
                   "best_val_map10": best["val_map10"], "split": split}, f, indent=2)


def load_checkpoint(cfg, output_dir, t, device="cpu"):
    ckpt = torch.load(os.path.join(output_dir, "model.pt"), map_location=device)
    model = build_model(cfg, t).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model
