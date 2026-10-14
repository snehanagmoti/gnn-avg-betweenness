"""
Step 3c: training loop + prediction.
"""
import copy
import json
import os
import time

import numpy as np
import torch

from src.losses import classification_loss, triplet_margin_ranking_loss
from src.metrics import average_precision_at_k
from src.model import EdgeRanker, mean_adjacency
from src.sampling import sample_edges, sample_triplets


def to_tensors(g, device):
    """Turn one loaded graph (dict from data_io) into the tensors the model needs."""
    n = g["G"].number_of_nodes()
    return {
        "x": torch.as_tensor(g["features"], device=device),
        "adj": mean_adjacency(g["edges"], n).to(device),
        "missing": torch.as_tensor(g["missing_edges"], device=device),
        "labels": torch.as_tensor(g["labels"], device=device),
        "targets": torch.as_tensor(g["reductions"], dtype=torch.float32, device=device),
        "reductions": g["reductions"],
        "labels_np": g["labels"],
    }


@torch.no_grad()
def predict(model, t, ranking_mode="classify_then_rank"):
    """Score every missing edge of one graph. Returns (final_scores, raw_scores, p_positive) as numpy.
    final_scores is what the ranking uses (higher = better)."""
    model.eval()
    logits, scores = model(t["x"], t["adj"], t["missing"])
    p_pos = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    scores = scores.cpu().numpy()
    if ranking_mode == "classify_then_rank":
        # paper: edges classified positive by MLP1 come first, ordered by MLP2 score
        offset = scores.max() - scores.min() + 1.0
        final = scores + offset * (p_pos > 0.5)
    else:
        final = scores
    return final, scores, p_pos


def val_map(model, val_tensors, k, ranking_mode):
    aps = []
    for t in val_tensors:
        final, _, _ = predict(model, t, ranking_mode)
        order = np.argsort(-final, kind="stable")
        aps.append(average_precision_at_k(order, t["reductions"], k))
    return float(np.mean(aps))


def train(cfg, train_graphs, val_graphs, device="cpu", verbose=True):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    train_t = [to_tensors(g, device) for g in train_graphs]
    val_t = [to_tensors(g, device) for g in val_graphs]

    in_dim = train_t[0]["x"].shape[1]
    model = EdgeRanker(in_dim, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = {k: [] for k in ["train_loss", "train_cls_loss", "train_rank_loss", "val_map10"]}
    best = {"val_map10": -1.0, "epoch": 0, "state": None}
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums = np.zeros(3)
        for i in rng.permutation(len(train_t)):
            t = train_t[i]
            idx = sample_edges(t["reductions"], cfg.edge_sample_size, cfg.sampling_power, rng)
            idx_t = torch.as_tensor(idx, device=device)
            logits, scores = model(t["x"], t["adj"], t["missing"][idx_t])

            loss_c = classification_loss(logits, t["labels"][idx_t])
            # ranking loss on ground-truth positive edges only (paper ranks the positive edges)
            pos_positions = np.where(t["labels_np"][idx] == 1)[0]
            trip = sample_triplets(pos_positions, cfg.triplets_per_edge * len(idx), rng)
            trip_t = torch.as_tensor(trip, device=device)
            loss_r = (triplet_margin_ranking_loss(scores, t["targets"][idx_t], trip_t, cfg.margin)
                      if len(trip) else torch.zeros((), device=device))
            loss = loss_c + loss_r                      # paper: unweighted sum

            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += [loss.item(), loss_c.item(), loss_r.item()]

        sums /= len(train_t)
        vmap = val_map(model, val_t, 10, cfg.ranking_mode)
        for key, v in zip(history, [*sums, vmap]):
            history[key].append(float(v))

        if vmap > best["val_map10"]:
            best = {"val_map10": vmap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:3d} | loss {sums[0]:.4f} (cls {sums[1]:.4f}, rank {sums[2]:.4f}) "
                  f"| val mAP@10 {vmap:.3f} | best {best['val_map10']:.3f} @ {best['epoch']} "
                  f"| {time.time() - start:.0f}s")
        if epoch - best["epoch"] >= cfg.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs)")
            break

    model.load_state_dict(best["state"])
    return model, history, best


def save_checkpoint(model, cfg, in_dim, history, best, split, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim}, os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best["epoch"],
                   "best_val_map10": best["val_map10"], "split": split}, f, indent=2)


def load_checkpoint(cfg, output_dir, device="cpu"):
    ckpt = torch.load(os.path.join(output_dir, "model.pt"), map_location=device)
    model = EdgeRanker(ckpt["in_dim"], cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model
