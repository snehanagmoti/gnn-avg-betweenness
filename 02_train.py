"""
STEP 2 + 3 - Model and training
Split the graphs into train / val / test, train GraphSAGE + MLP1 + MLP2, keep the epoch with
the best validation mAP@10, and save the model to `outputs/`.

Run:  python 02_train.py        (after 01_generate_dataset.py)
"""
import torch

from config import CFG
from src.data_io import load_all
from src.plots import plot_training_history
from src.trainer import save_checkpoint, train


def make_split(num_graphs, split):
    n_train, n_val, n_test = split
    assert n_train + n_val + n_test == num_graphs, "split must add up to the number of graphs"
    ids = list(range(num_graphs))
    return {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}


def main(cfg=CFG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = load_all(cfg.data_dir)
    split = make_split(len(graphs), cfg.split)
    print(f"Device: {device} | split: {split}")

    model, history, best = train(cfg, [graphs[i] for i in split["train"]],
                                 [graphs[i] for i in split["val"]], device)
    in_dim = graphs[0]["features"].shape[1]
    save_checkpoint(model, cfg, in_dim, history, best, split, cfg.output_dir)
    plot_training_history(history, cfg.output_dir)
    print(f"Best val mAP@10 = {best['val_map10']:.3f} at epoch {best['epoch']}. Saved to '{cfg.output_dir}/'")


if __name__ == "__main__":
    main()
