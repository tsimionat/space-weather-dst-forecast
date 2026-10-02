"""
Step 3 - train the LSTM.

The loop below is written out explicitly (no high-level "fit" function) so that
every step of learning is visible:
    forward pass -> loss -> backward pass (gradients) -> optimizer step

Usage:
    python 03_train.py
    python 03_train.py --epochs 5      # quick test run
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from src.features import Scaler, load_split
from src.models import DstLSTM, count_parameters
from src.plotting import plot_training_history


def set_seed(seed: int) -> None:
    """Make the run reproducible (same random initial weights, same shuffling)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")          # NVIDIA GPU (e.g. Google Colab)
    if torch.backends.mps.is_available():
        return torch.device("mps")           # Apple Silicon GPU
    return torch.device("cpu")


def make_loader(split, scaler: Scaler, shuffle: bool) -> DataLoader:
    X = torch.from_numpy(scaler.transform_x(split.X))
    y = torch.from_numpy(scaler.transform_y(split.y, split.dst_now))
    return DataLoader(TensorDataset(X, y), batch_size=config.BATCH_SIZE,
                      shuffle=shuffle, drop_last=False)


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)          # switches dropout on (train) or off (eval)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            prediction = model(X_batch)                 # 1. forward pass
            loss = loss_fn(prediction, y_batch)         # 2. how wrong are we?

            if training:
                optimizer.zero_grad()                   # 3. reset old gradients
                loss.backward()                         # 4. backpropagation
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()                        # 5. update the weights

            total += loss.item() * len(X_batch)
            n += len(X_batch)
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = pick_device()
    print(f"Using device: {device}")

    scaler = Scaler.load(config.DATA_PROCESSED / "scaler.json")
    train = load_split(config.DATA_PROCESSED / "train.npz")
    val = load_split(config.DATA_PROCESSED / "val.npz")
    train_loader = make_loader(train, scaler, shuffle=True)
    val_loader = make_loader(val, scaler, shuffle=False)
    print(f"Training windows: {len(train):,}   validation windows: {len(val):,}")

    model = DstLSTM(n_features=len(config.FEATURES), n_outputs=len(config.HORIZONS),
                    hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS,
                    dropout=config.DROPOUT).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    # halve the learning rate when the validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=3)

    config.MODELS.mkdir(parents=True, exist_ok=True)
    best_path = config.MODELS / "lstm_best.pt"
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, epochs_without_improvement = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        improved = val_loss < best_val
        if improved:
            best_val, epochs_without_improvement = val_loss, 0
            torch.save(model.state_dict(), best_path)   # keep the best weights
        else:
            epochs_without_improvement += 1

        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr:.1e} | {time.time() - t0:5.1f} s {'*' if improved else ''}")

        if epochs_without_improvement >= config.PATIENCE:
            print(f"Early stopping: no improvement for {config.PATIENCE} epochs.")
            break

    (config.MODELS / "history.json").write_text(json.dumps(history, indent=2))
    plot_training_history(history, config.FIGURES / "learning_curves.png")
    print(f"\nBest validation loss {best_val:.4f}; weights saved to {best_path}")


if __name__ == "__main__":
    main()
