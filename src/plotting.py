"""All figures of the project. Every function saves one PNG file."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed (works on servers/Colab)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Persistence": "#7f7f7f", "Physics (OM2000)": "#2ca02c",
          "Ridge": "#1f77b4", "LSTM": "#d62728"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="training loss")
    ax.plot(epochs, history["val_loss"], label="validation loss")
    best = int(np.argmin(history["val_loss"]))
    ax.axvline(best + 1, color="k", ls="--", lw=0.8, label=f"best epoch ({best + 1})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (standardised units)")
    ax.set_title("Learning curves")
    ax.legend()
    _save(fig, path)


def plot_rmse_vs_horizon(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, group in metrics.groupby("model", sort=False):
        c = COLORS.get(name)
        axes[0].plot(group["horizon_h"], group["RMSE"], "o-", color=c, label=name)
        axes[1].plot(group["horizon_h"], group["RMSE_storm"], "o-", color=c, label=name)
    axes[0].set_title("All test hours")
    axes[1].set_title("Storm hours only (Dst < -50 nT)")
    for ax in axes:
        ax.set_xlabel("forecast horizon [h]")
        ax.set_ylabel("RMSE [nT]")
        ax.grid(alpha=0.3)
    axes[0].legend()
    _save(fig, path)


def plot_storm_event(times_now: np.ndarray, y_true: np.ndarray,
                     predictions: dict[str, np.ndarray], horizon: int,
                     horizon_index: int, center: pd.Timestamp, path: Path,
                     days_before: float = 2.0, days_after: float = 4.0) -> None:
    """
    Observed vs forecast Dst around one storm. Each forecast is plotted at the
    time it is valid for (t + h), so curves can be compared point by point.
    """
    valid_time = pd.DatetimeIndex(times_now) + pd.Timedelta(hours=horizon)
    sel = ((valid_time >= center - pd.Timedelta(days=days_before))
           & (valid_time <= center + pd.Timedelta(days=days_after)))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(valid_time[sel], y_true[sel, horizon_index], color="k", lw=2.2,
            label="observed Dst")
    for name, pred in predictions.items():
        ax.plot(valid_time[sel], pred[sel, horizon_index], lw=1.3,
                color=COLORS.get(name), label=f"{name}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Dst [nT]")
    ax.set_title(f"Storm of {center:%d %b %Y}: {horizon}-hour-ahead forecasts")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    _save(fig, path)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, horizon: int,
                 model_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(y_true, y_pred, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect forecast")
    ax.set_xlabel("observed Dst [nT]")
    ax.set_ylabel(f"{model_name} forecast [nT]")
    ax.set_title(f"{horizon}-hour-ahead, test set")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    _save(fig, path)
