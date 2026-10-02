"""
Step 4 - compare all models on the TEST set (years never seen during training).

Models: Persistence, Physics (O'Brien & McPherron 2000), Ridge regression, LSTM.

Output (in results/):
    metrics.csv               every metric for every model and horizon
    metrics_table.md          RMSE table, ready to paste into your README
    figures/rmse_vs_horizon.png
    figures/storm_*.png       the strongest storms of the test period
    figures/scatter_lstm_3h.png
"""
import numpy as np
import pandas as pd
import torch

import config
from src.baselines import fit_ridge, obrien_mcpherron, persistence, predict_ridge
from src.features import Scaler, load_split
from src.metrics import evaluate_all
from src.models import DstLSTM
from src.plotting import plot_rmse_vs_horizon, plot_scatter, plot_storm_event


def lstm_predict(split, scaler: Scaler) -> np.ndarray:
    model = DstLSTM(n_features=len(config.FEATURES), n_outputs=len(config.HORIZONS),
                    hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS,
                    dropout=config.DROPOUT)
    model.load_state_dict(torch.load(config.MODELS / "lstm_best.pt", map_location="cpu"))
    model.eval()                                   # dropout OFF for prediction

    X = torch.from_numpy(scaler.transform_x(split.X))
    outputs = []
    with torch.no_grad():                          # no gradients needed: faster
        for start in range(0, len(X), 4096):
            outputs.append(model(X[start:start + 4096]).numpy())
    return scaler.inverse_y(np.concatenate(outputs), split.dst_now)


def strongest_storms(times: np.ndarray, dst: np.ndarray, n: int,
                     separation_days: float = 5.0) -> list[pd.Timestamp]:
    """Times of the n deepest Dst minima, at least `separation_days` apart."""
    t = pd.DatetimeIndex(times)
    order = np.argsort(dst)
    picked: list[pd.Timestamp] = []
    for i in order:
        if all(abs(t[i] - p) > pd.Timedelta(days=separation_days) for p in picked):
            picked.append(t[i])
        if len(picked) == n:
            break
    return picked


def markdown_table(metrics: pd.DataFrame, value: str) -> str:
    table = metrics.pivot(index="model", columns="horizon_h", values=value)
    table = table.loc[metrics["model"].unique()]          # keep model order
    header = "| Model | " + " | ".join(f"{h} h" for h in table.columns) + " |"
    sep = "|---|" + "---|" * len(table.columns)
    rows = ["| " + name + " | " + " | ".join(f"{v:.1f}" for v in row) + " |"
            for name, row in table.iterrows()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    scaler = Scaler.load(config.DATA_PROCESSED / "scaler.json")
    train = load_split(config.DATA_PROCESSED / "train.npz")
    test = load_split(config.DATA_PROCESSED / "test.npz")
    H = config.HORIZONS
    print(f"Test windows: {len(test):,}")

    print("Computing forecasts ...")
    ridge = fit_ridge(scaler.transform_x(train.X),
                      scaler.transform_y(train.y, train.dst_now), config.RIDGE_ALPHA)
    predictions = {
        "Persistence": persistence(test.dst_now, H),
        "Physics (OM2000)": obrien_mcpherron(test.dst_now, test.p_now, test.vbs_now, H),
        "Ridge": scaler.inverse_y(predict_ridge(ridge, scaler.transform_x(test.X)),
                                  test.dst_now),
        "LSTM": lstm_predict(test, scaler),
    }

    # a window counts as "storm time" if Dst is below the threshold at any
    # moment between t and t + max(horizon)
    storm_mask = (np.minimum(test.dst_now, test.y.min(axis=1))
                  < config.STORM_THRESHOLD)
    print(f"Storm-time windows: {storm_mask.sum():,} ({storm_mask.mean() * 100:.1f} %)")

    metrics = evaluate_all(test.y, predictions, H, storm_mask)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.RESULTS / "metrics.csv", index=False, float_format="%.4f")

    md = ("### RMSE [nT], all test hours\n\n" + markdown_table(metrics, "RMSE")
          + "\n\n### RMSE [nT], storm hours (Dst < "
          + f"{config.STORM_THRESHOLD:.0f} nT)\n\n"
          + markdown_table(metrics, "RMSE_storm") + "\n")
    (config.RESULTS / "metrics_table.md").write_text(md)
    print("\n" + md)

    # ---------------- figures ----------------
    plot_rmse_vs_horizon(metrics, config.FIGURES / "rmse_vs_horizon.png")

    j3 = H.index(3) if 3 in H else 0
    plot_scatter(test.y[:, j3], predictions["LSTM"][:, j3], H[j3], "LSTM",
                 config.FIGURES / f"scatter_lstm_{H[j3]}h.png")

    for k, center in enumerate(strongest_storms(test.times, test.dst_now, n=3), 1):
        path = config.FIGURES / f"storm_{k}_{center:%Y%m%d}.png"
        plot_storm_event(test.times, test.y, predictions, H[j3], j3, center, path)
        print(f"Storm {k}: minimum Dst near {center:%Y-%m-%d %H:%M} -> {path.name}")

    print(f"\nAll results saved in {config.RESULTS}")


if __name__ == "__main__":
    main()
