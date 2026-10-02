"""Error metrics, computed separately for every forecast horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.array([np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]
                     for j in range(y_true.shape[1])])


def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """
    MSE skill score relative to a reference forecast (here: persistence).
        SS = 1 - MSE_model / MSE_reference
    SS = 1 perfect, SS = 0 no better than the reference, SS < 0 worse.
    """
    mse_model = np.mean((y_true - y_pred) ** 2, axis=0)
    mse_ref = np.mean((y_true - y_ref) ** 2, axis=0)
    return 1.0 - mse_model / mse_ref


def evaluate_all(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                 horizons: list[int], storm_mask: np.ndarray,
                 reference: str = "Persistence") -> pd.DataFrame:
    """One row per (model, horizon) with all metrics, for all hours and storm hours."""
    rows = []
    y_ref = predictions[reference]
    for name, y_pred in predictions.items():
        r_all = rmse(y_true, y_pred)
        r_storm = rmse(y_true[storm_mask], y_pred[storm_mask])
        m_all = mae(y_true, y_pred)
        c_all = correlation(y_true, y_pred)
        ss_all = skill_score(y_true, y_pred, y_ref)
        ss_storm = skill_score(y_true[storm_mask], y_pred[storm_mask], y_ref[storm_mask])
        for j, h in enumerate(horizons):
            rows.append({
                "model": name, "horizon_h": h,
                "RMSE": r_all[j], "MAE": m_all[j], "corr": c_all[j],
                "skill_vs_persistence": ss_all[j],
                "RMSE_storm": r_storm[j], "skill_storm": ss_storm[j],
            })
    return pd.DataFrame(rows)
