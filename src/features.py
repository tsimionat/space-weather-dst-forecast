"""
From a raw hourly table to (input window, target) pairs that a model can learn from.

Steps
-----
1. put the data on a regular hourly grid (missing hours become NaN rows)
2. fill SHORT gaps by linear interpolation (long gaps stay NaN)
3. add physically motivated derived features (V*Bs)
4. cut the time series into sliding windows:
       input  = the last LOOKBACK hours of all features, up to time t
       target = Dst at t+1, t+2, ..., t+6 hours
5. split the windows chronologically into train / validation / test
6. standardise every feature (zero mean, unit variance) using TRAINING data only
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# --------------------------------------------------------------------------
# 1-3: cleaning and derived features
# --------------------------------------------------------------------------
def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """
    Linearly interpolate runs of NaN that are at most `max_gap` samples long.

    Note: pandas' interpolate(limit=k) is NOT enough, because inside a long gap
    it would still fill the first k points. Here long gaps are left untouched.
    """
    is_nan = series.isna()
    # give every run of consecutive NaNs its own id, then measure its length
    run_id = (is_nan != is_nan.shift()).cumsum()
    run_length = is_nan.groupby(run_id).transform("sum")
    fillable = is_nan & (run_length <= max_gap)

    interpolated = series.interpolate(method="time", limit_area="inside")
    out = series.copy()
    out[fillable] = interpolated[fillable]
    return out


def clean_and_engineer(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Regular hourly grid + short-gap filling + derived features."""
    df = df.asfreq("h")  # insert missing hours as NaN rows

    for col in df.columns:
        df[col] = interpolate_short_gaps(df[col], max_gap)

    # Rectified electric field: V*Bs in mV/m, with Bs = -Bz when Bz < 0 (southward)
    # and 0 otherwise. Only southward IMF reconnects efficiently with Earth's field.
    # V [km/s] * Bs [nT] * 1e-3 = mV/m   (same convention as OMNI's word 36)
    bs = np.clip(-df["Bz"], 0.0, None)
    df["VBs"] = df["V"] * bs * 1e-3
    return df


# --------------------------------------------------------------------------
# 4: sliding windows
# --------------------------------------------------------------------------
@dataclass
class WindowedData:
    X: np.ndarray        # (N, lookback, n_features)  inputs
    y: np.ndarray        # (N, n_horizons)            Dst(t+h)      [nT]
    dst_now: np.ndarray  # (N,)                       Dst(t)        [nT]
    p_now: np.ndarray    # (N,)                       pressure at t [nPa]   (physics baseline)
    vbs_now: np.ndarray  # (N,)                       V*Bs at t     [mV/m]  (physics baseline)
    times: np.ndarray    # (N,)                       time t (datetime64)

    def subset(self, mask: np.ndarray) -> "WindowedData":
        return WindowedData(self.X[mask], self.y[mask], self.dst_now[mask],
                            self.p_now[mask], self.vbs_now[mask], self.times[mask])

    def __len__(self) -> int:
        return len(self.y)


def make_windows(df: pd.DataFrame, features: list[str], target: str,
                 lookback: int, horizons: list[int]) -> WindowedData:
    """
    Build every possible (input window, future target) pair.

    For a "now" index t the input covers rows t-lookback+1 ... t (inclusive)
    and the targets are rows t+h for each h in horizons.
    Windows containing any NaN (in inputs or targets) are discarded.
    """
    values = df[features].to_numpy(dtype=np.float32)    # (T, F)
    dst = df[target].to_numpy(dtype=np.float32)          # (T,)
    T, F = values.shape
    h_max = max(horizons)

    # all valid "now" indices: enough past for the input, enough future for the target
    t_idx = np.arange(lookback - 1, T - h_max)

    # sliding_window_view gives windows[s] = values[s : s+lookback] without copying;
    # a window that ENDS at t STARTS at s = t - lookback + 1
    windows = sliding_window_view(values, window_shape=(lookback, F))[:, 0]
    X = windows[t_idx - lookback + 1]                                  # (N, L, F)
    y = np.stack([dst[t_idx + h] for h in horizons], axis=1)           # (N, H)

    valid = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y).any(axis=1)
    t_idx = t_idx[valid]

    return WindowedData(
        X=np.ascontiguousarray(X[valid]),
        y=y[valid],
        dst_now=dst[t_idx],
        p_now=df["P"].to_numpy(dtype=np.float32)[t_idx],
        vbs_now=df["VBs"].to_numpy(dtype=np.float32)[t_idx],
        times=df.index.to_numpy()[t_idx],
    )


# --------------------------------------------------------------------------
# 5: chronological split
# --------------------------------------------------------------------------
def year_mask(times: np.ndarray, years: tuple[int, int], h_max: int) -> np.ndarray:
    """
    True for windows whose "now" time AND last target time both fall inside
    `years` (inclusive). This keeps targets from leaking across split borders.
    """
    t = pd.DatetimeIndex(times)
    t_end = t + pd.Timedelta(hours=h_max)
    first, last = years
    return ((t.year >= first) & (t.year <= last)
            & (t_end.year >= first) & (t_end.year <= last))


# --------------------------------------------------------------------------
# 6: standardisation
# --------------------------------------------------------------------------
@dataclass
class Scaler:
    """
    Standardise inputs feature-by-feature and targets horizon-by-horizon.

    The model does not predict Dst(t+h) directly but the CHANGE
    Delta_h = Dst(t+h) - Dst(t). A model that outputs 0 is exactly the
    persistence forecast, so the network only has to learn the correction.
    """
    x_mean: np.ndarray   # (F,)
    x_std: np.ndarray    # (F,)
    d_mean: np.ndarray   # (H,)  mean of Delta_h
    d_std: np.ndarray    # (H,)  std  of Delta_h

    @classmethod
    def fit(cls, data: WindowedData) -> "Scaler":
        flat = data.X.reshape(-1, data.X.shape[-1])
        delta = data.y - data.dst_now[:, None]
        return cls(flat.mean(axis=0), flat.std(axis=0) + 1e-6,
                   delta.mean(axis=0), delta.std(axis=0) + 1e-6)

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def transform_y(self, y: np.ndarray, dst_now: np.ndarray) -> np.ndarray:
        delta = y - dst_now[:, None]
        return ((delta - self.d_mean) / self.d_std).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray, dst_now: np.ndarray) -> np.ndarray:
        """From the network's scaled output back to Dst in nT."""
        return y_scaled * self.d_std + self.d_mean + dst_now[:, None]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({k: v.tolist() for k, v in self.__dict__.items()},
                                   indent=2))

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        d = json.loads(path.read_text())
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


# --------------------------------------------------------------------------
# saving / loading the processed splits
# --------------------------------------------------------------------------
def save_split(data: WindowedData, path: Path) -> None:
    np.savez_compressed(path, X=data.X, y=data.y, dst_now=data.dst_now,
                        p_now=data.p_now, vbs_now=data.vbs_now,
                        times=data.times.astype("datetime64[s]").astype(np.int64))


def load_split(path: Path) -> WindowedData:
    z = np.load(path)
    return WindowedData(X=z["X"], y=z["y"], dst_now=z["dst_now"], p_now=z["p_now"],
                        vbs_now=z["vbs_now"],
                        times=z["times"].astype("datetime64[s]"))
