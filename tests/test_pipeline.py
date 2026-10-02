"""Unit tests. Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.baselines import (OM_B, OM_C, obrien_mcpherron, om_decay_time,
                           om_injection, persistence)
from src.features import (Scaler, clean_and_engineer, interpolate_short_gaps,
                          make_windows, year_mask)
from src.omni import load_omni, read_omni_file
from tests.synthetic import write_years


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("omni")
    write_years([2003, 2004], d, seed=1)
    return d


def test_reader_parses_time_and_fill_values(synthetic_dir):
    df = read_omni_file(synthetic_dir / "omni2_2004.dat")
    assert len(df) == 8784                                  # leap year
    assert df.index[0] == pd.Timestamp("2004-01-01 00:00")
    assert df.index[-1] == pd.Timestamp("2004-12-31 23:00")
    assert df["Bz"].isna().any()                            # fills became NaN
    assert (df["Bz"].dropna().abs() < 900).all()            # no fill value survived


def test_short_gaps_filled_long_gaps_kept():
    idx = pd.date_range("2000-01-01", periods=12, freq="h")
    s = pd.Series([0, 1, np.nan, np.nan, 4, 5, np.nan, np.nan, np.nan, np.nan,
                   np.nan, 11], index=idx, dtype=float)
    out = interpolate_short_gaps(s, max_gap=3)
    assert out.iloc[2] == pytest.approx(2) and out.iloc[3] == pytest.approx(3)
    assert out.iloc[6:11].isna().all()                      # 5-hour gap untouched


def test_windows_are_aligned_in_time():
    idx = pd.date_range("2000-01-01", periods=200, freq="h")
    df = pd.DataFrame({c: np.arange(200.0) for c in ["B", "By", "Bz", "V", "n", "P"]},
                      index=idx)
    df["Dst"] = -np.arange(200.0)
    df = clean_and_engineer(df, max_gap=3)
    data = make_windows(df, ["V", "Dst"], "Dst", lookback=24, horizons=[1, 3, 6])
    assert np.allclose(data.X[:, -1, 1], data.dst_now)      # last input row is "now"
    # targets are exactly h hours in the future (Dst decreases by 1 per hour here)
    assert np.allclose(data.y, data.dst_now[:, None] - np.array([1, 3, 6]))
    assert pd.Timestamp(data.times[0]) == idx[23]           # first window ends at hour 23
    assert len(data) == 200 - 23 - 6


def test_year_mask_prevents_target_leakage():
    times = pd.date_range("2014-12-31 15:00", periods=10, freq="h").to_numpy()
    mask = year_mask(times, (2014, 2014), h_max=6)
    # t + 6 h must still be in 2014  ->  only 15:00, 16:00, 17:00 are allowed
    assert mask.tolist() == [True, True, True] + [False] * 7


def test_om2000_analytic_solution_matches_numerical_integration():
    rng = np.random.default_rng(0)
    dst0 = rng.uniform(-200, 20, 50)
    p = rng.uniform(0.5, 10, 50)
    vbs = rng.uniform(0, 15, 50)
    analytic = obrien_mcpherron(dst0, p, vbs, horizons=[1, 3, 6])

    # brute force: tiny Euler steps of dDst*/dt = Q - Dst*/tau
    corr = OM_B * np.sqrt(p) - OM_C
    ds = dst0 - corr
    q, tau, dt = om_injection(vbs), om_decay_time(vbs), 1e-3
    results = {}
    for step in range(1, 6001):
        ds = ds + dt * (q - ds / tau)
        if step in (1000, 3000, 6000):
            results[step // 1000] = ds + corr
    numeric = np.stack([results[1], results[3], results[6]], axis=1)
    assert np.allclose(analytic, numeric, atol=0.05)


def test_scaler_round_trip(synthetic_dir):
    df = clean_and_engineer(load_omni([2003], synthetic_dir), max_gap=3)
    data = make_windows(df, ["B", "Bz", "V", "P", "VBs", "Dst"], "Dst", 24, [1, 6])
    sc = Scaler.fit(data)
    y_back = sc.inverse_y(sc.transform_y(data.y, data.dst_now), data.dst_now)
    assert np.allclose(y_back, data.y, atol=1e-3)
    assert np.allclose(persistence(data.dst_now, [1, 6])[:, 0], data.dst_now)


def test_lstm_output_shape():
    torch = pytest.importorskip("torch")
    from src.models import DstLSTM
    model = DstLSTM(n_features=8, n_outputs=6, hidden_size=16)
    out = model(torch.randn(5, 24, 8))
    assert out.shape == (5, 6)
