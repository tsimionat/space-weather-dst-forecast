"""
Generate fake OMNI2 files with exactly the same format as the real ones.

Used by the unit tests (and handy to try the whole pipeline offline).
The solar wind is random; Dst is produced by the O'Brien-McPherron equation
driven by that wind, plus noise, so there IS a learnable physical relation.
These numbers are NOT real data - never report results obtained on them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines import OM_B, OM_C, om_decay_time, om_injection
from src.omni import FILL_VALUES, N_WORDS, WORDS

COL = {name: word - 1 for word, name in WORDS.items()}   # name -> 0-based column


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def make_year(year: int, rng: np.random.Generator, dst0: float = 0.0,
              gap_prob: float = 0.002) -> tuple[np.ndarray, float]:
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    n_hours = 8784 if leap else 8760
    t = np.arange(n_hours)

    v = 420 + _ar1(n_hours, 0.98, 12, rng)
    n = np.clip(6 + _ar1(n_hours, 0.95, 1.0, rng), 0.5, None)
    bz = _ar1(n_hours, 0.9, 1.2, rng)
    by = _ar1(n_hours, 0.9, 1.5, rng)
    # a few "storms": intervals of strongly southward Bz and fast wind
    for start in rng.choice(n_hours - 40, size=8, replace=False):
        dur = rng.integers(6, 20)
        bz[start:start + dur] -= rng.uniform(8, 25)
        v[start:start + dur + 10] += rng.uniform(150, 350)
    v = np.clip(v, 250, None)
    b = np.sqrt(bz**2 + by**2 + 9.0)
    p = 1.67e-6 * n * v**2 * 1.04
    e = -v * bz * 1e-3
    vbs = np.clip(e, 0, None)

    dst_star = np.empty(n_hours)
    ds = dst0
    for i in range(n_hours):
        ds = ds + om_injection(vbs[i]) - ds / om_decay_time(vbs[i])
        dst_star[i] = ds
    dst = np.round(dst_star + OM_B * np.sqrt(p) - OM_C + rng.normal(0, 2, n_hours))

    words = np.zeros((n_hours, N_WORDS))
    words[:, 0] = year
    words[:, 1] = t // 24 + 1
    words[:, 2] = t % 24
    for name, arr in {"B": b, "By": by, "Bz": bz, "n": n, "V": v, "P": p,
                      "E": e, "Dst": dst}.items():
        words[:, COL[name]] = arr

    # random data gaps (short and long) written as fill values, like real OMNI
    for start in np.flatnonzero(rng.random(n_hours) < gap_prob):
        length = rng.choice([1, 2, 5, 12])
        for name in ["B", "By", "Bz", "n", "V", "P", "E"]:
            words[start:start + length, COL[name]] = FILL_VALUES[name]
    return words, float(dst_star[-1])


def write_years(years, out_dir: Path, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fmt = ["%d", "%d", "%d"] + ["%.2f"] * (N_WORDS - 3)
    dst0 = 0.0
    for year in years:
        words, dst0 = make_year(year, rng, dst0)
        np.savetxt(out_dir / f"omni2_{year}.dat", words, fmt=fmt)


if __name__ == "__main__":
    # python -m tests.synthetic  -> fills data/raw with FAKE files for an offline dry run
    import config
    write_years(range(config.START_YEAR, config.END_YEAR + 1), config.DATA_RAW)
    print(f"Synthetic (fake!) OMNI2 files written to {config.DATA_RAW}")
