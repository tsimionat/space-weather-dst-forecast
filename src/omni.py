"""
Download and read NASA OMNI2 hourly solar-wind data.

OMNI2 files are plain text: one line per hour, 55 numbers ("words") per line.
The full format description is at
https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text

Missing values are NOT empty: they are written as "fill values" such as 999.9
or 99999. We must convert them to NaN (Not a Number) before doing anything else,
otherwise the model would think Bz = 999.9 nT really happened!
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

N_WORDS = 55  # numbers per line in an OMNI2 file

# word number in the official description (1-based)  ->  our column name
WORDS = {
    1: "year",
    2: "doy",    # day of year (1 = 1 January)
    3: "hour",   # 0 ... 23
    9: "B",      # field magnitude average |B|           [nT]
    16: "By",    # By, GSM                                [nT]
    17: "Bz",    # Bz, GSM                                [nT]
    24: "n",     # proton density                         [cm^-3]
    25: "V",     # plasma (flow) speed                    [km/s]
    29: "P",     # flow pressure                          [nPa]
    36: "E",     # electric field -V*Bz*1e-3              [mV/m]
    41: "Dst",   # Dst index from Kyoto                   [nT]
}

# fill value used by OMNI for each quantity when the measurement is missing
FILL_VALUES = {
    "B": 999.9,
    "By": 999.9,
    "Bz": 999.9,
    "n": 999.9,
    "V": 9999.0,
    "P": 99.99,
    "E": 999.99,
    "Dst": 99999.0,
}


def download_year(year: int, dest_dir: Path, url_template: str,
                  overwrite: bool = False, timeout: float = 60.0) -> Path:
    """Download the OMNI2 file of one year into dest_dir and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"omni2_{year}.dat"
    if path.exists() and not overwrite:
        return path

    url = url_template.format(year=year)
    request = urllib.request.Request(url, headers={"User-Agent": "dst-forecast/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()

    tmp = path.with_suffix(".part")      # write to a temporary file first, so an
    tmp.write_bytes(content)             # interrupted download never leaves a
    tmp.replace(path)                    # half-written .dat file behind
    return path


def read_omni_file(path: Path) -> pd.DataFrame:
    """
    Read one OMNI2 yearly file.

    Returns a DataFrame indexed by UTC time (one row per hour) with the columns
    listed in WORDS (except year/doy/hour), fill values replaced by NaN.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    if raw.shape[1] != N_WORDS:
        raise ValueError(
            f"{path.name}: expected {N_WORDS} columns, found {raw.shape[1]}. "
            "Is this really an OMNI2 low-resolution (hourly) file?"
        )

    # keep only the words we need; pandas columns are 0-based, words are 1-based
    df = raw[[w - 1 for w in WORDS]].copy()
    df.columns = list(WORDS.values())

    # build a proper timestamp from year + day-of-year + hour
    time = (pd.to_datetime(df["year"].astype(int).astype(str), format="%Y")
            + pd.to_timedelta(df["doy"].astype(int) - 1, unit="D")
            + pd.to_timedelta(df["hour"].astype(int), unit="h"))
    df = df.drop(columns=["year", "doy", "hour"])
    df.index = pd.DatetimeIndex(time, name="time")

    # replace fill values by NaN
    for col, fill in FILL_VALUES.items():
        df[col] = df[col].astype(float)
        df.loc[np.isclose(df[col], fill), col] = np.nan

    return df


def load_omni(years: range | list[int], raw_dir: Path) -> pd.DataFrame:
    """Read several yearly files and stitch them into one continuous DataFrame."""
    frames = []
    for year in years:
        path = raw_dir / f"omni2_{year}.dat"
        if not path.exists():
            print(f"  [warning] {path.name} not found, skipping year {year}")
            continue
        frames.append(read_omni_file(path))
    if not frames:
        raise FileNotFoundError(
            f"No OMNI2 files found in {raw_dir}. Run 01_download_data.py first."
        )
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df
