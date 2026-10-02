"""
Step 2 - clean the raw data and turn it into training / validation / test windows.

Output (in data/processed/):
    train.npz, val.npz, test.npz   input windows + targets
    scaler.json                    means / standard deviations from the TRAINING set
"""
import numpy as np

import config
from src.features import (Scaler, clean_and_engineer, make_windows, save_split,
                          year_mask)
from src.omni import load_omni


def main() -> None:
    years = range(config.START_YEAR, config.END_YEAR + 1)
    print("Reading OMNI2 files ...")
    raw = load_omni(years, config.DATA_RAW)
    print(f"  {len(raw):,} hourly rows from {raw.index[0]} to {raw.index[-1]}")
    print("  fraction of missing values per column (before gap filling):")
    print((raw.isna().mean() * 100).round(1).to_string())

    df = clean_and_engineer(raw, config.MAX_GAP_HOURS)

    print("\nBuilding sliding windows ...")
    data = make_windows(df, config.FEATURES, config.TARGET,
                        config.LOOKBACK, config.HORIZONS)
    print(f"  {len(data):,} complete windows "
          f"(input {config.LOOKBACK} h x {len(config.FEATURES)} features, "
          f"{len(config.HORIZONS)} horizons)")

    h_max = max(config.HORIZONS)
    splits = {
        "train": data.subset(year_mask(data.times, config.TRAIN_YEARS, h_max)),
        "val": data.subset(year_mask(data.times, config.VAL_YEARS, h_max)),
        "test": data.subset(year_mask(data.times, config.TEST_YEARS, h_max)),
    }

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        storm = (split.dst_now < config.STORM_THRESHOLD).mean() * 100
        print(f"  {name:5s}: {len(split):7,} windows, "
              f"min Dst = {split.dst_now.min():6.0f} nT, "
              f"storm-time fraction = {storm:4.1f} %")
        save_split(split, config.DATA_PROCESSED / f"{name}.npz")

    # the scaler is fitted on TRAINING data only, then reused for val/test
    scaler = Scaler.fit(splits["train"])
    scaler.save(config.DATA_PROCESSED / "scaler.json")
    print(f"\nSaved processed data to {config.DATA_PROCESSED}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
