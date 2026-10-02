"""
Step 1 - download the OMNI2 hourly data from NASA (one file per year, ~3 MB each).

Usage:
    python 01_download_data.py
    python 01_download_data.py --start 2000 --end 2010
    python 01_download_data.py --overwrite      # re-download (e.g. to get updated recent years)
"""
import argparse
import sys
import urllib.error

import config
from src.omni import download_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=config.START_YEAR)
    parser.add_argument("--end", type=int, default=config.END_YEAR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    failed = []
    for year in range(args.start, args.end + 1):
        try:
            path = download_year(year, config.DATA_RAW, config.OMNI_URL, args.overwrite)
            print(f"  {year}: ok  ({path.stat().st_size / 1e6:.1f} MB)")
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            print(f"  {year}: FAILED ({err})")
            failed.append(year)

    if failed:
        print(f"\nCould not download: {failed}. Run the script again; finished "
              "years are skipped automatically.")
        sys.exit(1)          # non-zero exit code: GitHub Actions marks the step as failed
    else:
        print(f"\nAll files saved in {config.DATA_RAW}")


if __name__ == "__main__":
    main()
