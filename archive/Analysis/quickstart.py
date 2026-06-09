#!/usr/bin/env python3
"""Register any new measurement CSVs in Data/ into registry.csv.

Usage:
    python quickstart.py            # process only not-yet-registered files
    python quickstart.py --all      # re-process every file in Data/
"""

import sys

from registry import register_file, new_files_since_registry
from parity_lib import discover_data_files


def main() -> None:
    process_all = "--all" in sys.argv[1:]
    files = discover_data_files("Data") if process_all else new_files_since_registry("Data")

    if not files:
        print("Nothing to do — registry.csv is already up to date.")
        print("(Run `python quickstart.py --all` to force a full re-analysis.)")
        return

    for f in files:
        print(f"Registering {f} ...")
        register_file(f)

    print(f"\nDone. Updated registry.csv with {len(files)} file(s).")
    print("Open registry.csv to see per-QWP fidelity estimates (is_best_in_file = winning QWP).")


if __name__ == "__main__":
    main()
