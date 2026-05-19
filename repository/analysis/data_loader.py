"""Unified data loading for HDF5, Zarr, and legacy CSV formats.

Auto-detects format and returns a consistent dict structure regardless
of source. Supports loading by file path, RID, or date search.

Usage::

    from analysis.data_loader import load, find_runs, load_by_rid

    result = load("results/2026-04-03/15/000000123-ZParityExperiment.zarr")
    result = load("results/2026-04-03/15/000000123-ZParityExperiment.h5")
    # Both return: {"data": {...}, "metadata": {...}, "archive": {...}}
"""

import os
import glob
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
ANALYSIS_RESULTS_DIR = Path(__file__).parent.parent / "analysis_results"
LEGACY_CSV_DIR = Path(r"C:\Networking Experiment\artiq codes\artiq-master\repository\SPCM_CSV")


# =========================================================================
#  AUTO-DETECT LOADER
# =========================================================================

def load(filepath):
    """Auto-detect format and load. Works with .h5, .zarr, or .csv.

    Returns {"data": dict, "metadata": dict, "archive": dict}.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".zarr" or (filepath.is_dir() and (filepath / ".zgroup").exists()):
        return load_zarr(filepath)
    elif filepath.suffix == ".h5":
        return load_h5(filepath)
    elif filepath.suffix == ".csv":
        return {"data": load_csv(filepath), "metadata": {}, "archive": {}}
    else:
        raise ValueError(f"Unknown format: {filepath}")


# =========================================================================
#  HDF5 LOADER
# =========================================================================

def load_h5(filepath):
    """Load an HDF5 result file."""
    import h5py
    data = {}
    metadata = {}
    archive = {}

    with h5py.File(str(filepath), "r") as f:
        if "datasets" in f:
            for key in f["datasets"]:
                data[key] = f["datasets"][key][()]
            for key in f["datasets"].attrs:
                data[key] = f["datasets"].attrs[key]

        if "metadata" in f:
            for key in f["metadata"].attrs:
                metadata[key] = f["metadata"].attrs[key]

        if "archive" in f:
            for key in f["archive"]:
                archive[key] = f["archive"][key][()]
            for key in f["archive"].attrs:
                archive[key] = f["archive"].attrs[key]

    return {"data": data, "metadata": metadata, "archive": archive}


# =========================================================================
#  ZARR LOADER
# =========================================================================

def load_zarr(filepath):
    """Load a Zarr store. Returns same dict structure as load_h5."""
    import zarr

    root = zarr.open_group(str(filepath), mode="r")
    metadata = dict(root.attrs)

    data = {}
    if "datasets" in root:
        for key in root["datasets"]:
            data[key] = root["datasets"][key][:]
        for key in root["datasets"].attrs:
            data[key] = root["datasets"].attrs[key]

    archive = {}
    if "archive" in root:
        for key in root["archive"]:
            archive[key] = root["archive"][key][:]
        for key in root["archive"].attrs:
            archive[key] = root["archive"].attrs[key]

    return {"data": data, "metadata": metadata, "archive": archive}


def load_zarr_lazy(filepath):
    """Load a Zarr store with lazy (memory-mapped) arrays.

    Returns zarr arrays instead of numpy — data is only read when accessed.
    Useful for large datasets where you only need a subset.
    """
    import zarr

    root = zarr.open_group(str(filepath), mode="r")
    metadata = dict(root.attrs)

    data = {}
    if "datasets" in root:
        for key in root["datasets"]:
            data[key] = root["datasets"][key]  # zarr array, not loaded
        for key in root["datasets"].attrs:
            data[key] = root["datasets"].attrs[key]

    archive = {}
    if "archive" in root:
        for key in root["archive"]:
            archive[key] = root["archive"][key]
        for key in root["archive"].attrs:
            archive[key] = root["archive"].attrs[key]

    return {"data": data, "metadata": metadata, "archive": archive}


# =========================================================================
#  CSV LOADER (legacy)
# =========================================================================

def load_csv(filepath, **kwargs):
    """Load a legacy CSV file. Returns DataFrame."""
    import pandas as pd
    return pd.read_csv(str(filepath), **kwargs)


# =========================================================================
#  FILE DISCOVERY
# =========================================================================

def find_runs(date_str=None, experiment=None, format=None, results_dir=None):
    """Find result files by date and/or experiment name.

    Parameters
    ----------
    date_str : str, optional
        Date like "2026-04-03". If None, searches all dates.
    experiment : str, optional
        Experiment class name substring to filter by.
    format : str, optional
        "h5", "zarr", or None (both).
    results_dir : Path, optional
        Override results directory.

    Returns list of file paths sorted by modification time.
    """
    base = Path(results_dir) if results_dir else RESULTS_DIR
    files = []

    if format in (None, "h5"):
        if date_str:
            pattern = str(base / date_str / "**" / "*.h5")
        else:
            pattern = str(base / "**" / "*.h5")
        files.extend(glob.glob(pattern, recursive=True))

    if format in (None, "zarr"):
        if date_str:
            pattern = str(base / date_str / "**" / "*.zarr")
        else:
            pattern = str(base / "**" / "*.zarr")
        files.extend(glob.glob(pattern, recursive=True))

    files = list(dict.fromkeys(files))  # deduplicate, preserve order

    if experiment:
        files = [f for f in files if experiment.lower() in os.path.basename(f).lower()]

    return sorted(files, key=os.path.getmtime)


def load_by_rid(rid, format=None, results_dir=None):
    """Load result file by run ID number.

    Prefers .zarr if both exist, unless format is specified.
    """
    base = Path(results_dir) if results_dir else RESULTS_DIR

    if format != "h5":
        zarr_files = glob.glob(str(base / "**" / f"{rid:09d}-*.zarr"), recursive=True)
        if zarr_files:
            return load_zarr(zarr_files[0])

    if format != "zarr":
        h5_files = glob.glob(str(base / "**" / f"{rid:09d}-*.h5"), recursive=True)
        if h5_files:
            return load_h5(h5_files[0])

    return None


def find_csvs(pattern="*.csv", csv_dir=None):
    """Find CSV files matching a glob pattern in the legacy CSV directory."""
    base = Path(csv_dir) if csv_dir else LEGACY_CSV_DIR
    return sorted(glob.glob(str(base / pattern)), key=os.path.getmtime)
