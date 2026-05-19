"""Interactive Zarr store exploration utilities.

Provides tools for inspecting, lazy-loading, and converting Zarr stores
without loading entire datasets into memory.

Usage::

    from analysis.zarr_explorer import inspect, list_runs, to_xarray

    inspect("results/2026-04-03/15/000000123-ZParityExperiment.zarr")
    runs = list_runs("2026-04-03")
    ds = to_xarray("results/2026-04-03/15/000000123-ZParityExperiment.zarr")
"""

import os
import glob
import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
ANALYSIS_RESULTS_DIR = Path(__file__).parent.parent / "analysis_results"


def inspect(path, show_chunks=True, show_compressor=True):
    """Print detailed info about a Zarr store without loading data.

    Shows dataset names, shapes, dtypes, chunk sizes, compression, and metadata.
    """
    import zarr

    root = zarr.open_group(str(path), mode="r")

    print(f"Zarr store: {path}")
    print(f"{'=' * 70}")

    # Root metadata
    if root.attrs:
        print("Metadata:")
        for k, v in root.attrs.items():
            print(f"  {k}: {v}")
        print()

    # Walk all groups
    def _inspect_group(group, indent=0):
        prefix = "  " * indent
        for key in sorted(group):
            item = group[key]
            if hasattr(item, "shape"):
                # It's an array
                info = f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}"
                if show_chunks and item.chunks:
                    info += f", chunks={item.chunks}"
                if show_compressor and item.compressor:
                    info += f", compressor={item.compressor}"
                print(info)
                if item.attrs:
                    for ak, av in item.attrs.items():
                        print(f"{prefix}  @{ak}: {av}")
            else:
                # It's a group
                print(f"{prefix}{key}/")
                if item.attrs:
                    for ak, av in item.attrs.items():
                        print(f"{prefix}  @{ak}: {av}")
                _inspect_group(item, indent + 1)

    _inspect_group(root)


def list_runs(date_str=None, results_dir=None):
    """List all Zarr stores for a given date.

    Returns list of dicts with path, rid, experiment name, and metadata.
    """
    import zarr

    base = Path(results_dir) if results_dir else RESULTS_DIR
    pattern = str(base / (date_str or "**") / "**" / "*.zarr")
    stores = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)

    runs = []
    for store_path in stores:
        try:
            root = zarr.open_group(store_path, mode="r")
            runs.append({
                "path": store_path,
                "rid": root.attrs.get("rid", "?"),
                "experiment": root.attrs.get("experiment", "?"),
                "start_time": root.attrs.get("start_time", "?"),
                "n_datasets": len(root["datasets"]) if "datasets" in root else 0,
            })
        except Exception:
            runs.append({"path": store_path, "error": "could not read"})

    return runs


def print_runs(date_str=None, results_dir=None):
    """Pretty-print all runs for a date."""
    runs = list_runs(date_str, results_dir)
    if not runs:
        print(f"No Zarr stores found for {date_str or 'any date'}")
        return

    print(f"{'RID':>10}  {'Experiment':<30}  {'Time':<20}  {'Datasets':>8}")
    print("-" * 75)
    for r in runs:
        if "error" in r:
            print(f"  ERROR: {r['path']} - {r['error']}")
        else:
            print(f"{r['rid']:>10}  {r['experiment']:<30}  {r['start_time']:<20}  {r['n_datasets']:>8}")


def to_xarray(path):
    """Convert a Zarr store to an xarray.Dataset.

    Requires xarray: pip install xarray

    Returns xarray.Dataset with all datasets/ arrays as data variables
    and metadata as global attributes.
    """
    import xarray as xr
    import zarr
    import numpy as np

    root = zarr.open_group(str(path), mode="r")
    metadata = dict(root.attrs)

    data_vars = {}
    if "datasets" in root:
        for key in root["datasets"]:
            arr = root["datasets"][key][:]
            attrs = dict(root["datasets"][key].attrs) if root["datasets"][key].attrs else {}
            if arr.ndim == 0:
                data_vars[key] = xr.DataArray(arr.item(), attrs=attrs)
            elif arr.ndim == 1:
                data_vars[key] = xr.DataArray(arr, dims=[f"{key}_dim"], attrs=attrs)
            else:
                dims = [f"{key}_dim{i}" for i in range(arr.ndim)]
                data_vars[key] = xr.DataArray(arr, dims=dims, attrs=attrs)

    return xr.Dataset(data_vars, attrs=metadata)


def save_analysis_zarr(data_dict, metadata, output_path):
    """Save analysis results to a Zarr store in analysis_results/.

    Parameters
    ----------
    data_dict : dict
        Maps dataset_name -> numpy array or scalar.
    metadata : dict
        Metadata attributes (source_rids, analysis_date, script_name, etc.)
    output_path : str or Path
        Path for the .zarr output directory.
    """
    import zarr
    import numpy as np
    from numcodecs import Blosc

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(output_path), mode="w")

    # Metadata
    metadata["saved_at"] = datetime.datetime.now().isoformat()
    for k, v in metadata.items():
        if isinstance(v, (list, tuple)):
            root.attrs[k] = list(v)
        else:
            root.attrs[k] = v

    # Data
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            chunks = tuple(min(s, 1024) for s in value.shape) if value.ndim > 0 else None
            root.create_dataset(key, data=value, chunks=chunks,
                                compressor=compressor, overwrite=True)
        elif isinstance(value, (int, float, bool)):
            root.create_dataset(key, data=np.array(value), overwrite=True)
        elif isinstance(value, str):
            root.attrs[key] = value
        else:
            root.attrs[key] = str(value)

    print(f"Analysis results saved: {output_path}")
    return output_path
