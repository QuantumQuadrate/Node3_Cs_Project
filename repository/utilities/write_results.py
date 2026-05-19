"""Dual-format result saving for ARTIQ experiments (HDF5 + Zarr).

Provides an async RPC function that dumps all current datasets plus metadata
into both an HDF5 file (for ARTIQ dashboard/browser compatibility) and a
Zarr store (for modern analysis workflows with xarray/Dask).

Usage in experiments::

    from utilities.write_results import write_results
    # At end of run():
    write_results(self)                    # writes both .h5 and .zarr
    write_results(self, format="zarr")     # zarr only
    write_results(self, format="h5")       # h5 only
"""

import os
import datetime
from pathlib import Path

from artiq.language.core import rpc

RESULTS_BASE = Path(__file__).parent.parent / "results"


def _get_results_path(experiment, name=None):
    """Get (out_dir, base_name, rid) for saving results.

    Computes the path ONCE using the experiment start time and caches it
    on the experiment object. All subsequent calls return the same path,
    preventing duplicate files when an experiment crosses an hour boundary.
    """
    if name is None:
        name = type(experiment).__name__

    try:
        rid = experiment.scheduler.rid
    except AttributeError:
        rid = 0

    # Cache the output dir on first call so all saves go to the same folder
    if not hasattr(experiment, '_results_out_dir'):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")
        out_dir = RESULTS_BASE / date_str / hour_str
        out_dir.mkdir(parents=True, exist_ok=True)
        experiment._results_out_dir = out_dir
        experiment._results_start_time = now

    return experiment._results_out_dir, f"{rid:09d}-{name}", rid


@rpc(flags={"async"})
def write_results(experiment, name=None, format="both"):
    """Save all datasets and metadata to HDF5 and/or Zarr.

    Parameters
    ----------
    experiment : EnvExperiment
        The running experiment instance.
    name : str, optional
        Override for the filename. Defaults to the experiment class name.
    format : str
        Output format: "both" (default), "h5", or "zarr".
    """
    import numpy as np

    out_dir, base_name, rid = _get_results_path(experiment, name)
    if name is None:
        name = type(experiment).__name__

    # Collect all data once (shared between both writers)
    dataset_mgr = experiment._HasEnvironment__dataset_mgr
    start_time = getattr(experiment, '_results_start_time', datetime.datetime.now())
    metadata = {
        "rid": rid,
        "start_time": start_time.isoformat(),
        "experiment": name,
        "hostname": os.environ.get("COMPUTERNAME", "unknown"),
    }
    if hasattr(experiment, "_feedback_profile"):
        metadata["feedback_profile"] = experiment._feedback_profile

    local_data = _collect_local_datasets(dataset_mgr, np)
    archive_data = _collect_archive_datasets(dataset_mgr, np)

    if format in ("h5", "both"):
        filepath_h5 = out_dir / f"{base_name}.h5"
        _write_h5(filepath_h5, metadata, local_data, archive_data, np)
        print(f"HDF5 saved: {filepath_h5}")

    if format in ("zarr", "both"):
        try:
            filepath_zarr = out_dir / f"{base_name}.zarr"
            _write_zarr(filepath_zarr, metadata, local_data, archive_data, np)
            print(f"Zarr saved: {filepath_zarr}")
        except ImportError:
            if format == "zarr":
                print("WARNING: zarr/numcodecs not installed, cannot write Zarr")
            # silently skip zarr when format="both" and module missing


# =========================================================================
#  DATA COLLECTION (shared)
# =========================================================================

def _collect_local_datasets(dataset_mgr, np):
    """Extract local datasets from ARTIQ dataset manager."""
    data = {}
    try:
        for key, entry in dataset_mgr.local.items():
            try:
                value = entry[1] if isinstance(entry, tuple) and len(entry) >= 2 else entry
                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                data[key] = value
            except Exception:
                data[key] = str(entry)
    except Exception:
        pass
    return data


def _collect_archive_datasets(dataset_mgr, np):
    """Extract archive (persistent) datasets from ARTIQ dataset manager."""
    data = {}
    try:
        for key, value in dataset_mgr.archive.items():
            try:
                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                data[key] = value
            except Exception:
                data[key] = str(value)
    except AttributeError:
        pass
    return data


# =========================================================================
#  HDF5 WRITER
# =========================================================================

def _write_h5(filepath, metadata, local_data, archive_data, np):
    """Write datasets and metadata to an HDF5 file."""
    import h5py

    with h5py.File(str(filepath), "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        for k, v in metadata.items():
            meta.attrs[k] = v

        # Datasets
        datasets_group = f.create_group("datasets")
        for key, value in local_data.items():
            _write_h5_value(datasets_group, key, value, np)

        # Archive
        archive_group = f.create_group("archive")
        for key, value in archive_data.items():
            _write_h5_value(archive_group, key, value, np)


def _write_h5_value(group, key, value, np):
    """Write a single value to an HDF5 group."""
    try:
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, (int, float, bool, np.integer, np.floating)):
            group.create_dataset(key, data=value)
        elif isinstance(value, str):
            group.attrs[key] = value
        else:
            group.attrs[key] = str(value)
    except Exception:
        try:
            group.attrs[key] = str(value)
        except Exception:
            pass


# =========================================================================
#  ZARR WRITER
# =========================================================================

def _write_zarr(filepath, metadata, local_data, archive_data, np):
    """Write datasets and metadata to a Zarr store."""
    import zarr
    from numcodecs import Blosc

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    root = zarr.open_group(str(filepath), mode="w")

    # Metadata as root attributes
    for k, v in metadata.items():
        root.attrs[k] = v

    # Datasets
    datasets_group = root.create_group("datasets")
    for key, value in local_data.items():
        _write_zarr_value(datasets_group, key, value, np, compressor)

    # Archive
    archive_group = root.create_group("archive")
    for key, value in archive_data.items():
        _write_zarr_value(archive_group, key, value, np, compressor)


def _write_zarr_value(group, key, value, np, compressor):
    """Write a single value to a Zarr group."""
    try:
        if isinstance(value, np.ndarray):
            if value.size > 0:
                chunks = _auto_chunks(value)
                group.create_dataset(key, data=value, chunks=chunks,
                                     compressor=compressor, overwrite=True)
            else:
                group.create_dataset(key, data=value, overwrite=True)
        elif isinstance(value, (int, float, bool, np.integer, np.floating)):
            group.create_dataset(key, data=np.array(value), overwrite=True)
        elif isinstance(value, str):
            group.attrs[key] = value
        else:
            group.attrs[key] = str(value)
    except Exception:
        try:
            group.attrs[key] = str(value)
        except Exception:
            pass


def _auto_chunks(arr):
    """Pick reasonable chunk sizes for a numpy array."""
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return (min(len(arr), 1024),)
    # For 2D+, chunk along last axis
    return tuple(min(s, 1024) for s in arr.shape)


# =========================================================================
#  INCREMENTAL SAVE (callable from host-side code, not an RPC)
# =========================================================================

def _save_h5_now(experiment, name=None):
    """Write current datasets to HDF5. Plain function, not an RPC.

    Call this from host-side code (e.g. from within an async RPC) to
    save a snapshot of the current data mid-experiment.

    Uses the same cached output path as write_results() so all saves
    for one RID go to the same file regardless of clock changes.
    """
    import numpy as np

    out_dir, base_name, rid = _get_results_path(experiment, name)
    if name is None:
        name = type(experiment).__name__

    dataset_mgr = experiment._HasEnvironment__dataset_mgr
    start_time = getattr(experiment, '_results_start_time', datetime.datetime.now())
    metadata = {
        "rid": rid,
        "start_time": start_time.isoformat(),
        "experiment": name,
        "hostname": os.environ.get("COMPUTERNAME", "unknown"),
    }
    if hasattr(experiment, "_feedback_profile"):
        metadata["feedback_profile"] = experiment._feedback_profile

    local_data = _collect_local_datasets(dataset_mgr, np)
    archive_data = _collect_archive_datasets(dataset_mgr, np)

    filepath_h5 = out_dir / f"{base_name}.h5"
    _write_h5(filepath_h5, metadata, local_data, archive_data, np)


# =========================================================================
#  UTILITIES
# =========================================================================

@rpc(flags={"async"})
def print_async(experiment, *args):
    """Non-blocking print from kernel code. Call as print_async(self, 'msg')."""
    print(*args)
