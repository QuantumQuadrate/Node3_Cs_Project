# Plan: Zarr Integration for SNAQ-Node-3 Data Infrastructure

## Context

Currently the codebase uses HDF5 exclusively (via `write_h5.py`). The user is considering migrating to Zarr. After research, here are the key findings and proposed implementation.

---

## HDF5 vs Zarr: Findings for ARTIQ

### Why NOT full Zarr migration
- **ARTIQ master is hardwired to HDF5** — the master's result archival writes `.h5` files; the ARTIQ browser reads `.h5` for historical results. This is in ARTIQ's core, not configurable.
- **Dashboard uses LMDB** for live datasets (independent of file format), but the browser/result viewer expects HDF5.
- **No performance gain locally** — benchmarks show HDF5 ≈ Zarr on local filesystems. Zarr's speed advantage is cloud-native (S3, GCS).
- **Your data volumes are small** — MB-to-GB range per experiment day. Neither format is a bottleneck.

### Why Zarr IS attractive
- **Directory-based storage** — each dataset is a folder of chunk files + JSON metadata. You can `ls` and inspect raw data without any tool. HDF5 is an opaque binary blob.
- **No file locking** — HDF5 has single-writer limitation; Zarr supports concurrent reads/writes at chunk boundaries. Useful if you want a monitoring process reading while the experiment writes.
- **Unlimited metadata** — HDF5 has a ~64KB attribute size limit. Zarr has none. Relevant if you ever embed large config blobs or full calibration snapshots.
- **xarray/Dask native** — if analysis scales to larger datasets or you want labeled dimensions, xarray reads Zarr natively (HDF5 requires netCDF4 shim).
- **Better append semantics** — Zarr's `append()` and `resize()` are first-class operations, not afterthoughts.
- **Future-proofing** — scientific Python ecosystem (Pangeo, OME-NGFF) is standardizing on Zarr.

### Recommendation: Hybrid dual-write

Keep ARTIQ's native HDF5 untouched. Add Zarr as a parallel output from `write_results()`. Analysis tools gain the option to read from either format.

```
Experiment runs
  │
  ├── ARTIQ master (unchanged) → results/{date}/{hour}/{RID}.h5
  │
  └── write_results() RPC → results/{date}/{hour}/{RID}.zarr  (NEW)
                           → results/{date}/{hour}/{RID}.h5   (KEPT)

Analysis reads: .zarr (preferred) or .h5 (fallback/legacy)
Dashboard/browser: unchanged (reads ARTIQ's own .h5 + LMDB)
```

---

## Implementation Plan

### Step 1: Extend write_results() to dual-write

**File: `utilities/write_h5.py`** → rename to `utilities/write_results.py`

Add `write_zarr()` function alongside existing `write_results()`:

```python
@rpc(flags={"async"})
def write_results(experiment, name=None, format="both"):
    """Save datasets to HDF5 and/or Zarr.
    
    format: "h5", "zarr", or "both" (default)
    """
    if format in ("h5", "both"):
        _write_h5(experiment, name)
    if format in ("zarr", "both"):
        _write_zarr(experiment, name)
```

Zarr store structure (mirrors HDF5 layout):
```
{RID:09d}-{ExperimentName}.zarr/
  ├── .zgroup                          # root group metadata
  ├── .zattrs                          # {rid, start_time, experiment, hostname, feedback_profile}
  ├── datasets/
  │   ├── SPCM1_timestamps/            # zarr array (chunked, compressed)
  │   ├── counts_signal/
  │   ├── HWP_angles/
  │   └── ...
  └── archive/
      ├── nominal_trapdepth/           # scalar dataset
      ├── f_cooling_MOT/
      └── ...                          # all ~150 ExperimentVariables snapshot
```

### Step 2: Update analysis/h5_loader.py → analysis/data_loader.py

Rename and extend to support both formats:

```python
def load(filepath):
    """Auto-detect format and load. Works with .h5 or .zarr."""
    if str(filepath).endswith('.zarr') or Path(filepath).is_dir():
        return load_zarr(filepath)
    else:
        return load_h5(filepath)

def load_zarr(filepath):
    """Load a Zarr store. Returns same dict structure as load_h5."""
    import zarr
    root = zarr.open_group(str(filepath), mode='r')
    data = {key: root['datasets'][key][:] for key in root['datasets']}
    metadata = dict(root.attrs)
    archive = {key: root['archive'][key][:] for key in root['archive']}
    return {"data": data, "metadata": metadata, "archive": archive}
```

### Step 3: Add Zarr-native analysis capabilities

**New file: `analysis/zarr_explorer.py`**

Utility for interactive exploration of Zarr stores:
- `list_runs(date)` — list all .zarr stores for a date
- `inspect(path)` — print dataset names, shapes, dtypes, chunk sizes without loading data
- `load_lazy(path)` — return zarr arrays (lazy, memory-mapped) instead of numpy (useful for large datasets)
- `to_xarray(path)` — convert zarr store to xarray.Dataset with labeled dimensions

### Step 4: Analysis output storage

**New convention: analysis results saved as Zarr too**

```
useful_codes/
  results/                              # Raw experiment data
    2026-04-03/
      15/
        000000123-ZParityExperiment.h5   # ARTIQ + write_results HDF5
        000000123-ZParityExperiment.zarr/ # write_results Zarr
  
  analysis_results/                      # NEW: post-processing output
    parity/
      2026-04-03_Z_parity_QWP_scan.zarr/ # Fitted parity curves, visibility
        ├── .zattrs                       # {source_rids: [123,124,...], analysis_date, script}
        ├── fit_params/                   # C, V, phi, f per QWP
        ├── raw_data/                     # Aggregated from multiple runs
        └── figures/                      # Saved matplotlib PNGs (as byte arrays or paths)
    microwave/
      2026-04-03_MW_scan_00.zarr/
    trap_freq/
      2026-04-03_trap_freq.zarr/
```

Benefits of Zarr for analysis output:
- You can append results from multiple experiment runs into one analysis store
- Lazy loading means you can inspect without loading everything
- xarray integration gives you labeled axes (QWP_angle, HWP_angle, frequency_MHz)

### Step 5: Update all analysis scripts

Each analysis script gets a `save_results()` function:

```python
def analyze_and_save(source_files, output_path):
    """Run analysis, save results to Zarr, return figures."""
    # Load source data
    # Run fits
    # Save fit_params, raw_data, metadata to output_path.zarr
    # Save figures as files in analysis_results/figures/
```

Update: `microwave_analysis.py`, `parity_analysis.py`, `ramsey_analysis.py`, `trap_freq_analysis.py`

---

## Folder Structure (Final)

```
useful_codes/
  results/                              # Experiment output (dual format)
    {date}/{hour}/
      {RID}-{name}.h5                   # HDF5 (ARTIQ compat + our write)
      {RID}-{name}.zarr/                # Zarr (our write, NEW)

  analysis_results/                      # Analysis output (Zarr only, NEW)
    parity/
      {date}_{description}.zarr/
    microwave/
    trap_freq/
    ramsey/

  (everything else unchanged)
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `utilities/write_results.py` | **Rename** from write_h5.py | Dual-write: HDF5 + Zarr |
| `analysis/data_loader.py` | **Rename** from h5_loader.py | Auto-detect format, load_zarr(), load_h5() |
| `analysis/zarr_explorer.py` | **New** | inspect(), load_lazy(), to_xarray() |
| `analysis/microwave_analysis.py` | **Modify** | Add save_results() to Zarr |
| `analysis/parity_analysis.py` | **Modify** | Add save_results() to Zarr |
| `analysis/ramsey_analysis.py` | **Modify** | Add save_results() to Zarr |
| `analysis/trap_freq_analysis.py` | **Modify** | Add save_results() to Zarr |
| All experiment files | **Modify** | Update `from utilities.write_h5 import` → `from utilities.write_results import` |
| `utilities/__init__.py` | **Modify** | Update export |

**No changes to:** device_db, config/, subroutines/, base_experiment, fitting/, applets/, ndsp/

---

## Verification

1. Run ExperimentVariables → verify dataset_db.pyon unchanged
2. Run atom_load_monitor → verify both `.h5` and `.zarr` created in results/
3. Load both with `data_loader.load()` → verify identical data
4. Run parity analysis → verify output saved to `analysis_results/parity/`
5. Dashboard/browser → verify unchanged behavior (reads ARTIQ's native .h5)

---

## Dependencies

```
pip install zarr
pip install xarray  # optional, for to_xarray()
```

Zarr v2 is stable and widely used. Zarr v3 is in development but not yet recommended for production.
