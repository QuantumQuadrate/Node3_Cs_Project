"""Standalone MW scan analysis: load HDF5, summarize, plot, fit.

Primary workflow: set RID below, press F5 in VS Code.
Also works from command line:
    python -m analysis.analyze_mw_scan
    python -m analysis.analyze_mw_scan --rid 50
    python -m analysis.analyze_mw_scan --list
"""

# ===== CONFIGURE HERE — set RID or leave None for latest =====
RID = None              # e.g. 50 to load specific run, None for latest
LIST_RUNS = False       # True to list recent runs and exit
NO_FIT = False          # True to skip fitting
# ==============================================================

import sys
import os
import argparse
import glob
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.data_loader import load_h5, find_runs, RESULTS_DIR


# =========================================================================
#  FILE DISCOVERY
# =========================================================================

def list_recent_runs(n=30):
    """List recent HDF5 files with scan info peeked from inside each file."""
    files = find_runs(format="h5")
    if not files:
        print(f"No HDF5 files found in {RESULTS_DIR}")
        return

    files = files[-n:]

    print(f"\n{'='*90}")
    print(f"  Recent runs in {RESULTS_DIR}")
    print(f"{'='*90}")
    print(f"  {'RID':>6s}  {'Experiment':<22s}  {'Scan Mode':<12s}  "
          f"{'Type':<11s}  {'Date':<12s}  {'Time':<6s}  Folder")
    print(f"  {'─'*6}  {'─'*22}  {'─'*12}  {'─'*11}  {'─'*12}  {'─'*6}  {'─'*20}")

    for f in reversed(files):
        fname = os.path.basename(f)
        folder = str(Path(f).parent.relative_to(RESULTS_DIR))

        # Parse RID and experiment name from filename
        parts = fname.replace(".h5", "").split("-", 1)
        rid_str = parts[0] if len(parts) > 0 else "?"
        exp_name = parts[1] if len(parts) > 1 else "?"

        # File modification time
        mtime = os.path.getmtime(f)
        dt = datetime.datetime.fromtimestamp(mtime)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")

        # Peek inside HDF5 for scan mode/type
        scan_mode = ""
        scan_type = ""
        try:
            import h5py
            with h5py.File(f, "r") as hf:
                ds = hf.get("datasets")
                if ds is not None:
                    if "mw_scan_mode" in ds.attrs:
                        scan_mode = ds.attrs["mw_scan_mode"]
                    elif "mw_scan_mode" in ds:
                        val = ds["mw_scan_mode"][()]
                        scan_mode = val.decode() if isinstance(val, bytes) else str(val)
                    if "mw_scan_type" in ds.attrs:
                        scan_type = ds.attrs["mw_scan_type"]
                    elif "mw_scan_type" in ds:
                        val = ds["mw_scan_type"][()]
                        scan_type = val.decode() if isinstance(val, bytes) else str(val)
        except Exception:
            pass

        if isinstance(scan_mode, bytes):
            scan_mode = scan_mode.decode()
        if isinstance(scan_type, bytes):
            scan_type = scan_type.decode()

        print(f"  {rid_str:>6s}  {exp_name:<22s}  {scan_mode:<12s}  "
              f"{scan_type:<11s}  {date_str:<12s}  {time_str:<6s}  {folder}")

    print(f"{'='*90}\n")


def find_file_by_rid(rid):
    """Find HDF5 file by RID number."""
    pattern = f"{rid:09d}-*.h5"
    matches = glob.glob(str(RESULTS_DIR / "**" / pattern), recursive=True)
    if not matches:
        print(f"ERROR: No HDF5 file found for RID {rid}")
        print(f"  Searched in: {RESULTS_DIR}")
        sys.exit(1)
    return matches[0]


def find_latest():
    """Find the most recent HDF5 file."""
    files = find_runs(format="h5")
    if not files:
        print(f"ERROR: No HDF5 files found in {RESULTS_DIR}")
        sys.exit(1)
    return files[-1]


# =========================================================================
#  SUMMARY & DATA TABLE
# =========================================================================

def summarize(filepath, result):
    """Print metadata and MW scan summary."""
    data = result["data"]
    meta = result["metadata"]

    print(f"\n{'='*70}")
    print(f"  File: {os.path.basename(filepath)}")
    print(f"  Path: {filepath}")
    print(f"{'='*70}")

    if meta:
        print(f"\n  Metadata:")
        for k, v in meta.items():
            val = v.decode() if isinstance(v, bytes) else v
            print(f"    {k}: {val}")

    # MW scan specific
    if "mw_scan_x" in data:
        x = np.asarray(data["mw_scan_x"])
        unique_x = np.unique(x)
        n_points = len(unique_x)
        n_total = len(x)
        n_loops = n_total // max(n_points, 1)

        mode = data.get("mw_scan_mode", b"?")
        stype = data.get("mw_scan_type", b"?")
        if isinstance(mode, bytes): mode = mode.decode()
        if isinstance(stype, bytes): stype = stype.decode()

        print(f"\n  MW Scan:")
        print(f"    Mode: {mode}  |  Type: {stype}")
        print(f"    Scan points: {n_points}  |  Loops: {n_loops}  |  Total: {n_total}")
        if n_points > 0:
            if stype == "frequency":
                print(f"    Range: {unique_x[0]/1e6:.4f} .. {unique_x[-1]/1e6:.4f} MHz")
            else:
                print(f"    Range: {unique_x[0]:.1f} .. {unique_x[-1]:.1f} mu")

    # Inline fit results
    if "mw_fit_success" in data:
        success = data["mw_fit_success"]
        print(f"\n  Inline Fit: {'SUCCESS' if success else 'FAILED'}")
        if "mw_fit_summary" in data:
            s = data["mw_fit_summary"]
            if isinstance(s, bytes): s = s.decode()
            for line in s.split("\n"):
                print(f"    {line}")
        if success and "mw_fit_extracted_value" in data:
            print(f"    Extracted: {data['mw_fit_extracted_value']}")


def print_data_table(data):
    """Print averaged per-scan-point data as a readable table."""
    if "mw_scan_x" not in data:
        return

    x_raw = np.asarray(data["mw_scan_x"], dtype=float)
    if x_raw.size == 0:
        return

    stype = str(data.get("mw_scan_type", b"frequency"))
    if isinstance(stype, bytes): stype = stype.decode()

    unique_x = np.unique(x_raw)

    # Gather arrays
    def avg_by_x(key):
        if key not in data:
            return None
        arr = np.asarray(data[key], dtype=float)
        if arr.size != x_raw.size:
            return None
        return np.array([np.mean(arr[x_raw == xv]) for xv in unique_x])

    # Retention = N(shot1>thresh AND shot2>thresh) / N(shot1>thresh) per scan point
    ret_raw = np.asarray(data.get("mw_retention", []), dtype=float)
    load_rate = avg_by_x("mw_atom_loaded")
    retention = np.zeros(len(unique_x))
    for i, xv in enumerate(unique_x):
        vals = ret_raw[x_raw == xv]
        valid = vals[vals >= 0]  # exclude -1 (no atom loaded)
        retention[i] = valid.mean() if len(valid) > 0 else float('nan')
    s1_1 = avg_by_x("mw_shot1_spcm1")
    s1_2 = avg_by_x("mw_shot1_spcm2")
    s2_1 = avg_by_x("mw_shot2_spcm1")
    s2_2 = avg_by_x("mw_shot2_spcm2")
    bg_1 = avg_by_x("mw_bg_spcm1")
    bg_2 = avg_by_x("mw_bg_spcm2")
    n_per = np.array([np.sum(x_raw == xv) for xv in unique_x])

    # Compute combined counts
    shot1_tot = (s1_1 + s1_2) if (s1_1 is not None and s1_2 is not None) else None
    shot2_tot = (s2_1 + s2_2) if (s2_1 is not None and s2_2 is not None) else None
    bg_tot = (bg_1 + bg_2) if (bg_1 is not None and bg_2 is not None) else None

    # Binomial errors (using N_loaded, not N_total)
    err = np.zeros(len(unique_x))
    for i, xv in enumerate(unique_x):
        vals = ret_raw[x_raw == xv]
        n_loaded = np.sum(vals >= 0)
        r = retention[i]
        if n_loaded > 0 and not np.isnan(r):
            err[i] = np.sqrt(max(r*(1-r), 0) / n_loaded)
        else:
            err[i] = float('nan')

    # Format scan values
    if stype == "frequency":
        x_label = "Freq (MHz)"
        x_fmt = [f"{v/1e6:.4f}" for v in unique_x]
    elif stype in ("time", "ramsey"):
        x_label = "Time (mu)"
        x_fmt = [f"{v:.0f}" for v in unique_x]
    else:
        x_label = "Param"
        x_fmt = [f"{v:.4g}" for v in unique_x]

    # Print table
    print(f"\n  {'─'*80}")
    print(f"  {x_label:<14s}  {'N':>4s}  {'Ret':>6s}  {'Err':>6s}  "
          f"{'Load':>5s}  {'Shot1':>6s}  {'Shot2':>6s}  {'BG':>5s}")
    print(f"  {'─'*14}  {'─'*4}  {'─'*6}  {'─'*6}  "
          f"{'─'*5}  {'─'*6}  {'─'*6}  {'─'*5}")

    for i in range(len(unique_x)):
        row = f"  {x_fmt[i]:<14s}  {int(n_per[i]):>4d}"
        row += f"  {retention[i]:>6.3f}" if retention is not None else "       "
        row += f"  {err[i]:>6.3f}" if err is not None else "       "
        row += f"  {load_rate[i]:>5.2f}" if load_rate is not None else "      "
        row += f"  {shot1_tot[i]:>6.1f}" if shot1_tot is not None else "       "
        row += f"  {shot2_tot[i]:>6.1f}" if shot2_tot is not None else "       "
        row += f"  {bg_tot[i]:>5.1f}" if bg_tot is not None else "      "
        print(row)

    print(f"  {'─'*80}\n")


# =========================================================================
#  PLOTTING
# =========================================================================

def plot_mw_scan(data, fit_result=None, fitter=None, save_path=None):
    """Plot MW scan: retention, raw counts, and loading rate. Always shows."""
    if "mw_scan_x" not in data:
        print("No mw_scan_x dataset — cannot plot.")
        return

    x_raw = np.asarray(data["mw_scan_x"], dtype=float)
    if x_raw.size == 0:
        print("mw_scan_x is empty — nothing to plot.")
        return

    stype = str(data.get("mw_scan_type", b"frequency"))
    if isinstance(stype, bytes): stype = stype.decode()

    # Unit conversion
    if stype == "frequency":
        to_plot = lambda v: v / 1e6
        xlabel = "Frequency (MHz)"
    elif stype == "time":
        to_plot = lambda v: v
        xlabel = "Duration (mu)"
    elif stype == "ramsey":
        to_plot = lambda v: v
        xlabel = "Free evolution time (mu)"
    else:
        to_plot = lambda v: v
        xlabel = "Scan parameter"

    # Average by unique scan value
    unique_x = np.unique(x_raw)
    unique_x_plot = to_plot(unique_x)
    n_per = np.array([np.sum(x_raw == xv) for xv in unique_x])

    def avg_by_x(key):
        if key not in data:
            return None
        arr = np.asarray(data[key], dtype=float)
        if arr.size != x_raw.size:
            return None
        return np.array([np.mean(arr[x_raw == xv]) for xv in unique_x])

    # Retention = N(shot1>thresh AND shot2>thresh) / N(shot1>thresh)
    ret_raw = np.asarray(data.get("mw_retention", []), dtype=float)
    load_rate = avg_by_x("mw_atom_loaded")
    retention = np.zeros(len(unique_x))
    err_ret = np.zeros(len(unique_x))
    for i, xv in enumerate(unique_x):
        vals = ret_raw[x_raw == xv]
        valid = vals[vals >= 0]
        n_loaded = len(valid)
        if n_loaded > 0:
            retention[i] = valid.mean()
            err_ret[i] = np.sqrt(max(retention[i]*(1-retention[i]), 0) / n_loaded)
        else:
            retention[i] = float('nan')
            err_ret[i] = float('nan')

    # --- Figure: 3 subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})

    mode_name = data.get("mw_scan_mode", b"MW Scan")
    if isinstance(mode_name, bytes): mode_name = mode_name.decode()

    # ---- Top: Retention ----
    ax1 = axes[0]
    if retention is not None:
        ax1.errorbar(unique_x_plot, retention, yerr=err_ret,
                     fmt="o", color="C0", capsize=3, markersize=6,
                     label=f"Retention (n={int(n_per[0])} per pt)")

    # Fit overlay
    fit_shown = False
    if fit_result and fit_result.get("success") and fitter is not None:
        x_fine = np.linspace(unique_x.min(), unique_x.max(), 500)
        if stype == "frequency":
            x_fine_fit = x_fine / 1e6
        else:
            x_fine_fit = x_fine
        y_fine = fitter.evaluate(x_fine_fit, fit_result["params"])
        ax1.plot(to_plot(x_fine), y_fine, "r-", lw=2, alpha=0.8, label="Fit")
        fit_shown = True

        summary = fit_result.get("summary", "")
        if summary:
            ax1.text(0.02, 0.02, summary, transform=ax1.transAxes,
                     fontsize=8, va="bottom", fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    if not fit_shown:
        # Show why there's no fit
        if fit_result and not fit_result.get("success"):
            reason = "Fit failed"
        elif len(unique_x) < 3:
            reason = f"Not enough scan points ({len(unique_x)}) for fit"
        else:
            reason = "Fit skipped"
        ax1.text(0.98, 0.02, reason, transform=ax1.transAxes,
                 fontsize=9, va="bottom", ha="right", color="gray",
                 fontstyle="italic")

    ax1.set_ylabel("Retention")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{mode_name} — RID {data.get('rid', '?')}")

    # ---- Middle: Raw photon counts ----
    ax2 = axes[1]
    has_counts = False

    # Combined counts
    s1_1 = avg_by_x("mw_shot1_spcm1")
    s1_2 = avg_by_x("mw_shot1_spcm2")
    s2_1 = avg_by_x("mw_shot2_spcm1")
    s2_2 = avg_by_x("mw_shot2_spcm2")
    bg_1 = avg_by_x("mw_bg_spcm1")
    bg_2 = avg_by_x("mw_bg_spcm2")

    if s1_1 is not None and s1_2 is not None:
        ax2.plot(unique_x_plot, s1_1 + s1_2, "s-", color="C2", markersize=4,
                 label="Shot1 (loaded)")
        has_counts = True
    if s2_1 is not None and s2_2 is not None:
        ax2.plot(unique_x_plot, s2_1 + s2_2, "D-", color="C4", markersize=4,
                 label="Shot2 (post-MW)")
        has_counts = True
    if bg_1 is not None and bg_2 is not None:
        ax2.plot(unique_x_plot, bg_1 + bg_2, "x-", color="C7", markersize=4,
                 label="Background")
        has_counts = True

    if has_counts:
        ax2.set_ylabel("Avg total photon counts")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No per-SPCM count data",
                 transform=ax2.transAxes, ha="center", va="center", color="gray")

    # ---- Bottom: Loading rate ----
    ax3 = axes[2]
    if load_rate is not None:
        ax3.plot(unique_x_plot, load_rate, "o-", color="C1", markersize=4)
        ax3.set_ylabel("Loading rate")
        ax3.set_ylim(-0.05, 1.15)
        ax3.axhline(1.0, color="gray", ls="--", lw=0.5)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No loading data",
                 transform=ax3.transAxes, ha="center", va="center", color="gray")

    ax3.set_xlabel(xlabel)

    plt.tight_layout()

    # Save figure
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    plt.show()


# =========================================================================
#  FITTING (reuses existing infrastructure)
# =========================================================================

def try_fit(data):
    """Attempt to fit the scan data. Returns (fit_result, fitter) or (None, None)."""
    from analysis.microwave_analysis import fit_scan_result

    x = np.asarray(data["mw_scan_x"], dtype=float)
    retention = np.asarray(data.get("mw_retention", []), dtype=float)

    if retention.size != x.size:
        print("  Retention array size mismatch — skipping fit.")
        return None, None

    n_unique = len(np.unique(x))
    if n_unique < 3:
        print(f"  Only {n_unique} unique scan points — need >= 3 for fit.")
        return {"success": False, "summary": f"Only {n_unique} scan points"}, None

    stype = data.get("mw_scan_type", b"frequency")
    smode = data.get("mw_scan_mode", b"unknown")
    if isinstance(stype, bytes): stype = stype.decode()
    if isinstance(smode, bytes): smode = smode.decode()

    print(f"  Fitting: mode={smode}, type={stype}")
    result = fit_scan_result(x, retention, stype, smode)

    if result and result.get("success"):
        fitter = result.get("fitter")
        print(f"\n  FIT SUCCEEDED:")
        for line in result.get("summary", "").split("\n"):
            print(f"    {line}")
        return result, fitter
    else:
        print(f"  FIT FAILED")
        return result, None


# =========================================================================
#  MAIN
# =========================================================================

def main():
    # CLI args override the top-of-file config
    parser = argparse.ArgumentParser(description="MW scan analysis")
    parser.add_argument("--file", type=str)
    parser.add_argument("--rid", type=int)
    parser.add_argument("--list", action="store_true", dest="list_runs")
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    # Merge: CLI args take priority over top-of-file config
    do_list = args.list_runs or LIST_RUNS
    rid = args.rid if args.rid is not None else RID
    no_fit = args.no_fit or NO_FIT

    if do_list:
        list_recent_runs(30)
        return

    # Find file
    if args.file:
        filepath = args.file
    elif rid is not None:
        filepath = find_file_by_rid(rid)
    else:
        filepath = find_latest()
        print(f"(Using latest run — set RID at top of file to pick a specific one)")

    print(f"\nLoading: {filepath}")

    result = load_h5(filepath)
    data = result["data"]

    # Add RID from file to data for plot title
    if "rid" not in data:
        fname = os.path.basename(filepath)
        try:
            data["rid"] = int(fname.split("-")[0])
        except (ValueError, IndexError):
            pass

    summarize(filepath, result)

    if "mw_scan_x" not in data:
        print("This file does not contain MW scan data (no mw_scan_x dataset).")
        print("Datasets available:")
        for k in sorted(data.keys()):
            print(f"  {k}")
        return

    print_data_table(data)

    # Fit
    fit_result = None
    fitter = None
    if not no_fit:
        fit_result, fitter = try_fit(data)

    # Plot (always, regardless of fit outcome)
    if not args.no_plot:
        png_path = filepath.replace(".h5", ".png")
        plot_mw_scan(data, fit_result=fit_result, fitter=fitter, save_path=png_path)


if __name__ == "__main__":
    main()
