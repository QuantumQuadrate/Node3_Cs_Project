"""Ramsey free-evolution analysis.

Adapted from Old_py_analysis_codes/Ramsey_Analysis.py.
Fits damped cosine with Gaussian decay (n=2 fixed).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from analysis.statistics import classify_events, group_by_scan
from analysis.threshold import find_threshold_double_gaussian
from fitting.damped_cosine import RamseyFit


def analyze_ramsey(filepath, scan_axis_ns=None, threshold=12.0,
                   force_threshold=True):
    """Ramsey analysis with hybrid LSQ→MLE fitting.

    Parameters
    ----------
    filepath : str
        Path to CSV or HDF5 file.
    scan_axis_ns : list or None
        Time axis in nanoseconds. Required for CSV files where the
        scan axis is not stored in the data.
    threshold : float
        Atom detection threshold.

    Returns
    -------
    dict with 'grouped', 'fit_result', 'fitter'.
    """
    from analysis.data_loader import load, load_h5, load_csv

    if filepath.endswith(".h5"):
        result = load_h5(filepath)
        shot1 = result["data"].get("counts_signal", np.array([]))
        shot2 = result["data"].get("counts_background", np.array([]))
        bg = np.zeros_like(shot1)
        scan_ns = result["data"].get("scan_sequence", scan_axis_ns)
    else:
        df = load_csv(filepath)
        shot1 = df.iloc[:, 1].values.astype(float)
        shot2 = df.iloc[:, 2].values.astype(float)
        bg = df.iloc[:, 3].values.astype(float) if df.shape[1] > 3 else np.zeros_like(shot1)
        scan_ns = scan_axis_ns

    if scan_ns is None:
        raise ValueError("scan_axis_ns required for Ramsey analysis")

    events = classify_events(shot1, shot2, background=bg, threshold=threshold)
    grouped = group_by_scan(scan_ns, events["loaded"], events["retained"])

    # Convert ns to ms for fitting
    x_ms = grouped["x"].values / 1e6
    y = grouped["Ret_P"].values
    yerr = grouped["Ret_Err"].values
    k = grouped["N_retained"].values
    n = grouped["N_loaded"].values

    fitter = RamseyFit()
    fit_result = fitter.fit_hybrid(x_ms, y, yerr=yerr, k=k, n=n)

    return {"grouped": grouped, "fit_result": fit_result, "fitter": fitter}


def plot_ramsey(result, title="Ramsey Oscillation"):
    """Plot Ramsey analysis with fit overlay."""
    grouped = result["grouped"]
    fit_result = result["fit_result"]
    fitter = result["fitter"]

    x_ms = grouped["x"].values / 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(x_ms, grouped["Load_Rate"], yerr=grouped["Load_Err"],
                 fmt="o-", capsize=3)
    ax1.set_xlabel("Free evolution time (ms)")
    ax1.set_ylabel("Loading rate")
    ax1.set_title(f"{title} - Loading")

    ax2.errorbar(x_ms, grouped["Ret_P"], yerr=grouped["Ret_Err"],
                 fmt="o", capsize=3, label="Data")
    if fit_result["success"]:
        x_fine = np.linspace(x_ms.min(), x_ms.max(), 500)
        y_fine = fitter.evaluate(x_fine, fit_result["params"])
        ax2.plot(x_fine, y_fine, "r-", lw=2, label=f"{fit_result['method']}")

        pd = fitter.param_dict(fit_result)
        info = (f"f = {pd['f']['value']:.3f} ± {pd['f']['error']:.3f} kHz\n"
                f"τ = {pd['tau']['value']:.2f} ± {pd['tau']['error']:.2f} ms\n"
                f"R² = {fit_result['r_squared']:.4f}")
        vis, vis_err = fitter.visibility(fit_result)
        info += f"\nV = {vis:.3f} ± {vis_err:.3f}"
        ax2.text(0.02, 0.98, info, transform=ax2.transAxes, fontsize=9,
                 verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))

    ax2.set_xlabel("Free evolution time (ms)")
    ax2.set_ylabel("Retention probability")
    ax2.set_title(f"{title} - Retention")
    ax2.legend()

    plt.tight_layout()
    return fig
