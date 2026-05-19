"""Microwave scan analysis: Rabi, Ramsey, and Resonance scans.

Adapted from Old_py_analysis_codes/Microwave_var_scan_analysis_v5_hybrid_MLE.py.
Supports both new HDF5 format and legacy CSV.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from analysis.data_loader import load, load_h5, load_csv
from analysis.threshold import find_threshold_double_gaussian
from analysis.statistics import classify_events, group_by_scan, binomial_error
from fitting.damped_cosine import DampedCosineFit, RabiFit, RamseyFit
from fitting.gaussian_dip import GaussianDipFit


def analyze_microwave_scan(filepath, scan_type="rabi", threshold=None,
                           force_threshold=False, manual_threshold=12.0,
                           bin_count=30):
    """Full microwave scan analysis pipeline.

    Parameters
    ----------
    filepath : str
        Path to HDF5 or CSV file.
    scan_type : str
        One of 'rabi', 'ramsey', 'resonance'.
    threshold : float or None
        Atom detection threshold. If None, auto-detected.
    force_threshold : bool
        If True, use manual_threshold instead of auto-detection.
    manual_threshold : float
        Manual threshold value.
    bin_count : int
        Histogram bins for threshold detection.

    Returns
    -------
    dict with 'grouped' (DataFrame), 'fit_result', 'threshold', 'fitter'.
    """
    # Load data
    if filepath.endswith(".h5"):
        result = load_h5(filepath)
        shot1 = result["data"].get("counts_signal", np.array([]))
        shot2 = result["data"].get("counts_background", np.array([]))
        bg = np.zeros_like(shot1)
    else:
        df = load_csv(filepath)
        shot1 = df.iloc[:, 1].values.astype(float)
        shot2 = df.iloc[:, 2].values.astype(float)
        bg = df.iloc[:, 3].values.astype(float) if df.shape[1] > 3 else np.zeros_like(shot1)

    # Threshold
    if force_threshold:
        thresh = manual_threshold
    elif threshold is not None:
        thresh = threshold
    else:
        shot1_net = shot1 - bg
        thresh, _ = find_threshold_double_gaussian(shot1_net, n_bins=bin_count)

    # Classify
    events = classify_events(shot1, shot2, background=bg, threshold=thresh)

    # Build scan axis from filename or HDF5 metadata
    # (User should provide scan_values or they come from the data)

    # Select fitter
    if scan_type == "ramsey":
        fitter = RamseyFit()
    elif scan_type == "resonance":
        fitter = GaussianDipFit()
    else:
        fitter = RabiFit()

    return {
        "events": events,
        "threshold": thresh,
        "fitter": fitter,
    }


def plot_microwave_result(grouped, fit_result, fitter, scan_type="rabi",
                          title=None):
    """Plot microwave scan results with fit overlay.

    Parameters
    ----------
    grouped : DataFrame
        Output of group_by_scan.
    fit_result : dict
        Output of fitter.fit_hybrid.
    fitter : HybridFitter
        The fitter instance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = grouped["x"].values
    # Loading rate
    ax1.errorbar(x, grouped["Load_Rate"], yerr=grouped["Load_Err"],
                 fmt="o-", capsize=3, label="Loading rate")
    ax1.set_xlabel("Scan parameter")
    ax1.set_ylabel("Loading rate")
    ax1.set_title("Loading rate" if title is None else f"{title} - Loading")
    ax1.legend()

    # Retention with fit
    ax2.errorbar(x, grouped["Ret_P"], yerr=grouped["Ret_Err"],
                 fmt="o", capsize=3, label="Data")

    if fit_result["success"]:
        x_fine = np.linspace(x.min(), x.max(), 500)
        y_fine = fitter.evaluate(x_fine, fit_result["params"])
        ax2.plot(x_fine, y_fine, "r-", lw=2, label=f"Fit ({fit_result['method']})")

        # Info box
        info = f"R² = {fit_result['r_squared']:.4f}"
        if scan_type in ("rabi", "ramsey"):
            pd = fitter.param_dict(fit_result)
            info += f"\nf = {pd['f']['value']:.4f} ± {pd['f']['error']:.4f}"
            info += f"\nτ = {pd['tau']['value']:.2f} ± {pd['tau']['error']:.2f}"
        ax2.text(0.02, 0.98, info, transform=ax2.transAxes, fontsize=9,
                 verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))

    ax2.set_xlabel("Scan parameter")
    ax2.set_ylabel("Retention probability")
    ax2.set_title("Retention" if title is None else f"{title} - Retention")
    ax2.legend()

    plt.tight_layout()
    return fig


# =========================================================================
#  POST-RUN FITTING (called from microwave_scan.py _analyze_and_update)
# =========================================================================

def fit_scan_result(x, retention, mode_type, mode_name):
    """Fit MW scan data and extract physics values.

    Parameters
    ----------
    x : list or array
        Scan axis values (Hz for frequency, mu for time/Ramsey).
    retention : list or array
        Retention probability per scan point.
    mode_type : str
        "frequency", "time", or "ramsey".
    mode_name : str
        e.g. "Freq_00", "Time_11", "Ramsey_2ph".

    Returns
    -------
    dict with 'success', 'params', 'errors', 'extracted_value', 'summary'.
    """
    x = np.asarray(x, dtype=float)
    retention = np.asarray(retention, dtype=float)

    if len(x) < 3:
        return {"success": False, "summary": "Not enough data points"}

    # Average retention by unique scan value (handles multiple loops)
    # Filter out -1.0 sentinel (unloaded shots) — retention = P(ret|loaded)
    unique_x = np.unique(x)
    avg_ret = np.zeros(len(unique_x))
    n_per = np.zeros(len(unique_x))
    for i, xv in enumerate(unique_x):
        vals = retention[x == xv]
        valid = vals[vals >= 0]  # exclude -1.0 sentinel
        n_per[i] = len(valid) if len(valid) > 0 else np.sum(x == xv)
        avg_ret[i] = valid.mean() if len(valid) > 0 else 0.0
    err_ret = np.array([np.sqrt(max(r * (1 - r), 0) / max(n, 1)) for r, n in zip(avg_ret, n_per)])

    extracted = None
    summary_lines = [f"Mode: {mode_name}"]

    if mode_type == "frequency":
        fitter = GaussianDipFit()
        # Convert Hz to MHz for fitting
        x_mhz = unique_x / 1e6
        result = fitter.fit_lsq(x_mhz, avg_ret, yerr=err_ret)

        if result["success"]:
            center_mhz, center_err = fitter.center(result)
            fwhm_mhz, fwhm_err = fitter.fwhm(result)
            extracted = center_mhz * 1e6  # back to Hz for dataset storage
            summary_lines.append(f"Resonance: {center_mhz:.6f} +/- {center_err:.6f} MHz")
            summary_lines.append(f"FWHM: {fwhm_mhz:.6f} +/- {fwhm_err:.6f} MHz")
            summary_lines.append(f"R^2: {result.get('r_squared', 'N/A')}")

    elif mode_type == "time":
        fitter = RabiFit()
        result = fitter.fit_lsq(unique_x, avg_ret, yerr=err_ret)

        if result["success"]:
            pi_time = fitter.pi_time(result)
            vis, vis_err = fitter.visibility(result)
            pd = fitter.param_dict(result)
            extracted = pi_time  # in mu, ready for dataset
            summary_lines.append(f"Pi time: {pi_time:.0f} mu")
            summary_lines.append(f"Frequency: {pd['f']['value']:.6f} +/- {pd['f']['error']:.6f}")
            summary_lines.append(f"Visibility: {vis:.3f} +/- {vis_err:.3f}")

    elif mode_type == "ramsey":
        fitter = RamseyFit()
        # Convert mu to ms for fitting
        x_ms = unique_x / 1e6
        result = fitter.fit_lsq(x_ms, avg_ret, yerr=err_ret)

        if result["success"]:
            pd = fitter.param_dict(result)
            vis, vis_err = fitter.visibility(result)
            summary_lines.append(f"Frequency: {pd['f']['value']:.4f} +/- {pd['f']['error']:.4f} kHz")
            summary_lines.append(f"T2: {pd['tau']['value']:.2f} +/- {pd['tau']['error']:.2f} ms")
            summary_lines.append(f"Visibility: {vis:.3f} +/- {vis_err:.3f}")
            # Ramsey doesn't auto-update a calibration dataset
    else:
        return {"success": False, "summary": f"Unknown mode_type: {mode_type}"}

    if not result["success"]:
        summary_lines.append("FIT FAILED")
        print("\n".join(summary_lines))
        return {"success": False, "summary": "\n".join(summary_lines)}

    summary = "\n".join(summary_lines)
    print(summary)

    return {
        "success": True,
        "params": result["params"],
        "errors": result["errors"],
        "extracted_value": extracted,
        "summary": summary,
        "fitter": fitter,
        "fit_result": result,
    }
