"""Trap frequency analysis using the French method.

Adapted from Old_py_analysis_codes/Trap_freq_scan_French_Method.py.
Analyzes recapture probability vs oscillation time to extract trap frequency.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from fitting.trap_oscillation import TrapOscillationFit
from fitting.fit_base import binomial_error


def analyze_trap_frequency(filepath, threshold=15, dt1_us=10.0, dt2_us=60.0,
                           f_min_kHz=10.0, f_max_kHz=300.0, w0_um=None):
    """Trap frequency analysis pipeline.

    Parameters
    ----------
    filepath : str
        Path to CSV or HDF5 file.
    threshold : float
        Photon count threshold for atom detection.
    dt1_us, dt2_us : float
        First and second release durations in microseconds.
    f_min_kHz, f_max_kHz : float
        Frequency search range for FFT initial guess.
    w0_um : float or None
        Beam waist in micrometers for trap depth calculation.

    Returns
    -------
    dict with 'data', 'fit_result', 'fitter', 'trap_depth_mK' (if w0_um set).
    """
    from analysis.data_loader import load, load_h5, load_csv

    if filepath.endswith(".h5"):
        result = load_h5(filepath)
        shot1 = np.asarray(result["data"].get("shot1", []), dtype=float)
        shot2 = np.asarray(result["data"].get("shot2", []), dtype=float)
        delay = np.asarray(result["data"].get("delay", []), dtype=float)
    else:
        df = load_csv(filepath)
        shot1 = df["shot1"].values.astype(float)
        shot2 = df["shot2"].values.astype(float)
        delay = df["delay"].values.astype(float)

    # Classify
    atom1 = shot1 >= threshold
    atom2 = shot2 >= threshold

    # Group by delay
    unique_delays = np.unique(delay)
    rows = []
    for d in unique_delays:
        mask = delay == d
        n_total = np.sum(mask)
        n_atom1 = np.sum(atom1[mask])
        n_recaptured = np.sum(atom1[mask] & atom2[mask])

        load_frac = n_atom1 / n_total if n_total > 0 else 0.0
        recap_prob = n_recaptured / n_atom1 if n_atom1 > 0 else np.nan

        rows.append({
            "delay_us": d,
            "n_total": n_total,
            "n_atom1": n_atom1,
            "n_recaptured": n_recaptured,
            "load_frac": load_frac,
            "recap_prob": recap_prob,
            "recap_err": binomial_error(recap_prob, n_atom1) if n_atom1 > 0 else np.nan,
        })

    # Filter valid points
    valid = [r for r in rows if np.isfinite(r["recap_prob"]) and r["n_atom1"] >= 1]
    x = np.array([r["delay_us"] for r in valid])
    y = np.array([r["recap_prob"] for r in valid])
    yerr = np.array([r["recap_err"] for r in valid])

    # Fit
    fitter = TrapOscillationFit()
    fit_result = fitter.fit_lsq(x, y, yerr=yerr)

    output = {
        "rows": rows,
        "valid": valid,
        "fit_result": fit_result,
        "fitter": fitter,
    }

    if w0_um is not None and fit_result["success"]:
        depth, depth_err = fitter.trap_depth_mK(fit_result, w0_um)
        output["trap_depth_mK"] = depth
        output["trap_depth_mK_err"] = depth_err

    return output


def plot_trap_frequency(result, title="Trap Frequency (French Method)"):
    """Plot recapture probability and loading fraction."""
    rows = result["rows"]
    valid = result["valid"]
    fit_result = result["fit_result"]
    fitter = result["fitter"]

    delays_all = [r["delay_us"] for r in rows]
    load_fracs = [r["load_frac"] for r in rows]

    x = np.array([r["delay_us"] for r in valid])
    y = np.array([r["recap_prob"] for r in valid])
    yerr = np.array([r["recap_err"] for r in valid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Recapture probability
    ax1.errorbar(x, y, yerr=yerr, fmt="ko", capsize=3, label="Data")

    if fit_result["success"]:
        x_fine = np.linspace(x.min(), x.max(), 1000)
        y_fine = fitter.evaluate(x_fine, fit_result["params"])
        ax1.plot(x_fine, y_fine, "r-", lw=2, label="Fit")

        f_r, f_r_err = fitter.trap_frequency_kHz(fit_result)
        tau = fit_result["params"][3]
        info = f"f_r = {f_r:.1f} ± {f_r_err:.1f} kHz\nτ_decay = {tau:.1f} μs"
        if "trap_depth_mK" in result:
            info += f"\nU₀ = {result['trap_depth_mK']:.2f} ± {result['trap_depth_mK_err']:.2f} mK"
        ax1.text(0.02, 0.98, info, transform=ax1.transAxes, fontsize=9,
                 verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))

    ax1.set_xlabel("Oscillation time τ (μs)")
    ax1.set_ylabel("Recapture probability")
    ax1.set_title(title)
    ax1.legend()

    # Loading fraction (sanity check)
    ax2.plot(delays_all, load_fracs, "bo-", markersize=4)
    ax2.set_xlabel("Oscillation time τ (μs)")
    ax2.set_ylabel("Loading fraction")
    ax2.set_title("Loading fraction (sanity check)")

    plt.tight_layout()
    return fig
