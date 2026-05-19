"""Parity oscillation analysis for Z and X basis measurements.

Adapted from Old_py_analysis_codes/Parity_analysis_w_QWP_scan.py.
Classifies single-photon events and fits cos-4 parity oscillations
using binomial MLE with grid search.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from fitting.parity_cos4 import ParityCos4Fit
from fitting.fit_base import binomial_error


def classify_parity_events(spcm1, spcm2, counts, threshold=15):
    """Classify parity measurement events.

    Parameters
    ----------
    spcm1, spcm2 : array-like
        Single-photon detector values. -2 = no detection, >0 = timestamp/count.
    counts : array-like
        Atom detection counts.
    threshold : float
        Atom detection threshold.

    Returns
    -------
    dict with boolean arrays: single_click, atom, spcm1_fired, spcm2_fired
    """
    spcm1 = np.asarray(spcm1, dtype=float)
    spcm2 = np.asarray(spcm2, dtype=float)
    counts = np.asarray(counts, dtype=float)

    spcm1_fired = spcm1 > 0
    spcm2_fired = spcm2 > 0
    single_click = spcm1_fired ^ spcm2_fired  # XOR: exactly one fires
    atom = counts >= threshold

    return {
        "single_click": single_click,
        "atom": atom,
        "spcm1_fired": spcm1_fired,
        "spcm2_fired": spcm2_fired,
    }


def analyze_parity_per_qwp(hwp_angles, events, hwp_bins=None):
    """Analyze parity oscillation for a single QWP value.

    Classifies into 4 categories (S0_L, S1_L, S0_E, S1_E) and
    computes green/purple parity combinations.

    Returns dict with per-HWP-bin counts and fitted results.
    """
    valid = events["single_click"]
    hwp = np.asarray(hwp_angles)[valid]
    spcm1 = events["spcm1_fired"][valid]
    spcm2 = events["spcm2_fired"][valid]
    atom = events["atom"][valid]

    if hwp_bins is None:
        hwp_bins = np.unique(hwp)

    rows = []
    for h in hwp_bins:
        mask = hwp == h
        n_total = np.sum(mask)
        if n_total == 0:
            continue

        s0_l = np.sum(spcm1[mask] & ~atom[mask])  # SPCM0 fires, atom lost
        s1_l = np.sum(spcm2[mask] & ~atom[mask])  # SPCM1 fires, atom lost
        s0_e = np.sum(spcm1[mask] & atom[mask])   # SPCM0 fires, atom exists
        s1_e = np.sum(spcm2[mask] & atom[mask])   # SPCM1 fires, atom exists

        k_green = s0_l + s1_e
        k_purple = s1_l + s0_e

        rows.append({
            "hwp": h, "n": n_total,
            "S0_L": s0_l, "S1_L": s1_l, "S0_E": s0_e, "S1_E": s1_e,
            "k_green": k_green, "k_purple": k_purple,
            "p_green": k_green / n_total,
            "p_purple": k_purple / n_total,
        })

    return rows


def fit_parity(rows, fixed_freq=4.0):
    """Fit parity oscillations (green and purple) using MLE grid search.

    Returns dict with 'green' and 'purple' fit results.
    """
    hwp = np.array([r["hwp"] for r in rows])
    n = np.array([r["n"] for r in rows])
    k_green = np.array([r["k_green"] for r in rows])
    k_purple = np.array([r["k_purple"] for r in rows])

    fitter = ParityCos4Fit()

    result_green = fitter.fit_with_grid_search(hwp, k_green, n,
                                                fixed_freq=fixed_freq)
    result_purple = fitter.fit_with_grid_search(hwp, k_purple, n,
                                                 fixed_freq=fixed_freq)

    theta_star = fitter.optimal_hwp_angle(result_green, result_purple)

    return {
        "green": result_green,
        "purple": result_purple,
        "theta_star": theta_star,
        "fitter": fitter,
    }


def plot_parity(rows, fit_results, qwp_label=""):
    """Plot parity oscillation data with fit curves."""
    hwp = np.array([r["hwp"] for r in rows])
    n = np.array([r["n"] for r in rows])
    p_green = np.array([r["p_green"] for r in rows])
    p_purple = np.array([r["p_purple"] for r in rows])
    err_green = binomial_error(p_green, n)
    err_purple = binomial_error(p_purple, n)

    fitter = fit_results["fitter"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Green parity
    ax = axes[0]
    ax.errorbar(hwp, p_green, yerr=err_green, fmt="go", capsize=3, label="Green (S0_L + S1_E)")
    ax.errorbar(hwp, p_purple, yerr=err_purple, fmt="ms", capsize=3, label="Purple (S1_L + S0_E)")

    x_fine = np.linspace(hwp.min(), hwp.max(), 500)
    if fit_results["green"]["success"]:
        ax.plot(x_fine, fitter.evaluate(x_fine, fit_results["green"]["params"]),
                "g-", lw=2, alpha=0.7)
    if fit_results["purple"]["success"]:
        ax.plot(x_fine, fitter.evaluate(x_fine, fit_results["purple"]["params"]),
                "m-", lw=2, alpha=0.7)

    ax.axvline(fit_results["theta_star"], color="gray", ls="--", alpha=0.5,
               label=f"θ* = {fit_results['theta_star']:.1f}°")
    ax.set_xlabel("HWP angle (deg)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Parity oscillation {qwp_label}")
    ax.legend(fontsize=8)

    # Visibility summary
    ax = axes[1]
    labels = ["Green", "Purple"]
    for i, (key, color) in enumerate(zip(["green", "purple"], ["green", "purple"])):
        r = fit_results[key]
        if r["success"]:
            pd = fitter.param_dict(r)
            ax.bar(i, pd["V"]["value"], yerr=pd["V"]["error"],
                   color=color, alpha=0.7, capsize=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Visibility")
    ax.set_title(f"Visibility {qwp_label}")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig
