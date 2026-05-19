"""Automatic threshold detection using double-Gaussian fitting.

Fits a histogram of photon counts to a sum of two Gaussians to find
the valley between "no atom" and "atom loaded" populations.

Extracted from Microwave_var_scan_analysis_v5_hybrid_MLE.py.
"""

import numpy as np
from scipy.optimize import curve_fit


def double_gaussian(x, a1, mu1, sig1, a2, mu2, sig2):
    """Sum of two Gaussians."""
    g1 = a1 * np.exp(-(x - mu1) ** 2 / (2 * sig1 ** 2))
    g2 = a2 * np.exp(-(x - mu2) ** 2 / (2 * sig2 ** 2))
    return g1 + g2


def find_threshold_double_gaussian(counts, n_bins=30):
    """Find the threshold between two populations in photon count data.

    Fits a double-Gaussian to the histogram and returns the valley
    between the two peaks as the threshold.

    Parameters
    ----------
    counts : array-like
        Photon count values (e.g., shot1 - background).
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    threshold : float
        Optimal threshold value.
    fit_params : dict
        Fit parameters (mu1, sigma1, mu2, sigma2, amplitudes).
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts)]

    hist, bin_edges = np.histogram(counts, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initial guess: two peaks at 25th and 75th percentile
    q25 = np.percentile(counts, 25)
    q75 = np.percentile(counts, 75)
    spread = np.std(counts)

    p0 = [
        np.max(hist) * 0.6, q25, spread * 0.3,
        np.max(hist) * 0.4, q75, spread * 0.3,
    ]
    bounds = (
        [0, np.min(counts), 0.1, 0, np.min(counts), 0.1],
        [np.max(hist) * 2, np.max(counts), spread * 2,
         np.max(hist) * 2, np.max(counts), spread * 2],
    )

    try:
        popt, _ = curve_fit(double_gaussian, bin_centers, hist, p0=p0,
                            bounds=bounds, maxfev=10000)
        a1, mu1, sig1, a2, mu2, sig2 = popt

        # Ensure mu1 < mu2
        if mu1 > mu2:
            a1, mu1, sig1, a2, mu2, sig2 = a2, mu2, sig2, a1, mu1, sig1

        # Threshold = point between peaks where Gaussians cross
        x_fine = np.linspace(mu1, mu2, 1000)
        g1 = a1 * np.exp(-(x_fine - mu1) ** 2 / (2 * sig1 ** 2))
        g2 = a2 * np.exp(-(x_fine - mu2) ** 2 / (2 * sig2 ** 2))
        idx = np.argmin(g1 + g2)
        threshold = x_fine[idx]

        fit_params = {
            "mu1": mu1, "sigma1": sig1, "amp1": a1,
            "mu2": mu2, "sigma2": sig2, "amp2": a2,
        }
        return threshold, fit_params

    except Exception:
        # Fallback: midpoint of range
        threshold = np.median(counts)
        return threshold, {}
