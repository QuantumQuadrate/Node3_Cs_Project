"""Cos-4 parity oscillation model for waveplate angle scans.

Model: P(theta) = C * (1 + V * cos(f * theta_rad - phi))

where theta is HWP angle in degrees, f is typically 4 for HWP scans.

Extracted from Parity_analysis_w_QWP_scan.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fitting.fit_base import HybridFitter


class ParityCos4Fit(HybridFitter):
    """Parity cos-4 model: C * (1 + V * cos(f*theta - phi))

    Parameters: C (offset), V (visibility), phi (phase, degrees), f (frequency)
    """

    param_names = ["C", "V", "phi_deg", "f"]

    @staticmethod
    def model(x_deg, C, V, phi_deg, f):
        theta = np.deg2rad(x_deg)
        phi = np.deg2rad(phi_deg)
        return C * (1.0 + V * np.cos(f * theta - phi))

    def initial_guess(self, x, y, yerr=None):
        C0 = np.mean(y)
        V0 = (np.max(y) - np.min(y)) / (2.0 * C0) if C0 > 0 else 0.5
        V0 = min(V0, 0.99)
        p0 = [C0, V0, 0.0, 4.0]
        lo = [0.001, 0.0, -360.0, 3.0]
        hi = [0.999, 1.0, 360.0, 10.0]
        return p0, lo, hi

    def fit_with_grid_search(self, x_deg, k, n, n_phase_trials=72, fixed_freq=None):
        """MLE fit with grid search over phase for robust convergence.

        Parameters
        ----------
        x_deg : array
            HWP angles in degrees.
        k : array
            Success counts per angle bin.
        n : array
            Total trials per angle bin.
        n_phase_trials : int
            Number of phase values to try in grid search.
        fixed_freq : float or None
            If set, frequency is fixed to this value.

        Returns best fit result dict.
        """
        y = k / np.maximum(n, 1)
        C0 = np.mean(y)
        V0 = (np.max(y) - np.min(y)) / (2.0 * max(C0, 0.01))
        V0 = np.clip(V0, 0.01, 0.99)

        best_result = None
        best_nll = np.inf
        phases = np.linspace(0, 360, n_phase_trials, endpoint=False)

        for phi_trial in phases:
            freq = fixed_freq if fixed_freq is not None else 4.0
            p0 = [C0, V0, phi_trial, freq]

            if fixed_freq is not None:
                bounds = [(0.001, 0.999), (0.0, 1.0),
                          (-360.0, 360.0), (freq - 0.001, freq + 0.001)]
            else:
                bounds = [(0.001, 0.999), (0.0, 1.0),
                          (-360.0, 360.0), (3.0, 10.0)]

            result = self.fit_mle_binomial(x_deg, k, n, p0=p0, bounds=bounds)
            if result["success"] and result["nll"] < best_nll:
                best_nll = result["nll"]
                best_result = result

        if best_result is None:
            best_result = {"params": np.array([C0, V0, 0.0, 4.0]),
                           "errors": np.full(4, np.inf),
                           "nll": np.inf, "success": False}

        best_result["method"] = "Grid MLE"
        return best_result

    def optimal_hwp_angle(self, result_green, result_purple, x_range=(0, 360)):
        """Find HWP angle where |green - purple| is maximized."""
        x = np.linspace(x_range[0], x_range[1], 1000)
        y_green = self.model(x, *result_green["params"])
        y_purple = self.model(x, *result_purple["params"])
        idx = np.argmax(np.abs(y_green - y_purple))
        return x[idx]
