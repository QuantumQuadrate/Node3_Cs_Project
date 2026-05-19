"""Gaussian dip/peak fit model for microwave resonance spectroscopy.

Model: y = A * exp(-(x - x0)^2 / (2*sigma^2)) + y0

Extracted from Microwave_var_scan_analysis_v5_hybrid_MLE.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fitting.fit_base import HybridFitter


class GaussianDipFit(HybridFitter):
    """Gaussian: A * exp(-(x-x0)^2 / (2*sigma^2)) + y0

    Parameters: A, x0, sigma, y0

    Direction:
      - "auto" (default): detect dip vs peak from the data.
      - "dip":  force fit to a dip (A < 0, baseline = upper plateau).
                For Cs MW scan: use for Freq_00 and Freq_01 (final state F=4
                gets blown away, retention dips at resonance).
      - "peak": force fit to a peak (A > 0, baseline = lower plateau).
                For Cs MW scan: use for Freq_11 and Freq_2ph (atom returns
                to F=3 at resonance, retention rises).
    """

    param_names = ["A", "x0", "sigma", "y0"]

    def __init__(self, direction="auto"):
        if direction not in ("auto", "dip", "peak"):
            raise ValueError(
                f"direction must be 'auto', 'dip', or 'peak'; got {direction!r}"
            )
        self.direction = direction

    @staticmethod
    def model(x, A, x0, sigma, y0):
        return A * np.exp(-(x - x0) ** 2 / (2.0 * sigma ** 2)) + y0

    def initial_guess(self, x, y, yerr=None):
        """Robust auto-guess for Gaussian dip or peak.

        Improvements over a median-based guess that hurt narrow features
        like the 2ph resonance (~1 kHz FWHM in a ~10 kHz scan):
          - Baseline (y0) from the top/bottom quartile mean instead of
            the median — unbiased even when the dip is sharper than
            half the scan width.
          - Amplitude (A) measured from the chosen baseline to the
            extremum, not from the median (which understates dip depth
            for narrow features).
          - Sigma estimated from half-maximum crossings in the data when
            those crossings are clearly visible, falling back to
            x_range/10 otherwise.
          - Wider parameter bounds (A in [-1.5, 1.5], y0 in [-0.1, 1.5])
            so a noisy baseline near 1 doesn't bump into a hard wall.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_range = x[-1] - x[0] if len(x) > 1 else 1.0

        # Top/bottom quartile means -- robust to a narrow feature
        y_sorted = np.sort(y)
        q = max(2, len(y) // 4)
        y_top = float(np.mean(y_sorted[-q:]))
        y_bot = float(np.mean(y_sorted[:q]))
        y_med = float(np.median(y))

        # Pick dip vs peak. If direction is "auto", decide from the data
        # (whichever extreme is further from the median); otherwise honor
        # the caller's hint -- the physics-correct choice is more reliable
        # than data-driven detection on noisy scans.
        if self.direction == "dip":
            is_dip = True
        elif self.direction == "peak":
            is_dip = False
        else:
            is_dip = abs(y_bot - y_med) >= abs(y_top - y_med)

        if is_dip:
            # DIP: baseline is the upper plateau
            y0_0 = y_top
            A0 = y_bot - y_top                    # negative
            idx_extr = int(np.argmin(y))
        else:
            # PEAK: baseline is the lower plateau
            y0_0 = y_bot
            A0 = y_top - y_bot                    # positive
            idx_extr = int(np.argmax(y))
        x0_0 = float(x[idx_extr])

        # Sigma from half-max crossings
        half_level = y0_0 + A0 / 2.0
        if A0 < 0:
            in_feature = y < half_level
        else:
            in_feature = y > half_level
        if int(np.sum(in_feature)) >= 2:
            x_in = x[in_feature]
            fwhm_est = float(np.max(x_in) - np.min(x_in))
            sigma0 = fwhm_est / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        else:
            sigma0 = x_range / 10.0
        # Sanity clamps -- never zero, never larger than the scan
        sigma0 = max(sigma0, x_range / 1000.0)
        sigma0 = min(sigma0, x_range)

        # Bounds: when direction is forced we narrow A and y0 to the
        # correct half-plane -- protects against the fit slipping into
        # the wrong sign on a borderline shape.
        if self.direction == "dip":
            A_lo, A_hi = -1.5, 0.0
            y0_lo, y0_hi = 0.3, 1.5
        elif self.direction == "peak":
            A_lo, A_hi = 0.0, 1.5
            y0_lo, y0_hi = -0.1, 0.7
        else:
            A_lo, A_hi = -1.5, 1.5
            y0_lo, y0_hi = -0.1, 1.5

        p0 = [A0, x0_0, sigma0, y0_0]
        lo = [A_lo, x[0],  x_range / 1000.0, y0_lo]
        hi = [A_hi, x[-1], x_range * 2.0,     y0_hi]
        return p0, lo, hi

    def center(self, result):
        """Return resonance center frequency."""
        return result["params"][1], result["errors"][1]

    def fwhm(self, result):
        """Full width at half maximum = 2*sqrt(2*ln2)*sigma."""
        sigma = result["params"][2]
        sigma_err = result["errors"][2]
        factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        return factor * sigma, factor * sigma_err
