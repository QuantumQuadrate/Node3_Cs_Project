"""Damped cosine fit models for Rabi and Ramsey oscillations.

Model: y = A * cos(2*pi*f*x + phi) * exp(-(x/tau)^n) + y0

Extracted from Microwave_var_scan_analysis_v5_hybrid_MLE.py and Ramsey_Analysis.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fitting.fit_base import HybridFitter


class DampedCosineFit(HybridFitter):
    """Damped cosine: A*cos(2*pi*f*x + phi)*exp(-(x/tau)^n) + y0

    Parameters: A, f, phi, tau, y0, n
    """

    param_names = ["A", "f", "phi", "tau", "y0", "n"]

    @staticmethod
    def model(x, A, f, phi, tau, y0, n):
        return A * np.cos(2 * np.pi * f * x + phi) * np.exp(-(x / tau) ** n) + y0

    def initial_guess(self, x, y, yerr=None):
        A0 = (np.max(y) - np.min(y)) / 2.0
        y0_0 = np.mean(y)
        x_range = x[-1] - x[0] if len(x) > 1 else 1.0

        # FFT for frequency estimate
        dx = np.mean(np.diff(x))
        f0 = 1.0 / x_range  # fallback
        if dx > 0 and len(x) > 4:
            fft_vals = np.abs(np.fft.rfft(y - y0_0))
            freqs = np.fft.rfftfreq(len(x), d=dx)
            if len(fft_vals) > 1:
                idx = np.argmax(fft_vals[1:]) + 1
                f0 = freqs[idx] if freqs[idx] > 0 else f0

        # Estimate decay from envelope: find where oscillation amplitude drops
        # Use x_range/3 as reasonable tau guess (visible decay within scan)
        tau0 = x_range * 10.0  # start assuming minimal decay

        p0 = [A0, f0, 0.0, tau0, y0_0, 2.0]
        lo = [0.0, f0 / 20.0, -2 * np.pi, x_range * 1e-1, 0.0, 1.0]
        hi = [1.0, f0 * 20.0, 2 * np.pi, x_range * 1e6, 1.0, 2.0]
        return p0, lo, hi

    def visibility(self, result):
        """Visibility = 2*A."""
        A = result["params"][0]
        A_err = result["errors"][0]
        return 2.0 * A, 2.0 * A_err

    def pi_time(self, result):
        """Pi time = 1/(2*f)."""
        f = result["params"][1]
        return 0.5 / f if f > 0 else np.inf


class RamseyFit(HybridFitter):
    """Ramsey fringe fit with free decay exponent.

    Model: A * cos(2*pi*f*x + phi) * exp(-(x/tau)^n) + y0

    Expects x in milliseconds. Uses segment-aware FFT for frequency
    estimation on multi-segment Ramsey data (detects gaps automatically).
    """

    param_names = ["A", "f", "phi", "tau", "y0", "n"]

    @staticmethod
    def model(x, A, f, phi, tau, y0, n):
        return A * np.cos(2 * np.pi * f * x + phi) * np.exp(-(x / tau) ** n) + y0

    def initial_guess(self, x, y, yerr=None):
        x_range = x[-1] - x[0] if len(x) > 1 else 1.0
        A0 = (np.max(y) - np.min(y)) / 2.0
        y0_0 = np.mean(y)

        # Detect first segment (before first large gap) for clean FFT
        seg1_end = len(x)
        if len(x) > 4:
            diffs = np.diff(x)
            median_step = np.median(diffs)
            gap_idx = np.where(diffs > 3 * median_step)[0]
            if len(gap_idx) > 0:
                seg1_end = gap_idx[0] + 1

        # FFT on first segment only (no gap contamination)
        x_seg = x[:seg1_end]
        y_seg = y[:seg1_end]
        f0 = 1.0 / x_range  # fallback

        dx = np.mean(np.diff(x_seg)) if len(x_seg) > 1 else 1.0
        if dx > 0 and len(x_seg) > 3:
            fft_vals = np.abs(np.fft.rfft(y_seg - np.mean(y_seg)))
            freqs = np.fft.rfftfreq(len(x_seg), d=dx)
            if len(fft_vals) > 1:
                idx = np.argmax(fft_vals[1:]) + 1
                if freqs[idx] > 0:
                    f0 = freqs[idx]

        tau0 = x_range * 0.6
        n0 = 2.0

        p0 = [A0, f0, 0.0, tau0, y0_0, n0]
        lo = [0.0, f0 / 20.0, -2 * np.pi, x_range * 0.1, 0.0, 1.0]
        hi = [1.0, f0 * 20.0, 2 * np.pi, x_range * 1e6, 1.0, 2.0]
        return p0, lo, hi

    def visibility(self, result):
        """Visibility = 2*A."""
        A = result["params"][0]
        A_err = result["errors"][0]
        return 2.0 * A, 2.0 * A_err

    def frequency_khz(self, result):
        """Oscillation frequency in kHz (when x was in ms)."""
        return result["params"][1], result["errors"][1]

    def decay_time_ms(self, result):
        """Decay time in ms (when x was in ms)."""
        return result["params"][3], result["errors"][3]

    def decay_exponent(self, result):
        """Decay exponent n."""
        return result["params"][5], result["errors"][5]


class RabiFit(DampedCosineFit):
    """Rabi-specific: fixed decay exponent n=1 (exponential decay)."""

    param_names = ["A", "f", "phi", "tau", "y0"]

    @staticmethod
    def model(x, A, f, phi, tau, y0):
        return A * np.cos(2 * np.pi * f * x + phi) * np.exp(-x / tau) + y0

    def initial_guess(self, x, y, yerr=None):
        p0, lo, hi = super().initial_guess(x, y, yerr)
        return p0[:5], lo[:5], hi[:5]

    def fit_lsq(self, x, y, yerr=None, p0=None, bounds=None):
        """Phase-robust LSQ.

        Rabi data can start at any phase; a single fixed phi0 lets curve_fit
        settle into a sub-harmonic local minimum (observed: nearly-identical
        retention curves fitting to f vs ~0.7f, R^2 ~ 0.01). Try several
        starting phases and keep the highest-R^2 result. phi0 = 0 is among the
        starts, so this can only match or beat the single fixed-phase fit.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if p0 is None or bounds is None:
            p0a, lo, hi = self.initial_guess(x, y, yerr)
            if p0 is None:
                p0 = p0a
            if bounds is None:
                bounds = (lo, hi)
        best = None
        for phi0 in np.linspace(-np.pi, np.pi, 9):
            trial = list(p0)
            trial[2] = float(phi0)
            res = super().fit_lsq(x, y, yerr=yerr, p0=trial, bounds=bounds)
            if not res["success"]:
                continue
            yp = self.evaluate(x, res["params"])
            ss = float(np.sum((y - yp) ** 2))
            st = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss / st if st > 0 else 0.0
            if best is None or r2 > best[0]:
                best = (r2, res)
        if best is not None:
            return best[1]
        return super().fit_lsq(x, y, yerr=yerr, p0=p0, bounds=bounds)
