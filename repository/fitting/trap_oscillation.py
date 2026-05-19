"""Damped cosine model for trap frequency measurement (French method).

Model: P(tau) = offset + A * exp(-tau/tau_decay) * cos(2*pi*f_osc*tau + phi)

where f_osc = 2 * f_trap (the observed frequency is twice the trap frequency).

Includes FFT-based initial frequency estimation.

Extracted from Trap_freq_scan_French_Method.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fitting.fit_base import HybridFitter

# Cs-133 physical constants
M_CS = 133 * 1.66054e-27    # kg
K_B = 1.380649e-23          # J/K


class TrapOscillationFit(HybridFitter):
    """Trap oscillation: offset + A*exp(-tau/tau_decay)*cos(2*pi*f_osc*tau + phi)

    Parameters: offset, A, f_osc_kHz, tau_decay_us, phi
    f_trap = f_osc / 2
    """

    param_names = ["offset", "A", "f_osc_kHz", "tau_decay_us", "phi"]

    @staticmethod
    def model(x_us, offset, A, f_osc_kHz, tau_decay_us, phi):
        f = f_osc_kHz / 1000.0  # kHz to 1/us
        return offset + A * np.exp(-x_us / tau_decay_us) * np.cos(2 * np.pi * f * x_us + phi)

    def initial_guess(self, x, y, yerr=None, f_min_kHz=10.0, f_max_kHz=300.0):
        """Estimate initial parameters using FFT for frequency."""
        offset0 = np.mean(y)
        A0 = (np.max(y) - np.min(y)) / 2.0

        # FFT frequency estimation
        y_centered = y - offset0
        dx = np.mean(np.diff(x))
        if dx > 0 and len(x) > 4:
            fft_vals = np.abs(np.fft.rfft(y_centered))
            freqs_kHz = np.fft.rfftfreq(len(x), d=dx) * 1000.0  # us -> kHz

            mask = (freqs_kHz >= f_min_kHz) & (freqs_kHz <= f_max_kHz)
            if np.any(mask):
                masked_fft = fft_vals.copy()
                masked_fft[~mask] = 0
                f_osc_0 = freqs_kHz[np.argmax(masked_fft)]
            else:
                f_osc_0 = (f_min_kHz + f_max_kHz) / 2.0
        else:
            f_osc_0 = (f_min_kHz + f_max_kHz) / 2.0

        tau0 = np.max(x) / 2.0

        p0 = [offset0, A0, f_osc_0, tau0, 0.0]
        lo = [0.0, 0.0, f_min_kHz, 1.0, -np.pi]
        hi = [1.0, 0.5, f_max_kHz, np.max(x) * 5, np.pi]
        return p0, lo, hi

    def trap_frequency_kHz(self, result):
        """Extract trap frequency f_r = f_osc / 2 in kHz."""
        f_osc = result["params"][2]
        f_osc_err = result["errors"][2]
        return f_osc / 2.0, f_osc_err / 2.0

    def coherent_cycles(self, result):
        """Number of coherent oscillation cycles before 1/e decay."""
        f_osc_kHz = result["params"][2]
        tau_us = result["params"][3]
        return tau_us * f_osc_kHz / 1000.0

    def trap_depth_mK(self, result, w0_um):
        """Calculate trap depth in mK from trap frequency and beam waist.

        U0 = M_cs * (2*pi*f_r)^2 * w0^2 / 4
        """
        f_r_kHz, f_r_err = self.trap_frequency_kHz(result)
        f_r_Hz = f_r_kHz * 1e3
        w0_m = w0_um * 1e-6

        U0_J = M_CS * (2 * np.pi * f_r_Hz) ** 2 * w0_m ** 2 / 4.0
        U0_mK = U0_J / (K_B * 1e-3)

        # Error propagation (f_r dominates)
        U0_err = U0_mK * 2 * f_r_err / f_r_kHz
        return U0_mK, U0_err
