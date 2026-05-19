"""Unit conversion helpers for ARTIQ experiments."""

import numpy as np


def dB_to_V(dB):
    """Convert RF power in dBm to DDS amplitude voltage (into 50 ohm)."""
    return np.sqrt(2.0 * 50.0 * 10.0 ** ((dB - 30.0) / 10.0))


def sampler_mu_to_V(mu_val):
    """Convert Sampler ADC raw machine units to voltage.

    Sampler range is +/- 10V mapped to 16-bit signed integer.
    """
    return mu_val * (20.0 / 65536.0)
