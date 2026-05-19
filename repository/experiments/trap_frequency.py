"""Trap frequency measurement using the French method (parametric excitation).

Modulates the trap depth at varying frequencies and measures atom survival
to find the trap oscillation frequency.

Source: archive/Characterization/HS_code_20260331_Trap_freq_FrenchMethod.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom, imaging_pgc,
    spcm_image_dual, beams_off_for_imaging, reset_to_mot,
)
from utilities.write_results import write_results


class TrapFrequencyMeasurement(EnvExperiment):
    """Trap frequency measurement via parametric excitation (French method)."""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("n_measurements",
                              NumberValue(50, min=1, max=100000, step=1),
                              group="Experiment")
        self.setattr_argument("modulation_freq_min_kHz",
                              NumberValue(10.0, min=0.1, max=1000.0, ndecimals=3),
                              group="Modulation Scan")
        self.setattr_argument("modulation_freq_max_kHz",
                              NumberValue(200.0, min=0.1, max=1000.0, ndecimals=3),
                              group="Modulation Scan")
        self.setattr_argument("modulation_freq_step_kHz",
                              NumberValue(5.0, min=0.01, max=100.0, ndecimals=3),
                              group="Modulation Scan")
        self.setattr_argument("modulation_depth",
                              NumberValue(0.05, min=0.0, max=0.5, ndecimals=3),
                              group="Modulation")
        self.setattr_argument("modulation_duration_ms",
                              NumberValue(10.0, min=0.1, max=1000.0, ndecimals=1),
                              group="Modulation")

        # Rebalancer created and configured by BaseExperiment.build()

    def prepare(self):
        self.base.prepare()
        v, s = self.modulation_freq_min_kHz, self.modulation_freq_step_kHz
        n = int(round((self.modulation_freq_max_kHz - v) / s)) + 1
        self.freq_scan_kHz = [v + i * s for i in range(n)]

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)
        # Parametric excitation scan kernel to be completed from
        # archive/Characterization/HS_code_20260331_Trap_freq_FrenchMethod.py
        # Framework: for each modulation frequency, load atom, apply
        # trap amplitude modulation at that frequency, check survival.
        pass
