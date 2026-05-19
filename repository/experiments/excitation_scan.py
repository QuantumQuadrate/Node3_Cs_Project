"""Excitation rate measurement with variable trap-off duration.

Scans the trap-off time during excitation to characterize
photon emission rates and optimize excitation parameters.

Source: archive/Calibration/HS_code_20260225_ext_long_trap_off.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom, imaging_pgc,
    spcm_image_dual, beams_off_for_imaging,
    optical_pumping_full, excitation_pulse, reset_to_mot,
)
from utilities.write_results import write_results


class ExcitationScan(EnvExperiment):
    """Excitation_Scan"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("n_measurements",
                              NumberValue(100, min=1, max=600000, step=1),
                              group="Experiment")
        self.setattr_argument("max_excitation_cycles",
                              NumberValue(200, min=1, max=8000, step=1),
                              group="Excitation")
        self.setattr_argument("t_trap_off_min_mu",
                              NumberValue(10, min=0, max=8000, ndecimals=0),
                              group="Trap-off Scan")
        self.setattr_argument("t_trap_off_max_mu",
                              NumberValue(200, min=0, max=8000, ndecimals=0),
                              group="Trap-off Scan")
        self.setattr_argument("t_trap_off_step_mu",
                              NumberValue(10, min=1, max=1000, ndecimals=0),
                              group="Trap-off Scan")
        self.setattr_argument("t_spcm_collection_mu",
                              NumberValue(250, min=0, max=8000, ndecimals=0),
                              group="Excitation")
        self.setattr_argument("length_offset_mu",
                              NumberValue(30, min=0, max=8000, ndecimals=0),
                              group="Excitation")

        # Rebalancer created and configured by BaseExperiment.build()

    def prepare(self):
        self.base.prepare()
        v, s = self.t_trap_off_min_mu, self.t_trap_off_step_mu
        n = int(round((self.t_trap_off_max_mu - v) / s)) + 1
        self.trap_off_scan = [int(v + i * s) for i in range(n)]

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)
        # Excitation scan kernel to be completed from
        # archive/Calibration/HS_code_20260225_ext_long_trap_off.py
        # Framework: for each trap-off duration, load atom, run excitation
        # cycles, record photon detection rate.
        pass
