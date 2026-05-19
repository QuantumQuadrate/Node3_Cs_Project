"""General 1D/2D parameter sweep framework.

Scan any ExperimentVariable over a user-defined range while running a
configurable experiment function. Supports 1D scans and nested 2D scans.

Modeled on qn_artiq_routines GeneralVariableScan pattern.
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


class GeneralVariableScan(EnvExperiment):
    """General_Varaible_Scan"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("scan_variable_name",
                              StringValue("nominal_trapdepth"),
                              group="Scan Config")
        self.setattr_argument("scan_min",
                              NumberValue(0.1, min=-1e6, max=1e6, ndecimals=6),
                              group="Scan Config")
        self.setattr_argument("scan_max",
                              NumberValue(0.3, min=-1e6, max=1e6, ndecimals=6),
                              group="Scan Config")
        self.setattr_argument("scan_steps",
                              NumberValue(10, min=1, max=10000, step=1, ndecimals=0, type="int"),
                              group="Scan Config")
        self.setattr_argument("measurements_per_point",
                              NumberValue(50, min=1, max=100000, step=1, ndecimals=0, type="int"),
                              group="Scan Config")

        # 2D scan (optional)
        self.setattr_argument("enable_2D_scan", BooleanValue(default=False),
                              group="2D Scan")
        self.setattr_argument("scan_variable_2_name",
                              StringValue("shim_X_idle"),
                              group="2D Scan")
        self.setattr_argument("scan_2_min",
                              NumberValue(0.0, min=-1e6, max=1e6, ndecimals=6),
                              group="2D Scan")
        self.setattr_argument("scan_2_max",
                              NumberValue(0.5, min=-1e6, max=1e6, ndecimals=6),
                              group="2D Scan")
        self.setattr_argument("scan_2_steps",
                              NumberValue(5, min=1, max=1000, step=1, ndecimals=0, type="int"),
                              group="2D Scan")

        # Rebalancer created and configured by BaseExperiment.build()

    def prepare(self):
        self.base.prepare()
        step = (self.scan_max - self.scan_min) / max(1, self.scan_steps)
        self.scan_sequence = [self.scan_min + i * step
                              for i in range(self.scan_steps + 1)]
        self.set_dataset("scan_variable", self.scan_variable_name, broadcast=True)
        self.set_dataset("scan_sequence", self.scan_sequence, broadcast=True)

        if self.enable_2D_scan:
            step2 = (self.scan_2_max - self.scan_2_min) / max(1, self.scan_2_steps)
            self.scan_2_sequence = [self.scan_2_min + i * step2
                                    for i in range(self.scan_2_steps + 1)]

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)
        # General variable scan kernel to be completed.
        # Framework: for each scan point, update the dataset variable,
        # run load+measure cycles, record loading/survival rates.
        # See qn_artiq_routines GeneralVariableScan.py for the full pattern.
        pass
