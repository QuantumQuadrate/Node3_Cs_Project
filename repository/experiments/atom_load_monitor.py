"""Continuous atom loading monitor with dual-SPCM readout.

Uses shared subroutines (load_atom, prepare_for_feedback, etc.) from
sequences.py with ExperimentVariables datasets, plus atom_load-specific
shim overrides.

Goes straight into readout loop after loading (no separate atom check) —
matches the original's pattern where the readout loop IS the atom check.

Applet command:
  ${python} "<repo_path>/applets/plot_xyline.py" spcm_counts1 --pts 300
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    load_atom, imaging_pgc, beams_off_for_imaging,
    spcm_image_dual_sequential,
)
from utilities.write_results import write_results


class AtomLoadMonitor(EnvExperiment):
    """Atom_Load_Real_Time_Monitor"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("num_runs",
                              NumberValue(100, min=1, max=600000, step=1),
                              group="Experiment")
        self.setattr_argument("feedback_every_n",
                              NumberValue(10, min=1, max=1000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        self.setattr_argument("readout_threshold",
                              NumberValue(60, min=1, max=10000, ndecimals=0),
                              group="Readout")

    def prepare(self):
        self.base.prepare()

        # Override shims with atom_load_monitor-optimized values
        # (ExperimentVariables has parity values which differ)
        self.shim_Y_idle = 0.2
        self.shim_Z1_idle = 4.27
        self.shim_X_compression = 0.25
        self.shim_Y_compression = 0.0
        self.shim_Z1_compression = 4.27
        self.nominal_trapdepth = 0.22

        self.set_dataset("spcm_counts1", [], broadcast=True)
        self.set_dataset("spcm_counts2", [], broadcast=True)

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        # Initial feedback before loop — MOT-only (rebalance_3ch) so the
        # FORT amplitude is not perturbed during atom loading; see
        # subroutines/FEEDBACK_README.md "Rebalancer Methods".
        prepare_for_feedback(self)
        self.rebalancer.rebalance_3ch()
        restore_after_feedback(self)
        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            self.core.break_realtime()

            # --- Periodic feedback (MOT-only) ---
            if i % self.feedback_every_n == 0:
                prepare_for_feedback(self)
                self.rebalancer.rebalance_3ch()
                restore_after_feedback(self)

            # --- Load atom using shared subroutine ---
            load_atom(self)

            # --- Beams off before readout ---
            beams_off_for_imaging(self)

            # --- Readout loop (no separate atom check — loop IS the check) ---
            c0 = 900
            while c0 > int(self.readout_threshold):
                count1_r, count2_r = spcm_image_dual_sequential(self, self.t_readout_monitor)

                c0 = count1_r + count2_r

                self.append_to_dataset("spcm_counts1", count1_r)
                self.append_to_dataset("spcm_counts2", count2_r)
                self.set_dataset("SPCM1_counts", count1_r, broadcast=True)
                self.set_dataset("SPCM2_counts", count2_r, broadcast=True)

                # Brief PGC between readouts
                imaging_pgc(self)
                beams_off_for_imaging(self)

            self.set_dataset("measurement_progress", i, broadcast=True)

        write_results(self)
