"""Continuous atom loading monitor -- general-variables edition.

Same trap -> watch-fluorescence-until-loss -> reload loop as atom_load_monitor.py,
but with three changes requested for alignment work:

  1. ALL shim coil values (and nominal_trapdepth) come from ExperimentVariables
     -- NO hard-coded overrides. So this monitor uses the exact same fields as
     the rest of your experiments (single source of truth). t_readout_monitor
     also comes from ExperimentVariables, as before.

  2. The loss threshold `readout_threshold` (the value the summed fluorescence
     c0 is compared against) is a FRONT-PANEL argument -- edit it live in the
     dashboard instead of a hard-coded number.

  3. Feedback uses the latest 5-channel rebalancer (self.rebalancer.rebalance())
     instead of the 3-channel MOT-only one. NOTE: the 5-channel rebalance also
     re-references the FORT amplitude, so it is run only BETWEEN atoms (top of
     the outer loop), never during the readout-hold loop -- it will not perturb
     an atom currently being watched, but it does re-lock the FORT each time.

Applet command (live plot):
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


class AtomLoadMonitorGeneral(EnvExperiment):
    """Atom Load Monitor (general vars, 5ch feedback)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("num_runs",
                              NumberValue(100, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        self.setattr_argument("feedback_every_n",
                              NumberValue(10, min=1, max=1000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        # Loss threshold: summed-SPCM fluorescence (c0) below this = atom lost.
        # Front-panel so you can tune it live from the fluorescence plot,
        # especially after changing t_readout_monitor (counts scale with gate).
        self.setattr_argument("readout_threshold",
                              NumberValue(60, min=1, max=100000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # Rolling-window size: the fluorescence datasets keep only the most
        # recent `max_points` readings (drop the oldest when a new one arrives),
        # so the applet never accumulates unbounded data. Front-panel adjustable.
        self.setattr_argument("max_points",
                              NumberValue(300, min=1, max=1000000, step=1, ndecimals=0, type="int"),
                              group="Readout")

    def prepare(self):
        # Loads ALL parameters (shims, t_readout_monitor, trap depth, ...) from
        # ExperimentVariables. No overrides -> same fields as every other experiment.
        self.base.prepare()

        # Host-side rolling buffers (the dataset mirrors these, capped at max_points).
        self._buf1 = []
        self._buf2 = []
        self.set_dataset("spcm_counts1", [], broadcast=True)
        self.set_dataset("spcm_counts2", [], broadcast=True)

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        # Initial 5-channel feedback before the loop (locks beam powers + FORT).
        prepare_for_feedback(self)
        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
        restore_after_feedback(self)
        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            self.core.break_realtime()

            # --- Periodic 5-channel feedback (between atoms only) ---
            if i % self.feedback_every_n == 0:
                prepare_for_feedback(self)
                amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                restore_after_feedback(self)

            # --- Load atom (all shims from ExperimentVariables) ---
            load_atom(self)

            # --- Beams off before readout ---
            beams_off_for_imaging(self)

            # --- Readout loop: watch fluorescence until the atom is lost. ---
            # Seed c0 above the front-panel threshold so the loop always runs at
            # least once; the loop exits when measured fluorescence drops below it.
            c0 = int(self.readout_threshold) + 1
            while c0 > int(self.readout_threshold):
                count1_r, count2_r = spcm_image_dual_sequential(self, self.t_readout_monitor)
                c0 = count1_r + count2_r

                # Push into the rolling window (drops the oldest past max_points).
                self._push(count1_r, count2_r)

                # Brief PGC between readouts
                imaging_pgc(self)
                beams_off_for_imaging(self)
                delay(500 * ms)

            self.set_dataset("measurement_progress", i, broadcast=True)

        write_results(self)

    @rpc(flags={"async"})
    def _push(self, c1, c2):
        """Append one dual-SPCM reading to the rolling window and re-broadcast.

        Keeps only the most recent `max_points` samples so the applet shows a
        fixed-length trace (e.g. the last 300) instead of an ever-growing one.
        """
        self._buf1.append(c1)
        self._buf2.append(c2)
        n = int(self.max_points)
        if len(self._buf1) > n:
            del self._buf1[:len(self._buf1) - n]
            del self._buf2[:len(self._buf2) - n]
        # Copy so the stored dataset isn't aliased to the mutating buffer.
        self.set_dataset("spcm_counts1", list(self._buf1), broadcast=True)
        self.set_dataset("spcm_counts2", list(self._buf2), broadcast=True)
        self.set_dataset("SPCM1_counts", c1, broadcast=True)
        self.set_dataset("SPCM2_counts", c2, broadcast=True)
