"""Align SPCM1/2 by monitoring the summed atom fluorescence (trapped windows only).

Same load -> watch-until-loss -> reload loop as atom_load_monitor_general.py, but
instead of two separate traces it pushes the SUM

    spcm_tot = SPCM1 + SPCM2

to a single dataset, and ONLY for windows where an atom is actually trapped
(spcm_tot > readout_threshold). So the live trace is a clean record of the
atom-fluorescence total -- maximize it while aligning the SPCM1/2 collection
optics. Background / atom-lost windows are not pushed (that's the "trapped only").

Live applet:
  ${python} "<repo>/applets/plot_spcm_tot.py" spcm_tot
  # or a built-in big number of the running mean:
  ${artiq_applet}big_number spcm_tot_mean

Datasets (broadcast, rolling window capped at max_points):
  spcm_tot       : list -- summed SPCM1+2 counts, trapped windows only
  spcm_tot_last  : scalar -- the most recent value (for a big-number applet)
  spcm_tot_mean  : scalar -- rolling mean of the last `mean_window` trapped windows

Self-contained: only CALLS shared subroutines; modifies nothing shared.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    load_atom, load_atom_deterministic_spcm3, imaging_pgc, beams_off_for_imaging,
    spcm_image_dual_sequential,
)
from utilities.write_results import write_results


class AtomAlignSPCM12TrappedOnly(EnvExperiment):
    """SPCM1/2 Alignment (spcm_tot, trapped only)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # CCB (Client Control Broker): lets the experiment auto-create the live
        # applet in the dashboard when it runs. Only present when launched from
        # the dashboard; prepare() guards the call so headless runs don't fail.
        self.setattr_device("ccb")

        # SPCM3 lives on ttl2 (input-capable). Registered here (self-contained)
        # so it can be read as the atom-survival reference without touching any
        # shared config or subroutine.
        self.setattr_device("ttl2")
        # ttl2_counter is ttl2's gateware EDGE COUNTER (a separate device from
        # the ttl2 TTLInOut). load_atom_deterministic_spcm3 polls SPCM3 through
        # it, so it must be registered here too.
        self.setattr_device("ttl2_counter")

        self.setattr_argument("num_runs",
                              NumberValue(200, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        self.setattr_argument("feedback_every_n",
                              NumberValue(10, min=1, max=1000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        # Atom-present cutoff on SPCM1+SPCM2: only windows above this are pushed.
        self.setattr_argument("readout_threshold",
                              NumberValue(60, min=1, max=100000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # Rolling-window size for the live plot.
        self.setattr_argument("max_points",
                              NumberValue(300, min=1, max=1000000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # spcm_tot_mean is the mean of the last `mean_window` trapped windows --
        # a RECENT (rolling) average so it tracks what you just changed while
        # aligning, instead of a cumulative average that barely moves.
        self.setattr_argument("mean_window",
                              NumberValue(10, min=1, max=1000000, step=1, ndecimals=0, type="int"),
                              group="Readout")

    def prepare(self):
        # All parameters (shims, t_readout_monitor, imaging freq/amp, ...) from
        # ExperimentVariables. No overrides.
        self.base.prepare()

        self._buf = []            # rolling window of trapped-window totals
                                  # (spcm_tot_mean = mean of its last mean_window)

        self.set_dataset("spcm_tot", [], broadcast=True)
        self.set_dataset("spcm_tot_last", 0.0, broadcast=True)
        self.set_dataset("spcm_tot_mean", 0.0, broadcast=True)
        self.set_dataset("measurement_progress", 0, broadcast=True)

        self._spawn_applets()

    def _spawn_applets(self):
        """Auto-create the live applets in the dashboard (via CCB) so they pop up
        when the experiment runs -- no need to add them by hand. Guarded so a
        headless run (no dashboard/ccb) just prints a note instead of failing."""
        applet = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "applets", "plot_spcm_tot.py"))
        try:
            # Scrolling spcm_tot trace.
            self.ccb.issue("create_applet", "spcm_tot (align)",
                           '${python} "' + applet + '" spcm_tot',
                           group="alignment")
            # Running-mean as a big number (built-in applet), to maximize.
            self.ccb.issue("create_applet", "spcm_tot mean",
                           "${artiq_applet}big_number spcm_tot_mean",
                           group="alignment")
        except Exception as e:
            print("Could not auto-create applet (no dashboard/ccb?):", e)

    @kernel
    def _image_triple(self, gate_time):
        """Imaging in ONE window: gate SPCM1 (ttl0), SPCM2 (ttl1) and SPCM3
        (ttl2) sequentially with the imaging beams on; returns
        (count1, count2, count3). Inlined here (self-contained) so SPCM3 can be
        read without modifying the shared spcm_image_dual_sequential. Same beams
        and sequential (overflow-safe) gating as that subroutine."""
        self.dds_cooling_DP.set(frequency=self.f_cooling_imaging,
                                amplitude=self.amp_cooling_imaging)
        self.dds_cooling_DP.sw.on()
        delay(1 * us)
        self.dds_repump.sw.on()
        self.ttl_repump_switch.off()
        delay(1 * us)

        t_end0 = self.ttl_SPCM1.gate_rising(gate_time)
        count1 = self.ttl_SPCM1.count(t_end0)
        delay(1 * ms)
        t_end1 = self.ttl_SPCM2.gate_rising(gate_time)
        count2 = self.ttl_SPCM2.count(t_end1)
        delay(1 * ms)
        t_end2 = self.ttl2.gate_rising(gate_time)
        count3 = self.ttl2.count(t_end2)
        delay(1 * ms)

        return count1, count2, count3

    @kernel
    def run(self):
        self.base.initialize_hardware()
        self.ttl2.input()          # SPCM3 direction = input
        delay(1000 * ms)

        # Initial 5-channel feedback (locks beam powers + FORT).
        prepare_for_feedback(self)
        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
        restore_after_feedback(self)
        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            self.core.break_realtime()

            # Periodic 5-channel feedback (between atoms only).
            if i % self.feedback_every_n == 0:
                prepare_for_feedback(self)
                amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                restore_after_feedback(self)

            # Load atom.
            # load_atom(self)
            load_atom_deterministic_spcm3(self)
            beams_off_for_imaging(self)

            # Readout loop: watch until the atom is lost. Push spcm_tot ONLY for
            # windows where the atom is trapped (sum above threshold).
            c0 = int(self.readout_threshold) + 1
            while c0 > int(self.readout_threshold):
                count1_r, count2_r, count3_r = self._image_triple(self.t_readout_monitor)
                c0 = count3_r
                c1 = count1_r + count2_r
                if c0 > int(self.readout_threshold):     # atom trapped this window
                    self._push(c1)
                imaging_pgc(self)
                beams_off_for_imaging(self)

            self.set_dataset("measurement_progress", i, broadcast=True)

        write_results(self)

    @rpc(flags={"async"})
    def _push(self, tot):
        """Append one trapped-window total to the rolling window and broadcast
        spcm_tot plus the latest value and the running mean."""
        self._buf.append(tot)
        n = int(self.max_points)
        if len(self._buf) > n:
            del self._buf[:len(self._buf) - n]

        # Rolling mean of the last `mean_window` trapped windows (responsive live
        # feedback -- unlike a cumulative mean, it tracks recent changes).
        w = int(self.mean_window)
        window = self._buf[-w:]
        mean_val = sum(window) / len(window)

        self.set_dataset("spcm_tot", list(self._buf), broadcast=True)
        self.set_dataset("spcm_tot_last", tot, broadcast=True)
        self.set_dataset("spcm_tot_mean", mean_val, broadcast=True)
