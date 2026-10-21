"""SPCM3 alignment tool, SPCM3 ONLY: no SPCM1/SPCM2 readout at all.

SPCM3 (on ttl2) is both the detector being aligned AND the only atom reference:
load an atom -> image repeatedly, reading ONLY SPCM3 -> when SPCM3 drops below
`readout_threshold` the atom is treated as lost -> take a few atom-free windows
-> reload. Use it when SPCM1/2 are unavailable (blocked, unplugged, reading
zero) or when you simply want SPCM3 judged on its own.

Align by maximizing `align_contrast` -- or better, by watching the HISTOGRAM of
spcm_counts3 separate into two peaks (no atom / atom) in the analysis.

    align_signal      mean SPCM3 counts in windows ABOVE readout_threshold
    align_background  mean SPCM3 counts in windows AT/BELOW readout_threshold
    align_contrast    align_signal - align_background

!! READ THIS ABOUT THE METRICS. With no independent reference, SPCM3 both
   DEFINES and MEASURES "atom present", so the split is self-referential: every
   signal window is above the threshold by construction and every background
   window below it. align_contrast therefore can never be smaller than roughly
   the threshold spacing, and it shifts when you change `readout_threshold` even
   if the optics never moved. It is a useful RELATIVE indicator while you walk
   the detector at a FIXED threshold -- it is NOT an absolute figure of merit,
   and it cannot be compared against the two-detector version in
   atom_align_spcm3.py (where SPCM1+2 gave an independent reference, so the
   contrast there was a real measurement). The honest readout is the histogram:
   a genuinely aligned SPCM3 shows two separated peaks, and that is visible no
   matter where the threshold sits. There is likewise NO correlation metric here
   -- correlating SPCM3 against itself is meaningless, so align_corr is gone.

THRESHOLD -- `readout_threshold` is a front-panel (dashboard) argument, in
COUNTS PER WINDOW of SPCM3 over the `t_readout_monitor` gate. ~60 is the current
working value. It scales with the gate: change t_readout_monitor and you must
retune it. Set it so it lands in the valley between the two histogram peaks.

Live applets:
  # the SPCM3 trace
  ${python} "<repo>/applets/plot_xyline.py" spcm_counts3
  # or with a pinned y-axis
  ${python} "<repo>/applets/plot_spcm_counts.py" spcm_counts3 --ymin 0 --ymax 200
  # the number to maximize
  ${artiq_applet}big_number align_contrast

DATA -- `spcm_counts3` is broadcast as a rolling window capped at `max_points`
so the applet never accumulates unbounded data; `spcm_all3` records EVERY window
of the run uncapped and archive-only (broadcast=False), so it reaches the run's
HDF5 without any dashboard traffic. Offline histograms use the complete run.

NOTE ON ANALYSIS -- analysis/analyze_align_spcm3.py expects the THREE-detector
datasets (spcm_all1/2/3 + spcm_sum12) and will report "no windows" for runs of
this experiment, which honestly records only SPCM3. It needs a small
single-detector branch; ask for it.

DETECTION -- SPCM3 is read with the gateware edge counter `ttl2_counter`, not
with ttl2.gate_rising()/count(). TTLInOut buffers one timestamp per photon in a
64-deep FIFO that the CPU must drain event by event, and it overflows
(RTIOOverflow on channel 2) when photons arrive faster than that loop -- gating
a single channel does NOT make it safe. The counter counts in gateware and holds
one integer, so no photon rate can overflow it.

This experiment does NOT modify any subroutine or shared code (it only registers
ttl2 / ttl2_counter and inlines its own single-detector imaging window).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    load_atom, imaging_pgc, beams_off_for_imaging,
)
from utilities.write_results import write_results


class AtomAlignSPCM3Only(EnvExperiment):
    """SPCM3 Alignment (SPCM3 only)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # SPCM3 lives on ttl2. ttl2_counter is its gateware EDGE COUNTER
        # (device_db channel 0xa) -- that, not ttl2 itself, is what reads the
        # counts (see _image_spcm3_only). Registered here so we do NOT have to
        # touch device_aliases.json / any shared config.
        self.setattr_device("ttl2")
        self.setattr_device("ttl2_counter")

        self.setattr_argument("num_runs",
                              NumberValue(200, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        self.setattr_argument("feedback_every_n",
                              NumberValue(10, min=1, max=1000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        # ATOM THRESHOLD on SPCM3 itself, in counts per t_readout_monitor window.
        # Dashboard-adjustable; ~60 is the current working value. The readout loop
        # keeps imaging while SPCM3 stays ABOVE this and treats the first window
        # at/below it as "atom lost". Retune after changing t_readout_monitor
        # (counts scale with the gate).
        self.setattr_argument("readout_threshold",
                              NumberValue(60, min=1, max=100000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # Rolling-window size for the live plot.
        self.setattr_argument("max_points",
                              NumberValue(300, min=1, max=1000000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # After each atom is lost, take this many extra no-atom windows so the
        # "background" (atom-absent) bucket is well populated for the contrast.
        self.setattr_argument("n_bg_windows",
                              NumberValue(5, min=0, max=1000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # Safety cap on the watch loop. Without it the loop is "while SPCM3 >
        # threshold", which NEVER EXITS if the no-atom background sits above the
        # threshold (stray light, threshold set too low) -- the kernel would hang
        # with no error. This bounds one atom's watch; raise it to follow a
        # long-lived atom.
        self.setattr_argument("max_windows_per_atom",
                              NumberValue(50, min=1, max=1000000, step=1, ndecimals=0, type="int"),
                              group="Readout")

    def prepare(self):
        # All parameters (shims, t_readout_monitor, imaging freq/amp, ...) from
        # ExperimentVariables. No overrides.
        self.base.prepare()

        # Host-side rolling buffer (the broadcast dataset mirrors this, capped
        # at max_points).
        self._buf3 = []

        # UNCAPPED record of EVERY window measured in this run, for offline
        # histograms. The rolling buffer above is what the applet sees and stays
        # capped at max_points; this list is never trimmed.
        self._all3 = []

        # Running conditional stats for the alignment metrics.
        self._n_sig = 0
        self._sum3_sig = 0.0       # sum of SPCM3 over above-threshold windows
        self._n_bg = 0
        self._sum3_bg = 0.0        # sum of SPCM3 over at/below-threshold windows

        self.set_dataset("spcm_counts3", [], broadcast=True)
        # TOTAL windows recorded so far -- the absolute shot counter. spcm_counts3
        # is a ROLLING window, so once it hits max_points its index 0 stops
        # meaning "shot 0" and the array slides one place per shot. An applet
        # averaging groups of shots (plot_spcm_counts --bin N) then regroups
        # different shots on every update and its points visibly drift. Passing
        # this counter as `--total spcm_n_windows` lets it anchor the bins to
        # absolute shot numbers, so plotted points stay put.
        self.set_dataset("spcm_n_windows", 0, broadcast=True)
        # Archive-only (broadcast=False): saved into the run's HDF5, never sent
        # to the dashboard -> no extra applet traffic. Registered ONCE here: the
        # dataset manager keeps a REFERENCE to the list, so the appends in
        # _push() cost nothing and the HDF5 gets the complete, untruncated run.
        self.set_dataset("spcm_all3", self._all3, broadcast=False, archive=True)
        for key in ("align_signal", "align_background", "align_contrast"):
            self.set_dataset(key, 0.0, broadcast=True)
        self.set_dataset("measurement_progress", 0, broadcast=True)

    @kernel
    def _image_spcm3_only(self, gate_time):
        """One imaging window read by SPCM3 ALONE, via the gateware counter.

        Beam handling is the same as spcm_image_dual_sequential (cooling DP at
        the imaging frequency/amplitude + repump on, repump RF switch open).

        Detection uses ttl2_counter (EdgeCounter), NOT ttl2.gate_rising()/count().
        A TTLInOut stores ONE TIMESTAMP PER PHOTON in a 64-deep input FIFO and
        the kernel CPU must pop them one at a time (~microseconds each); if
        photons arrive faster than that loop drains them, the FIFO overflows ->
        RTIOOverflow on channel 2. Gating ONE channel does not avoid this -- it
        only removes the extra penalty of channels buffering while another is
        being read. An EdgeCounter counts in GATEWARE and keeps a single
        integer, so there is no FIFO and no rate at which it can overflow. This
        is the same mechanism load_atom has always used for its poll
        (sequences.py: ttl0_counter / ttl1_counter).

        Returns count3.
        """
        self.dds_cooling_DP.set(frequency=self.f_cooling_imaging,
                                amplitude=self.amp_cooling_imaging)
        self.dds_cooling_DP.sw.on()
        delay(1 * us)
        self.dds_repump.sw.on()
        self.ttl_repump_switch.off()
        delay(1 * us)

        self.ttl2_counter.gate_rising(gate_time)
        count3 = self.ttl2_counter.fetch_count()
        delay(1 * ms)          # rebuild RTIO slack after the blocking read

        return count3

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
            load_atom(self)
            beams_off_for_imaging(self)

            # --- Readout loop: watch SPCM3 until it drops below the threshold
            #     (atom lost), bounded by max_windows_per_atom. Seeded above the
            #     threshold so the loop always runs at least once. ---
            c0 = int(self.readout_threshold) + 1
            k = 0
            while c0 > int(self.readout_threshold) and k < int(self.max_windows_per_atom):
                count3_r = self._image_spcm3_only(self.t_readout_monitor)
                c0 = count3_r
                self._push(count3_r)
                k += 1

                imaging_pgc(self)
                beams_off_for_imaging(self)
                delay(100 * ms)

            # --- Background: a few no-atom windows so the contrast is reliable. ---
            for _ in range(int(self.n_bg_windows)):
                b3 = self._image_spcm3_only(self.t_readout_monitor)
                self._push(b3)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                delay(100 * ms)

            self.set_dataset("measurement_progress", i, broadcast=True)

        write_results(self)

    @rpc(flags={"async"})
    def _push(self, c3):
        """Store one SPCM3 window: append to both records, update the metrics.

        The signal/background split uses SPCM3 against readout_threshold, which
        is self-referential (see the module docstring) -- fine as a relative
        indicator at a fixed threshold, not an absolute measurement.
        """
        # Uncapped record of the whole run (archive-only; this list IS the
        # spcm_all3 dataset registered in prepare(), so nothing else is needed).
        self._all3.append(c3)

        # Rolling window (capped at max_points).
        self._buf3.append(c3)
        n = int(self.max_points)
        if len(self._buf3) > n:
            del self._buf3[:len(self._buf3) - n]

        # Conditional means, split on the threshold.
        if c3 > int(self.readout_threshold):
            self._n_sig += 1; self._sum3_sig += c3
        else:
            self._n_bg += 1; self._sum3_bg += c3

        sig = (self._sum3_sig / self._n_sig) if self._n_sig else 0.0
        bg = (self._sum3_bg / self._n_bg) if self._n_bg else 0.0

        # Copy so the broadcast dataset isn't aliased to the mutating buffer.
        self.set_dataset("spcm_counts3", list(self._buf3), broadcast=True)
        # Absolute shot count (never trimmed) -- see prepare().
        self.set_dataset("spcm_n_windows", len(self._all3), broadcast=True)
        self.set_dataset("align_signal", sig, broadcast=True)
        self.set_dataset("align_background", bg, broadcast=True)
        self.set_dataset("align_contrast", sig - bg, broadcast=True)
