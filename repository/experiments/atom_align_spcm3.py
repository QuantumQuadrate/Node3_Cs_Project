"""SPCM3 alignment tool: align a new SPCM (ttl2) against SPCM1/2.

A third SPCM (SPCM3, on ttl2) is being added on the opposite side of the current
SPCM1/2. This experiment helps you ALIGN SPCM3 onto the atom image, using SPCM1+2
as the reference for "is there a trapped atom".

Based on atom_load_monitor_general.py (load -> watch-until-loss -> reload), with:
  - a third SPCM read in the SAME imaging window as SPCM1/2 (self-contained
    triple imaging; the shared subroutines are only called, never modified),
  - SPCM1+2 kept as the atom-present reference (the loss threshold), and
  - live alignment metrics you maximize while physically moving SPCM3:
      align_signal      : mean SPCM3 counts when an atom IS present (SPCM1+2 > thr)
      align_background  : mean SPCM3 counts when NO atom (SPCM1+2 <= thr)
      align_contrast    : align_signal - align_background   <-- maximize this
      align_corr        : Pearson correlation of SPCM3 vs (SPCM1+SPCM2)  (-> 1)
  When SPCM3 is aligned onto the atom, it shows high counts exactly when SPCM1+2
  do, so contrast and correlation both climb.

Live applets:
  # three scrolling traces
  ${python} "<repo>/applets/plot_spcm_counts.py" spcm_counts1 --y2 spcm_counts2 --y3 spcm_counts3
  # SPCM3 vs (SPCM1+SPCM2) scatter with live correlation
  ${python} "<repo>/applets/plot_align_scatter.py" spcm_counts3 --x spcm_sum12
  # the numbers to maximize (built-in big-number applets)
  ${artiq_applet}big_number align_contrast
  ${artiq_applet}big_number align_corr

The BROADCAST datasets (spcm_counts1/2/3, spcm_sum12 -- the ones the applets
read) are rolling windows capped at max_points, so the live plots never
accumulate unbounded data. Alongside them, spcm_all1/spcm_all2/spcm_all3 record
EVERY window of the run with no cap; they are archive-only (broadcast=False), so
they are saved into the run's HDF5 but never sent to the dashboard -- offline
histograms therefore use the complete run, not just the last max_points shots.
Analyze with analysis/analyze_align_spcm3.py, which prefers spcm_all* and falls
back to the rolling datasets for runs recorded before spcm_all* existed.

SPCM1/2 remain the reference; this experiment does NOT modify any subroutine or
shared code (it only registers ttl2 and inlines the third gate).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    load_atom_deterministic, imaging_pgc, beams_off_for_imaging,
)
from utilities.write_results import write_results


class AtomAlignSPCM3(EnvExperiment):
    """SPCM3 Alignment (vs SPCM1/2)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # SPCM3 lives on ttl2 (input-capable). Register it here so we do NOT
        # have to touch device_aliases.json / any shared config.
        self.setattr_device("ttl2")

        self.setattr_argument("num_runs",
                              NumberValue(200, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        self.setattr_argument("feedback_every_n",
                              NumberValue(10, min=1, max=1000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        # Atom-present reference threshold on SPCM1+SPCM2 (the loss cutoff).
        self.setattr_argument("readout_threshold",
                              NumberValue(60, min=1, max=100000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # Rolling-window size for the live plots.
        self.setattr_argument("max_points",
                              NumberValue(300, min=1, max=1000000, step=1, ndecimals=0, type="int"),
                              group="Readout")
        # After each atom is lost, take this many extra no-atom windows so the
        # "background" (atom-absent) bucket is well populated for the contrast.
        self.setattr_argument("n_bg_windows",
                              NumberValue(5, min=0, max=1000, step=1, ndecimals=0, type="int"),
                              group="Readout")

    def prepare(self):
        # All parameters (shims, t_readout_monitor, imaging freq/amp, ...) from
        # ExperimentVariables. No overrides.
        self.base.prepare()

        # Host-side rolling buffers (datasets mirror these, capped at max_points).
        self._buf1 = []
        self._buf2 = []
        self._buf3 = []
        self._bufsum = []          # SPCM1 + SPCM2 (reference), paired with buf3

        # UNCAPPED record of EVERY window measured in this run, for offline
        # histograms. The rolling buffers above are what the applets see and stay
        # capped at max_points; these are separate lists that are never trimmed.
        self._all1 = []
        self._all2 = []
        self._all3 = []

        # Running conditional stats for the alignment metrics.
        self._n_sig = 0
        self._sum3_sig = 0.0       # sum of SPCM3 over atom-present windows
        self._n_bg = 0
        self._sum3_bg = 0.0        # sum of SPCM3 over atom-absent windows
        # Correlation accumulators (SPCM3 vs SPCM1+SPCM2).
        self._cn = 0
        self._cx = 0.0; self._cy = 0.0
        self._cxx = 0.0; self._cyy = 0.0; self._cxy = 0.0

        for key in ("spcm_counts1", "spcm_counts2", "spcm_counts3", "spcm_sum12"):
            self.set_dataset(key, [], broadcast=True)
        # Archive-only (broadcast=False): saved into the run's HDF5, never sent
        # to the dashboard -> the applets and their network traffic are
        # unchanged. Registered ONCE here: the dataset manager keeps a REFERENCE
        # to each list, so the appends in _push() cost nothing and the HDF5 gets
        # the complete, untruncated run.
        for key, buf in (("spcm_all1", self._all1),
                         ("spcm_all2", self._all2),
                         ("spcm_all3", self._all3)):
            self.set_dataset(key, buf, broadcast=False, archive=True)
        for key in ("align_signal", "align_background", "align_contrast", "align_corr"):
            self.set_dataset(key, 0.0, broadcast=True)
        self.set_dataset("measurement_progress", 0, broadcast=True)

    @kernel
    def _image_triple(self, gate_time):
        """Dual+one SPCM imaging in ONE window: gate SPCM1 (ttl0), SPCM2 (ttl1)
        and SPCM3 (ttl2) sequentially with the imaging beams on. Sequential
        gating matches spcm_image_dual_sequential (overflow-safe for long gates).
        Returns (count1, count2, count3). Inlined here so all three counts come
        from the same atom window -- required for the correlation."""
        # Imaging beams on (cooling DP at imaging frequency + repump) -- same as
        # spcm_image_dual_sequential, but this copy is local to this experiment.
        self.dds_cooling_DP.set(frequency=self.f_cooling_imaging,
                                amplitude=self.amp_cooling_imaging)
        self.dds_cooling_DP.sw.on()
        delay(1 * us)
        self.dds_repump.sw.on()
        self.ttl_repump_switch.off()
        delay(1 * us)

        t_end2 = self.ttl2.gate_rising(gate_time)
        count3 = self.ttl2.count(t_end2)
        delay(1 * ms)
        t_end0 = self.ttl_SPCM1.gate_rising(gate_time)
        count1 = self.ttl_SPCM1.count(t_end0)
        delay(1 * ms)
        t_end1 = self.ttl_SPCM2.gate_rising(gate_time)
        count2 = self.ttl_SPCM2.count(t_end1)
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
            load_atom_deterministic(self)
            beams_off_for_imaging(self)

            # --- Readout loop: watch (SPCM1+2 reference) until the atom is lost,
            #     reading SPCM3 in the same window every time. ---
            c0 = int(self.readout_threshold) + 1
            while c0 > int(self.readout_threshold):
                count1_r, count2_r, count3_r = self._image_triple(self.t_readout_monitor)
                # count1_r, count2_r, count3_r = self._image_triple(20 * ms)
                c0 = count1_r + count2_r
                self._push(count1_r, count2_r, count3_r)
                
                imaging_pgc(self)
                beams_off_for_imaging(self)
                delay(100 * ms)
            # # --- Background: a few no-atom windows so the contrast is reliable. ---
            for _ in range(int(self.n_bg_windows)):
                b1, b2, b3 = self._image_triple(self.t_readout_monitor)
                self._push(b1, b2, b3)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                delay(100 * ms)

            self.set_dataset("measurement_progress", i, broadcast=True)

        write_results(self)

    @rpc(flags={"async"})
    def _push(self, c1, c2, c3):
        """Append one triple reading, update the rolling windows and the live
        alignment metrics (conditional means + correlation), then broadcast."""
        s12 = c1 + c2
        present = s12 > int(self.readout_threshold)

        # Uncapped record of the whole run (archive-only; these lists ARE the
        # spcm_all* datasets registered in prepare(), so nothing else is needed).
        self._all1.append(c1); self._all2.append(c2); self._all3.append(c3)

        # Rolling windows (capped at max_points).
        self._buf1.append(c1); self._buf2.append(c2)
        self._buf3.append(c3); self._bufsum.append(s12)
        n = int(self.max_points)
        if len(self._buf1) > n:
            k = len(self._buf1) - n
            del self._buf1[:k]; del self._buf2[:k]
            del self._buf3[:k]; del self._bufsum[:k]

        # Conditional means: SPCM3 with / without an atom.
        if present:
            self._n_sig += 1; self._sum3_sig += c3
        else:
            self._n_bg += 1; self._sum3_bg += c3

        # Correlation of SPCM3 (y) vs SPCM1+SPCM2 (x).
        x = float(s12); y = float(c3)
        self._cn += 1
        self._cx += x; self._cy += y
        self._cxx += x * x; self._cyy += y * y; self._cxy += x * y

        sig = (self._sum3_sig / self._n_sig) if self._n_sig else 0.0
        bg = (self._sum3_bg / self._n_bg) if self._n_bg else 0.0
        corr = self._pearson()

        self.set_dataset("spcm_counts1", list(self._buf1), broadcast=True)
        self.set_dataset("spcm_counts2", list(self._buf2), broadcast=True)
        self.set_dataset("spcm_counts3", list(self._buf3), broadcast=True)
        self.set_dataset("spcm_sum12", list(self._bufsum), broadcast=True)
        self.set_dataset("align_signal", sig, broadcast=True)
        self.set_dataset("align_background", bg, broadcast=True)
        self.set_dataset("align_contrast", sig - bg, broadcast=True)
        self.set_dataset("align_corr", corr, broadcast=True)

    def _pearson(self):
        """Pearson r from the running sums (host-side, no BLAS)."""
        n = self._cn
        if n < 2:
            return 0.0
        cov = self._cxy - self._cx * self._cy / n
        vx = self._cxx - self._cx * self._cx / n
        vy = self._cyy - self._cy * self._cy / n
        denom = (vx * vy) ** 0.5
        if denom <= 0.0:
            return 0.0
        r = cov / denom
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
        return r
