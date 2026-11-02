"""Deterministic-loading MOT-formation SCAN (single run).

Sweeps the MOT-formation time passed to `load_atom_deterministic` over a range
and records the loading diagnostics at each point -- all in ONE run, so the whole
scan lives in a single .h5 and `analyze_atom_load_deterministic.py` can plot the
time-vs-MOT curve directly.

Why one run beats many: the scan is INTERLEAVED. The outer loop (`n_loops`)
sweeps every MOT value once, then repeats, so each MOT point is sampled across
the whole run instead of in one time block. Run-to-run loading drift (which
badly confounded the old one-MOT-per-run scans) therefore averages out per point
instead of aliasing onto the MOT axis.

Each shot is: `background_measurement_dual` (FORT switch OFF -> releases any atom
from the previous shot so the trap starts EMPTY -- reset_to_mot does NOT turn the
FORT off -- and measures the no-atom background), then one
`load_atom_deterministic(exp, t_mot_formation)` call (returns success, n_tries,
feedback rounds, winning-gate SPCM counts, first-try MOT-only counts, poll time),
then a CONFIRMATION image to check whether an atom is REALLY in the trap, then
`reset_to_mot`. The confirmation (signal - background > atom_threshold, same as
pgc_lowtrap_scan) is ground truth independent of the loader's own threshold
verdict -- which can false-trigger on a bright MOT at long formation times, so
`det_load_success` can read 100% while `det_confirmed` does not. Compare the two.
(Skipping the pre-load release is what makes n_tries collapse to 1 -- the loader's
first poll gate sees the leftover atom immediately.)

Scan is SINGLE-POINT when mot_min == mot_max (or step <= 0): then this is just the
old baseline test at one MOT time, `n_loops` shots.

Per-shot datasets (all broadcast; one entry per (MOT point x loop)):
  det_mot_formation_s        float  the MOT-formation time used for THIS shot [s]
  det_loop_index             int    which outer sweep (0..n_loops-1)
  det_load_success           int    loader returned atom_loaded=True (1/0)
  det_n_tries                int    total polling gates this shot
  det_n_feedback_rounds      int    MOT-feedback fallbacks inside the loader
  det_spcm1_at_load          int    SPCM1 count in the winning gate (0 on miss)
  det_spcm2_at_load          int    SPCM2 count in the winning gate (0 on miss)
  det_load_time_s            float  poll time (EXCLUDES the MOT formation) [s]
  det_first_unloaded_counts  int    SPCM1+SPCM2 in the FIRST polling try (MOT-only)
  det_confirmed              int    GROUND TRUTH: confirmation image says an atom
                                    is really trapped (1/0) -- vs the loader's
                                    det_load_success, which can false-trigger
  det_confirm_count1/2       int    SPCM1/2 in the confirmation image
  det_confirm_bg1/2          int    SPCM1/2 no-atom background (FORT off)

Scalars (effective values actually used):
  det_threshold_rate_used    counts/s
  det_gate_time_used         s

Dashboard overrides (0 => use dataset value):
  threshold_rate_override    overrides single_atom_threshold_for_loading (counts/s)
  gate_time_us_override      overrides t_atom_check_time (us)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, beams_off_for_imaging, spcm_image_dual, reset_to_mot,
)
from utilities.write_results import write_results


class AtomLoadDeterministicTest(EnvExperiment):
    """Atom Deterministic Loading MOT scan"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- MOT-formation scan range (ms). Single point if min==max/step<=0. ---
        self.setattr_argument("mot_min_ms",
                              NumberValue(20, min=1, max=2000, step=1, ndecimals=0),
                              group="MOT scan")
        self.setattr_argument("mot_max_ms",
                              NumberValue(150, min=1, max=2000, step=1, ndecimals=0),
                              group="MOT scan")
        self.setattr_argument("mot_step_ms",
                              NumberValue(10, min=0, max=2000, step=1, ndecimals=0),
                              group="MOT scan")
        # Number of interleaved sweeps = shots per MOT point. More = better
        # per-point statistics AND better drift averaging.
        self.setattr_argument("n_loops",
                              NumberValue(20, min=1, max=100000, step=1, ndecimals=0, type="int"),
                              group="MOT scan")

        # Live overrides you tune during calibration (0 => use dataset value).
        self.setattr_argument("threshold_rate_override",
                              NumberValue(0.0, min=0.0, max=1.0e8, ndecimals=1),
                              group="Deterministic Tuning")
        self.setattr_argument("gate_time_us_override",
                              NumberValue(0.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Deterministic Tuning")

        # Rebalancer is created/configured by BaseExperiment.build()

    def prepare(self):
        self.base.prepare(feedback_profile="FORT_loading")

        # Apply dashboard overrides on top of dataset values.
        if self.threshold_rate_override > 0.0:
            self.single_atom_threshold_for_loading = float(self.threshold_rate_override)
        if self.gate_time_us_override > 0.0:
            self.t_atom_check_time = float(self.gate_time_us_override) * 1e-6

        # --- Build the MOT-formation scan list (seconds). ---
        vmin = float(self.mot_min_ms)
        vmax = float(self.mot_max_ms)
        step = float(self.mot_step_ms)
        if step > 0.0 and vmax > vmin:
            n = int(round((vmax - vmin) / step)) + 1
        else:
            n = 1
        if n < 1:
            n = 1
        self.mot_list = [(vmin + i * step) * 1e-3 for i in range(n)]   # seconds
        self.n_points = n

        # Per-shot result datasets.
        for key in (
            "det_mot_formation_s",
            "det_loop_index",
            "det_load_success",
            "det_n_tries",
            "det_n_feedback_rounds",
            "det_spcm1_at_load",
            "det_spcm2_at_load",
            "det_load_time_s",
            "det_first_unloaded_counts",
            "det_confirmed",
            "det_confirm_count1",
            "det_confirm_count2",
            "det_confirm_bg1",
            "det_confirm_bg2",
        ):
            self.set_dataset(key, [], broadcast=True)

        self.set_dataset("det_threshold_rate_used",
                         float(self.single_atom_threshold_for_loading),
                         broadcast=True, persist=True)
        self.set_dataset("det_gate_time_used",
                         float(self.t_atom_check_time),
                         broadcast=True, persist=True)
        self.set_dataset("measurement_progress", 0, broadcast=True)
        self.set_dataset("mot_scan_complete", 0, broadcast=True)

        self._shots_done = 0
        self._total_shots = self.n_points * int(self.n_loops)

        mot_ms_list = [m * 1e3 for m in self.mot_list]
        print(f"MOT-formation scan: {self.n_points} point(s) "
              f"{mot_ms_list[0]:.0f}..{mot_ms_list[-1]:.0f} ms x {int(self.n_loops)} loops "
              f"= {self._total_shots} shots (interleaved).")
        print(f"  threshold = {self.single_atom_threshold_for_loading:.3e} counts/s, "
              f"gate = {self.t_atom_check_time*1e6:.1f} us, "
              f"max_polling_tries = {int(self.max_polling_tries)}, "
              f"loading_timeout_rounds = {int(self.loading_timeout_rounds)}")

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        for loop in range(int(self.n_loops)):
            self.core.break_realtime()

            # One 5-channel feedback per sweep (locks FORT amplitude). The loader
            # only does 3-ch MOT feedback internally, so FORT stability needs this.
            prepare_for_feedback(self)
            amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
            restore_after_feedback(self)
            self.core.break_realtime()

            for j in range(self.n_points):
                self.core.break_realtime()
                mot = self.mot_list[j]

                # Release the previous shot's atom AND measure the no-atom
                # background (FORT switch OFF -- reset_to_mot does not do this).
                # Also the background subtracted in the confirmation check below.
                count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)

                # The headline action: one deterministic load at this MOT time.
                (success, n_tries, n_fb, s1_load, s2_load,
                 first_unloaded, t_load) = load_atom_deterministic(self, mot)

                # GROUND TRUTH: image the trap and check whether an atom is really
                # there, independent of the loader's own threshold verdict (which
                # can false-trigger on a bright MOT at long formation times). Same
                # confirmation as pgc_lowtrap_scan: (signal - background) > threshold.
                imaging_pgc(self)
                beams_off_for_imaging(self)
                count1_conf, count2_conf = spcm_image_dual(self, self.t_imaging_gate)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                confirmed = (count1_conf + count2_conf
                             - count1_bg - count2_bg) > self.atom_threshold

                success_int = 1 if success else 0
                confirmed_int = 1 if confirmed else 0
                self._save_point(mot, loop, success_int, n_tries, n_fb,
                                 s1_load, s2_load, t_load, first_unloaded,
                                 confirmed_int, count1_conf, count2_conf,
                                 count1_bg, count2_bg)

                # Release the atom before the next point so each load is fresh.
                reset_to_mot(self)

        self._finish()
        write_results(self)

    @rpc(flags={"async"})
    def _save_point(self, mot_s, loop, success, n_tries, n_fb,
                    s1_load, s2_load, t_load, first_unloaded,
                    confirmed, count1_conf, count2_conf, count1_bg, count2_bg):
        """One async RPC per shot: append the per-shot fields + progress."""
        self.append_to_dataset("det_mot_formation_s", mot_s)
        self.append_to_dataset("det_loop_index", loop)
        self.append_to_dataset("det_load_success", success)
        self.append_to_dataset("det_n_tries", n_tries)
        self.append_to_dataset("det_n_feedback_rounds", n_fb)
        self.append_to_dataset("det_spcm1_at_load", s1_load)
        self.append_to_dataset("det_spcm2_at_load", s2_load)
        self.append_to_dataset("det_load_time_s", t_load)
        self.append_to_dataset("det_first_unloaded_counts", first_unloaded)
        self.append_to_dataset("det_confirmed", confirmed)
        self.append_to_dataset("det_confirm_count1", count1_conf)
        self.append_to_dataset("det_confirm_count2", count2_conf)
        self.append_to_dataset("det_confirm_bg1", count1_bg)
        self.append_to_dataset("det_confirm_bg2", count2_bg)

        self._shots_done += 1
        self.set_dataset("measurement_progress", self._shots_done, broadcast=True)
        if self._shots_done % 20 == 0:
            print(f"  shot {self._shots_done}/{self._total_shots}: "
                  f"MOT={mot_s*1e3:.0f} ms loop={loop} success={success} "
                  f"confirmed={confirmed} n_tries={n_tries} t_load={t_load*1e3:.1f} ms "
                  f"first_unloaded={first_unloaded} conf=({count1_conf},{count2_conf}) "
                  f"bg=({count1_bg},{count2_bg})")

    @rpc
    def _finish(self):
        self.set_dataset("mot_scan_complete", self._total_shots, broadcast=True)
