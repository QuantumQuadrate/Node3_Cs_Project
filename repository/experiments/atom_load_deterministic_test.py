"""Test bench for the deterministic atom-loading subroutine.

Exercises `subroutines.sequences.load_atom_deterministic` once per outer
iteration, then takes a confirmation fluorescence shot, then logs full
diagnostics. Use this experiment to:

  1. Calibrate the `single_atom_threshold_for_loading` rate threshold
     via the SPCM count histogram (see `applets/plot_loading_histogram.py`).
  2. Compare loading time and confirmed-atom rate against the statistical
     loader (`atom_load_monitor.py`).
  3. Validate edge cases (timeout returns False, feedback-fallback runs).

Per-shot dataset contract (all broadcast=True, used by the applet):
  det_load_success           bool   loader returned atom_loaded=True
  det_n_tries                int    total polling gates this shot (across all rounds)
  det_n_feedback_rounds      int    times rebalance_3ch was invoked
  det_spcm1_at_load          int    SPCM1 count in the winning gate (0 on miss)
  det_spcm2_at_load          int    SPCM2 count in the winning gate (0 on miss)
  det_load_time_s            float  wall-clock seconds from MOT-on to atom-trigger
  det_first_unloaded_counts  int    SPCM1+SPCM2 in the FIRST polling try (always
                                    captured, indicative of MOT-only background)
  det_confirm_count1         int    SPCM1 count in confirmation imaging shot
  det_confirm_count2         int    SPCM2 count in confirmation imaging shot
  det_confirm_bg1            int    SPCM1 background count (FORT off, beams on)
  det_confirm_bg2            int    SPCM2 background count (same)

Dashboard overrides:
  threshold_rate_override    if != 0, overrides single_atom_threshold_for_loading
                             (counts/s) for this run without touching the dataset
  gate_time_us_override      if != 0, overrides t_atom_check_time (in us)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, beams_off_for_imaging, spcm_image_dual,
    reset_to_mot,
)
from utilities.write_results import write_results


class AtomLoadDeterministicTest(EnvExperiment):
    """Atom Deterministic Loading TEST"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("num_runs",
                              NumberValue(200, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")

        # Live overrides for the two parameters you tune most during calibration.
        # Set to 0 to use the dataset value (the dashboard ExperimentVariables value).
        self.setattr_argument("threshold_rate_override",
                              NumberValue(0.0, min=0.0, max=1.0e8, ndecimals=1),
                              group="Deterministic Tuning")
        self.setattr_argument("gate_time_us_override",
                              NumberValue(0.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Deterministic Tuning")

        # Speed-optimization sweep knobs. Set to 0 to use the dataset value.
        # Sweep these (one at a time) to find the shortest setting at which
        # the loading rate stays on the saturation plateau.
        # NOTE: no unit= here on purpose. In this ARTIQ build unit="ms" would
        # itself scale the entered number by 1e-3, and we also scale by 1e-3
        # below -- that double-converts (entering 100 gives 0.1 ms, not 100).
        # Matches the gate_time_us_override / threshold_rate_override pattern:
        # bare number, explicit manual scale in prepare().
        self.setattr_argument("t_MOT2PGC_override_ms",
                              NumberValue(0.0, min=0.0, max=2000.0,
                                          ndecimals=0),
                              group="Optimization")

        # Rebalancer is created/configured by BaseExperiment.build()

    def prepare(self):
        # Use the "FORT_loading" calibration profile, same as atom_load_monitor.
        # This is the high-FORT-depth feedback profile used during loading.
        self.base.prepare(feedback_profile="FORT_loading")

        # Apply dashboard overrides on top of dataset values.
        # We assign to the same attribute names load_atom_deterministic reads,
        # so the kernel doesn't need to know whether the value came from dataset
        # or dashboard override.
        if self.threshold_rate_override > 0.0:
            self.single_atom_threshold_for_loading = float(self.threshold_rate_override)
        if self.gate_time_us_override > 0.0:
            self.t_atom_check_time = float(self.gate_time_us_override) * 1e-6
        if self.t_MOT2PGC_override_ms > 0.0:
            self.t_MOT2PGC = float(self.t_MOT2PGC_override_ms) * 1e-3

        # Result datasets -- all broadcast=True so the applet sees live updates.
        for key in (
            "det_load_success",
            "det_n_tries",
            "det_n_feedback_rounds",
            "det_spcm1_at_load",
            "det_spcm2_at_load",
            "det_load_time_s",
            "det_first_unloaded_counts",
            "det_confirm_count1",
            "det_confirm_count2",
            "det_confirm_bg1",
            "det_confirm_bg2",
        ):
            self.set_dataset(key, [], broadcast=True)

        # Echo the effective values so the applet can compute the threshold line
        # without depending on whether overrides were applied, and so the data
        # archive records what was actually used for each sweep point.
        self.set_dataset("det_threshold_rate_used",
                         float(self.single_atom_threshold_for_loading),
                         broadcast=True, persist=True)
        self.set_dataset("det_gate_time_used",
                         float(self.t_atom_check_time),
                         broadcast=True, persist=True)
        self.set_dataset("det_t_MOT2PGC_used",
                         float(self.t_MOT2PGC),
                         broadcast=True, persist=True)

        print(f"Deterministic test: threshold_rate = {self.single_atom_threshold_for_loading:.3e} counts/s, "
              f"gate_time = {self.t_atom_check_time*1e6:.1f} us, "
              f"t_MOT2PGC = {self.t_MOT2PGC*1e3:.0f} ms, "
              f"max_polling_tries = {int(self.max_polling_tries)}, "
              f"loading_timeout_rounds = {int(self.loading_timeout_rounds)}")
        print(f"Total budget per shot ~ "
              f"{int(self.max_polling_tries) * int(self.loading_timeout_rounds) * self.t_atom_check_time * 1e3:.0f} ms polling"
              f" + {self.t_MOT2PGC*1e3:.0f} ms MOT load + overhead")

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        # One initial 5-channel feedback to lock FORT amplitude for the run.
        # Inside the deterministic loader, only 3-ch (MOT-only) feedback runs,
        # so FORT stability needs this outer call.
        prepare_for_feedback(self)
        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
        restore_after_feedback(self)
        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            self.core.break_realtime()

            # Background measurement (cooling+repump on, FORT off, no atom).
            count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)

            # The headline action: deterministic loading.
            # Returns 7 values; see load_atom_deterministic docstring.
            (success, n_tries, n_fb, s1_load, s2_load,
             first_unloaded, t_load) = load_atom_deterministic(self)

            # Confirmation fluorescence shot (matches Yellow scope of parity).
            imaging_pgc(self)
            beams_off_for_imaging(self)
            delay(self.t_SPCM_exposure)
            count1_conf, count2_conf = spcm_image_dual(self, self.t_imaging_gate)
            imaging_pgc(self)
            beams_off_for_imaging(self)

            # Log everything in a single async RPC -- one kernel→host crossing per shot,
            # no per-poll RTIO event explosion. Use 1-if/else-0 (not int(bool)) because
            # the ARTIQ kernel compiler doesn't accept int() on a bool argument.
            success_int = 1 if success else 0
            self._log_shot(success_int, n_tries, n_fb, s1_load, s2_load, t_load,
                           first_unloaded,
                           count1_conf, count2_conf, count1_bg, count2_bg, i)

            self.set_dataset("measurement_progress", i + 1, broadcast=True)
            reset_to_mot(self)

        write_results(self)

    @rpc(flags={"async"})
    def _log_shot(self, success, n_tries, n_fb, s1_load, s2_load, t_load,
                  first_unloaded, count1_conf, count2_conf,
                  count1_bg, count2_bg, shot_index):
        """One async RPC per shot, 11 appends + periodic progress print.

        first_unloaded is the SPCM1+SPCM2 sum from the FIRST polling try of
        this shot -- a MOT-only background sample, populated on both success
        and timeout. The applet uses it for the "MOT only" histogram.
        s1_load / s2_load are the per-SPCM counts at the moment the threshold
        was crossed (success only; 0 on timeout). The applet uses
        s1_load + s2_load (filtered to >0) for the "atom loaded" histogram.
        """
        self.append_to_dataset("det_load_success", success)
        self.append_to_dataset("det_n_tries", n_tries)
        self.append_to_dataset("det_n_feedback_rounds", n_fb)
        self.append_to_dataset("det_spcm1_at_load", s1_load)
        self.append_to_dataset("det_spcm2_at_load", s2_load)
        self.append_to_dataset("det_load_time_s", t_load)
        self.append_to_dataset("det_first_unloaded_counts", first_unloaded)
        self.append_to_dataset("det_confirm_count1", count1_conf)
        self.append_to_dataset("det_confirm_count2", count2_conf)
        self.append_to_dataset("det_confirm_bg1", count1_bg)
        self.append_to_dataset("det_confirm_bg2", count2_bg)

        if shot_index % 10 == 0:
            # first_unloaded = SPCM1+SPCM2 in the FIRST polling try this shot.
            # Equivalent rate (counts/s) = first_unloaded / (2 * gate_time).
            gate = float(self.t_atom_check_time)
            mot_rate = 0.5 * first_unloaded / gate if gate > 0 else 0.0
            print(f"  shot {shot_index:4d}: success={success}, n_tries={n_tries}, "
                  f"n_fb={n_fb}, s1={s1_load}, s2={s2_load}, t_load={t_load*1e3:.1f} ms, "
                  f"first_unloaded={first_unloaded} (~{mot_rate/1e3:.1f} kcps MOT-only), "
                  f"conf=({count1_conf},{count2_conf}), bg=({count1_bg},{count2_bg})")
