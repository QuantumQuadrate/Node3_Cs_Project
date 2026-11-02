"""Release-and-recapture: single-atom TEMPERATURE calibration.

Per trial, on a fresh atom:
  1. 5-channel feedback (beam powers + FORT; also returns the low-trap amp3)
  2. background image (FORT off)                         -> bg
  3. deterministic load (load_atom_deterministic)
  4. PGC -> image 1 -> PGC                               -> shot1 (loaded?)
  5. [optional] adiabatic ramp DOWN to the low trap (amp3 from the rebalancer)
  6. RELEASE: FORT RF switch off for t_release, then back on (recapture)
  7. [optional] adiabatic ramp back UP to trap_ramp_start_amp
  8. PGC -> image 2 -> PGC                               -> shot2 (recaptured?)
  9. reset_to_mot

While the FORT is off the atom flies ballistically; it is recaptured only if its
kinetic + potential energy at switch-on is still negative. Recapture probability
vs t_release therefore falls faster for a hotter atom, and fitting the curve to a
Monte Carlo of a thermal atom in the Gaussian trap gives the temperature in uK
(analysis/analyze_release_recapture.py, after Tuchendler et al., PRA 78, 033425
(2008)). That conversion needs the trap depth U0 and waist w0 AT THE RELEASE
DEPTH -- which is why the release depth is an explicit switch here:

  release_at_low_trap = False : release from the loading/imaging trap
      (FORT at nominal_trapdepth, as in the archive R&R codes) -> temperature
      after loading + PGC.
  release_at_low_trap = True  : ramp down to the low trap first (exactly the
      Rate_measurement / parity ramp: trap_ramp_start_amp -> amp3), release
      there, ramp back up for image 2 -> temperature in the trap the excitation
      runs in.

Release-time scan: t_release_min_us .. max .. step (us), plus t = 0 ALWAYS (the
no-release reference: survival through two images, which the fit uses as the
recapture amplitude). Times are snapped to whole ns (1 mu = 1 ns on Kasli) and
the value logged per shot is the exact snapped value. Each of the
n_measurements sweeps visits every release time once in a freshly shuffled
order, so slow drifts do not alias onto the release axis. t = 0 does not toggle
the FORT switch at all (an off/on pair at the same timestamp is not a release).

The release is timed on the FORT RF switch. The AOM's acoustic delay shifts
switch-off and switch-on equally, so the optical off time equals the RF off
time to within the AOM rise/fall (~0.1-0.2 us) -- negligible for the 5-100 us
releases that matter for temperatures of order 10 uK.

Sources: archive/Calibration/Release_n_recap_optimize_20260413.py (sequence),
experiments/compression_shim_scan.py (modern R&R step), experiments/
Rate_measurement.py (feedback, imaging and low-trap ramp conventions).

Data (one row per trial, all broadcast):
  rr_t_release_us   exact release time of the shot (us)
  rr_loaded         1 if image 1 shows an atom (shot1 - bg > atom_threshold)
  rr_survived       1 if image 2 shows it again (only meaningful if loaded)
  rr_bg_spcm1/2, rr_shot1_spcm1/2, rr_shot2_spcm1/2   raw counts (re-threshold
                    offline); shot2 is 0 when not loaded (image 2 is skipped)
  rr_release_amp    FORT DDS amplitude the atom was released from
Live curve (applet): rr_curve_t_us, rr_curve_p, rr_curve_err (recaptured /
loaded per release time, binomial error).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, beams_off_for_imaging, spcm_image_dual,
    ramp_trap_amp, reset_to_mot,
)
from utilities.write_results import write_results


class ReleaseRecapture(EnvExperiment):
    """Release & Recapture (temperature)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # CCB: auto-create the live recapture-curve applet (guarded in prepare).
        self.setattr_device("ccb")

        self.setattr_argument("n_measurements",
                              NumberValue(100, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")

        # --- Release-time scan, in MICROSECONDS (converted once, in prepare) ---
        self.setattr_argument("t_release_min_us",
                              NumberValue(5.0, min=0.0, max=10000.0, ndecimals=3, step=1.0),
                              group="Release time scan (us)")
        self.setattr_argument("t_release_max_us",
                              NumberValue(100.0, min=0.0, max=10000.0, ndecimals=3, step=1.0),
                              group="Release time scan (us)")
        self.setattr_argument("t_release_step_us",
                              NumberValue(5.0, min=0.001, max=10000.0, ndecimals=3, step=1.0),
                              group="Release time scan (us)")

        # --- Where to release from ---
        self.setattr_argument("release_at_low_trap", BooleanValue(False),
                              group="Release depth")
        # Low-trap ramp, used only when release_at_low_trap is True. Declared
        # after base.build() so they shadow the same-named datasets, exactly as
        # in Rate_measurement; defaults are the live values (RID 3649, expid).
        self.setattr_argument("trap_ramp_start_amp",
                              NumberValue(0.21, min=0.0, max=0.3, ndecimals=3),
                              group="Release depth")
        self.setattr_argument("trap_ramp_steps",
                              NumberValue(200, min=0, max=100000, ndecimals=0),
                              group="Release depth")
        self.setattr_argument("trap_ramp_dt_us",
                              NumberValue(50, min=0.001, max=1000, ndecimals=3),
                              group="Release depth")

    def prepare(self):
        # Same feedback profile as Rate_measurement / ExcitationScan, so amp3 is
        # the same low-trap amplitude the excitation experiments ramp to.
        self.base.prepare(feedback_profile="low_FORT")

        # --- Release times: whole ns, 0 always included, sorted, unique ---
        lo_us = float(self.t_release_min_us)
        hi_us = float(self.t_release_max_us)
        st_us = float(self.t_release_step_us)
        if hi_us < lo_us:
            lo_us, hi_us = hi_us, lo_us
        n = int(np.floor((hi_us - lo_us) / st_us + 1e-9)) + 1
        ns = {0}
        for i in range(n):
            ns.add(int(round((lo_us + i * st_us) * 1000.0)))   # us -> ns (= mu)
        self.release_ns = sorted(ns)
        self.release_us = [v / 1000.0 for v in self.release_ns]
        self.n_points = len(self.release_ns)

        # Interleaved schedule: every sweep visits each release time once, in a
        # new shuffled order. Kept in SECONDS for the kernel, which converts with
        # core.seconds_to_mu (exact for whole ns) -> int64 for delay_mu.
        rng = np.random.RandomState(12345)
        sched = []
        for _ in range(int(self.n_measurements)):
            for j in rng.permutation(self.n_points):
                sched.append(self.release_ns[int(j)] * 1e-9)
        self.schedule_s = sched
        self.n_trials = len(sched)

        # --- Per-trial datasets ---
        for key in ("rr_t_release_us", "rr_loaded", "rr_survived",
                    "rr_bg_spcm1", "rr_bg_spcm2",
                    "rr_shot1_spcm1", "rr_shot1_spcm2",
                    "rr_shot2_spcm1", "rr_shot2_spcm2",
                    "rr_release_amp"):
            self.set_dataset(key, [], broadcast=True)
        self.set_dataset("rr_progress", 0, broadcast=True)

        # --- Live recapture curve (applet) ---
        self._agg = {us: [0, 0] for us in self.release_us}   # us -> [recaptured, loaded]
        self.set_dataset("rr_curve_t_us", list(self.release_us), broadcast=True)
        self.set_dataset("rr_curve_p", [0.0] * self.n_points, broadcast=True)
        self.set_dataset("rr_curve_err", [0.0] * self.n_points, broadcast=True)

        # --- Metadata so the .h5 is self-describing (archive only) ---
        self.set_dataset("rr_release_times_us", list(self.release_us))
        self.set_dataset("rr_release_at_low_trap", int(bool(self.release_at_low_trap)))
        self.set_dataset("rr_n_measurements", int(self.n_measurements))
        self.set_dataset("rr_atom_threshold", float(self.atom_threshold))
        self.set_dataset("rr_nominal_trapdepth", float(self.nominal_trapdepth))
        self.set_dataset("rr_trap_ramp_start_amp", float(self.trap_ramp_start_amp))
        self.set_dataset("rr_trap_ramp_steps", float(self.trap_ramp_steps))
        self.set_dataset("rr_trap_ramp_dt_us", float(self.trap_ramp_dt_us))

        print(f"ReleaseRecapture: {self.n_points} release times x "
              f"{int(self.n_measurements)} sweeps = {self.n_trials} trials, "
              f"release from the {'LOW' if self.release_at_low_trap else 'HIGH'} trap")
        print(f"  release times (us): {self.release_us}")

        self._spawn_applet()

    def _spawn_applet(self):
        """Live recapture curve via ARTIQ's built-in plot_xy (no new applet code).
        Guarded so a headless run prints a note instead of failing."""
        cmd = ("${artiq_applet}plot_xy rr_curve_p --x rr_curve_t_us "
               "--error rr_curve_err")
        try:
            self.ccb.issue("create_applet", "release-recapture curve", cmd,
                           group="release-recapture")
        except Exception as e:
            print("Could not auto-create applet (no dashboard/ccb?):", e)

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        trap_ramp_end_amp = self.trap_ramp_start_amp   # safe init; real value = amp3

        for i in range(len(self.schedule_s)):
            self.core.break_realtime()
            t_rel_s = self.schedule_s[i]
            t_rel_mu = self.core.seconds_to_mu(t_rel_s)   # int64, exact whole ns

            # === Feedback (locks beams + FORT; amp3 = low-trap amplitude) ===
            prepare_for_feedback(self)
            amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
            trap_ramp_end_amp = amp3
            restore_after_feedback(self)

            # === Background (FORT off, beams on) + deterministic load ===
            c1_bg, c2_bg = background_measurement_dual(self, self.t_imaging_gate)
            bg = c1_bg + c2_bg
            load_atom_deterministic(self)

            # === Image 1: is an atom trapped? ===
            imaging_pgc(self)
            beams_off_for_imaging(self)
            c1a, c2a = spcm_image_dual(self, self.t_imaging_gate)
            imaging_pgc(self)
            beams_off_for_imaging(self)       # beams dark + repump blocked: release in the dark
            loaded = ((c1a + c2a) - bg) > self.atom_threshold

            c1b = 0
            c2b = 0
            survived = False
            release_amp = self.nominal_trapdepth
            if loaded:
                if self.release_at_low_trap:
                    ramp_trap_amp(self, self.trap_ramp_start_amp, trap_ramp_end_amp,
                                  self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)
                    release_amp = trap_ramp_end_amp

                # === RELEASE and RECAPTURE ===
                if t_rel_mu > 0:
                    self.dds_FORT.sw.off()        # trap OFF: free flight
                    delay_mu(t_rel_mu)
                    self.dds_FORT.sw.on()         # trap ON: recapture
                delay(1 * ms)                     # settle (unbound atoms leave)

                if self.release_at_low_trap:
                    ramp_trap_amp(self, trap_ramp_end_amp, self.trap_ramp_start_amp,
                                  self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)

                # === Image 2: was it recaptured? ===
                imaging_pgc(self)
                beams_off_for_imaging(self)
                c1b, c2b = spcm_image_dual(self, self.t_imaging_gate)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                survived = ((c1b + c2b) - bg) > self.atom_threshold

            self._log(t_rel_s, 1 if loaded else 0, 1 if survived else 0,
                      c1_bg, c2_bg, c1a, c2a, c1b, c2b, release_amp)

            reset_to_mot(self)

        write_results(self)

    @rpc(flags={"async"})
    def _log(self, t_rel_s, loaded, survived, c1_bg, c2_bg, c1a, c2a, c1b, c2b,
             release_amp):
        # Exact snapped value in us (t_rel_s came from whole ns in prepare).
        t_us = round(t_rel_s * 1e9) / 1000.0
        self.append_to_dataset("rr_t_release_us", t_us)
        self.append_to_dataset("rr_loaded", loaded)
        self.append_to_dataset("rr_survived", survived)
        self.append_to_dataset("rr_bg_spcm1", c1_bg)
        self.append_to_dataset("rr_bg_spcm2", c2_bg)
        self.append_to_dataset("rr_shot1_spcm1", c1a)
        self.append_to_dataset("rr_shot1_spcm2", c2a)
        self.append_to_dataset("rr_shot2_spcm1", c1b)
        self.append_to_dataset("rr_shot2_spcm2", c2b)
        self.append_to_dataset("rr_release_amp", release_amp)

        # Live curve: recaptured / loaded per release time.
        if loaded and t_us in self._agg:
            self._agg[t_us][1] += 1
            self._agg[t_us][0] += survived
            ps, es = [], []
            for us in self.release_us:
                k, n = self._agg[us]
                p = k / n if n else 0.0
                ps.append(p)
                es.append((p * (1.0 - p) / n) ** 0.5 if n else 0.0)
            self.set_dataset("rr_curve_p", ps, broadcast=True)
            self.set_dataset("rr_curve_err", es, broadcast=True)

        done = len(self.get_dataset("rr_loaded"))
        self.set_dataset("rr_progress", done, broadcast=True)
        # Incremental HDF5 save after every full sweep (a crash keeps prior sweeps).
        if done % self.n_points == 0:
            self._incremental_save()
            print(f"  [{done}/{self.n_trials}] sweep {done // self.n_points} saved")

    def _incremental_save(self):
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
        except Exception as e:
            print("[incremental save]", e)
