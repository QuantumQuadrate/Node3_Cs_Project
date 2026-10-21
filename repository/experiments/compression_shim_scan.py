"""Compression-shim 4D scan: loading probability & release-recapture survival.

Scans the four compression-stage shim fields and, at each grid point, runs many
trials of:  deterministic load -> image (trapped?) -> RELEASE & RECAPTURE ->
image (survived?).

Release-and-recapture is a temperature probe: the FORT is switched OFF for a
short time `t_release` (default 100 us), during which the atom flies freely --
hot atoms travel far and are NOT recaptured when the trap switches back on. So
the survival probability after R&R is a proxy for how COLD / well-trapped the
atom is. Scanning the compression shims and maximizing R&R survival finds the
bias field that gives the coldest, best-held atoms.

Per trial we record:
  - loaded   : 1 if the first image confirms a trapped atom, else 0
  - survived : 1 if the atom is still there AFTER release-and-recapture, else 0
               (only meaningful/collected when loaded == 1; 0 otherwise)

SAFETY BOUNDS (hard-limited in the dashboard AND clamped in prepare):
  shim_X_compression : -0.2 .. 0.6
  shim_Y_compression : -0.2 .. 0.4
  shim_Z1_compression: 4.12 .. 4.44
  shim_Z2_compression: 4.12 .. 4.44

The loader reads exp.shim_*_compression, so we reassign those attributes per grid
point (not kernel_invariants) and reuse the existing load_atom_deterministic.

Data saved per trial (all broadcast): cap_X, cap_Y, cap_Z1, cap_Z2, cap_loaded,
cap_survived. Analysis: analysis/analyze_compression_shim_scan.py.
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


# Hard safety limits (also used as NumberValue min/max below).
X_LO, X_HI = -0.2, 0.8
Y_LO, Y_HI = -0.2, 1.4
Z1_LO, Z1_HI = 4.12, 4.44
Z2_LO, Z2_HI = 4.12, 4.44


class CompressionShimScan(EnvExperiment):
    """Compression Shim Scan (R&R survival)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("n_measurements",
                              NumberValue(100, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")
        # Release-and-recapture free-flight time (FORT off). Entered in us.
        self.setattr_argument("t_release_us",
                              NumberValue(100, min=1, max=100000, step=1, ndecimals=1),
                              group="Release & Recapture")

        # --- 4D scan ranges (min/max are clamped to the safety bounds) ---
        self.setattr_argument("X_min", NumberValue(-0.2, min=X_LO, max=X_HI, ndecimals=3), group="Scan X")
        self.setattr_argument("X_max", NumberValue(0.8,  min=X_LO, max=X_HI, ndecimals=3), group="Scan X")
        self.setattr_argument("X_step", NumberValue(0.2, min=0.001, max=1.0, ndecimals=3), group="Scan X")

        self.setattr_argument("Y_min", NumberValue(-0.2, min=Y_LO, max=Y_HI, ndecimals=3), group="Scan Y")
        self.setattr_argument("Y_max", NumberValue(1.4,  min=Y_LO, max=Y_HI, ndecimals=3), group="Scan Y")
        self.setattr_argument("Y_step", NumberValue(0.15, min=0.001, max=1.0, ndecimals=3), group="Scan Y")

        self.setattr_argument("Z1_min", NumberValue(4.12, min=Z1_LO, max=Z1_HI, ndecimals=3), group="Scan Z1")
        self.setattr_argument("Z1_max", NumberValue(4.44, min=Z1_LO, max=Z1_HI, ndecimals=3), group="Scan Z1")
        self.setattr_argument("Z1_step", NumberValue(0.08, min=0.001, max=1.0, ndecimals=3), group="Scan Z1")

        self.setattr_argument("Z2_min", NumberValue(4.12, min=Z2_LO, max=Z2_HI, ndecimals=3), group="Scan Z2")
        self.setattr_argument("Z2_max", NumberValue(4.44, min=Z2_LO, max=Z2_HI, ndecimals=3), group="Scan Z2")
        self.setattr_argument("Z2_step", NumberValue(0.08, min=0.001, max=1.0, ndecimals=3), group="Scan Z2")

        # Rebalancer is created/configured by BaseExperiment.build()

    @staticmethod
    def _axis(vmin, vmax, step, lo, hi):
        """Inclusive scan list, clamped to [lo, hi] (safety) and ordered."""
        vmin = min(max(vmin, lo), hi)
        vmax = min(max(vmax, lo), hi)
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        step = abs(step) if step else (vmax - vmin or 1.0)
        n = int(round((vmax - vmin) / step)) + 1
        vals = [min(max(vmin + i * step, lo), hi) for i in range(max(n, 1))]
        out = []
        for v in vals:
            if not out or abs(v - out[-1]) > 1e-9:
                out.append(v)
        return out

    def prepare(self):
        # High-FORT-depth loading profile (same as atom_load_deterministic_test).
        self.base.prepare(feedback_profile="FORT_loading")

        # Release time (us -> seconds for the kernel delay()).
        self.t_release = float(self.t_release_us) * 1e-6

        self.scan_X = self._axis(self.X_min, self.X_max, self.X_step, X_LO, X_HI)
        self.scan_Y = self._axis(self.Y_min, self.Y_max, self.Y_step, Y_LO, Y_HI)
        self.scan_Z1 = self._axis(self.Z1_min, self.Z1_max, self.Z1_step, Z1_LO, Z1_HI)
        self.scan_Z2 = self._axis(self.Z2_min, self.Z2_max, self.Z2_step, Z2_LO, Z2_HI)

        self.n_points = (len(self.scan_X) * len(self.scan_Y)
                         * len(self.scan_Z1) * len(self.scan_Z2))
        self.n_shots_total = self.n_points * int(self.n_measurements)

        # Per-trial result datasets
        for key in ("cap_X", "cap_Y", "cap_Z1", "cap_Z2", "cap_loaded",
                    "cap_survived", "cap_load_time_s"):
            self.set_dataset(key, [], broadcast=True)
        self.set_dataset("cap_progress", 0, broadcast=True)

        # Metadata so the .h5 is self-describing
        self.set_dataset("cap_scan_X", self.scan_X, broadcast=True, persist=True)
        self.set_dataset("cap_scan_Y", self.scan_Y, broadcast=True, persist=True)
        self.set_dataset("cap_scan_Z1", self.scan_Z1, broadcast=True, persist=True)
        self.set_dataset("cap_scan_Z2", self.scan_Z2, broadcast=True, persist=True)
        self.set_dataset("cap_n_measurements", int(self.n_measurements), broadcast=True, persist=True)
        self.set_dataset("cap_n_points", self.n_points, broadcast=True, persist=True)
        self.set_dataset("cap_t_release_us", float(self.t_release_us), broadcast=True, persist=True)

        print(f"CompressionShimScan (R&R survival): grid "
              f"{len(self.scan_X)}x{len(self.scan_Y)}x{len(self.scan_Z1)}x{len(self.scan_Z2)} "
              f"= {self.n_points} points x {int(self.n_measurements)} trials "
              f"= {self.n_shots_total} trials")
        print(f"  release time = {self.t_release_us:.1f} us")
        print(f"  X  = {self.scan_X}")
        print(f"  Y  = {self.scan_Y}")
        print(f"  Z1 = {self.scan_Z1}")
        print(f"  Z2 = {self.scan_Z2}")

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        # Loop order: Z2 outermost so each completed Z2 gives a full 3D (X,Y,Z1)
        # map -- matches the analysis (one 3D map per Z2).
        for z2 in self.scan_Z2:
            for z1 in self.scan_Z1:
                for y in self.scan_Y:
                    for x in self.scan_X:
                        # Reassign the compression shims the loader reads.
                        self.shim_X_compression = x
                        self.shim_Y_compression = y
                        self.shim_Z1_compression = z1
                        self.shim_Z2_compression = z2

                        # One 5-channel feedback per grid point (lock FORT/beams).
                        self.core.break_realtime()
                        prepare_for_feedback(self)
                        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                        restore_after_feedback(self)

                        for m in range(int(self.n_measurements)):
                            self.core.break_realtime()

                            # Background (FORT off, beams on, no atom)
                            count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                            bg = count1_bg + count2_bg

                            # Deterministic load (leaves atom PGC-cooled, FORT on)
                            (success, n_tries, n_fb, s1, s2,
                             first_unloaded, t_load) = load_atom_deterministic(self)

                            # --- Image 1: is an atom actually trapped? ---
                            imaging_pgc(self)
                            beams_off_for_imaging(self)
                            delay(self.t_SPCM_exposure)
                            c1a, c2a = spcm_image_dual(self, self.t_imaging_gate)
                            imaging_pgc(self)
                            beams_off_for_imaging(self)
                            loaded = ((c1a + c2a) - bg) > self.atom_threshold

                            survived = False
                            if loaded:
                                # --- Release and recapture ---
                                self.dds_FORT.sw.off()      # trap OFF (free flight)
                                delay(self.t_release)
                                self.dds_FORT.sw.on()       # trap ON (recapture)
                                delay(1 * ms)               # settle before imaging

                                # --- Image 2: did the atom survive? ---
                                imaging_pgc(self)
                                beams_off_for_imaging(self)
                                delay(self.t_SPCM_exposure)
                                c1b, c2b = spcm_image_dual(self, self.t_imaging_gate)
                                imaging_pgc(self)
                                beams_off_for_imaging(self)
                                survived = ((c1b + c2b) - bg) > self.atom_threshold

                            loaded_int = 1 if loaded else 0
                            survived_int = 1 if survived else 0
                            # t_load = loader's time from poll-start to detection.
                            self._log(x, y, z1, z2, loaded_int, survived_int, t_load)

                            reset_to_mot(self)

        write_results(self)

    @rpc(flags={"async"})
    def _log(self, x, y, z1, z2, loaded, survived, load_time_s):
        self.append_to_dataset("cap_X", x)
        self.append_to_dataset("cap_Y", y)
        self.append_to_dataset("cap_Z1", z1)
        self.append_to_dataset("cap_Z2", z2)
        self.append_to_dataset("cap_loaded", loaded)
        self.append_to_dataset("cap_survived", survived)
        self.append_to_dataset("cap_load_time_s", load_time_s)

        n = len(self.get_dataset("cap_loaded"))
        self.set_dataset("cap_progress", n, broadcast=True)
        if n % 25 == 0:
            self._incremental_save()
            print(f"  [{n}/{self.n_shots_total}] "
                  f"X={x:.3f} Y={y:.3f} Z1={z1:.3f} Z2={z2:.3f}  "
                  f"loaded={loaded} survived={survived} t_load={load_time_s*1e3:.0f}ms")

    def _incremental_save(self):
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
        except Exception as e:
            print("[incremental save]", e)
