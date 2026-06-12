"""Single-atom IMAGING-condition scan at the LOW trap (release-and-recapture + histogram).

Finds the cooling-DP frequency + amplitude that image a single Cs atom best when the
image is taken at the LOW (shallow) FORT, using a dedicated low-trap gate
(t_imaging_gate_lowtrap, longer than the high-trap gate for enough signal). Goal:
remove the in-cycle trap ramp up/down from parity -- ramp DOWN
once after loading, then image at the low trap (the high trap is used only for loading).
The FORT light-shifts the imaging transition ~proportionally to trap intensity, so the
optimal imaging detuning/intensity at the low trap differ from f_cooling_imaging /
amp_cooling_imaging (tuned for the nominal trap).

Per scan point (f, a, release), fresh atom:
  1. Load + confirm at the HIGH trap (standard loader; loaded = counts > atom_threshold).
  2. Ramp trap DOWN to amp3 (the low trap).
  3. (optional) low-trap PGC pre-cool (use_pgc_precool, default OFF).
  4. shot1 = image at the LOW trap with the scanned (f, a), low-trap gate (atom present).
  5. release-and-recapture at the low trap (FORT off release us, back on).
  6. shot2 = image again at (f, a) (heating/survival probe).
  7. long release (~1 ms) to EJECT the atom, then bg = image at (f, a) with FORT OFF
     (the no-atom reference under the identical imaging condition).
  8. ramp trap UP (restore the high trap for the next load).

Two metrics (computed in analyze_imaging_lowtrap):
  - Heating: survival (shot2 vs shot1) at release=0 ("survives two low-trap images") and at
    release_us (temperature sensitivity). Higher = less heating.
  - Separation / detection fidelity: histogram of shot1 (atom) vs bg (no atom) per (f, a)
    -> a double-Gaussian threshold and fidelity = 0.5*(P_detect + P_reject).

CRITICAL: imaging light reaches the atom ONLY via the MOT fiber AOMs. Every low-trap
image is wrapped (_img) with B~0 (shim_off) + turn_fiber_aom_on + repump; without it
shot ~= bg (false null). Reuses the machinery built for pgc_lowtrap_scan.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    spcm_image_dual, imaging_pgc, imaging_pgc_low_trap, beams_off_for_imaging,
    set_qaxis, ramp_trap_amp, turn_fiber_aom_on, reset_mw_scan,
    image_low_trap,
)
from utilities.write_results import write_results


class ImagingLowTrapScan(EnvExperiment):
    """Imaging_LowTrap_Scan"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Imaging-DP frequency scan (Hz, displayed MHz), centred on f_cooling_imaging ---
        self.setattr_argument("f_img_min",
                              NumberValue(110e6, min=50e6, max=200e6, unit="MHz", ndecimals=4),
                              group="Imaging scan")
        self.setattr_argument("f_img_max",
                              NumberValue(130e6, min=50e6, max=200e6, unit="MHz", ndecimals=4),
                              group="Imaging scan")
        self.setattr_argument("f_img_step",
                              NumberValue(2e6, min=0.05e6, max=50e6, unit="MHz", ndecimals=4),
                              group="Imaging scan")
        # --- Imaging-DP amplitude scan ---
        self.setattr_argument("amp_img_min",
                              NumberValue(0.20, min=0.0, max=1.0, ndecimals=3),
                              group="Imaging scan")
        self.setattr_argument("amp_img_max",
                              NumberValue(0.40, min=0.0, max=1.0, ndecimals=3),
                              group="Imaging scan")
        self.setattr_argument("amp_img_step",
                              NumberValue(0.02, min=0.001, max=1.0, ndecimals=3),
                              group="Imaging scan")

        # --- Heating probe: 2-point release-and-recapture (0.0 is always added). Set
        #     min=max=0 for a pure double-image survival check; a range = temperature curve. ---
        self.setattr_argument("release_min",
                              NumberValue(0.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Heating probe")
        self.setattr_argument("release_max",
                              NumberValue(15.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Heating probe")
        self.setattr_argument("release_step",
                              NumberValue(15.0, min=0.1, max=10000.0, ndecimals=1),
                              group="Heating probe")

        # --- Long release (ms) to EJECT the atom before the no-atom background shot ---
        self.setattr_argument("eject_release_ms",
                              NumberValue(1.0, min=0.0, max=100.0, ndecimals=3),
                              group="Background")

        # --- Optional low-trap PGC pre-cool before shot1 (matches the parity-restructure
        #     state; default OFF to isolate the imaging physics) ---
        self.setattr_argument("use_pgc_precool",
                              BooleanValue(False), group="Atom state")

        # --- Repetitions (binomial statistics) ---
        self.setattr_argument("n_loops",
                              NumberValue(50, min=1, max=600000, ndecimals=0, step=1),
                              group="Experiment")
        self.setattr_argument("skip_on_no_atom",
                              BooleanValue(True), group="Experiment")

        # --- Adiabatic trap ramp (follows the parity experiments) ---
        self.setattr_argument("trap_ramp_start_amp",
                              NumberValue(0.24, min=0.0, max=0.3, ndecimals=3),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_steps",
                              NumberValue(200, min=0, max=100000, ndecimals=0),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_dt_us",
                              NumberValue(50, min=0.001, max=1000, ndecimals=3),
                              group="Adiabatic")

        # All other physics (t_imaging_gate, atom_threshold, shims, f_cooling_imaging,
        # f_cooling_PGC, the deep-PGC baselines) load from datasets via BaseExperiment;
        # the rebalancer is created there too.

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # --- Build the 2-D (frequency, amplitude) scan lists ---
        vf, sf = float(self.f_img_min), float(self.f_img_step)
        nf = int(round((float(self.f_img_max) - vf) / sf)) + 1
        self.f_list = [vf + i * sf for i in range(nf)]

        va, sa = float(self.amp_img_min), float(self.amp_img_step)
        na = int(round((float(self.amp_img_max) - va) / sa)) + 1
        self.a_list = [va + i * sa for i in range(na)]

        # --- Release-and-recapture list (us). 0.0 ALWAYS included (the "survives two
        # 4 ms images" reference); the >0 point(s) give the temperature sensitivity. ---
        vr, sr = float(self.release_min), float(self.release_step)
        nr = (int(round((float(self.release_max) - vr) / sr)) + 1) if sr > 0.0 else 1
        if nr < 1:
            nr = 1
        rel_pts = [round(vr + i * sr, 3) for i in range(nr)]
        self.release_list = sorted(set([0.0] + rel_pts))

        # --- Self-describing snapshot ---
        snapshot_params = {
            "f_img_min": float(self.f_img_min),
            "f_img_max": float(self.f_img_max),
            "f_img_step": float(self.f_img_step),
            "amp_img_min": float(self.amp_img_min),
            "amp_img_max": float(self.amp_img_max),
            "amp_img_step": float(self.amp_img_step),
            "release_min": float(self.release_min),
            "release_max": float(self.release_max),
            "release_step": float(self.release_step),
            "eject_release_ms": float(self.eject_release_ms),
            "use_pgc_precool": int(self.use_pgc_precool),
            "n_loops": int(self.n_loops),
            "skip_on_no_atom": int(self.skip_on_no_atom),
            # baselines (the nominal-trap imaging the scan is re-optimizing)
            "f_cooling_imaging": self.f_cooling_imaging,
            "amp_cooling_imaging": self.amp_cooling_imaging,
            "f_cooling_PGC": self.f_cooling_PGC,
            "amp_cooling_PGC": self.amp_cooling_PGC,
            "t_imaging_gate": self.t_imaging_gate,
            "t_imaging_gate_lowtrap": self.t_imaging_gate_lowtrap,
            "f_cooling_imaging_lowtrap": self.f_cooling_imaging_lowtrap,
            "amp_cooling_imaging_lowtrap": self.amp_cooling_imaging_lowtrap,
            "atom_threshold": self.atom_threshold,
            "atom_threshold_lowtrap": self.atom_threshold_lowtrap,
            "trap_ramp_start_amp": self.trap_ramp_start_amp,
            "trap_ramp_steps": self.trap_ramp_steps,
            "trap_ramp_dt_us": self.trap_ramp_dt_us,
            "shim_X_off": self.shim_X_off,
            "shim_Y_off": self.shim_Y_off,
            "shim_Z1_off": self.shim_Z1_off,
            "shim_Z2_off": self.shim_Z2_off,
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Result datasets (per point) ---
        for k in ("img_f", "img_amp", "img_release_us",
                  "img_loadbg_spcm1", "img_loadbg_spcm2",
                  "img_shot1_spcm1", "img_shot1_spcm2",
                  "img_shot2_spcm1", "img_shot2_spcm2",
                  "img_bg_spcm1", "img_bg_spcm2",
                  "img_loaded", "img_survived", "img_loop_index"):
            self.set_dataset(k, [], broadcast=True, persist=True)
        self.set_dataset("img_scan_complete", 0, broadcast=True)

        # --- Metadata ---
        self.set_dataset("img_n_freq_points", len(self.f_list), persist=True, broadcast=True)
        self.set_dataset("img_n_amp_points", len(self.a_list), persist=True, broadcast=True)
        self.set_dataset("img_n_release_points", len(self.release_list), persist=True, broadcast=True)
        self.set_dataset("img_n_loops", int(self.n_loops), persist=True, broadcast=True)
        self.set_dataset("img_best_f", 0.0, persist=True, broadcast=True)
        self.set_dataset("img_best_amp", 0.0, persist=True, broadcast=True)
        self.set_dataset("img_best_fidelity", 0.0, persist=True, broadcast=True)

        self._n_pts_per_round = max(
            len(self.f_list) * len(self.a_list) * len(self.release_list), 1)
        print(f"ImagingLowTrapScan: {len(self.f_list)} freq x {len(self.a_list)} amp x "
              f"{len(self.release_list)} release x {int(self.n_loops)} loops | "
              f"release={self.release_list} us | eject={float(self.eject_release_ms):g} ms | "
              f"precool={int(self.use_pgc_precool)} | low-trap gate={self.t_imaging_gate_lowtrap} s")
        print(f"  baseline imaging (nominal trap): f_cooling_imaging={self.f_cooling_imaging/1e6:.4f} MHz, "
              f"amp_cooling_imaging={self.amp_cooling_imaging:.3f}  "
              f"(centre the scan here)")

    # =====================================================================
    #  KERNEL HELPERS
    # =====================================================================

    @kernel
    def _img(self, f, a):
        """Image at the scanned (f, a) on the CURRENT trap depth. Opens the fiber-AOM
        imaging path then closes the beams. ASSUMES B~0 (shim_off) is already set by
        the caller (done once per point, since nothing in the imaging block changes the
        shims). Returns (count1, count2)."""
        turn_fiber_aom_on(self)                            # imaging light path ON (must-fix)
        delay(1 * us)
        c1, c2 = image_low_trap(self, f, a, self.t_imaging_gate_lowtrap)   # repump managed inside
        beams_off_for_imaging(self)                        # cooling + repump OFF after the gate
        return c1, c2

    @kernel
    def _pgc_precool(self):
        """Optional low-trap PGC pre-cool (use_pgc_precool). Same beam path as _img but
        a PGC pulse instead of a gated image (uses f_cooling_PGC/amp_cooling_PGC)."""
        turn_fiber_aom_on(self)
        delay(1 * us)
        self.dds_repump.sw.on()
        delay(1 * us)
        self.ttl_repump_switch.off()
        imaging_pgc_low_trap(self)
        beams_off_for_imaging(self)

    # =====================================================================
    #  RUN
    # =====================================================================

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_loops)
        eject = self.eject_release_ms

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)
            # === Feedback (gives amp3 = low trap) ===
            prepare_for_feedback(self)
            amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
            trap_end = amp3
            restore_after_feedback(self)

            for ifr in range(len(self.f_list)):
                f = self.f_list[ifr]

                for ia in range(len(self.a_list)):
                    a = self.a_list[ia]

                    for irel in range(len(self.release_list)):
                        rel = self.release_list[irel]
                        # === Load + confirm at the HIGH trap (standard loader) ===
                        loadbg1, loadbg2 = background_measurement_dual(self, self.t_imaging_gate)
                        load_atom_deterministic(self)
                        imaging_pgc(self)
                        beams_off_for_imaging(self)
                        count1, count2 = spcm_image_dual(self, self.t_imaging_gate)
                        imaging_pgc(self)
                        beams_off_for_imaging(self)

                        loaded = (count1 + count2 - loadbg1 - loadbg2) > self.atom_threshold

                        s1a = 0
                        s1b = 0
                        s2a = 0
                        s2b = 0
                        bga = 0
                        bgb = 0
                        if loaded or (not self.skip_on_no_atom):
                            # --- ramp DOWN to the low trap (ONCE) ---
                            ramp_trap_amp(self, self.trap_ramp_start_amp, trap_end,
                                          self.trap_ramp_steps, self.trap_ramp_dt_us)
                            delay(1 * ms)

                            # --- B~0 for the whole imaging block (nothing below moves shims) ---
                            set_qaxis(self, self.shim_X_off, self.shim_Y_off,
                                      self.shim_Z1_off, self.shim_Z2_off)

                            # --- optional low-trap PGC pre-cool ---
                            if self.use_pgc_precool:
                                self._pgc_precool()

                            # --- shot1: image at the low trap (atom present) ---
                            s1a, s1b = self._img(f, a)

                            # --- heating probe: release-and-recapture at the low trap ---
                            self.core.break_realtime()
                            if rel > 0.0:
                                self.dds_FORT.sw.off()
                                delay(rel * us)
                                self.dds_FORT.sw.on()
                            delay(1 * ms)

                            # --- shot2: image again (survival/heating) ---
                            s2a, s2b = self._img(f, a)

                            # --- long release to EJECT the atom, then bg with FORT OFF ---
                            self.core.break_realtime()
                            self.dds_FORT.sw.off()
                            delay(eject)
                            bga, bgb = self._img(f, a)          # no atom, no trap
                            self.dds_FORT.sw.on()

                            # --- restore the HIGH trap for the next load ---
                            self.core.break_realtime()
                            ramp_trap_amp(self, trap_end, self.trap_ramp_start_amp,
                                          self.trap_ramp_steps, self.trap_ramp_dt_us)
                            delay(1 * ms)

                        # === Save ===
                        self._save_point(f, a, rel,
                                         loadbg1, loadbg2,
                                         s1a, s1b, s2a, s2b, bga, bgb,
                                         (1 if loaded else 0), loop)

                        # === Reset for next point ===
                        self.core.break_realtime()
                        reset_mw_scan(self)

        self._summarize()
        write_results(self)

    # =====================================================================
    #  HOST-SIDE SAVE / SUMMARY
    # =====================================================================

    @rpc
    def _save_point(self, f, a, rel_us, loadbg1, loadbg2,
                    s1a, s1b, s2a, s2b, bga, bgb, loaded, loop_idx):
        """Append one point; survived flag (atom present in shot2, scanned bg subtracted);
        incremental HDF5 save once per full (freq x amp x release) round."""
        self.append_to_dataset("img_f", f)
        self.append_to_dataset("img_amp", a)
        self.append_to_dataset("img_release_us", rel_us)
        self.append_to_dataset("img_loadbg_spcm1", loadbg1)
        self.append_to_dataset("img_loadbg_spcm2", loadbg2)
        self.append_to_dataset("img_shot1_spcm1", s1a)
        self.append_to_dataset("img_shot1_spcm2", s1b)
        self.append_to_dataset("img_shot2_spcm1", s2a)
        self.append_to_dataset("img_shot2_spcm2", s2b)
        self.append_to_dataset("img_bg_spcm1", bga)
        self.append_to_dataset("img_bg_spcm2", bgb)
        self.append_to_dataset("img_loop_index", loop_idx)

        shot2 = s2a + s2b
        bg = bga + bgb
        survived = int((shot2 - bg) > self.atom_threshold) if loaded > 0 else 0
        self.append_to_dataset("img_loaded", loaded)
        self.append_to_dataset("img_survived", survived)

        if not hasattr(self, "_points_saved"):
            self._points_saved = 0
        self._points_saved += 1
        self.set_dataset("img_scan_complete", self._points_saved, broadcast=True)
        if self._points_saved % self._n_pts_per_round == 0:
            self._incremental_save()

    def _incremental_save(self):
        """Write current data to HDF5 mid-run (crash-safe per-round save)."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
            print(f"[incremental save] {self._points_saved} points saved to HDF5")
        except Exception as e:
            print(f"[incremental save] FAILED: {e}")

    @rpc
    def _summarize(self):
        """Per (f, a): survival@release=0 / @probe and the histogram gap = mean(shot1) -
        mean(bg) over loaded points. The REAL survival-gated fidelity is computed in
        analyze_imaging_lowtrap (double-Gaussian threshold). BLAS-free (sums only)."""
        fs = self.get_dataset("img_f")
        amps = self.get_dataset("img_amp")
        rels = self.get_dataset("img_release_us")
        loaded = self.get_dataset("img_loaded")
        survived = self.get_dataset("img_survived")
        s1a = self.get_dataset("img_shot1_spcm1")
        s1b = self.get_dataset("img_shot1_spcm2")
        bga = self.get_dataset("img_bg_spcm1")
        bgb = self.get_dataset("img_bg_spcm2")
        if len(fs) == 0:
            print("ImagingLowTrapScan: no data to summarize.")
            return

        surv = {}       # (f, a, rel) -> [n_survived, n_loaded]
        hist = {}       # (f, a) -> [sum_shot1, sum_bg, n]
        rel_set = set()
        for i in range(len(fs)):
            if loaded[i] <= 0:
                continue
            fa = (round(float(fs[i]), 0), round(float(amps[i]), 4))
            r = round(float(rels[i]), 1)
            rel_set.add(r)
            sk = (fa[0], fa[1], r)
            surv.setdefault(sk, [0, 0])
            surv[sk][0] += int(survived[i])
            surv[sk][1] += 1
            hist.setdefault(fa, [0.0, 0.0, 0])
            hist[fa][0] += float(s1a[i] + s1b[i])
            hist[fa][1] += float(bga[i] + bgb[i])
            hist[fa][2] += 1

        rel_probe = max(rel_set) if rel_set else 0.0
        print(f"ImagingLowTrapScan: survival@0us / @{rel_probe:g}us and "
              f"gap=mean(shot1)-mean(bg) per (f_MHz, amp):")
        best_fa = None
        best_gap = -1.0e18
        for fa in sorted(hist.keys()):
            h = hist[fa]
            gap = (h[0] - h[1]) / h[2] if h[2] > 0 else float("nan")
            d0 = surv.get((fa[0], fa[1], 0.0))
            dp = surv.get((fa[0], fa[1], rel_probe))
            s0 = (d0[0] / d0[1]) if (d0 and d0[1] > 0) else float("nan")
            sp = (dp[0] / dp[1]) if (dp and dp[1] > 0) else float("nan")
            print(f"  f={fa[0]/1e6:.4f} MHz  amp={fa[1]:.3f}  "
                  f"surv@0={s0:.3f}  surv@{rel_probe:g}={sp:.3f}  gap={gap:.1f}")
            if h[2] > 0 and gap > best_gap:
                best_gap = gap
                best_fa = fa

        if best_fa is not None:
            self.set_dataset("img_best_f", float(best_fa[0]), persist=True, broadcast=True)
            self.set_dataset("img_best_amp", float(best_fa[1]), persist=True, broadcast=True)
            print(f"ImagingLowTrapScan: highest gap = {best_gap:.1f} at "
                  f"f {best_fa[0]/1e6:.4f} MHz, amp {best_fa[1]:.3f} (gap != fidelity; run "
                  f"analyze_imaging_lowtrap for the survival-gated detection fidelity).")
