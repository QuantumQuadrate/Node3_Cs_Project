"""PGC-at-low-trap parameter scan (release-and-recapture).

Finds the cooling-DP frequency + amplitude that cool best when PGC is applied at
the LOW (excitation) trap depth -- a prerequisite for the parity speed-up that
PGC-cools at the low trap and only ramps up for imaging. The deep-PGC params
(f_cooling_deep_PGC / amp_cooling_deep_PGC) are tuned for the NOMINAL trap; the
FORT light-shifts the cooling transition ~proportionally to trap intensity, so at
the shallow trap the optimal detuning, intensity AND equilibration time differ. This
scans four axes (set any to a single point to fix it): cooling-DP frequency,
amplitude, PGC duration t_PGC, and release time. Intended use after the (freq, amp)
ridge is known: FIX (freq, amp) and scan (t_PGC x release) -- the release scan at
each t_PGC gives an actual temperature, so T(t_PGC) reveals the cooling 'knee' = the
shortest PGC time still reaching the cold floor. release=0 is always included as the
retention/survival reference (separates 'cold' from 'lost').

Per scan point (f, a, t_PGC, release), fresh atom:
  1. load + image (shot1).
  2. Q-axis ON, ramp trap DOWN to amp3, optical_pumping + excitation_pulse
     (x n_excite) to heat the atom at the low trap.
  3. shim OFF (B~0) + turn_fiber_aom_on + repump (so the PGC light actually
     reaches the atom) -> PGC at the low trap with (f, a) for t_PGC -> beams off.
     The trap is NOT ramped up for the PGC.
  4. release-and-recapture at the low trap (FORT off for release us, back on).
  5. ramp trap UP + image (shot2).

Figure of merit = recapture (shot2-bg)/(shot1-bg). At release=0 it is the survival
(did PGC keep the atom); at release>0 it is the temperature proxy (higher = colder).
Release-and-recapture is measured AT the low trap (before the up-ramp) so it reflects
the temperature at the conditions the parity restructure will actually use.

Reuses the machinery built for excitation_survival_scan.py / excitation_scan.py
(BaseExperiment, the load/shot1 pattern, _excite_drain, incremental HDF5 save).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    spcm_image_dual, imaging_pgc, beams_off_for_imaging,
    set_qaxis, optical_pumping, ramp_trap_amp, excitation_pulse,
    turn_fiber_aom_on, final_state_readout, reset_mw_scan,
)
from utilities.write_results import write_results


class PGCLowTrapScan(EnvExperiment):
    """PGC_LowTrap_Scan"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Cooling-DP PGC frequency scan (Hz), centred on the deep-PGC value ---
        self.setattr_argument("f_lowtrap_min",
                              NumberValue(120e6, min=50e6, max=200e6, unit="MHz", ndecimals=4),
                              group="PGC scan")
        self.setattr_argument("f_lowtrap_max",
                              NumberValue(135e6, min=50e6, max=200e6, unit="MHz", ndecimals=4),
                              group="PGC scan")
        self.setattr_argument("f_lowtrap_step",
                              NumberValue(1e6, min=0.05e6, max=50e6, unit="MHz", ndecimals=4),
                              group="PGC scan")
        # --- Cooling-DP PGC amplitude scan ---
        self.setattr_argument("amp_lowtrap_min",
                              NumberValue(0.28, min=0.0, max=1.0, ndecimals=3),
                              group="PGC scan")
        self.setattr_argument("amp_lowtrap_max",
                              NumberValue(0.42, min=0.0, max=1.0, ndecimals=3),
                              group="PGC scan")
        self.setattr_argument("amp_lowtrap_step",
                              NumberValue(0.02, min=0.001, max=1.0, ndecimals=3),
                              group="PGC scan")

        # --- PGC duration SCAN at the low trap (us). Set min=max to FIX t_PGC. ---
        self.setattr_argument("t_pgc_min",
                              NumberValue(200, min=1, max=200000, ndecimals=1),
                              group="PGC time scan")
        self.setattr_argument("t_pgc_max",
                              NumberValue(2000, min=1, max=200000, ndecimals=1),
                              group="PGC time scan")
        self.setattr_argument("t_pgc_step",
                              NumberValue(200, min=1, max=200000, ndecimals=1),
                              group="PGC time scan")

        # --- Release-and-recapture SCAN (temperature probe), us. release=0 is
        #     ALWAYS added as the retention/survival reference. Set min=max=0 for
        #     a pure retention check; a range gives the full recapture curve. ---
        self.setattr_argument("release_min",
                              NumberValue(0.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Release scan")
        self.setattr_argument("release_max",
                              NumberValue(40.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Release scan")
        self.setattr_argument("release_step",
                              NumberValue(5.0, min=0.1, max=10000.0, ndecimals=1),
                              group="Release scan")

        # --- How many excitation pulses to heat the atom (user's sketch: once) ---
        self.setattr_argument("n_excite",
                              NumberValue(1, min=0, max=8000, ndecimals=0, step=1),
                              group="Experiment")
        # --- Repetitions (binomial statistics) ---
        self.setattr_argument("n_loops",
                              NumberValue(50, min=1, max=600000, ndecimals=0, step=1),
                              group="Experiment")
        self.setattr_argument("skip_on_no_atom",
                              BooleanValue(True), group="Experiment")

        # --- Excitation timing (follows the parity experiments) ---
        self.setattr_argument("t_trap_off_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("t_spcm_collection_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("length_offset_mu",
                              NumberValue(12, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")

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

        # All other physics (f_excitation_pulse, exc_* offsets, shims,
        # atom_threshold, t_imaging_gate, the deep-PGC baselines, ...) load from
        # datasets via BaseExperiment; the rebalancer is created there too.

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # --- Build the 2-D (frequency, amplitude) scan lists ---
        vf, sf = float(self.f_lowtrap_min), float(self.f_lowtrap_step)
        nf = int(round((float(self.f_lowtrap_max) - vf) / sf)) + 1
        self.f_list = [vf + i * sf for i in range(nf)]

        va, sa = float(self.amp_lowtrap_min), float(self.amp_lowtrap_step)
        na = int(round((float(self.amp_lowtrap_max) - va) / sa)) + 1
        self.a_list = [va + i * sa for i in range(na)]

        # --- PGC-duration scan list (us). Single point if min==max. ---
        vt, st = float(self.t_pgc_min), float(self.t_pgc_step)
        nt = (int(round((float(self.t_pgc_max) - vt) / st)) + 1) if st > 0.0 else 1
        if nt < 1:
            nt = 1
        self.t_pgc_list = [vt + i * st for i in range(nt)]

        # --- Release-and-recapture scan list (us). 0.0 is ALWAYS included as the
        # retention/survival reference (separates 'cold' from 'lost'); the >0
        # points give the recapture curve = temperature proxy at each t_PGC. ---
        vr, sr = float(self.release_min), float(self.release_step)
        nr = (int(round((float(self.release_max) - vr) / sr)) + 1) if sr > 0.0 else 1
        if nr < 1:
            nr = 1
        rel_pts = [round(vr + i * sr, 3) for i in range(nr)]
        self.release_list = sorted(set([0.0] + rel_pts))

        # --- Self-describing snapshot ---
        snapshot_params = {
            "f_lowtrap_min": float(self.f_lowtrap_min),
            "f_lowtrap_max": float(self.f_lowtrap_max),
            "f_lowtrap_step": float(self.f_lowtrap_step),
            "amp_lowtrap_min": float(self.amp_lowtrap_min),
            "amp_lowtrap_max": float(self.amp_lowtrap_max),
            "amp_lowtrap_step": float(self.amp_lowtrap_step),
            "t_pgc_min": float(self.t_pgc_min),
            "t_pgc_max": float(self.t_pgc_max),
            "t_pgc_step": float(self.t_pgc_step),
            "release_min": float(self.release_min),
            "release_max": float(self.release_max),
            "release_step": float(self.release_step),
            "n_excite": int(self.n_excite),
            "n_loops": int(self.n_loops),
            "skip_on_no_atom": int(self.skip_on_no_atom),
            # baselines (deep-trap PGC) for reference
            "f_cooling_deep_PGC": self.f_cooling_deep_PGC,
            "amp_cooling_deep_PGC": self.amp_cooling_deep_PGC,
            "t_PGC_after_img": self.t_PGC_after_img,
            "f_excitation_pulse": self.f_excitation_pulse,
            "amp_excitation_pulse": self.amp_excitation_pulse,
            "t_trap_off_mu": self.t_trap_off_mu,
            "t_spcm_collection_mu": self.t_spcm_collection_mu,
            "length_offset_mu": self.length_offset_mu,
            "trap_ramp_start_amp": self.trap_ramp_start_amp,
            "trap_ramp_steps": self.trap_ramp_steps,
            "trap_ramp_dt_us": self.trap_ramp_dt_us,
            "shim_X_qaxis": self.shim_X_qaxis,
            "shim_Y_qaxis": self.shim_Y_qaxis,
            "shim_Z1_qaxis": self.shim_Z1_qaxis,
            "shim_Z2_qaxis": self.shim_Z2_qaxis,
            "shim_X_off": self.shim_X_off,
            "shim_Y_off": self.shim_Y_off,
            "shim_Z1_off": self.shim_Z1_off,
            "shim_Z2_off": self.shim_Z2_off,
            "atom_threshold": self.atom_threshold,
            "t_imaging_gate": self.t_imaging_gate,
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Result datasets (per point) ---
        for k in ("pgc_f", "pgc_amp", "pgc_t_pgc_us", "pgc_release_us",
                  "pgc_bg_spcm1", "pgc_bg_spcm2",
                  "pgc_shot1_spcm1", "pgc_shot1_spcm2",
                  "pgc_shot2_spcm1", "pgc_shot2_spcm2",
                  "pgc_atom_loaded", "pgc_atom_retained", "pgc_loop_index"):
            self.set_dataset(k, [], broadcast=True, persist=True)
        self.set_dataset("pgc_scan_complete", 0, broadcast=True)

        # --- Metadata ---
        self.set_dataset("pgc_n_freq_points", len(self.f_list), persist=True, broadcast=True)
        self.set_dataset("pgc_n_amp_points", len(self.a_list), persist=True, broadcast=True)
        self.set_dataset("pgc_n_tpgc_points", len(self.t_pgc_list), persist=True, broadcast=True)
        self.set_dataset("pgc_n_release_points", len(self.release_list), persist=True, broadcast=True)
        self.set_dataset("pgc_n_loops", int(self.n_loops), persist=True, broadcast=True)
        self.set_dataset("pgc_best_f", 0.0, persist=True, broadcast=True)
        self.set_dataset("pgc_best_amp", 0.0, persist=True, broadcast=True)
        self.set_dataset("pgc_best_tpgc", 0.0, persist=True, broadcast=True)

        self._n_pts_per_round = max(
            len(self.f_list) * len(self.a_list) * len(self.t_pgc_list)
            * len(self.release_list), 1)
        print(f"PGCLowTrapScan: {len(self.f_list)} freq x {len(self.a_list)} amp x "
              f"{len(self.t_pgc_list)} t_PGC x {len(self.release_list)} release x "
              f"{int(self.n_loops)} loops | t_PGC={self.t_pgc_list} us | "
              f"release={self.release_list} us | n_excite={int(self.n_excite)}")

    # =====================================================================
    #  KERNEL HELPERS
    # =====================================================================

    @kernel
    def _excite_drain(self):
        """One excitation attempt to heat the atom; photon clicks DISCARDED.
        Reads both SPCM timestamps INLINE (drains the gate, avoids the channel-39
        underflow) then runs parity's no-detect dead-time tail."""
        t_end1, t_end2, t0 = excitation_pulse(self)
        click1 = self.ttl_SPCM1.timestamp_mu(t_end1)   # drain inline (discarded)
        click2 = self.ttl_SPCM2.timestamp_mu(t_end2)   # drain inline (discarded)
        delay(5 * us)
        delay(2 * us)
        self.ttl_MW_switch.on()
        delay(1 * us)

    @kernel
    def _pgc_lowtrap(self, f, a, t_pgc_us):
        """PGC at the LOW trap with scanned cooling-DP (f, a). Trap is NOT ramped
        up. CRITICAL: the cooling light reaches the atom only through the MOT
        fiber AOMs (turn_fiber_aom_on) + repump must be on -- optical_pumping
        leaves them OFF, so without this the PGC does nothing. Ends with cooling
        OFF (beams_off_for_imaging) so the following release is light-free."""
        set_qaxis(self, self.shim_X_off, self.shim_Y_off,
                  self.shim_Z1_off, self.shim_Z2_off)      # B~0 for PGC
        delay(1 * ms)   # coil relexation time
        turn_fiber_aom_on(self)                            # PGC light path ON (must-fix)
        delay(1 * us)
        self.dds_repump.sw.on()
        delay(1 * us)
        self.ttl_repump_switch.off()                       # unblock repump
        self.dds_cooling_DP.set(frequency=f, amplitude=a)
        self.dds_cooling_DP.sw.on()
        delay(t_pgc_us * us)
        beams_off_for_imaging(self)                        # cooling + repump OFF before release

    @kernel
    def _release_lowtrap(self, rel_us):
        """Release-and-recapture AT the low trap (no light, FORT off->on).
        rel_us==0 -> no release (FORT stays on). Sequential sw.off/delay/sw.on
        with break_realtime first; the short release sits in ample slack."""
        self.core.break_realtime()
        if rel_us > 0.0:
            self.dds_FORT.sw.off()
            delay(rel_us * us)
            self.dds_FORT.sw.on()
        delay(1 * ms)                                      # settle at low trap before up-ramp

    # =====================================================================
    #  RUN
    # =====================================================================

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_loops)
        n_excite = int(self.n_excite)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            # === Feedback ===
            prepare_for_feedback(self)
            amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
            trap_end = amp3
            restore_after_feedback(self)
            
            for ifr in range(len(self.f_list)):
                f = self.f_list[ifr]

                for ia in range(len(self.a_list)):
                    a = self.a_list[ia]

                    for it in range(len(self.t_pgc_list)):
                        t_pgc = self.t_pgc_list[it]

                        for irel in range(len(self.release_list)):
                            rel = self.release_list[irel]



                            # === Background + load + confirm image (shot1) ===
                            count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                            load_atom_deterministic(self)
                            imaging_pgc(self)
                            beams_off_for_imaging(self)
                            count1, count2 = spcm_image_dual(self, self.t_imaging_gate)
                            imaging_pgc(self)
                            beams_off_for_imaging(self)

                            loaded = (count1 + count2 - count1_bg - count2_bg) > self.atom_threshold

                            count1_post = 0
                            count2_post = 0
                            if loaded or (not self.skip_on_no_atom):
                                # --- heat the atom at the low trap ---
                                set_qaxis(self, self.shim_X_qaxis, self.shim_Y_qaxis,
                                          self.shim_Z1_qaxis, self.shim_Z2_qaxis)   # Q-axis ON
                                ramp_trap_amp(self, self.trap_ramp_start_amp, trap_end,
                                              self.trap_ramp_steps, self.trap_ramp_dt_us)   # DOWN
                                for _ in range(n_excite):
                                    self.core.break_realtime()
                                    optical_pumping(self)
                                    self._excite_drain()

                                # --- PGC at the low trap (scanned f, a, t_PGC) ---
                                self._pgc_lowtrap(f, a, t_pgc)

                                # --- release-and-recapture at the low trap (rel us) ---
                                self._release_lowtrap(rel)

                                # --- ramp UP + image (shot2) ---
                                self.core.break_realtime()
                                ramp_trap_amp(self, trap_end, self.trap_ramp_start_amp,
                                              self.trap_ramp_steps, self.trap_ramp_dt_us)   # UP
                                delay(1 * ms)
                                count1_post, count2_post = final_state_readout(self)

                            # === Save ===
                            self._save_point(f, a, t_pgc, rel,
                                             count1_bg, count2_bg,
                                             count1, count2,
                                             count1_post, count2_post,
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
    def _save_point(self, f, a, t_pgc_us, rel_us,
                    bg_s1, bg_s2, shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2, loaded, loop_idx):
        """Append one point; threshold-based loaded/retained; incremental HDF5
        save once per full (freq x amp x t_PGC) round."""
        self.append_to_dataset("pgc_f", f)
        self.append_to_dataset("pgc_amp", a)
        self.append_to_dataset("pgc_t_pgc_us", t_pgc_us)
        self.append_to_dataset("pgc_release_us", rel_us)
        self.append_to_dataset("pgc_bg_spcm1", bg_s1)
        self.append_to_dataset("pgc_bg_spcm2", bg_s2)
        self.append_to_dataset("pgc_shot1_spcm1", shot1_s1)
        self.append_to_dataset("pgc_shot1_spcm2", shot1_s2)
        self.append_to_dataset("pgc_shot2_spcm1", shot2_s1)
        self.append_to_dataset("pgc_shot2_spcm2", shot2_s2)
        self.append_to_dataset("pgc_loop_index", loop_idx)

        bg = bg_s1 + bg_s2
        shot2 = shot2_s1 + shot2_s2
        retained = int((shot2 - bg) > self.atom_threshold) if loaded > 0 else 0
        self.append_to_dataset("pgc_atom_loaded", loaded)
        self.append_to_dataset("pgc_atom_retained", retained)

        if not hasattr(self, "_points_saved"):
            self._points_saved = 0
        self._points_saved += 1
        self.set_dataset("pgc_scan_complete", self._points_saved, broadcast=True)
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
        """Per (f, a, t_PGC): retention at release=0 (did PGC keep the atom) AND
        recapture at the largest release (the temperature proxy; higher = colder).
        Across t_PGC, recapture rising to a plateau = the cooling 'knee' (shortest
        PGC time reaching the cold floor) -- see analyze_pgc_time_release for the
        full T(t_PGC) curve. BLAS-free (sums only)."""
        fs = self.get_dataset("pgc_f")
        amps = self.get_dataset("pgc_amp")
        tps = self.get_dataset("pgc_t_pgc_us")
        rels = self.get_dataset("pgc_release_us")
        loaded = self.get_dataset("pgc_atom_loaded")
        retained = self.get_dataset("pgc_atom_retained")
        if len(fs) == 0:
            print("PGCLowTrapScan: no data to summarize.")
            return

        tab = {}        # (f, a, t_pgc) -> {rel -> [n_retained, n_loaded]}
        rel_set = set()
        for i in range(len(fs)):
            if loaded[i] <= 0:
                continue
            key = (round(float(fs[i]), 0), round(float(amps[i]), 4),
                   round(float(tps[i]), 1))
            r = round(float(rels[i]), 1)
            rel_set.add(r)
            tab.setdefault(key, {}).setdefault(r, [0, 0])
            tab[key][r][0] += int(retained[i])
            tab[key][r][1] += 1

        rel_probe = max(rel_set) if rel_set else 0.0   # the largest release = temperature proxy
        print(f"PGCLowTrapScan: retention@0us and recapture@{rel_probe:g}us per "
              f"(f_MHz, amp, t_PGC_us):")
        best_key = None
        best_p = -1.0
        for key in sorted(tab.keys()):
            d0 = tab[key].get(0.0)
            dp = tab[key].get(rel_probe)
            p0 = (d0[0] / d0[1]) if (d0 and d0[1] > 0) else float("nan")
            pp = (dp[0] / dp[1]) if (dp and dp[1] > 0) else float("nan")
            print(f"  f={key[0]/1e6:.4f} MHz  amp={key[1]:.3f}  t_PGC={key[2]:.0f} us  "
                  f"retention@0={p0:.3f}  recapture@{rel_probe:g}={pp:.3f}")
            if dp and dp[1] > 0 and pp > best_p:
                best_p = pp
                best_key = key

        if best_key is not None:
            self.set_dataset("pgc_best_f", float(best_key[0]), persist=True, broadcast=True)
            self.set_dataset("pgc_best_amp", float(best_key[1]), persist=True, broadcast=True)
            self.set_dataset("pgc_best_tpgc", float(best_key[2]), persist=True, broadcast=True)
            print(f"PGCLowTrapScan: highest recapture@{rel_probe:g}us = {best_p:.3f} at "
                  f"f {best_key[0]/1e6:.4f} MHz, amp {best_key[1]:.3f}, "
                  f"t_PGC {best_key[2]:.0f} us (highest recapture != necessarily the "
                  f"knee; see analyze_pgc_time_release for T(t_PGC)).")
