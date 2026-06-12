"""Optical-pumping TIME scan -- mF=0 state-preparation fidelity vs. pump time.

Goal: find the SHORTEST optical-pumping window that still fully prepares
|F=3, mF=0>. A shorter pump means less state-prep scattering/heating at the low
trap (what bounds how many excitation cycles parity can afford), so we want the
KNEE of the fidelity-vs-time curve, not just "more pumping".

Probe (user's design): at the Q-axis B-field the mw00 pi-pulse drives ONLY
|F=3, mF=0> -> |F=4, mF=0> (Zeeman-resolved, mF=0-selective). Pump for the
scanned time, apply the fixed 00 pi, blow away F=4, then image:
  - atom pumped to mF=0  -> transferred to F=4 -> blown away -> reads DARK (lost),
  - atom in mF!=0        -> stays in F=3        -> survives    -> reads BRIGHT.
So lost-fraction vs. pump-time is a SATURATING curve whose level = mF=0 fidelity.

This is microwave_scan's Time_00 mode with two swaps: the scan axis is the PUMP
TIME (not the MW duration), and the MW pulse is a FIXED 00 pi
(mw_pulse_00(self, f_mw_00, t_pi_00_mu)). Everything else is identical.

Interleaved no-pi reference: at each pump-time point we run BOTH configs back-to-
back on successive fresh atoms within the same loop -- cfg=1 applies the pi,
cfg=0 skips it (a matched dark-hold, MW stays blocked). The dark fraction can
also include atoms left in F=4 by INCOMPLETE pumping (a background, largest at
short times); the no-pi run measures that background directly, so the clean mF=0
signal is  lost_pi(t) - lost_noPi(t)  (see analyze_optical_pump_scan).

The pumping-time variant lives in subroutines.sequences.optical_pumping_timed
(an exact copy of optical_pumping with the pump window as a kernel argument);
optical_pumping() itself is untouched.

Optimum is PRINTED, not written -- set t_optical_pump on the dashboard yourself.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    spcm_image_dual, imaging_pgc, beams_off_for_imaging,
    optical_pumping_F3, set_qaxis, optical_pumping_timed, ramp_trap_amp,
    blowaway_f4, final_state_readout, reset_mw_scan, mw_pulse_00,
)
from utilities.write_results import write_results


class OpticalPumpScan(EnvExperiment):
    """OpticalPumpScan_Exp"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Optical-pump TIME scan (us). Plain NumberValue (raw us floats, NO
        #     unit=) to avoid any dashboard unit double-scaling; the logged value
        #     is exact us. prepare() prints the live t_optical_pump so you can
        #     centre the range on it. Set min==max to fix a single time. ---
        self.setattr_argument("t_pump_min_us",
                              NumberValue(0.0, min=0.0, max=100000.0, ndecimals=2, step=1.0),
                              group="Pump-time scan (us)")
        self.setattr_argument("t_pump_max_us",
                              NumberValue(2000.0, min=0.0, max=100000.0, ndecimals=2, step=1.0),
                              group="Pump-time scan (us)")
        self.setattr_argument("t_pump_step_us",
                              NumberValue(200.0, min=0.01, max=100000.0, ndecimals=2, step=1.0),
                              group="Pump-time scan (us)")

        # --- Repetitions (binomial statistics; each loop runs BOTH pi/no-pi
        #     configs at every pump time, so shots-per-(time,config) ~ n_loops) ---
        self.setattr_argument("n_measurements",
                              NumberValue(300, min=1, max=600000, ndecimals=0, step=1),
                              group="Experiment")

        # All physics (f_mw_00, t_pi_00_mu, amp_mw_pulse, shim_*_qaxis,
        # trap_ramp_*, nominal_trapdepth, atom_threshold, t_imaging_gate,
        # t_optical_pump, the pump-beam params) load from datasets via
        # BaseExperiment; the rebalancer is created there too.

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # --- Build the pump-time scan list (us). Single point if min==max. ---
        vt, st = float(self.t_pump_min_us), float(self.t_pump_step_us)
        nt = (int(round((float(self.t_pump_max_us) - vt) / st)) + 1) if st > 0.0 else 1
        if nt < 1:
            nt = 1
        self.pump_times_us = [vt + i * st for i in range(nt)]

        # --- Fixed 00-pi probe (SI, like microwave_scan) ---
        self._f00_hz = self.f_mw_00            # already Hz (SI) from BaseExperiment
        self._t00_mu = int(self.t_pi_00_mu)    # pi duration in machine units

        # --- Self-describing snapshot ---
        snapshot_params = {
            "t_pump_min_us": float(self.t_pump_min_us),
            "t_pump_max_us": float(self.t_pump_max_us),
            "t_pump_step_us": float(self.t_pump_step_us),
            "t_optical_pump": self.t_optical_pump,   # live value (seconds) for reference
            "f_mw_00": self.f_mw_00,
            "t_pi_00_mu": self.t_pi_00_mu,
            "amp_mw_pulse": self.amp_mw_pulse,
            "shim_X_qaxis": self.shim_X_qaxis,
            "shim_Y_qaxis": self.shim_Y_qaxis,
            "shim_Z1_qaxis": self.shim_Z1_qaxis,
            "shim_Z2_qaxis": self.shim_Z2_qaxis,
            "shim_X_off": self.shim_X_off,
            "shim_Y_off": self.shim_Y_off,
            "shim_Z1_off": self.shim_Z1_off,
            "shim_Z2_off": self.shim_Z2_off,
            "trap_ramp_start_amp": self.trap_ramp_start_amp,
            "trap_ramp_steps": self.trap_ramp_steps,
            "trap_ramp_dt_us": self.trap_ramp_dt_us,
            "nominal_trapdepth": self.nominal_trapdepth,
            "atom_threshold": self.atom_threshold,
            "t_imaging_gate": self.t_imaging_gate,
            "n_measurements": int(self.n_measurements),
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Result datasets (per point) ---
        for k in ("op_t_pump_us", "op_apply_pi",
                  "op_bg_spcm1", "op_bg_spcm2",
                  "op_shot1_spcm1", "op_shot1_spcm2",
                  "op_shot2_spcm1", "op_shot2_spcm2",
                  "op_atom_loaded", "op_atom_retained", "op_loop_index"):
            self.set_dataset(k, [], broadcast=True, persist=True)
        self.set_dataset("op_scan_complete", 0, broadcast=True)

        # --- Metadata ---
        self.set_dataset("op_n_time_points", len(self.pump_times_us), persist=True, broadcast=True)
        self.set_dataset("op_n_loops", int(self.n_measurements), persist=True, broadcast=True)
        self.set_dataset("op_best_t_pump_us", 0.0, persist=True, broadcast=True)
        self.set_dataset("op_best_signal", 0.0, persist=True, broadcast=True)

        # Two configs (no-pi reference + pi) per pump time -> one full round =
        # all pump times x both configs. Incremental HDF5 save once per round.
        self._n_pts_per_round = max(len(self.pump_times_us) * 2, 1)

        print(f"OpticalPumpScan: {len(self.pump_times_us)} pump-time pts x 2 cfg "
              f"(no-pi/pi) x {int(self.n_measurements)} loops | "
              f"t_pump={self.pump_times_us} us | live t_optical_pump = "
              f"{self.t_optical_pump * 1e6:.2f} us")

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_measurements)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            for scan_idx in range(len(self.pump_times_us)):
                t_pump_us = self.pump_times_us[scan_idx]

                # Interleave the two MW configs back-to-back at each pump time:
                #   cfg 0 = no-pi reference (matched dark-hold, MW blocked)
                #   cfg 1 = apply the fixed 00 pi
                for cfg in range(2):
                    apply_pi = cfg == 1

                    # === Feedback (per point, like microwave_scan) ===
                    prepare_for_feedback(self)
                    amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                    trap_ramp_end_amp = amp3
                    restore_after_feedback(self)

                    # === Background + load + confirm image (shot1) @ HIGH trap ===
                    count1_bg = 0
                    count2_bg = 0
                    count1 = 0
                    count2 = 0
                    count1_post = 0
                    count2_post = 0
                    count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                    load_atom_deterministic(self)
                    imaging_pgc(self)
                    beams_off_for_imaging(self)
                    count1, count2 = spcm_image_dual(self, self.t_imaging_gate)

                    # === Re-cool after the imaging shot (PGC) ===
                    imaging_pgc(self)
                    beams_off_for_imaging(self)   # block repump before F=3 prep

                    # === F=3 state preparation ===
                    optical_pumping_F3(self)

                    # === Q-axis -> trap ramp DOWN -> SCANNED optical pumping ===
                    set_qaxis(self, self.shim_X_qaxis, self.shim_Y_qaxis,
                              self.shim_Z1_qaxis, self.shim_Z2_qaxis)
                    ramp_trap_amp(self, self.trap_ramp_start_amp, trap_ramp_end_amp,
                                  self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)
                    optical_pumping_timed(self, t_pump_us * us)

                    # === MW: fixed 00 pi (cfg 1) or matched dark-hold (cfg 0) ===
                    if apply_pi:
                        mw_pulse_00(self, self._f00_hz, self._t00_mu)
                    else:
                        # Match the pi-branch timeline advance so the two configs
                        # differ ONLY by whether the MW actually fires (MW stays
                        # blocked). The few-us DDS-set difference is immaterial to
                        # F=3/F=4 populations.
                        delay_mu(4100 + self._t00_mu)

                    # === Blowaway F=4 + trap ramp UP + final readout (shot2) ===
                    blowaway_f4(self, amp2)
                    ramp_trap_amp(self, trap_ramp_end_amp, self.trap_ramp_start_amp,
                                  self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)
                    count1_post, count2_post = final_state_readout(self)

                    # === Save ===
                    delay(1 * ms)
                    self._save_point(t_pump_us, cfg,
                                     count1_bg, count2_bg,
                                     count1, count2,
                                     count1_post, count2_post,
                                     scan_idx, loop)

                    # === Reset for next point ===
                    self.core.break_realtime()
                    reset_mw_scan(self)

        self._summarize()
        write_results(self)

    # =====================================================================
    #  HOST-SIDE SAVE / SUMMARY
    # =====================================================================

    @rpc
    def _save_point(self, t_pump_us, apply_pi,
                    bg_s1, bg_s2, shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2, scan_idx, loop_idx):
        """Append one point; threshold-based loaded/retained (same convention as
        microwave_scan); incremental HDF5 save once per full (pump-time x cfg)
        round."""
        self.append_to_dataset("op_t_pump_us", t_pump_us)
        self.append_to_dataset("op_apply_pi", apply_pi)
        self.append_to_dataset("op_bg_spcm1", bg_s1)
        self.append_to_dataset("op_bg_spcm2", bg_s2)
        self.append_to_dataset("op_shot1_spcm1", shot1_s1)
        self.append_to_dataset("op_shot1_spcm2", shot1_s2)
        self.append_to_dataset("op_shot2_spcm1", shot2_s1)
        self.append_to_dataset("op_shot2_spcm2", shot2_s2)
        self.append_to_dataset("op_loop_index", loop_idx)

        bg = bg_s1 + bg_s2
        shot1 = shot1_s1 + shot1_s2
        shot2 = shot2_s1 + shot2_s2
        threshold = self.atom_threshold
        loaded = int((shot1 - bg) > threshold)
        retained = int((shot2 - bg) > threshold) if loaded > 0 else 0
        self.append_to_dataset("op_atom_loaded", loaded)
        self.append_to_dataset("op_atom_retained", retained)

        if not hasattr(self, "_points_saved"):
            self._points_saved = 0
        self._points_saved += 1
        self.set_dataset("op_scan_complete", self._points_saved, broadcast=True)
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
        """Per pump time: lost-fraction (1 - retention among loaded) for the pi
        and no-pi configs, and the clean mF=0 signal = lost_pi - lost_noPi. The
        signal SATURATES with pump time; the knee = the shortest sufficient pump
        time. (Cheap table only -- the saturating fit + recommended t_98 is in
        analyze_optical_pump_scan.) BLAS-free (sums only)."""
        ts = self.get_dataset("op_t_pump_us")
        pis = self.get_dataset("op_apply_pi")
        loaded = self.get_dataset("op_atom_loaded")
        retained = self.get_dataset("op_atom_retained")
        if len(ts) == 0:
            print("OpticalPumpScan: no data to summarize.")
            return

        tab = {}        # (t_us, cfg) -> [n_retained, n_loaded]
        for i in range(len(ts)):
            if loaded[i] <= 0:
                continue
            key = (round(float(ts[i]), 2), int(pis[i]))
            tab.setdefault(key, [0, 0])
            tab[key][0] += int(retained[i])
            tab[key][1] += 1

        times = sorted({k[0] for k in tab})
        print("OpticalPumpScan: lost-fraction (1 - retention among loaded) per pump time:")
        best_t = None
        best_sig = -2.0
        for t in times:
            dpi = tab.get((t, 1))
            dno = tab.get((t, 0))
            lost_pi = (1.0 - dpi[0] / dpi[1]) if (dpi and dpi[1] > 0) else float("nan")
            lost_no = (1.0 - dno[0] / dno[1]) if (dno and dno[1] > 0) else float("nan")
            npi = dpi[1] if dpi else 0
            nno = dno[1] if dno else 0
            if dpi and dno and dpi[1] > 0 and dno[1] > 0:
                sig = lost_pi - lost_no
            else:
                sig = float("nan")
            print(f"  t={t:9.2f} us  lost_pi={lost_pi:.3f} (n={npi})  "
                  f"lost_noPi={lost_no:.3f} (n={nno})  signal={sig:.3f}")
            if sig == sig and sig > best_sig:   # sig==sig screens NaN
                best_sig = sig
                best_t = t

        if best_t is not None:
            self.set_dataset("op_best_t_pump_us", float(best_t), persist=True, broadcast=True)
            self.set_dataset("op_best_signal", float(best_sig), persist=True, broadcast=True)
            print(f"OpticalPumpScan: max mF=0 signal = {best_sig:.3f} at t_pump = "
                  f"{best_t:g} us (highest signal != necessarily the knee; see "
                  f"analyze_optical_pump_scan for the saturating fit + recommended t_98).")
