"""Excitation-amplitude scan: photon collection efficiency vs excitation power.

Scans the excitation-pulse AMPLITUDE (the excitation DDS at 145 MHz) and, for
each amplitude, fires many excitation attempts on a trapped, optically-pumped
atom and counts the spontaneously-emitted photons collected by SPCM1/SPCM2.
Efficiency = detections / attempts. Use it to find the excitation amplitude
that maximizes photon collection for the parity experiments.

Structure: cloned from microwave_scan.py (the modern unified calibration
experiment) for parameter setting, atom preparation, optical pumping, imaging,
the interleaved scan loop, and per-round HDF5 saving. The MW-pulse science block
is replaced by an excitation + dual-SPCM photon-collection block (inlined from
subroutines.sequences.excitation_pulse, with the amplitude scanned), and the
blowaway is dropped (we measure photon collection, not the final F-state).

The fixed excitation-timing and adiabatic trap-ramp parameters strictly follow
the parity experiments (t_trap_off_mu / t_spcm_collection_mu / length_offset_mu
= 400 / 400 / 12; trap ramp 0.24 / 200 / 50). The amplitude-scan idea and the
145 MHz excitation come from archive HS_code_20260225_ext_long_trap_off.py
(where the excitation amplitude was confusingly named self.Trap_amps).

Data saved per measurement: excitation amplitude, dual-SPCM background /
atom-present / survival counts, per-SPCM detection counts, attempts, and
atom-loaded/-retained flags. Data saved per detected photon: arrival time
(relative to excitation start) tagged with its amplitude, for the temporal
profile. Intermediate HDF5 state is written after each round (full sweep),
mirroring microwave_scan, so a mid-run crash never loses prior rounds.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    spcm_image_dual, imaging_pgc, beams_off_for_imaging,
    optical_pumping_F3, set_qaxis, optical_pumping, ramp_trap_amp,
    final_state_readout, reset_mw_scan,
)
from utilities.write_results import write_results


class ExcitationScan(EnvExperiment):
    """Excitation_Amp_Scan"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # =============================================================
        #  SCAN-SPECIFIC ARGUMENTS
        # =============================================================

        # --- Excitation amplitude scan (defaults from the archive) ---
        self.setattr_argument("exc_amp_min",
                              NumberValue(0.150, min=0.0, max=0.5, ndecimals=3),
                              group="Excitation Amp Scan")
        self.setattr_argument("exc_amp_max",
                              NumberValue(0.250, min=0.0, max=0.6, ndecimals=3),
                              group="Excitation Amp Scan")
        self.setattr_argument("exc_amp_step",
                              NumberValue(0.010, min=0.0001, max=0.5, ndecimals=3),
                              group="Excitation Amp Scan")

        # --- Experiment control ---
        self.setattr_argument("n_measurements",
                              NumberValue(100, min=1, max=600000, step=1),
                              group="Experiment")
        self.setattr_argument("max_excitation_cycles",
                              NumberValue(200, min=1, max=8000, step=1),
                              group="Excitation")

        # --- Excitation timing (strictly follows the parity experiments) ---
        # Defined as build args so they shadow BaseExperiment's dataset binds,
        # exactly as the parity experiments do.
        self.setattr_argument("t_trap_off_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("t_spcm_collection_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("length_offset_mu",
                              NumberValue(12, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")

        # --- Adiabatic trap ramp (strictly follows the parity experiments) ---
        self.setattr_argument("trap_ramp_start_amp",
                              NumberValue(0.24, min=0.0, max=0.3, ndecimals=3),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_steps",
                              NumberValue(200, min=0, max=100000, ndecimals=0),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_dt_us",
                              NumberValue(50, min=0.001, max=1000, ndecimals=3),
                              group="Adiabatic")

        # =============================================================
        #  All other physics parameters (f_excitation_pulse, the exc_*
        #  offset datasets, shims, atom_threshold, etc.) are loaded by
        #  BaseExperiment from datasets. Rebalancer is created there too.
        # =============================================================

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # =============================================================
        #  PRE-RUN SNAPSHOT: record the excitation conditions so the data
        #  file is fully self-describing.
        # =============================================================
        snapshot_params = {
            "f_excitation_pulse": self.f_excitation_pulse,
            "amp_excitation_pulse": self.amp_excitation_pulse,
            "t_trap_off_mu": self.t_trap_off_mu,
            "t_spcm_collection_mu": self.t_spcm_collection_mu,
            "length_offset_mu": self.length_offset_mu,
            "exc_trap_off_offset_mu": self.exc_trap_off_offset_mu,
            "exc_d1_switch_offset_mu": self.exc_d1_switch_offset_mu,
            "exc_atten_offset_mu": self.exc_atten_offset_mu,
            "exc_beam_on_offset_mu": self.exc_beam_on_offset_mu,
            "exc_beam_duration_mu": self.exc_beam_duration_mu,
            "exc_spcm_gate_offset_mu": self.exc_spcm_gate_offset_mu,
            "trap_ramp_start_amp": self.trap_ramp_start_amp,
            "trap_ramp_steps": self.trap_ramp_steps,
            "trap_ramp_dt_us": self.trap_ramp_dt_us,
            "shim_X_qaxis": self.shim_X_qaxis,
            "shim_Y_qaxis": self.shim_Y_qaxis,
            "shim_Z1_qaxis": self.shim_Z1_qaxis,
            "shim_Z2_qaxis": self.shim_Z2_qaxis,
            "atom_threshold": self.atom_threshold,
            "max_excitation_cycles": int(self.max_excitation_cycles),
            "exc_amp_min": self.exc_amp_min,
            "exc_amp_max": self.exc_amp_max,
            "exc_amp_step": self.exc_amp_step,
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Build excitation-amplitude scan list ---
        v, s = self.exc_amp_min, self.exc_amp_step
        n = int(round((self.exc_amp_max - v) / s)) + 1
        self.exc_amps = [v + i * s for i in range(n)]

        # --- Result datasets (per measurement) ---
        self.set_dataset("exc_scan_amp", [], broadcast=True, persist=True)
        self.set_dataset("exc_bg_spcm1", [], broadcast=True, persist=True)
        self.set_dataset("exc_bg_spcm2", [], broadcast=True, persist=True)
        self.set_dataset("exc_shot1_spcm1", [], broadcast=True, persist=True)
        self.set_dataset("exc_shot1_spcm2", [], broadcast=True, persist=True)
        self.set_dataset("exc_shot2_spcm1", [], broadcast=True, persist=True)
        self.set_dataset("exc_shot2_spcm2", [], broadcast=True, persist=True)
        self.set_dataset("exc_det_spcm1", [], broadcast=True, persist=True)
        self.set_dataset("exc_det_spcm2", [], broadcast=True, persist=True)
        self.set_dataset("exc_attempts", [], broadcast=True, persist=True)
        self.set_dataset("exc_atom_loaded", [], broadcast=True, persist=True)
        self.set_dataset("exc_atom_retained", [], broadcast=True, persist=True)
        self.set_dataset("exc_scan_complete", 0, broadcast=True)

        # --- Result datasets (per detected photon) ---
        self.set_dataset("SPCM1_timestamps", [], broadcast=True, persist=True)
        self.set_dataset("SPCM1_ts_amp", [], broadcast=True, persist=True)
        self.set_dataset("SPCM2_timestamps", [], broadcast=True, persist=True)
        self.set_dataset("SPCM2_ts_amp", [], broadcast=True, persist=True)

        # --- Metadata ---
        self.set_dataset("exc_scan_mode", "excitation_amp", persist=True, broadcast=True)
        self.set_dataset("exc_n_scan_points", len(self.exc_amps), persist=True, broadcast=True)
        self.set_dataset("exc_n_loops", int(self.n_measurements), persist=True, broadcast=True)
        self.set_dataset("exc_best_amp", 0.0, persist=True, broadcast=True)
        self.set_dataset("exc_best_efficiency", 0.0, persist=True, broadcast=True)

        print(f"ExcitationScan: {len(self.exc_amps)} amplitudes, "
              f"{int(self.n_measurements)} loops")

    @kernel
    def _excite_and_collect(self, amp_exc):
        """One excitation attempt at amplitude `amp_exc`, with dual-SPCM photon
        collection. Inlined from subroutines.sequences.excitation_pulse; the ONLY
        difference is the opening dds_excitation amplitude (amp_exc instead of
        self.amp_excitation_pulse). Returns (click1, click2, t0); a click <= 0
        means no photon detected on that SPCM.
        """
        self.dds_excitation.set(frequency=self.f_excitation_pulse, amplitude=amp_exc)
        delay(5 * us)

        t0 = now_mu()

        # Unblock MW switch once at t0 (kept identical to excitation_pulse; the
        # MW source is off here, so this is a harmless no-op for the scan).
        at_mu(t0)
        self.ttl_MW_switch.off()

        # Trap off
        at_mu(t0 + int(self.exc_trap_off_offset_mu))
        self.dds_FORT.sw.off()
        at_mu(t0 + int(self.exc_trap_off_offset_mu) + int(self.t_trap_off_mu))
        self.dds_FORT.sw.on()

        # D1/excitation switch pulse
        at_mu(t0 + int(self.exc_d1_switch_offset_mu))
        self.ttl_D1_exc_switch.off()
        at_mu(t0 + int(self.exc_d1_switch_offset_mu) + int(self.length_offset_mu))
        self.ttl_D1_exc_switch.on()

        # Excitation attenuator pulse
        at_mu(t0 + int(self.exc_atten_offset_mu))
        with parallel:
            self.ttl_exc_atten2.off()
            self.ttl_exc_atten1.off()
        at_mu(t0 + int(self.exc_atten_offset_mu) + 100)
        with parallel:
            self.ttl_exc_atten2.on()
            self.ttl_exc_atten1.on()

        # Excitation beam pulse
        at_mu(t0 + int(self.exc_beam_on_offset_mu))
        self.dds_excitation.sw.on()
        at_mu(t0 + int(self.exc_beam_on_offset_mu) + int(self.exc_beam_duration_mu))
        self.dds_excitation.sw.off()

        # SPCM gate
        at_mu(t0 + int(self.exc_spcm_gate_offset_mu))
        with parallel:
            t_end1 = self.ttl_SPCM1.gate_rising(self.t_spcm_collection_mu * ns)
            t_end2 = self.ttl_SPCM2.gate_rising(self.t_spcm_collection_mu * ns)

        click1 = self.ttl_SPCM1.timestamp_mu(t_end1)
        click2 = self.ttl_SPCM2.timestamp_mu(t_end2)
        delay(5 * us)

        return click1, click2, t0

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_measurements)
        n_cyc = int(self.max_excitation_cycles)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            for scan_idx in range(len(self.exc_amps)):
                amp_exc = self.exc_amps[scan_idx]

                # === Feedback ===
                prepare_for_feedback(self)
                amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                trap_ramp_end_amp = amp3
                restore_after_feedback(self)

                # === Background + Load (no retry — matches microwave_scan) ===
                count1_bg = 0
                count2_bg = 0
                count1 = 0
                count2 = 0
                count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                load_atom_deterministic(self)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                count1, count2 = spcm_image_dual(self, self.t_imaging_gate)

                # === Re-cool after the imaging shot (PGC) ===
                imaging_pgc(self)
                beams_off_for_imaging(self)   # block repump before F=3 prep

                # === F=3 state preparation (parity's optical_pumping_F3) ===
                optical_pumping_F3(self)

                # === Q-axis → trap ramp down ===
                set_qaxis(self, self.shim_X_qaxis, self.shim_Y_qaxis,
                          self.shim_Z1_qaxis, self.shim_Z2_qaxis)
                ramp_trap_amp(self, self.trap_ramp_start_amp, trap_ramp_end_amp,
                              self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)

                # === Excitation cycle loop: pump → excite → collect photons ===
                det1 = 0
                det2 = 0
                for cyc in range(n_cyc):
                    self.core.break_realtime()   # refresh RTIO slack each cycle
                    optical_pumping(self)        # re-prep the atom before each attempt
                    click1, click2, t0 = self._excite_and_collect(amp_exc)
                    if click1 > 0:
                        det1 += 1
                        self._append_timestamp(0, self.core.mu_to_seconds(click1 - t0), amp_exc)
                    if click2 > 0:
                        det2 += 1
                        self._append_timestamp(1, self.core.mu_to_seconds(click2 - t0), amp_exc)

                # === Trap ramp back up ===
                self.core.break_realtime()
                ramp_trap_amp(self, trap_ramp_end_amp, self.trap_ramp_start_amp,
                              self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)

                # === Survival readout (no blowaway — measuring collection, not F-state) ===
                count1_post, count2_post = final_state_readout(self)

                # === Save data ===
                delay(1 * ms)
                self._save_point(amp_exc,
                                 count1_bg, count2_bg,
                                 count1, count2,
                                 count1_post, count2_post,
                                 det1, det2, n_cyc, scan_idx)

                # === Reset for next scan point ===
                self.core.break_realtime()
                reset_mw_scan(self)

        self._summarize()
        write_results(self)

    @rpc(flags={"async"})
    def _append_timestamp(self, spcm_id, ts_s, amp):
        """Append one detected photon's arrival time (s, relative to excitation
        start) tagged with its excitation amplitude."""
        if spcm_id == 0:
            self.append_to_dataset("SPCM1_timestamps", ts_s)
            self.append_to_dataset("SPCM1_ts_amp", amp)
        else:
            self.append_to_dataset("SPCM2_timestamps", ts_s)
            self.append_to_dataset("SPCM2_ts_amp", amp)

    @rpc
    def _save_point(self, amp,
                    bg_s1, bg_s2,
                    shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2,
                    det1, det2, attempts, scan_idx):
        """Append one measurement with threshold-based atom detection, and write
        intermediate HDF5 state after each round (full sweep)."""
        self.append_to_dataset("exc_scan_amp", amp)
        self.append_to_dataset("exc_bg_spcm1", bg_s1)
        self.append_to_dataset("exc_bg_spcm2", bg_s2)
        self.append_to_dataset("exc_shot1_spcm1", shot1_s1)
        self.append_to_dataset("exc_shot1_spcm2", shot1_s2)
        self.append_to_dataset("exc_shot2_spcm1", shot2_s1)
        self.append_to_dataset("exc_shot2_spcm2", shot2_s2)
        self.append_to_dataset("exc_det_spcm1", det1)
        self.append_to_dataset("exc_det_spcm2", det2)
        self.append_to_dataset("exc_attempts", attempts)

        bg = bg_s1 + bg_s2
        shot1 = shot1_s1 + shot1_s2
        shot2 = shot2_s1 + shot2_s2
        threshold = self.atom_threshold

        loaded = int((shot1 - bg) > threshold)
        retained = int((shot2 - bg) > threshold)
        self.append_to_dataset("exc_atom_loaded", loaded)
        self.append_to_dataset("exc_atom_retained", retained)
        self.set_dataset("exc_scan_complete", scan_idx + 1, broadcast=True)

        # Incremental HDF5 save after each round (full sweep), like microwave_scan
        if not hasattr(self, '_points_saved'):
            self._points_saved = 0
        self._points_saved += 1
        n_pts = max(len(self.exc_amps), 1)
        if self._points_saved % n_pts == 0:
            self._incremental_save()

    def _incremental_save(self):
        """Save current data to HDF5 mid-run (called from _save_point)."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
            print(f"[incremental save] {self._points_saved} measurements saved to HDF5")
        except Exception as e:
            print(f"[incremental save] FAILED: {e}")

    @rpc
    def _summarize(self):
        """Aggregate detections per amplitude over surviving measurements and
        report the peak-efficiency amplitude. BLAS-free (sums + argmax only)."""
        amps = self.get_dataset("exc_scan_amp")
        det1 = self.get_dataset("exc_det_spcm1")
        det2 = self.get_dataset("exc_det_spcm2")
        attempts = self.get_dataset("exc_attempts")
        retained = self.get_dataset("exc_atom_retained")

        if len(amps) == 0:
            print("ExcitationScan: no data to summarize.")
            return

        totals = {}   # rounded amp -> [sum_detections, sum_attempts]
        for i in range(len(amps)):
            if retained[i] <= 0:
                continue
            a = round(float(amps[i]), 6)
            if a not in totals:
                totals[a] = [0.0, 0.0]
            totals[a][0] += float(det1[i]) + float(det2[i])
            totals[a][1] += float(attempts[i])

        best_amp = 0.0
        best_eff = -1.0
        print("ExcitationScan efficiency (detected/attempt) per amplitude:")
        for a in sorted(totals.keys()):
            sd, sa = totals[a]
            eff = sd / sa if sa > 0 else 0.0
            print(f"  amp={a:.3f}  eff={eff:.4f}  ({int(sd)}/{int(sa)})")
            if eff > best_eff:
                best_eff = eff
                best_amp = a

        if best_eff < 0.0:
            best_eff = 0.0
        self.set_dataset("exc_best_amp", best_amp, persist=True, broadcast=True)
        self.set_dataset("exc_best_efficiency", best_eff, persist=True, broadcast=True)
        print(f"ExcitationScan: BEST excitation amplitude = {best_amp:.3f} "
              f"(efficiency {best_eff:.4f})")
