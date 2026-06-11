"""Parity oscillation measurement -- X / Y / Z basis selectable.

Scans HWP and QWP waveplate angles, for each angle:
  - Loads atom (Yellow scope)
  - Attempts excitation with photon detection (Orange scope)
  - On photon detection: analysis pulse, blowaway, state readout (Blue scope)

The analysis basis is chosen by the `basis_axis` argument:
  - Z: single MW pi pulse on the 11 transition (microwave_pi_pulse_z).
  - X: MW pi pulse + two-photon pi/2 basis rotation (microwave_two_photon_x).
  - Y: as X but RF phase = 0.25, rotating the basis X -> Y (microwave_two_photon_y).

All three pre-stage the leading pi pulse at f_mw_11 (the 11 transition,
F=3,mF=1 -> F=4,mF=1; t_pi_11_mu calibrated live via the ARTIQ dashboard). The
former separate f_mw_parity_00 value was an artifact and has been removed.

Build defaults are a single shared set: excitation timing from
Y_Parity_Optimized_v2 (t_trap_off_mu / t_spcm_collection_mu / length_offset_mu
= 400 / 400 / 12) and the narrow Y/Z waveplate window. For an X-basis
full-visibility scan, widen the Angle-Scan range to HWP 0-360 / QWP 0-360 in
the dashboard before running.

Data saved: photon timestamps, SPCM counts, trial counts, and the basis_axis
tag, per HDF5 run.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.k10cr1_functions import move_to_target_deg
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic, imaging_pgc,
    spcm_image_dual, beams_off_for_imaging,
    ramp_trap_amp, excitation_pulse, optical_pumping_F3, optical_pumping,
    microwave_pi_pulse_z, microwave_two_photon_x, microwave_two_photon_y, blowaway_f4,
    recool_and_check_survival, pgc_between_excitations,
    final_state_readout, reset_to_mot, turn_fiber_aom_on
)
from utilities.write_results import write_results


class ParityExperiment(EnvExperiment):
    """Parity Experiment (X/Y/Z)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Basis selector ---
        # Maps to an int axis_code in prepare() (kernel-safe: the kernel never
        # compares the string, only the int). Z is the default.
        self.setattr_argument("basis_axis",
                              EnumerationValue(["Z", "X", "Y"], default="Z"),
                              group="Basis")

        # --- Experiment-specific arguments ---
        self.setattr_argument("n_measurements",
                              NumberValue(100, min=1, max=600000, step=1),
                              group="Experiment")
        self.setattr_argument("max_excitation_cycles",
                              NumberValue(4, min=1, max=8000, step=1),
                              group="Excitation")
        self.setattr_argument("Check_img_every",
                              NumberValue(12, min=1, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        # Excitation timing defaults: shared set from Y_Parity_Optimized_v2.
        self.setattr_argument("t_trap_off_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("t_spcm_collection_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("length_offset_mu",
                              NumberValue(12, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")

        # Adiabatic trap ramp
        self.setattr_argument("trap_ramp_start_amp",
                              NumberValue(0.24, min=0.0, max=0.3, ndecimals=3),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_steps",
                              NumberValue(200, min=0, max=100000, ndecimals=0),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_dt_us",
                              NumberValue(50, min=0.001, max=1000, ndecimals=3),
                              group="Adiabatic")

        # Waveplate angle scan -- default to the narrow Y/Z window.
        # For an X-basis full parity oscillation, widen to HWP 0-360 / QWP 0-360.
        self.setattr_argument("HWP_angle_min",
                              NumberValue(10.0, min=0.0, max=360.0), group="Angle Scan")
        self.setattr_argument("HWP_angle_max",
                              NumberValue(130.0, min=0.0, max=3600.0), group="Angle Scan")
        self.setattr_argument("HWP_angle_step",
                              NumberValue(15.0, min=0.1, max=360.0), group="Angle Scan")
        self.setattr_argument("QWP_angle_min",
                              NumberValue(257.5, min=0.0, max=360.0), group="Angle Scan")
        self.setattr_argument("QWP_angle_max",
                              NumberValue(262.5, min=0.0, max=360.0), group="Angle Scan")
        self.setattr_argument("QWP_angle_step",
                              NumberValue(2.5, min=0.1, max=360.0), group="Angle Scan")

        # Rebalancer created and configured by BaseExperiment.build()

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # Two-photon frequencies are DERIVED from the transition frequencies +
        # detunings (no longer independent inputs) -- identical formula to
        # microwave_scan.py:195-201 (_fm10_2ph_hz / _rf_2ph_hz). Arithmetic in
        # MHz, converted to Hz once at the end (dds.set expects Hz).
        f00_mhz = self.f_mw_00 / 1e6                           # Hz  -> MHz
        f01_mhz = self.f_mw_01 / 1e6                           # Hz  -> MHz
        detune_mhz = self.rf_detune_khz / 1000.0              # kHz -> MHz
        small_detune_mhz = self.rf_small_detune_khz / 1000.0  # kHz -> MHz
        self.f_mw_2ph = (2.0 * f00_mhz - f01_mhz + detune_mhz) * 1e6
        self.f_RF_2ph = (f01_mhz - f00_mhz - detune_mhz + small_detune_mhz) * 1e6
        self.set_dataset("f_mw_2ph", self.f_mw_2ph, broadcast=True)
        self.set_dataset("f_RF_2ph", self.f_RF_2ph, broadcast=True)

        # Map the basis selector to a kernel-safe int (the kernel branches on
        # this int only, never on the string).
        if self.basis_axis == "X":
            self.axis_code = 1
        elif self.basis_axis == "Y":
            self.axis_code = 2
        else:  # "Z"
            self.axis_code = 0
        # Record which basis this run used, so the .h5 is self-describing.
        self.set_dataset("basis_axis", self.basis_axis, broadcast=True)

        # Result datasets (must be initialized before append_to_dataset)
        self.set_dataset("SPCM1_timestamps", [], broadcast=True)
        self.set_dataset("SPCM2_timestamps", [], broadcast=True)
        self.set_dataset("trials_per_meas", [], broadcast=True)
        self.set_dataset("counts_signal", [], broadcast=True)
        self.set_dataset("counts_background", [], broadcast=True)
        self.set_dataset("HWP_angles", [], broadcast=True)
        self.set_dataset("QWP_angles", [], broadcast=True)
        # Build scan sequences
        v1, s1 = self.HWP_angle_min, self.HWP_angle_step
        n1 = int(round((self.HWP_angle_max - v1) / s1)) + 1
        self.HWP_scan = [v1 + i * s1 for i in range(n1)]

        v2, s2 = self.QWP_angle_min, self.QWP_angle_step
        n2 = int(round((self.QWP_angle_max - v2) / s2)) + 1
        self.QWP_scan = [v2 + i * s2 for i in range(n2)]

    @kernel
    def _analysis_pulse(self, click_time_mu):
        """Axis-selected analysis pulse (the only axis-dependent kernel code).

        All three branch targets anchor every event on absolute
        at_mu(click_time_mu + ...), so the timeline is deterministic regardless
        of which branch runs.
        """
        if self.axis_code == 1:     # X
            microwave_two_photon_x(self, click_time_mu)
        elif self.axis_code == 2:   # Y
            microwave_two_photon_y(self, click_time_mu)
        else:                       # Z
            microwave_pi_pulse_z(self, click_time_mu)

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        for qwp in self.QWP_scan:
            delay(100 * ms)
            move_to_target_deg(self, name="852_QWP", target_deg=qwp)
            delay(15000 * ms)
            self.core.break_realtime()

            for hwp in self.HWP_scan:
                delay(100 * ms)
                move_to_target_deg(self, name="852_HWP", target_deg=hwp)
                if hwp == self.HWP_scan[0]:
                    delay(15000 * ms)
                else:
                    delay(5000 * ms)
                self.core.break_realtime()

                measurements = 0
                while measurements < int(self.n_measurements):
                    self.core.break_realtime()

                    # ===== YELLOW SCOPE: Loading =====
                    trap_ramp_end_amp = self.trap_ramp_start_amp   # safe FORT value; real value (amp3) set by the rebalancer in the load loop below
                    amp2 = 0.0   # definite-assignment init; real value set in the load loop below
                    count1_bg = 0   # definite-assignment init; real value set in the load loop below
                    count2_bg = 0   # definite-assignment init; real value set in the load loop below
                    while True:
                        # Feedback
                        prepare_for_feedback(self)
                        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                        trap_ramp_end_amp = amp3
                        restore_after_feedback(self)

                        # Background + atom loading
                        count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                        load_atom_deterministic(self)

                        # Imaging PGC
                        imaging_pgc(self)
                        beams_off_for_imaging(self)

                        # Wait
                        delay(self.t_SPCM_exposure)

                        # Image
                        count1_1, count2_1 = spcm_image_dual(self, self.t_imaging_gate)

                        # PGC after image
                        imaging_pgc(self)
                        beams_off_for_imaging(self)

                        # Check atom
                        if (count1_1 + count2_1 - count1_bg - count2_bg) > self.atom_threshold:
                            break
                    # ===== END YELLOW =====

                    # ===== ORANGE SCOPE: Excitation retry =====
                    photon_detected = False

                    trials = 0
                    n = 0   # cycle counter for Check_img_every gating (v2:512); definite-assignment safe int
                    atom_alive = True   # definite-assignment init; real value set in recovery branch below
                    final_ts1 = 0.0
                    final_ts2 = 0.0
                    while True:
                        optical_pumping_F3(self)                       # F=3 prep (v2:516-522)
                        self.zotino0.write_dac(0, self.shim_X_qaxis)   # Q-axis on (v2:524-530)
                        self.zotino0.write_dac(1, self.shim_Y_qaxis)
                        self.zotino0.write_dac(2, self.shim_Z1_qaxis)
                        self.zotino0.write_dac(3, self.shim_Z2_qaxis)
                        self.zotino0.load()
                        delay(1 * ms)
                        ramp_trap_amp(self, self.trap_ramp_start_amp,   # trap DOWN (v2:532-534)
                                    trap_ramp_end_amp,
                                    self.trap_ramp_steps,
                                    self.trap_ramp_dt_us)

                        # Excitation cycles
                        # v2 per-cycle order (archive {Z,Y}_Parity_Optimized_v2
                        # lines 543-792): n+=1 -> shared optical pump (q-axis +
                        # trap ramp now hoisted ONCE above the retry loop) ->
                        # excitation + analysis pulse + blowaway. The shared
                        # optical_pumping() reproduces the v2 pump steps exactly
                        # (minus q-axis + MW-IF, which are staged by the caller
                        # above/before the call).
                        for exc_cycle in range(int(self.max_excitation_cycles)):
                            n += 1   # v2:544
                            self.core.break_realtime()   # refresh RTIO slack each cycle (v2:559)

                            # Pre-stage MW IF for the leading pi pulse -- same
                            # 11-transition freq for all axes -- then optical pump.
                            self.dds_mw_IF.set(frequency=self.f_mw_11, amplitude=self.amp_mw_pulse)
                            optical_pumping(self)

                            # --- Excitation (v7:589 onward) ---
                            trials += 1
                            t_end1, t_end2, t0 = excitation_pulse(self)
                            # Read SPCM timestamps HERE (not inside
                            # excitation_pulse) so the function-return boundary
                            # is out of the slack-critical window: the empty-SPCM
                            # timestamp_mu blocks to gate close, pinning the wall
                            # clock with ~0 slack, and the analysis pulse anchors
                            # only 2 us after the photon. Reading inline keeps the
                            # read -> first MW edge minimal so click+2000 doesn't
                            # underflow (channel 39 RTIO underflow fix).
                            click1 = self.ttl_SPCM1.timestamp_mu(t_end1)
                            click2 = self.ttl_SPCM2.timestamp_mu(t_end2)

                            # Process photon detection (axis-selected analysis pulse)
                            if click1 > 0 and click2 < 0:
                                if self.axis_code == 1:     # X
                                    microwave_two_photon_x(self, click1)
                                elif self.axis_code == 2:   # Y
                                    microwave_two_photon_y(self, click1)
                                else:                       # Z
                                    microwave_pi_pulse_z(self, click1)
                                # self._analysis_pulse(click1)
                                blowaway_f4(self, amp2)
                                final_ts1 = self.core.mu_to_seconds(click1 - t0)
                                final_ts2 = 0.0
                                photon_detected = True
                                break

                            if click1 < 0 and click2 > 0:
                                if self.axis_code == 1:     # X
                                    microwave_two_photon_x(self, click2)
                                elif self.axis_code == 2:   # Y
                                    microwave_two_photon_y(self, click2)
                                else:                       # Z
                                    microwave_pi_pulse_z(self, click2)
                                #self._analysis_pulse(click2)
                                blowaway_f4(self, amp2)
                                final_ts1 = 0.0
                                final_ts2 = self.core.mu_to_seconds(click2 - t0)
                                photon_detected = True
                                break

                            delay(5 * us)   # preserves the no-detection dead-time
                                            # the trailing delay in excitation_pulse
                                            # used to add (reads now run in caller)
                            delay(2 * us)
                            self.ttl_MW_switch.on()
                            delay(1 * us)

                            # No per-cycle PGC: v2:790-792 skips pgc_between_excitations
                            # inside the cycle loop.

                        if photon_detected:
                            break

                        # Recovery (max cycles, no photon) -- single-recool
                        # decomposition of v2:797-862. ramp UP every round
                        # (v2:807-810); neither subroutine ramps the trap.
                        self.core.break_realtime()   # rebuild RTIO slack drained by the excitation cycles
                        self.zotino0.write_dac(0, self.shim_X_off)
                        self.zotino0.write_dac(1, self.shim_Y_off)
                        self.zotino0.write_dac(2, self.shim_Z1_off)
                        self.zotino0.write_dac(3, self.shim_Z2_off)
                        self.zotino0.load()
                        delay(1 * ms)
                        
                        ramp_trap_amp(self, trap_ramp_end_amp,
                                      self.trap_ramp_start_amp,
                                      self.trap_ramp_steps,
                                      self.trap_ramp_dt_us)
                        turn_fiber_aom_on(self)   # PGC/imaging path ON for recovery (v2:805-806)
                        
                        delay(1 * us)
                        self.dds_repump.sw.on()
                        delay(1 * us)
                        self.ttl_repump_switch.off()         # unblock repump
                        imaging_pgc(self)

                        if n % int(self.Check_img_every) == 0:
                            n = 0
                            # shim OFF (v2:799-805) -- recool_and_check_survival
                            # does NOT zero shims

                            delay(1 * ms)
                            count_bg_total = count1_bg + count2_bg
                            atom_alive = recool_and_check_survival(self, count_bg_total)   # recool+image+threshold (v2:812-847,857)
                            if not atom_alive:
                                break    # atom lost -> back to Yellow

                        beams_off_for_imaging(self)
                    # ===== END ORANGE =====

                    # ===== BLUE SCOPE: Success =====
                    if photon_detected:
                        delay(1 * us)
                        self.core.break_realtime()   # rebuild RTIO slack before the readout ramp
                        ramp_trap_amp(self, trap_ramp_end_amp,
                                      self.trap_ramp_start_amp,
                                      self.trap_ramp_steps,
                                      self.trap_ramp_dt_us)
                        delay(1 * ms)

                        count1_final, count2_final = final_state_readout(self)

                        # Save data
                        self._save_measurement(
                            final_ts1, final_ts2, trials,
                            count1_final + count2_final,
                            count1_bg + count2_bg,
                            hwp, qwp
                        )
                        trials = 0
                        measurements += 1
                        self.set_dataset("measurement_progress",
                                         measurements, broadcast=True)
                    # ===== END BLUE =====

                    reset_to_mot(self)

        write_results(self)

    @rpc(flags={"async"})
    def _save_measurement(self, ts1, ts2, trials, counts, counts_bg, hwp, qwp):
        """Append measurement to datasets."""
        self.append_to_dataset("SPCM1_timestamps", ts1)
        self.append_to_dataset("SPCM2_timestamps", ts2)
        self.append_to_dataset("trials_per_meas", trials)
        self.append_to_dataset("counts_signal", counts)
        self.append_to_dataset("counts_background", counts_bg)
        self.append_to_dataset("HWP_angles", hwp)
        self.append_to_dataset("QWP_angles", qwp)
        # Intermediate HDF5 save (mirrors microwave_scan's incremental save)
        self._incremental_save()

    def _incremental_save(self):
        """Write current datasets to HDF5 mid-run (mirrors microwave_scan)."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
        except Exception as e:
            print("[incremental save]", e)
