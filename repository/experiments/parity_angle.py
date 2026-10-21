"""Parity oscillation measurement -- FREE analysis angle (no X/Y/Z toggle).

Same heralded single-atom sequence as parity.py:
  - Loads atom (Yellow scope)
  - Attempts excitation with photon detection (Orange scope)
  - On photon detection: analysis pulse, blowaway, state readout (Blue scope)

The analysis basis is a FREELY ASSIGNABLE angle instead of an X/Y/Z selector.
Set `basis_phase_deg` (degrees) in the dashboard: it becomes the RF phase of the
two-photon pi/2 basis rotation, which chooses the equatorial measurement axis on
the Bloch sphere. Reference points:
  -   0 deg  ->  X basis   (equivalent to parity.py X)
  -  90 deg  ->  Y basis   (equivalent to parity.py Y)
  - 180 deg  -> -X ,  270 deg -> -Y ,  and anything in between.

There is NO Z basis here (Z is a plain pi pulse with no basis rotation, which
this experiment does not do). Every trial pre-stages the leading pi pulse at
f_mw_11 (the 11 transition, F=3,mF=1 -> F=4,mF=1; t_pi_11_mu calibrated live via
the ARTIQ dashboard).

Build defaults match parity.py: excitation timing t_trap_off_mu /
t_spcm_collection_mu / length_offset_mu = 400 / 400 / 12.

Data saved: photon timestamps, SPCM counts, trial counts, and the basis_phase_deg
tag, per HDF5 run.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.k10cr1_functions import move_to_target_deg
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic, imaging_pgc, imaging_pgc_low_trap,
    spcm_image_dual, beams_off_for_imaging,
    ramp_trap_amp, excitation_pulse, optical_pumping_F3, optical_pumping,
    microwave_two_photon_phase, blowaway_f4,
    recool_and_check_survival, pgc_between_excitations, recool_and_check_survival_lowtrap,
    final_state_readout, reset_to_mot, turn_fiber_aom_on
)
from utilities.write_results import write_results


class ParityAngleExperiment(EnvExperiment):
    """Parity Experiment (free angle)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Free analysis angle (degrees) ---
        # 0 = X basis, 90 = Y basis; converted to DDS turns in prepare().
        self.setattr_argument("basis_phase_deg",
                              NumberValue(0.0, min=0.0, max=360.0, ndecimals=2, step=1.0),
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
        # detunings (identical formula to parity.py / microwave_scan.py).
        f00_mhz = self.f_mw_00 / 1e6                           # Hz  -> MHz
        f01_mhz = self.f_mw_01 / 1e6                           # Hz  -> MHz
        detune_mhz = self.rf_detune_khz / 1000.0              # kHz -> MHz
        small_detune_mhz = self.rf_small_detune_khz / 1000.0  # kHz -> MHz
        self.f_mw_2ph = (2.0 * f00_mhz - f01_mhz + detune_mhz) * 1e6
        self.f_RF_2ph = (f01_mhz - f00_mhz - detune_mhz + small_detune_mhz) * 1e6
        self.set_dataset("f_mw_2ph", self.f_mw_2ph, broadcast=True)
        self.set_dataset("f_RF_2ph", self.f_RF_2ph, broadcast=True)

        # Convert the free analysis angle from degrees to DDS turns (0..1).
        # The kernel uses this float only; the RF phase wraps mod 1 in hardware.
        self.basis_phase_turns = self.basis_phase_deg / 360.0
        # Record the chosen angle so the .h5 is self-describing.
        self.set_dataset("basis_phase_deg", self.basis_phase_deg, broadcast=True)

        # Result datasets (must be initialized before append_to_dataset)
        self.set_dataset("SPCM1_timestamps", [], broadcast=True)
        self.set_dataset("SPCM2_timestamps", [], broadcast=True)
        self.set_dataset("trials_per_meas", [], broadcast=True)
        self.set_dataset("counts_signal", [], broadcast=True)
        self.set_dataset("counts_background", [], broadcast=True)
        self.set_dataset("HWP_angles", [], broadcast=True)
        self.set_dataset("QWP_angles", [], broadcast=True)
        self.set_dataset("how_many_atoms", [], broadcast=True)
        # Build scan sequences
        v1, s1 = self.HWP_angle_min, self.HWP_angle_step
        n1 = int(round((self.HWP_angle_max - v1) / s1)) + 1
        self.HWP_scan = [v1 + i * s1 for i in range(n1)]

        v2, s2 = self.QWP_angle_min, self.QWP_angle_step
        n2 = int(round((self.QWP_angle_max - v2) / s2)) + 1
        self.QWP_scan = [v2 + i * s2 for i in range(n2)]

    @kernel
    def run(self):
        self.base.initialize_hardware()
        # delay(1000 * ms)
        self.core.break_realtime()

        for qwp in self.QWP_scan:
            delay(100 * ms)
            move_to_target_deg(self, name="852_QWP", target_deg=qwp)
            delay(500 * ms)
            self.core.break_realtime()

            for hwp in self.HWP_scan:
                delay(100 * ms)
                move_to_target_deg(self, name="852_HWP", target_deg=hwp)
                if hwp == self.HWP_scan[0]:
                    delay(1000 * ms)
                else:
                    delay(500 * ms)
                self.core.break_realtime()

                measurements = 0
                how_many_atoms = 0   # how many atom load attempts for each measurement
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
                        how_many_atoms += 1

                        # Imaging PGC
                        imaging_pgc(self)
                        beams_off_for_imaging(self)

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
                    n = 0   # cycle counter for Check_img_every gating; definite-assignment safe int
                    atom_alive = True   # definite-assignment init; real value set in recovery branch below
                    final_ts1 = 0.0
                    final_ts2 = 0.0
                    while True:
                        optical_pumping_F3(self)                       # F=3 prep
                        self.zotino0.write_dac(0, self.shim_X_qaxis)   # Q-axis on
                        self.zotino0.write_dac(1, self.shim_Y_qaxis)
                        self.zotino0.write_dac(2, self.shim_Z1_qaxis)
                        self.zotino0.write_dac(3, self.shim_Z2_qaxis)
                        self.zotino0.load()
                        delay(1 * ms)
                        if n == 0:   # hoist trap ramp ONCE above the retry loop
                            ramp_trap_amp(self, self.trap_ramp_start_amp,   # trap DOWN
                                        trap_ramp_end_amp,
                                        self.trap_ramp_steps,
                                        self.trap_ramp_dt_us)

                        # Excitation cycles
                        for exc_cycle in range(int(self.max_excitation_cycles)):
                            n += 1
                            self.core.break_realtime()   # refresh RTIO slack each cycle

                            # Pre-stage MW IF for the leading pi pulse -- same
                            # 11-transition freq -- then optical pump.
                            self.dds_mw_IF.set(frequency=self.f_mw_11, amplitude=self.amp_mw_pulse)
                            optical_pumping(self)

                            # --- Excitation ---
                            trials += 1
                            t_end1, t_end2, t0 = excitation_pulse(self)
                            # Read SPCM timestamps inline (slack-critical; see parity.py note).
                            click1 = self.ttl_SPCM1.timestamp_mu(t_end1)
                            click2 = self.ttl_SPCM2.timestamp_mu(t_end2)

                            # Process photon detection (free-angle analysis pulse)
                            if click1 > 0 and click2 < 0:
                                microwave_two_photon_phase(self, click1, self.basis_phase_turns)
                                blowaway_f4(self, amp2)
                                final_ts1 = self.core.mu_to_seconds(click1 - t0)
                                final_ts2 = 0.0
                                photon_detected = True
                                break

                            if click1 < 0 and click2 > 0:
                                microwave_two_photon_phase(self, click2, self.basis_phase_turns)
                                blowaway_f4(self, amp2)
                                final_ts1 = 0.0
                                final_ts2 = self.core.mu_to_seconds(click2 - t0)
                                photon_detected = True
                                break

                            delay(5 * us)   # preserves the no-detection dead-time
                            delay(2 * us)
                            self.ttl_MW_switch.on()
                            delay(1 * us)

                        if photon_detected:
                            break

                        # Recovery (max cycles, no photon): PGC-recool at the LOW trap.
                        self.core.break_realtime()   # rebuild RTIO slack drained by the excitation cycles
                        self.zotino0.write_dac(0, self.shim_X_off)
                        self.zotino0.write_dac(1, self.shim_Y_off)
                        self.zotino0.write_dac(2, self.shim_Z1_off)
                        self.zotino0.write_dac(3, self.shim_Z2_off)
                        self.zotino0.load()
                        delay(1 * ms)

                        turn_fiber_aom_on(self)   # PGC/imaging path ON for recovery

                        delay(1 * us)
                        self.dds_repump.sw.on()
                        delay(1 * us)
                        self.ttl_repump_switch.off()         # unblock repump
                        imaging_pgc_low_trap(self)

                        if n % int(self.Check_img_every) == 0:
                            # shims were zeroed at recovery start (above) -- the low-trap
                            # survival image needs B~0 and does NOT zero shims itself.
                            delay(1 * ms)

                            atom_alive = recool_and_check_survival_lowtrap(self)   # recool+image+threshold
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
                            hwp, qwp, how_many_atoms
                        )
                        trials = 0
                        measurements += 1
                        how_many_atoms = 0
                        self.set_dataset("measurement_progress",
                                         measurements, broadcast=True)
                    # ===== END BLUE =====

                    reset_to_mot(self)

        write_results(self)

    @rpc(flags={"async"})
    def _save_measurement(self, ts1, ts2, trials, counts, counts_bg, hwp, qwp, how_many_atoms):
        """Append measurement to datasets. how_many_atoms = atom load attempts for
        this heralded measurement (reset to 0 after each success in run())."""
        self.append_to_dataset("SPCM1_timestamps", ts1)
        self.append_to_dataset("SPCM2_timestamps", ts2)
        self.append_to_dataset("trials_per_meas", trials)
        self.append_to_dataset("counts_signal", counts)
        self.append_to_dataset("counts_background", counts_bg)
        self.append_to_dataset("HWP_angles", hwp)
        self.append_to_dataset("QWP_angles", qwp)
        self.append_to_dataset("how_many_atoms", how_many_atoms)
        # Intermediate HDF5 save (mirrors parity.py / microwave_scan's incremental save)
        self._incremental_save()

    def _incremental_save(self):
        """Write current datasets to HDF5 mid-run (mirrors parity.py)."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
        except Exception as e:
            print("[incremental save]", e)
