"""Y-basis parity oscillation measurement.

Identical to X-basis parity except the RF antenna is set with phase=0.25
during the two-photon pulse, which rotates the measurement basis from X
to Y on the Bloch sphere.

Y-basis sequence after photon detection:
  1. MW pi pulse on 11 transition (t_pi_parity_mu, default 5120 mu)
  2. Set MW IF to f_mw_2ph (249.383 MHz), RF to f_RF_2ph (0.608 MHz)
     with phase=0.25
  3. Simultaneous MW + RF pulse for t_2ph_pulse_mu (default 106000 mu)
     for basis rotation
  4. Blowaway + final state readout

Default dashboard values come from archive Y_Parity_Optimized_v2
(20260414). Override per-run as needed.

TODO (v2 optimizations not yet ported, would benefit from cross-cutting
subroutine changes):
  - Check_img_every: survival check every N cycles vs. every cycle
  - Asymmetric blowaway delay: 9 ms (SPCM1 branch) vs 1 ms (SPCM2 branch)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.k10cr1_functions import move_to_target_deg
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom, imaging_pgc,
    spcm_image_dual, beams_off_for_imaging,
    optical_pumping_full, ramp_trap_amp, excitation_pulse,
    microwave_two_photon_y, blowaway_f4,
    recool_and_check_survival, pgc_between_excitations,
    final_state_readout, reset_to_mot,
)
from utilities.write_results import write_results


class YParityExperiment(EnvExperiment):
    """Y-basis parity Experiment"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Experiment-specific arguments (defaults from Y_Parity_Optimized_v2) ---
        self.setattr_argument("n_measurements",
                              NumberValue(100, min=1, max=600000, step=1),
                              group="Experiment")
        self.setattr_argument("max_excitation_cycles",
                              NumberValue(4, min=1, max=8000, step=1),
                              group="Excitation")
        self.setattr_argument("t_trap_off_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("t_spcm_collection_mu",
                              NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("length_offset_mu",
                              NumberValue(12, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")

        # Adiabatic trap ramp (v2 optimized: deeper start, finer steps, slower)
        self.setattr_argument("trap_ramp_start_amp",
                              NumberValue(0.24, min=0.0, max=0.3, ndecimals=3),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_steps",
                              NumberValue(200, min=0, max=100000, ndecimals=0),
                              group="Adiabatic")
        self.setattr_argument("trap_ramp_dt_us",
                              NumberValue(50, min=0.001, max=1000, ndecimals=3),
                              group="Adiabatic")

        # Waveplate angle scan (v2 optimized: narrow QWP window around 260 deg)
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
        # Result datasets
        self.set_dataset("SPCM1_timestamps", [], broadcast=True)
        self.set_dataset("SPCM2_timestamps", [], broadcast=True)
        self.set_dataset("trials_per_meas", [], broadcast=True)
        self.set_dataset("counts_signal", [], broadcast=True)
        self.set_dataset("counts_background", [], broadcast=True)
        self.set_dataset("HWP_angles", [], broadcast=True)
        self.set_dataset("QWP_angles", [], broadcast=True)
        v1, s1 = self.HWP_angle_min, self.HWP_angle_step
        n1 = int(round((self.HWP_angle_max - v1) / s1)) + 1
        self.HWP_scan = [v1 + i * s1 for i in range(n1)]

        v2, s2 = self.QWP_angle_min, self.QWP_angle_step
        n2 = int(round((self.QWP_angle_max - v2) / s2)) + 1
        self.QWP_scan = [v2 + i * s2 for i in range(n2)]

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

                    # ===== YELLOW: Loading =====
                    while True:
                        prepare_for_feedback(self)
                        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                        trap_ramp_end_amp = amp3
                        restore_after_feedback(self)
                        count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                        load_atom(self)
                        imaging_pgc(self)
                        beams_off_for_imaging(self)
                        delay(self.t_SPCM_exposure)
                        count1_1, count2_1 = spcm_image_dual(self, self.t_imaging_gate)
                        imaging_pgc(self)
                        beams_off_for_imaging(self)
                        if (count1_1 + count2_1 - count1_bg - count2_bg) > self.atom_threshold:
                            break

                    # ===== ORANGE: Excitation retry =====
                    photon_detected = False
                    final_ts1 = 0.0
                    final_ts2 = 0.0
                    trials = 0

                    while True:
                        optical_pumping_full(self)

                        for exc_cycle in range(int(self.max_excitation_cycles)):
                            trials += 1
                            click1, click2, t0 = excitation_pulse(self)

                            # Y-basis: two-photon rotation with RF phase=0.25
                            if click1 > 0 and click2 < 0:
                                microwave_two_photon_y(self, click1)
                                blowaway_f4(self, amp2)
                                final_ts1 = self.core.mu_to_seconds(click1 - t0)
                                final_ts2 = 0.0
                                photon_detected = True
                                break

                            if click1 < 0 and click2 > 0:
                                microwave_two_photon_y(self, click2)
                                blowaway_f4(self, amp2)
                                final_ts1 = 0.0
                                final_ts2 = self.core.mu_to_seconds(click2 - t0)
                                photon_detected = True
                                break

                            delay(2 * us)
                            self.ttl_MW_switch.on()
                            delay(1 * us)
                            pgc_between_excitations(self)

                        if photon_detected:
                            break

                        count_bg_total = count1_bg + count2_bg
                        atom_alive = recool_and_check_survival(self, count_bg_total)
                        if not atom_alive:
                            break

                    # ===== BLUE: Success =====
                    if photon_detected:
                        delay(1 * us)
                        ramp_trap_amp(self, trap_ramp_end_amp,
                                      self.trap_ramp_start_amp,
                                      self.trap_ramp_steps,
                                      self.trap_ramp_dt_us)
                        delay(1 * ms)
                        count1_final, count2_final = final_state_readout(self)
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

                    reset_to_mot(self)

        write_results(self)

    @rpc(flags={"async"})
    def _save_measurement(self, ts1, ts2, trials, counts, counts_bg, hwp, qwp):
        self.append_to_dataset("SPCM1_timestamps", ts1)
        self.append_to_dataset("SPCM2_timestamps", ts2)
        self.append_to_dataset("trials_per_meas", trials)
        self.append_to_dataset("counts_signal", counts)
        self.append_to_dataset("counts_background", counts_bg)
        self.append_to_dataset("HWP_angles", hwp)
        self.append_to_dataset("QWP_angles", qwp)
