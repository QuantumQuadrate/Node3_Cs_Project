"""Microwave spectroscopy: frequency, time, and Ramsey scans.

Supports 11 scan modes across 4 MW transitions (00, 01, 11, two-photon):
  - Frequency scans: find resonance peaks (GaussianDip fit)
  - Time scans: measure Rabi oscillations (RabiFit), extract pi-pulse times
  - Ramsey scans: measure coherence and detuning (RamseyFit)

Physics parameters (MW frequencies, pi-pulse times, Q-axis shims, trap ramp)
are read from ExperimentVariables persistent datasets — NOT redefined here.
Only scan-specific parameters (mode selection, scan ranges) are dashboard args.


Auto-calibration: when enabled, fitted resonance frequencies and pi-pulse
times are written back to persistent datasets for use by other experiments.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom, load_atom_deterministic, check_atom_loaded,
    imaging_readout_dual, spcm_image_dual,
    imaging_pgc, beams_off_for_imaging,
    f3_state_prep, set_qaxis, optical_pumping_mw_scan, ramp_trap_amp,
    blowaway_mw_scan, pgc_long,
    reload_beams, reset_mw_scan,
    mw_pulse_00, mw_pulse_01, mw_two_step_11, mw_two_step_2ph, mw_two_step_m10,
    ramsey_00, ramsey_11, ramsey_2ph,
)
from utilities.write_results import write_results

# Mode constants (integers for kernel dispatch)
MODE_FREQ_00 = 0
MODE_FREQ_01 = 1
MODE_FREQ_11 = 2
MODE_FREQ_2PH = 3
MODE_TIME_00 = 4
MODE_TIME_01 = 5
MODE_TIME_11 = 6
MODE_TIME_2PH = 7
MODE_RAMSEY_00 = 8
MODE_RAMSEY_11 = 9
MODE_RAMSEY_2PH = 10
MODE_FREQ_M10 = 11
MODE_TIME_M10 = 12


class MicrowaveScan(EnvExperiment):
    """MicrowaveScan_Exp"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # =============================================================
        #  SCAN-SPECIFIC ARGUMENTS (unique to MW scan, NOT in ExperimentVariables)
        # =============================================================

        # --- Scan mode selection (select exactly one) ---
        self.setattr_argument("Frequency_00_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Frequency_01_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Frequency_11_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Frequency_2ph_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Time_00_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Time_01_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Time_11_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Time_2ph_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Ramsey_00_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Ramsey_11_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Ramsey_2ph_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        # m10 transition |3,-1> <-> |4,0> (00 pi-pulse prep, then probe). Uses
        # the Frequency Scan / Time Scan ranges like the other freq/time modes.
        self.setattr_argument("Frequency_m10_Scan", BooleanValue(default=False), "Scan Mode (select one)")
        self.setattr_argument("Time_m10_Scan", BooleanValue(default=False), "Scan Mode (select one)")

        # --- Frequency scan range (MHz, dashboard display units) ---
        self.setattr_argument("Microwave_freq_min", NumberValue(250.4, min=190.0, max=260.0, step=0.0001, ndecimals=6), group="Frequency Scan")
        self.setattr_argument("Microwave_freq_max", NumberValue(250.6, min=190.0, max=260.0, step=0.0001, ndecimals=6), group="Frequency Scan")
        self.setattr_argument("Microwave_freq_step", NumberValue(0.01, min=0.000001, max=10.0, step=0.0001, ndecimals=6), group="Frequency Scan")
        self.setattr_argument("RF_freq_detun_min_kHz", NumberValue(-15.0, min=-50.0, max=50.0, step=0.01, ndecimals=6), group="Frequency Scan")
        self.setattr_argument("RF_freq_detun_max_kHz", NumberValue(-5.0, min=-50.0, max=50.0, step=0.01, ndecimals=6), group="Frequency Scan")
        self.setattr_argument("RF_freq_detun_step_kHz", NumberValue(0.5, min=0.001, max=50.0, step=0.01, ndecimals=6), group="Frequency Scan")

        # --- Time scan range (machine units) ---
        self.setattr_argument("time_min", NumberValue(0.0, min=0.0, max=1e7), group="Time Scan")
        self.setattr_argument("time_max", NumberValue(1e6, min=0.0, max=1e7), group="Time Scan")
        self.setattr_argument("time_step", NumberValue(50000.0, min=1.0, max=1e7), group="Time Scan")

        # --- Ramsey scan range ---
        # ramsey_time_step is used by Ramsey_00 (multi-segment) and Ramsey_11
        # (linear). Archive defaults: Ramsey_00 = 6000 mu, Ramsey_11 = 50000 mu.
        # ramsey_time_step_2ph is used by Ramsey_2ph only (multi-segment),
        # because the two-photon pi pulse is ~50x longer than the 00 pi pulse
        # so it needs a correspondingly coarser free-evolution step. Archive
        # default: 25000 mu.
        self.setattr_argument("ramsey_time_min", NumberValue(0.0, min=0.0, max=1e8), group="Ramsey Scan")
        self.setattr_argument("ramsey_time_max", NumberValue(1e6, min=0.0, max=1e8), group="Ramsey Scan")
        self.setattr_argument("ramsey_time_step", NumberValue(6000.0, min=1.0, max=1e8), group="Ramsey Scan")
        self.setattr_argument("ramsey_time_step_2ph", NumberValue(25000.0, min=1.0, max=1e8), group="Ramsey Scan")
        self.setattr_argument("ramsey_detuning_MHz", NumberValue(0.01, min=-1.0, max=1.0, ndecimals=6), group="Ramsey Scan")

        # --- Experiment control ---
        self.setattr_argument("n_measurements", NumberValue(300, min=1, max=600000, step=1), group="Experiment")
        self.setattr_argument("auto_update_calibration", BooleanValue(default=False), group="Calibration")

        # =============================================================
        #  All physics parameters (MW frequencies, pi-pulse times,
        #  Q-axis shims, trap ramp, feedback, optical pumping, etc.) are
        #  loaded automatically by BaseExperiment from datasets and JSON.
        #  Rebalancer is created and configured by BaseExperiment.build().
        # =============================================================

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # =============================================================
        #  PRE-RUN SNAPSHOT: record all physics parameters used for this run
        #  These are saved to the data file so it's fully self-describing.
        #  Values come from ExperimentVariables datasets (loaded by BaseExperiment).
        # =============================================================
        snapshot_params = {
            "f_mw_00": self.f_mw_00,
            "f_mw_01": self.f_mw_01,
            "f_mw_11": self.f_mw_11,
            "f_mw_m10": self.f_mw_m10,
            "f_mw_parity_00": self.f_mw_parity_00,
            "t_pi_00_mu": self.t_pi_00_mu,
            "t_pi_01_mu": self.t_pi_01_mu,
            "t_pi_11_mu": self.t_pi_11_mu,
            "t_pi_m10_mu": self.t_pi_m10_mu,
            "t_pi_2ph_mu": self.t_pi_2ph_mu,
            "rf_detune_khz": self.rf_detune_khz,
            "rf_small_detune_khz": self.rf_small_detune_khz,
            "shim_X_qaxis": self.shim_X_qaxis,
            "shim_Y_qaxis": self.shim_Y_qaxis,
            "shim_Z1_qaxis": self.shim_Z1_qaxis,
            "shim_Z2_qaxis": self.shim_Z2_qaxis,
            "trap_ramp_start_amp": self.trap_ramp_start_amp,
            "trap_ramp_steps": self.trap_ramp_steps,
            "trap_ramp_dt_us": self.trap_ramp_dt_us,
            "nominal_trapdepth": self.nominal_trapdepth,
            "atom_threshold": self.atom_threshold,
            "amp_mw_pulse": self.amp_mw_pulse,
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Determine active mode ---
        self._mode = -1
        self._mode_name = "none"
        self._mode_type = "none"
        self._target_dataset = ""

        mode_map = [
            (self.Frequency_00_Scan, MODE_FREQ_00, "Freq_00", "frequency", "f_mw_00"),
            (self.Frequency_01_Scan, MODE_FREQ_01, "Freq_01", "frequency", "f_mw_01"),
            (self.Frequency_11_Scan, MODE_FREQ_11, "Freq_11", "frequency", "f_mw_11"),
            (self.Frequency_2ph_Scan, MODE_FREQ_2PH, "Freq_2ph", "frequency", "f_RF_2ph"),
            (self.Time_00_Scan, MODE_TIME_00, "Time_00", "time", "t_pi_00_mu"),
            (self.Time_01_Scan, MODE_TIME_01, "Time_01", "time", "t_pi_01_mu"),
            (self.Time_11_Scan, MODE_TIME_11, "Time_11", "time", "t_pi_11_mu"),
            (self.Time_2ph_Scan, MODE_TIME_2PH, "Time_2ph", "time", "t_pi_2ph_mu"),
            (self.Ramsey_00_Scan, MODE_RAMSEY_00, "Ramsey_00", "ramsey", ""),
            (self.Ramsey_11_Scan, MODE_RAMSEY_11, "Ramsey_11", "ramsey", ""),
            (self.Ramsey_2ph_Scan, MODE_RAMSEY_2PH, "Ramsey_2ph", "ramsey", ""),
            (self.Frequency_m10_Scan, MODE_FREQ_M10, "Freq_m10", "frequency", "f_mw_m10"),
            (self.Time_m10_Scan, MODE_TIME_M10, "Time_m10", "time", "t_pi_m10_mu"),
        ]
        for flag, mode_id, name, mtype, target in mode_map:
            if flag:
                self._mode = mode_id
                self._mode_name = name
                self._mode_type = mtype
                self._target_dataset = target
                break

        if self._mode == -1:
            raise ValueError("No scan mode selected. Enable exactly one scan mode boolean.")

        # --- Derive internal frequencies from ExperimentVariables datasets ---
        # All self.f_mw_* are already in Hz (SI) from BaseExperiment._load_datasets()
        self._f00_hz = self.f_mw_00
        self._f01_hz = self.f_mw_01
        self._f11_hz = self.f_mw_11
        self._t00_mu = int(self.t_pi_00_mu)
        self._t01_mu = int(self.t_pi_01_mu)
        self._t11_mu = int(self.t_pi_11_mu)
        self._t2ph_mu = int(self.t_pi_2ph_mu)
        # m10 transition (|3,-1> <-> |4,0>): probe frequency + pi time
        self._fm10_hz = self.f_mw_m10
        self._tm10_mu = int(self.t_pi_m10_mu)

        # Two-photon derived frequencies (rf_detune_khz is raw kHz, not SI)
        f00_mhz = self._f00_hz / 1e6
        f01_mhz = self._f01_hz / 1e6
        detune_mhz = self.rf_detune_khz / 1000.0
        small_detune_mhz = self.rf_small_detune_khz / 1000.0
        self._fm10_2ph_hz = (2.0 * f00_mhz - f01_mhz + detune_mhz) * 1e6
        self._rf_2ph_hz = (f01_mhz - f00_mhz - detune_mhz + small_detune_mhz) * 1e6

        # Ramsey detuned frequencies (match original code exactly)
        # Old hardcoded detunings (kept for reference):
        # self._f00_ramsey_hz = (f00_mhz - 0.01 + 0.012) * 1e6
        # self._f11_ramsey_hz = (self._f11_hz / 1e6 - 0.03) * 1e6
        # self._rf_2ph_ramsey_hz = (self._rf_2ph_hz / 1e6 - 0.0002) * 1e6
        self._f00_ramsey_hz = (f00_mhz - self.ramsey_detuning_MHz) * 1e6
        self._f11_ramsey_hz = (self._f11_hz / 1e6 - self.ramsey_detuning_MHz) * 1e6
        self._rf_2ph_ramsey_hz = (self._rf_2ph_hz / 1e6 - self.ramsey_detuning_MHz) * 1e6

        # --- Build scan axis ---
        if self._mode_type == "frequency":
            if self._mode == MODE_FREQ_2PH:
                rf_min = self.RF_freq_detun_min_kHz / 1000.0
                rf_max = self.RF_freq_detun_max_kHz / 1000.0
                rf_step = self.RF_freq_detun_step_kHz / 1000.0
                n = int(round((rf_max - rf_min) / rf_step)) + 1
                self.scan_values = [
                    (f01_mhz - f00_mhz - detune_mhz + (rf_min + i * rf_step)) * 1e6
                    for i in range(n)
                ]
            else:
                n = int(round((self.Microwave_freq_max - self.Microwave_freq_min)
                              / self.Microwave_freq_step)) + 1
                self.scan_values = [
                    (self.Microwave_freq_min + i * self.Microwave_freq_step) * 1e6
                    for i in range(n)
                ]
        elif self._mode_type == "time":
            n = int(round((self.time_max - self.time_min) / self.time_step)) + 1
            self.scan_values = [float(int(self.time_min + i * self.time_step)) for i in range(n)]
        elif self._mode_type == "ramsey":
            # Pick the step appropriate to the mode (Finding #2).
            if self._mode == MODE_RAMSEY_2PH:
                step = self.ramsey_time_step_2ph
            else:
                step = self.ramsey_time_step
            n_per_seg = int(round((self.ramsey_time_max - self.ramsey_time_min) / step)) + 1
            if self._mode == MODE_RAMSEY_11:
                # Ramsey_11: single unsegmented linear scan (matches archive,
                # which used self.time_scan from time_min/time_max/time_step).
                self.scan_values = [float(int(self.ramsey_time_min + i * step)) for i in range(n_per_seg)]
            else:
                # Ramsey_00, Ramsey_2ph: 4-segment scan at 0/8/16/24 ms to
                # match archive time_scan_ramsey / time_scan_ramsey_2photon
                # (Finding #1). Each segment spans ramsey_time_min..max with
                # n_per_seg points; segments are 8 ms apart in absolute time.
                seg0 = [float(int(self.ramsey_time_min + i * step)) for i in range(n_per_seg)]
                seg1 = [float(int(8000000 + i * step)) for i in range(n_per_seg)]
                seg2 = [float(int(16000000 + i * step)) for i in range(n_per_seg)]
                seg3 = [float(int(24000000 + i * step)) for i in range(n_per_seg)]
                self.scan_values = seg0 + seg1 + seg2 + seg3

        # --- Result datasets ---
        self.set_dataset("mw_scan_x", [], broadcast=True, persist=True)
        self.set_dataset("mw_bg_spcm1", [], broadcast=True, persist=True)
        self.set_dataset("mw_bg_spcm2", [], broadcast=True, persist=True)
        self.set_dataset("mw_shot1_spcm1", [], broadcast=True, persist=True)
        self.set_dataset("mw_shot1_spcm2", [], broadcast=True, persist=True)
        self.set_dataset("mw_shot2_spcm1", [], broadcast=True, persist=True)
        self.set_dataset("mw_shot2_spcm2", [], broadcast=True, persist=True)
        self.set_dataset("mw_atom_loaded", [], broadcast=True, persist=True)
        self.set_dataset("mw_atom_retained", [], broadcast=True, persist=True)
        self.set_dataset("mw_retention", [], broadcast=True, persist=True)
        self.set_dataset("mw_load_attempts", [], broadcast=True, persist=True)
        self.set_dataset("mw_scan_complete", 0, broadcast=True)

        # --- Scan metadata ---
        self.set_dataset("mw_scan_mode", self._mode_name, persist=True, broadcast=True)
        self.set_dataset("mw_scan_type", self._mode_type, persist=True, broadcast=True)
        self.set_dataset("mw_target_dataset", self._target_dataset, persist=True, broadcast=True)
        self.set_dataset("mw_n_scan_points", len(self.scan_values), persist=True, broadcast=True)
        self.set_dataset("mw_n_loops", int(self.n_measurements), persist=True, broadcast=True)

        # --- Fit results (populated after run) ---
        self.set_dataset("mw_fit_success", False, persist=True, broadcast=True)
        self.set_dataset("mw_fit_extracted_value", 0.0, persist=True, broadcast=True)
        self.set_dataset("mw_fit_r_squared", 0.0, persist=True, broadcast=True)
        self.set_dataset("mw_fit_summary", "", persist=True, broadcast=True)

        print(f"MicrowaveScan: mode={self._mode_name}, {len(self.scan_values)} points")

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_measurements)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            for scan_idx in range(len(self.scan_values)):
                scan_val = self.scan_values[scan_idx]

                # === Feedback ===
                prepare_for_feedback(self)
                amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                trap_ramp_end_amp = amp3
                restore_after_feedback(self)

                # === Background + Load (no retry — matches original) ===
                count1_bg = 0
                count2_bg = 0
                count1 = 0
                count2 = 0
                load_attempts = 1
                count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                # load_atom(self)
                load_atom_deterministic(self)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                count1, count2 = spcm_image_dual(self, self.t_imaging_gate)

                # === Re-cool after the imaging shot (PGC) ===

                imaging_pgc(self)

                # === F=3 state preparation ===
                f3_state_prep(self)

                # === Q-axis → trap ramp down → optical pumping ===
                set_qaxis(self, self.shim_X_qaxis, self.shim_Y_qaxis,
                          self.shim_Z1_qaxis, self.shim_Z2_qaxis)
                ramp_trap_amp(self, self.trap_ramp_start_amp, trap_ramp_end_amp,
                              self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)
                optical_pumping_mw_scan(self)

                # === MODE DISPATCH: MW pulse ===
                if self._mode == 0:
                    mw_pulse_00(self, scan_val, self._t00_mu)
                elif self._mode == 1:
                    mw_pulse_01(self, scan_val, self._t01_mu)
                elif self._mode == 2:
                    mw_two_step_11(self, self._f01_hz, self._t01_mu,
                                   scan_val, self._t11_mu)
                elif self._mode == 3:
                    mw_two_step_2ph(self, self._f01_hz, self._t01_mu,
                                    self._fm10_2ph_hz, scan_val, self._t2ph_mu)
                elif self._mode == 4:
                    mw_pulse_00(self, self._f00_hz, int(scan_val))
                elif self._mode == 5:
                    mw_pulse_01(self, self._f01_hz, int(scan_val))
                elif self._mode == 6:
                    mw_two_step_11(self, self._f01_hz, self._t01_mu,
                                   self._f11_hz, int(scan_val))
                elif self._mode == 7:
                    mw_two_step_2ph(self, self._f01_hz, self._t01_mu,
                                    self._fm10_2ph_hz, self._rf_2ph_hz,
                                    int(scan_val))
                elif self._mode == 8:
                    ramsey_00(self, self._f00_ramsey_hz,
                              self._t00_mu // 2, int(scan_val))
                elif self._mode == 9:
                    ramsey_11(self, self._f01_hz, self._t01_mu,
                              self._f11_ramsey_hz,
                              self._t11_mu // 2, int(scan_val))
                elif self._mode == 10:
                    ramsey_2ph(self, self._f01_hz, self._t01_mu,
                               self._fm10_2ph_hz, self._rf_2ph_ramsey_hz,
                               self._t2ph_mu // 2, int(scan_val))
                elif self._mode == 11:
                    # Freq_m10: 00 pi prep, scan probe freq at fixed probe duration
                    mw_two_step_m10(self, self._f00_hz, self._t00_mu,
                                    scan_val, self._tm10_mu)
                elif self._mode == 12:
                    # Time_m10: 00 pi prep, fixed probe resonance, scan duration
                    mw_two_step_m10(self, self._f00_hz, self._t00_mu,
                                    self._fm10_hz, int(scan_val))

                # === Blowaway + trap ramp back ===
                blowaway_mw_scan(self, amp2)
                ramp_trap_amp(self, trap_ramp_end_amp, self.trap_ramp_start_amp,
                              self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)

                # === Long PGC after blowaway ===
                pgc_long(self)

                # === Shot 2: final state readout (dual SPCM) ===
                count1_post, count2_post = imaging_readout_dual(self, self.t_imaging_gate)
                # Alternate: spcm_image_dual(self, self.t_imaging_gate) — bare
                # parallel SPCM gates without imaging-beam restoration; swap in
                # for dual-SPCM mismatch debugging.

                # === Save data ===
                delay(1 * ms)
                self._save_point(scan_val,
                                 count1_bg, count2_bg,
                                 count1, count2,
                                 count1_post, count2_post,
                                 scan_idx, load_attempts)

                # === Reset for next scan point ===
                self.core.break_realtime()
                reset_mw_scan(self)

        # --- Post-run analysis ---
        self._analyze_and_update()
        write_results(self)

    @rpc
    def _save_point(self, scan_val,
                    bg_s1, bg_s2,
                    shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2,
                    scan_idx, load_attempts):
        """Append one scan point with threshold-based atom detection."""
        self.append_to_dataset("mw_scan_x", scan_val)
        self.append_to_dataset("mw_load_attempts", load_attempts)
        self.append_to_dataset("mw_bg_spcm1", bg_s1)
        self.append_to_dataset("mw_bg_spcm2", bg_s2)
        self.append_to_dataset("mw_shot1_spcm1", shot1_s1)
        self.append_to_dataset("mw_shot1_spcm2", shot1_s2)
        self.append_to_dataset("mw_shot2_spcm1", shot2_s1)
        self.append_to_dataset("mw_shot2_spcm2", shot2_s2)

        bg = bg_s1 + bg_s2
        shot1 = shot1_s1 + shot1_s2
        shot2 = shot2_s1 + shot2_s2
        threshold = self.atom_threshold

        loaded = int((shot1 - bg) > threshold)
        retained = int((shot2 - bg) > threshold)

        self.append_to_dataset("mw_atom_loaded", loaded)
        self.append_to_dataset("mw_atom_retained", retained)

        ret = float(retained) if loaded else -1.0
        self.append_to_dataset("mw_retention", ret)
        self.set_dataset("mw_scan_complete", scan_idx + 1, broadcast=True)

        # Incremental HDF5 save after each complete scan sweep
        if not hasattr(self, '_points_saved'):
            self._points_saved = 0
        self._points_saved += 1
        n_pts = max(len(self.scan_values), 1)
        if self._points_saved % n_pts == 0:
            self._incremental_save()

    def _incremental_save(self):
        """Save current data to HDF5 mid-run (called from _save_point)."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
            print(f"[incremental save] {self._points_saved} points saved to HDF5")
        except Exception as e:
            print(f"[incremental save] FAILED: {e}")

    @rpc
    def _analyze_and_update(self):
        """Post-run: fit scan data, save results, optionally update calibration."""
        import datetime
        from analysis.microwave_analysis import fit_scan_result

        x = self.get_dataset("mw_scan_x")
        retention = self.get_dataset("mw_retention")

        if len(x) < 3:
            print("Not enough data points for fitting.")
            self.set_dataset("mw_fit_success", False, persist=True, broadcast=True)
            self.set_dataset("mw_fit_summary", "Not enough data points", persist=True)
            return

        result = fit_scan_result(x, retention, self._mode_type, self._mode_name)

        if result is None or not result.get("success", False):
            print("Fit failed — calibration not updated.")
            self.set_dataset("mw_fit_success", False, persist=True, broadcast=True)
            self.set_dataset("mw_fit_summary", "Fit failed", persist=True)
            return

        # Persist fit results
        self.set_dataset("mw_fit_success", True, persist=True, broadcast=True)
        self.set_dataset("mw_fit_r_squared", result.get("r_squared", 0.0), persist=True, broadcast=True)
        self.set_dataset("mw_fit_summary", result.get("summary", ""), persist=True, broadcast=True)
        extracted = result.get("extracted_value")
        if extracted is not None:
            self.set_dataset("mw_fit_extracted_value", float(extracted), persist=True, broadcast=True)
        if "params" in result:
            self.set_dataset("mw_fit_params", list(result["params"]), persist=True)
        if "errors" in result:
            self.set_dataset("mw_fit_errors", list(result["errors"]), persist=True)

        # Auto-calibration: update persistent dataset + provenance
        if self.auto_update_calibration and self._target_dataset and extracted is not None:
            self.set_dataset(self._target_dataset, extracted, persist=True, broadcast=True)
            rid = self.scheduler.rid
            timestamp = datetime.datetime.now().isoformat()
            self.set_dataset(f"{self._target_dataset}_source_rid", rid, persist=True)
            self.set_dataset(f"{self._target_dataset}_source_time", timestamp, persist=True)
            self.set_dataset(f"{self._target_dataset}_source_mode", self._mode_name, persist=True)
            print(f"AUTO-CALIBRATION: {self._target_dataset} = {extracted} (RID {rid}, {timestamp})")
        elif not self.auto_update_calibration and self._target_dataset and extracted is not None:
            print(f"Suggested update: {self._target_dataset} = {extracted} (auto_update_calibration is OFF)")
