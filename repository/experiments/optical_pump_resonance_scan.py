"""Optical-pumping resonance scan: find the D1 pump resonance by F=4 blowaway.

Scans the D1 optical-pumping beam frequency (dds_D1_pump, i.e. f_pump_D1) and
looks for a DIP in atom retention.

WHY A DIP. Normal optical pumping runs the D1 pump TOGETHER with the pump
repumper, which cycles the atom until it parks in a dark m_F sublevel of F=3.
Here the repumper is OFF (amplitude 0), so the cycle is broken:

  * pump ON RESONANCE -> the D1 beam drives atoms out of F=3, they spontaneously
    decay, and whatever lands in F=4 STAYS there (nothing pumps it back). F=4
    population builds up -> the F=4 blowaway ejects the atom -> LOST.
  * pump OFF RESONANCE -> nothing is driven, the atom stays in F=3, which the
    blowaway does not touch -> RETAINED.

So retention vs pump frequency has a dip, and the dip centre is the pumping
resonance -> the value to put in `f_pump_D1`.

Sequence per shot (exactly the house pattern, as in optical_pump_scan.py):
    feedback (5-channel, per point)
    background window (FORT off)                        -> bg
    load_atom_deterministic
    PGC -> image 1 -> PGC                               -> shot1 (confirm + recool)
    optical_pumping_F3                                  (prepare F=3)
    set_qaxis -> ramp trap DOWN
    optical_pumping_timed(..., amp_repump = OP_repump_amp, f_d1 = SCANNED)
    blowaway_f4                                         (eject F=4)
    ramp trap UP
    final_state_readout                                 -> shot2 (survival)

The pumping step reuses the shared `optical_pumping_timed` kernel with
amp_repump = 0 rather than inlining a third copy of the pump choreography --
sequences.py already carries two hand-synced copies (optical_pumping and
optical_pumping_timed) and a third would be one more thing to keep in step.
`OP_repump_amp` is front-panel so you can run the repump-ON case as a control:
with the repumper restored the dip should wash out (pumping returns to F=3).

CAVEAT on amp_repump = 0: the repumper's RF switch is still switched ON by the
shared kernel, only its DDS amplitude is zero. That is how every other "beam
off by amplitude" case in this repo behaves, but it is not the same as a closed
switch -- if you suspect RF leakage through the repumper AOM, compare against a
run with OP_time = 0 (no pump window at all).

Datasets (one row per shot, appended, never truncated):
    opr_f_op          scanned D1-pump frequency actually played [Hz]
    opr_bg1/2         background window, per SPCM
    opr_shot1_spcm1/2 image 1 (after loading)
    opr_shot2_spcm1/2 image 2 (after blowaway)
    opr_loaded        1 if (shot1 - bg) > atom_threshold
    opr_retained      1 if (shot2 - bg) > atom_threshold
    opr_retention     retained, or -1.0 on shots that never loaded
    opr_loop_index    which repeat of the scan
Retention is only meaningful on loaded shots; the -1.0 sentinel marks the rest
(the same convention microwave_scan / optical_pump_scan use, and what
analysis/microwave_analysis.fit_scan_result expects).

After the run a Gaussian-dip fit is applied to retention vs frequency (the same
`fit_scan_result(..., "frequency", ...)` path microwave_scan uses) and the dip
centre is REPORTED ONLY -- it is printed and stored in opr_fit_resonance_Hz for
you to copy into `f_pump_D1` by hand if you want it.

AUTO-CALIBRATION IS DEPRECATED HERE. microwave_scan writes its fitted value
straight back into the calibration dataset; this experiment deliberately does
not. The write-back argument and code are kept COMMENTED OUT below (search
"DEPRECATED: auto-calibration") so the behaviour can be restored, rather than
deleted. Set fit_after_run = False to skip the fit entirely.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, beams_off_for_imaging, spcm_image_dual, mw_pulse_00,
    optical_pumping_F3, set_qaxis, optical_pumping_timed, ramp_trap_amp,
    blowaway_f4, final_state_readout, reset_mw_scan,
)
from utilities.write_results import write_results


class OpticalPumpResoScan(EnvExperiment):
    """OP Resonance Scan (D1 pump, no repump)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # =============================================================
        #  SCAN RANGE -- D1 pump frequency, in MHz.
        #  Plain NumberValue (no unit=), scaled by 1e6 in prepare(), which is
        #  the convention microwave_scan / optical_pump_scan already use for
        #  their MHz ranges. The live f_pump_D1 is 145.0 MHz, so the default
        #  brackets it by +-2 MHz.
        # =============================================================
        self.setattr_argument(
            "OP_freq_min", NumberValue(143.0, min=50.0, max=300.0, step=0.1, ndecimals=4),
            group="Frequency Scan (MHz)")
        self.setattr_argument(
            "OP_freq_max", NumberValue(147.0, min=50.0, max=300.0, step=0.1, ndecimals=4),
            group="Frequency Scan (MHz)")
        self.setattr_argument(
            "OP_freq_step", NumberValue(0.1, min=0.0001, max=50.0, step=0.01, ndecimals=4),
            group="Frequency Scan (MHz)")

        # --- Pump conditions ---
        # Pump window, microseconds. The live t_optical_pump is 350 us; 0 gives a
        # no-pump control (retention should then show no dip at all).
        self.setattr_argument(
            "OP_time_us", NumberValue(350.0, min=0.0, max=100000.0, step=10.0, ndecimals=2),
            group="Pump")
        # D1-pump amplitude. Live amp_pump_D1 is 0.04.
        self.setattr_argument(
            "OP_amp", NumberValue(0.04, min=0.0, max=1.0, step=0.005, ndecimals=3),
            group="Pump")
        # Repumper amplitude DURING the pump window. 0 = the measurement (no
        # repump, so F=4 accumulates). Set to amp_pump_repumper_active (~0.38)
        # as a control: the dip should wash out.
        self.setattr_argument(
            "OP_repump_amp", NumberValue(0.0, min=0.0, max=1.0, step=0.01, ndecimals=3),
            group="Pump")

        # --- Experiment control ---
        self.setattr_argument(
            "n_measurements", NumberValue(50, min=1, max=600000, step=1, ndecimals=0),
            group="Experiment")
        # Fit the dip after the run and PRINT the resonance. Reporting only --
        # nothing is written back to any calibration dataset.
        self.setattr_argument("fit_after_run", BooleanValue(default=True),
                              group="Calibration")
        # DEPRECATED: auto-calibration. Kept commented out so the write-back can
        # be restored without rebuilding it. Re-enabling needs BOTH this
        # argument and the matching block in _fit_and_report().
        # self.setattr_argument("auto_update_calibration",
        #                       BooleanValue(default=False), group="Calibration")

        # Everything else (shims, Q-axis, trap ramp, pump/blowaway parameters,
        # atom_threshold, the rebalancer) comes from BaseExperiment.

    # =====================================================================
    #  HOST SETUP
    # =====================================================================
    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # --- Self-describing snapshot of what this run actually used ---
        snapshot_params = {
            "f_pump_D1": self.f_pump_D1,
            "amp_pump_D1": self.amp_pump_D1,
            "f_pump_excitation": self.f_pump_excitation,
            "amp_pump_excitation": self.amp_pump_excitation,
            "f_pump_repumper": self.f_pump_repumper,
            "amp_pump_repumper_active": self.amp_pump_repumper_active,
            "t_optical_pump": self.t_optical_pump,
            "t_blowaway": self.t_blowaway,
            "shim_X_qaxis": self.shim_X_qaxis,
            "shim_Y_qaxis": self.shim_Y_qaxis,
            "shim_Z1_qaxis": self.shim_Z1_qaxis,
            "shim_Z2_qaxis": self.shim_Z2_qaxis,
            "trap_ramp_start_amp": self.trap_ramp_start_amp,
            "trap_ramp_steps": self.trap_ramp_steps,
            "trap_ramp_dt_us": self.trap_ramp_dt_us,
            "nominal_trapdepth": self.nominal_trapdepth,
            "atom_threshold": self.atom_threshold,
        }
        self._f00_hz = self.f_mw_00
        self._t00_mu = int(self.t_pi_00_mu)
        for key, val in snapshot_params.items():
            self.set_dataset("snapshot_{}".format(key), val, persist=True)

        # --- Scan axis: MHz front-panel -> Hz for the kernel ---
        if self.OP_freq_max < self.OP_freq_min:
            raise ValueError("OP_freq_max is below OP_freq_min.")
        n = int(round((self.OP_freq_max - self.OP_freq_min)
                      / self.OP_freq_step)) + 1
        self.scan_values = [(self.OP_freq_min + i * self.OP_freq_step) * 1e6
                            for i in range(n)]

        # Pump window in SECONDS: optical_pumping_timed takes t_pump in seconds.
        self._t_pump_s = float(self.OP_time_us) * 1e-6
        self._f_repump_hz = float(self.f_pump_repumper)
        self._amp_repump = float(self.OP_repump_amp)
        self._amp_d1 = float(self.OP_amp)

        # --- Result datasets: one row per shot ---
        for key in ("opr_f_op", "opr_bg1", "opr_bg2",
                    "opr_shot1_spcm1", "opr_shot1_spcm2",
                    "opr_shot2_spcm1", "opr_shot2_spcm2",
                    "opr_loaded", "opr_retained", "opr_retention",
                    "opr_loop_index"):
            self.set_dataset(key, [], broadcast=True)
        self.set_dataset("opr_scan_complete", 0, broadcast=True)
        self.set_dataset("opr_retention_live", 0.0, broadcast=True)

        # --- Scan metadata ---
        self.set_dataset("opr_n_points", len(self.scan_values),
                         persist=True, broadcast=True)
        self.set_dataset("opr_n_loops", int(self.n_measurements),
                         persist=True, broadcast=True)
        self.set_dataset("opr_t_pump_s", self._t_pump_s,
                         persist=True, broadcast=True)
        self.set_dataset("opr_amp_d1", self._amp_d1,
                         persist=True, broadcast=True)
        self.set_dataset("opr_amp_repump", self._amp_repump,
                         persist=True, broadcast=True)

        # --- Fit results (filled after the run) ---
        self.set_dataset("opr_fit_success", False, persist=True, broadcast=True)
        self.set_dataset("opr_fit_resonance_Hz", 0.0, persist=True, broadcast=True)
        self.set_dataset("opr_fit_summary", "", persist=True, broadcast=True)

        self._points_saved = 0
        self._n_loaded = 0
        self._n_retained = 0

        print("OpticalPumpResoScan: {} points, {:.4f} .. {:.4f} MHz step {:.4f}"
              .format(len(self.scan_values), self.OP_freq_min,
                      self.OP_freq_max, self.OP_freq_step))
        print("  pump window {:.1f} us, D1 amp {:.3f}, repump amp {:.3f}{}"
              .format(self.OP_time_us, self._amp_d1, self._amp_repump,
                      "  <- NO REPUMP (the measurement)"
                      if self._amp_repump == 0.0 else "  <- repump ON (control)"))
        if self.OP_time_us == 0.0:
            print("  NOTE: OP_time_us = 0 -> no pump window at all. This is the "
                  "null control; expect a flat curve with no dip.")

    # =====================================================================
    #  RUN
    # =====================================================================
    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_measurements)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            for scan_idx in range(len(self.scan_values)):
                f_op = self.scan_values[scan_idx]

                # === Feedback (per point, as in microwave_scan) ===
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
                count1_bg, count2_bg = background_measurement_dual(
                    self, self.t_imaging_gate)
                load_atom_deterministic(self)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                count1, count2 = spcm_image_dual(self, self.t_imaging_gate)

                # === Re-cool after the imaging shot (PGC) ===
                imaging_pgc(self)
                beams_off_for_imaging(self)   # block repump before F=3 prep

                # === F=3 state preparation ===
                self.dds_cooling_DP.sw.on()
                delay(10000 * us)
                self.dds_cooling_DP.sw.off()
                delay(1 * us)
                self.dds_D1_onoff_L.sw.off()
                delay(1 * us)


                # === Q-axis -> trap ramp DOWN -> SCANNED optical pumping ===
                set_qaxis(self, self.shim_X_qaxis, self.shim_Y_qaxis,
                          self.shim_Z1_qaxis, self.shim_Z2_qaxis)
                ramp_trap_amp(self, self.trap_ramp_start_amp, trap_ramp_end_amp,
                              self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)
                # The D1-pump FREQUENCY is the scan axis; the repumper amplitude
                # is 0 for the measurement, so F=4 accumulates on resonance.
                optical_pumping_timed(self, self._t_pump_s, self._f_repump_hz,
                                      self._amp_repump, f_op, self._amp_d1)
                mw_pulse_00(self, self._f00_hz, self._t00_mu)
                # === Blowaway F=4 + trap ramp UP + final readout (shot2) ===
                blowaway_f4(self, amp2)
                ramp_trap_amp(self, trap_ramp_end_amp, self.trap_ramp_start_amp,
                              self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)
                count1_post, count2_post = final_state_readout(self)

                # === Save ===
                delay(1 * ms)
                self._save_point(f_op,
                                 count1_bg, count2_bg,
                                 count1, count2,
                                 count1_post, count2_post,
                                 loop)

                # === Reset for the next point ===
                self.core.break_realtime()
                reset_mw_scan(self)

        self._fit_and_report()
        write_results(self)

    # =====================================================================
    #  HOST-SIDE SAVE / FIT
    # =====================================================================
    @rpc
    def _save_point(self, f_op,
                    bg_s1, bg_s2, shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2, loop_idx):
        """Append one shot, with threshold-based load/retain decisions."""
        self.append_to_dataset("opr_f_op", f_op)
        self.append_to_dataset("opr_bg1", bg_s1)
        self.append_to_dataset("opr_bg2", bg_s2)
        self.append_to_dataset("opr_shot1_spcm1", shot1_s1)
        self.append_to_dataset("opr_shot1_spcm2", shot1_s2)
        self.append_to_dataset("opr_shot2_spcm1", shot2_s1)
        self.append_to_dataset("opr_shot2_spcm2", shot2_s2)
        self.append_to_dataset("opr_loop_index", loop_idx)

        bg = bg_s1 + bg_s2
        shot1 = shot1_s1 + shot1_s2
        shot2 = shot2_s1 + shot2_s2
        threshold = self.atom_threshold

        loaded = int((shot1 - bg) > threshold)
        retained = int((shot2 - bg) > threshold)
        self.append_to_dataset("opr_loaded", loaded)
        self.append_to_dataset("opr_retained", retained)
        # -1.0 marks "never loaded", so the fitter can drop those shots.
        self.append_to_dataset("opr_retention",
                               float(retained) if loaded else -1.0)

        if loaded:
            self._n_loaded += 1
            self._n_retained += retained
        self.set_dataset("opr_retention_live",
                         (self._n_retained / self._n_loaded)
                         if self._n_loaded else 0.0, broadcast=True)

        self._points_saved += 1
        self.set_dataset("opr_scan_complete", self._points_saved,
                         broadcast=True)
        # Crash-safe HDF5 write once per completed sweep (a full-file write per
        # shot would back up the RPC queue).
        n_pts = max(len(self.scan_values), 1)
        if self._points_saved % n_pts == 0:
            self._incremental_save()

    def _incremental_save(self):
        """Write current datasets to HDF5 mid-run."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
            print("[incremental save] {} shots".format(self._points_saved))
        except Exception as e:
            print("[incremental save] FAILED: {}".format(e))

    @rpc
    def _fit_and_report(self):
        """Fit the retention dip vs pump frequency and REPORT the resonance.

        Reuses analysis/microwave_analysis.fit_scan_result in "frequency" mode,
        which is a Gaussian-dip fit -- the same code path microwave_scan uses
        for its resonance scans, so it is proven on this worker. Wrapped in
        try/except so a fit failure cannot lose the run's data (which is already
        on disk from the per-sweep incremental saves).

        Reporting only: the fitted centre lands in opr_fit_resonance_Hz and on
        stdout. Nothing is written back to f_pump_D1 -- see "DEPRECATED:
        auto-calibration" at the end of this method.
        """
        if not self.fit_after_run:
            print("fit_after_run is OFF -- data saved, no fit attempted.")
            return

        x = self.get_dataset("opr_f_op")
        retention = self.get_dataset("opr_retention")
        if len(x) < 3:
            print("Not enough data points for fitting.")
            self.set_dataset("opr_fit_summary", "Not enough data points",
                             persist=True, broadcast=True)
            return

        try:
            from analysis.microwave_analysis import fit_scan_result
            result = fit_scan_result(x, retention, "frequency", "OP_Resonance")
        except Exception as e:
            print("Fit raised: {} -- data is saved, fit skipped.".format(e))
            self.set_dataset("opr_fit_summary", "Fit error: {}".format(e),
                             persist=True, broadcast=True)
            return

        if result is None or not result.get("success", False):
            print("Fit failed -- calibration not updated.")
            self.set_dataset("opr_fit_summary", "Fit failed",
                             persist=True, broadcast=True)
            return

        self.set_dataset("opr_fit_success", True, persist=True, broadcast=True)
        self.set_dataset("opr_fit_summary", result.get("summary", ""),
                         persist=True, broadcast=True)
        extracted = result.get("extracted_value")
        print(result.get("summary", ""))
        if extracted is not None:
            self.set_dataset("opr_fit_resonance_Hz", float(extracted),
                             persist=True, broadcast=True)
            print("OP resonance (dip centre) = {:.6f} MHz".format(
                float(extracted) / 1e6))
            print("  f_pump_D1 is currently {:.6f} MHz -- set it by hand if you"
                  " trust this fit.".format(float(self.f_pump_D1) / 1e6))

        # ------------------------------------------------------------------
        # DEPRECATED: auto-calibration.
        # This experiment deliberately does NOT write the fit back; the number
        # above is for you to apply by hand. Restoring the write-back needs this
        # block AND the commented-out auto_update_calibration argument in
        # build(). Kept rather than deleted so it is revertable.
        #
        # import datetime
        # if self.auto_update_calibration and extracted is not None:
        #     self.set_dataset("f_pump_D1", float(extracted),
        #                      persist=True, broadcast=True)
        #     rid = self.scheduler.rid
        #     stamp = datetime.datetime.now().isoformat()
        #     self.set_dataset("f_pump_D1_source_rid", rid, persist=True)
        #     self.set_dataset("f_pump_D1_source_time", stamp, persist=True)
        #     self.set_dataset("f_pump_D1_source_mode", "OP_Resonance",
        #                      persist=True)
        #     print("AUTO-CALIBRATION: f_pump_D1 = {} (RID {}, {})".format(
        #         extracted, rid, stamp))
        # ------------------------------------------------------------------
