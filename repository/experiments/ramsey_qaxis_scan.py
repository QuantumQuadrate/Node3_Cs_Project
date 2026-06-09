"""Ramsey-vs-Q-axis scan: a Ramsey free-evolution scan at each of N magnetic-field
(quantization-axis) settings.

The 00 (clock) transition is field-insensitive, so its frequency is pinned to
F00_MHZ = 250 MHz and its pi time comes from the t_pi_00_mu dataset. The
field-sensitive 01 / two-photon parameters (f01, t01, small_detune, t2ph) are
supplied per field through fixed dashboard slots.

One Ramsey mode per run (Ramsey_00 OR Ramsey_2ph). For each Q-axis value the
experiment sweeps the Ramsey free-evolution time and records loading/retention.

Reuses subroutines/datasets from the repository without modifying any other
file. Mirrors the per-shot sequence of microwave_scan.py, INCLUDING the
post-image PGC recool that protects coherence.

Units (kept explicit on purpose):
  - dashboard f01 in MHz  -> *1e6 -> Hz for the subroutines
  - small_detune in kHz, rf_detune (dataset) in kHz -> /1000 -> MHz
  - ramsey fringe detunings in MHz
  - pulse / free-evolution times in machine units (int); half-pi = t_pi // 2
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, beams_off_for_imaging, spcm_image_dual,
    optical_pumping_F3, set_qaxis, optical_pumping, ramp_trap_amp,
    blowaway_f4, final_state_readout, reset_mw_scan,
    ramsey_00, ramsey_2ph,
)
from utilities.write_results import write_results

# Maximum number of Q-axis (field) slots shown on the dashboard.
N_QAXIS_MAX = 8

# Kernel-dispatch mode integers (only two modes in this experiment).
MODE_RAMSEY_00 = 0
MODE_RAMSEY_2PH = 1

# 00 (clock) transition is field-insensitive -> frequency pinned here (MHz).
F00_MHZ = 250.0


class RamseyQaxisScan(EnvExperiment):
    """RamseyQaxisScan_Exp"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Mode selection (select exactly one) ---
        self.setattr_argument("Ramsey_00_Scan", BooleanValue(default=True),
                              "Ramsey Mode (select one)")
        self.setattr_argument("Ramsey_2ph_Scan", BooleanValue(default=False),
                              "Ramsey Mode (select one)")

        # --- How many of the fixed Q-axis slots to actually use ---
        self.setattr_argument(
            "n_qaxis",
            NumberValue(1, min=1, max=N_QAXIS_MAX, step=1, ndecimals=0, type="int"),
            group="Q-axis Selection")

        # Shuffle the field VISIT ORDER (decorrelates field order from slow
        # drift). Each field still runs its full loop + free-time scan; only the
        # order of fields is permuted. seed >= 0 is reproducible; seed < 0 draws
        # a random seed and records it (so a "random" run is reproducible after).
        self.setattr_argument("shuffle_qaxis_order", BooleanValue(default=False),
                              group="Q-axis Selection")
        self.setattr_argument("shuffle_seed",
                              NumberValue(-1, min=-1, max=2147483647, step=1, ndecimals=0, type="int"),
                              group="Q-axis Selection")

        # --- Fixed per-field slots (only the first n_qaxis are used) ---
        # Each slot carries the 4 shim voltages AND the 2ph calibration
        # (f01, t01, small detune, t2ph). For Ramsey_00 only the shims matter.
        for i in range(1, N_QAXIS_MAX + 1):
            ii = "{:02d}".format(i)
            grp = "Q-axis {}".format(ii)
            self.setattr_argument("qa{}_shimX".format(ii),
                                  NumberValue(-0.013, min=-10.0, max=10.0, ndecimals=4), group=grp)
            self.setattr_argument("qa{}_shimY".format(ii),
                                  NumberValue(0.0, min=-10.0, max=10.0, ndecimals=4), group=grp)
            self.setattr_argument("qa{}_shimZ1".format(ii),
                                  NumberValue(-0.06, min=-10.0, max=10.0, ndecimals=4), group=grp)
            self.setattr_argument("qa{}_shimZ2".format(ii),
                                  NumberValue(0.06, min=-10.0, max=10.0, ndecimals=4), group=grp)
            self.setattr_argument("qa{}_f01_MHz".format(ii),
                                  NumberValue(250.0, min=190.0, max=260.0, ndecimals=6), group=grp)
            self.setattr_argument("qa{}_t01_mu".format(ii),
                                  NumberValue(4530, min=0, max=10000000, step=1, ndecimals=0, type="int"), group=grp)
            self.setattr_argument("qa{}_small_detune_kHz".format(ii),
                                  NumberValue(0.0, min=-1000.0, max=1000.0, ndecimals=6), group=grp)
            self.setattr_argument("qa{}_t2ph_mu".format(ii),
                                  NumberValue(200000, min=0, max=10000000, step=1, ndecimals=0, type="int"), group=grp)

        # --- Ramsey free-evolution scan range (machine units) ---
        self.setattr_argument("ramsey_time_min", NumberValue(0.0, min=0.0, max=1e8), group="Ramsey Scan")
        self.setattr_argument("ramsey_time_max", NumberValue(1e6, min=0.0, max=1e8), group="Ramsey Scan")
        self.setattr_argument("ramsey_time_step", NumberValue(6000.0, min=1.0, max=1e8), group="Ramsey Scan")
        # 2ph pi is ~50x longer than the 00 pi, so it needs a coarser step.
        self.setattr_argument("ramsey_time_step_2ph", NumberValue(25000.0, min=1.0, max=1e8), group="Ramsey Scan")
        # Separate fringe detunings: 00 and 2ph need different values.
        self.setattr_argument("ramsey_detuning_00_MHz", NumberValue(0.01, min=-1.0, max=1.0, ndecimals=6), group="Ramsey Scan")
        self.setattr_argument("ramsey_detuning_2ph_MHz", NumberValue(0.01, min=-1.0, max=1.0, ndecimals=6), group="Ramsey Scan")

        # --- Experiment control ---
        self.setattr_argument(
            "n_measurements",
            NumberValue(300, min=1, max=600000, step=1, ndecimals=0, type="int"),
            group="Experiment")

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # --- Mode selection + validation ---
        if self.Ramsey_00_Scan and self.Ramsey_2ph_Scan:
            raise ValueError("Select exactly ONE Ramsey mode, not both.")
        if self.Ramsey_00_Scan:
            self._mode = MODE_RAMSEY_00
            self._mode_name = "Ramsey_00"
        elif self.Ramsey_2ph_Scan:
            self._mode = MODE_RAMSEY_2PH
            self._mode_name = "Ramsey_2ph"
        else:
            raise ValueError("No Ramsey mode selected (choose Ramsey_00 or Ramsey_2ph).")

        # --- Active slot count ---
        self._n_q = max(1, min(int(self.n_qaxis), N_QAXIS_MAX))

        # --- Field visit order (optionally shuffled; recorded for reference) ---
        self._field_order = list(range(self._n_q))
        if self.shuffle_qaxis_order:
            seed = int(self.shuffle_seed)
            if seed < 0:
                seed = random.randrange(2**31)
            random.Random(seed).shuffle(self._field_order)
            self._shuffle_seed_used = int(seed)
        else:
            self._shuffle_seed_used = -1

        # --- Globals (units handled explicitly) ---
        # rf_detune_khz is the GLOBAL big 2ph detuning (dataset); kHz -> MHz.
        rf_detune_mhz = float(self.rf_detune_khz) / 1000.0
        # 00 transition is field-insensitive -> scalar parameters.
        self._f00_ramsey_hz = float((F00_MHZ - float(self.ramsey_detuning_00_MHz)) * 1e6)
        self._t_half_00_mu = int(int(self.t_pi_00_mu) // 2)

        # --- Per-field arrays (homogeneously typed: force float()/int()) ---
        self._q_shimX = []
        self._q_shimY = []
        self._q_shimZ1 = []
        self._q_shimZ2 = []
        self._q_f01_hz = []
        self._q_t01_mu = []
        self._q_fm10_2ph_hz = []
        self._q_rf_2ph_ramsey_hz = []
        self._q_t_half_2ph_mu = []
        self._q_small_detune_khz = []
        self._q_t2ph_mu = []

        for i in range(1, self._n_q + 1):
            ii = "{:02d}".format(i)
            sx = float(getattr(self, "qa{}_shimX".format(ii)))
            sy = float(getattr(self, "qa{}_shimY".format(ii)))
            sz1 = float(getattr(self, "qa{}_shimZ1".format(ii)))
            sz2 = float(getattr(self, "qa{}_shimZ2".format(ii)))
            f01_mhz = float(getattr(self, "qa{}_f01_MHz".format(ii)))
            t01 = int(getattr(self, "qa{}_t01_mu".format(ii)))
            sdet_khz = float(getattr(self, "qa{}_small_detune_kHz".format(ii)))
            sdet_mhz = sdet_khz / 1000.0
            t2ph = int(getattr(self, "qa{}_t2ph_mu".format(ii)))

            # Two-photon frequency derivation (mirrors microwave_scan.py:184-198,
            # with f00 pinned to F00_MHZ). Compute in MHz, then *1e6 -> Hz.
            fm10_2ph_hz = (2.0 * F00_MHZ - f01_mhz + rf_detune_mhz) * 1e6
            rf_2ph_hz = (f01_mhz - F00_MHZ - rf_detune_mhz + sdet_mhz) * 1e6
            rf_2ph_ramsey_hz = (rf_2ph_hz / 1e6 - float(self.ramsey_detuning_2ph_MHz)) * 1e6

            self._q_shimX.append(float(sx))
            self._q_shimY.append(float(sy))
            self._q_shimZ1.append(float(sz1))
            self._q_shimZ2.append(float(sz2))
            self._q_f01_hz.append(float(f01_mhz * 1e6))
            self._q_t01_mu.append(int(t01))
            self._q_fm10_2ph_hz.append(float(fm10_2ph_hz))
            self._q_rf_2ph_ramsey_hz.append(float(rf_2ph_ramsey_hz))
            self._q_t_half_2ph_mu.append(int(t2ph // 2))
            self._q_small_detune_khz.append(float(sdet_khz))
            self._q_t2ph_mu.append(int(t2ph))

        # --- Free-evolution scan axis: 4-segment scan at ~0/8/16/24 ms ---
        # Matches microwave_scan's Ramsey_00/Ramsey_2ph (lines 221-241): sample
        # near 0, 8e6, 16e6, 24e6 mu so fast fringes are captured at several
        # absolute-time windows. 2ph uses a coarser step (its pi is ~50x longer).
        if self._mode == MODE_RAMSEY_2PH:
            step = float(self.ramsey_time_step_2ph)
        else:
            step = float(self.ramsey_time_step)
        n_per_seg = max(1, int(round((float(self.ramsey_time_max) - float(self.ramsey_time_min)) / step)) + 1)
        seg_anchors_mu = [float(self.ramsey_time_min), 8000000.0, 16000000.0, 24000000.0]
        self.scan_values = []
        for anchor in seg_anchors_mu:
            for i in range(n_per_seg):
                self.scan_values.append(float(int(anchor + i * step)))

        # --- Per-shot result datasets ---
        for key in (
            # Exact per-shot field config so each data point is self-describing
            # even when the visit order is shuffled: shimY + all corresponding
            # per-field values.
            "rqs_shimY", "rqs_shimX", "rqs_shimZ1", "rqs_shimZ2",
            "rqs_f01_hz", "rqs_t01_mu", "rqs_small_detune_khz", "rqs_t2ph_mu",
            "rqs_free_time_mu",
            "rqs_bg_spcm1", "rqs_bg_spcm2",
            "rqs_shot1_spcm1", "rqs_shot1_spcm2",
            "rqs_shot2_spcm1", "rqs_shot2_spcm2",
            "rqs_atom_loaded", "rqs_atom_retained", "rqs_retention",
        ):
            self.set_dataset(key, [], broadcast=True, persist=True)
        self.set_dataset("rqs_scan_complete", 0, broadcast=True)

        # --- Metadata ---
        self.set_dataset("rqs_scan_mode", self._mode_name, broadcast=True, persist=True)
        self.set_dataset("rqs_n_qaxis", self._n_q, broadcast=True, persist=True)
        self.set_dataset("rqs_n_free_points", len(self.scan_values), broadcast=True, persist=True)
        self.set_dataset("rqs_n_loops", int(self.n_measurements), broadcast=True, persist=True)
        self.set_dataset("rqs_free_time_axis", list(self.scan_values), broadcast=True, persist=True)
        self.set_dataset("rqs_field_order", list(self._field_order), broadcast=True, persist=True)
        self.set_dataset("rqs_shuffle_seed", self._shuffle_seed_used, broadcast=True, persist=True)

        # --- Per-field snapshot (makes the data file self-describing) ---
        self.set_dataset("rqs_field_shimX", list(self._q_shimX), broadcast=True, persist=True)
        self.set_dataset("rqs_field_shimY", list(self._q_shimY), broadcast=True, persist=True)
        self.set_dataset("rqs_field_shimZ1", list(self._q_shimZ1), broadcast=True, persist=True)
        self.set_dataset("rqs_field_shimZ2", list(self._q_shimZ2), broadcast=True, persist=True)
        self.set_dataset("rqs_field_f01_hz", list(self._q_f01_hz), broadcast=True, persist=True)
        self.set_dataset("rqs_field_t01_mu", list(self._q_t01_mu), broadcast=True, persist=True)
        self.set_dataset("rqs_field_t_half_2ph_mu", list(self._q_t_half_2ph_mu), broadcast=True, persist=True)
        self.set_dataset("rqs_field_fm10_2ph_hz", list(self._q_fm10_2ph_hz), broadcast=True, persist=True)
        self.set_dataset("rqs_field_rf_2ph_ramsey_hz", list(self._q_rf_2ph_ramsey_hz), broadcast=True, persist=True)

        # --- Global snapshot ---
        self.set_dataset("snapshot_F00_MHz", float(F00_MHZ), broadcast=True, persist=True)
        self.set_dataset("snapshot_rf_detune_khz", float(self.rf_detune_khz), broadcast=True, persist=True)
        self.set_dataset("snapshot_ramsey_detuning_00_MHz", float(self.ramsey_detuning_00_MHz), broadcast=True, persist=True)
        self.set_dataset("snapshot_ramsey_detuning_2ph_MHz", float(self.ramsey_detuning_2ph_MHz), broadcast=True, persist=True)
        self.set_dataset("snapshot_ramsey_step_used", float(step), broadcast=True, persist=True)
        self.set_dataset("snapshot_t_pi_00_mu", int(self.t_pi_00_mu), broadcast=True, persist=True)
        self.set_dataset("snapshot_f00_ramsey_hz", float(self._f00_ramsey_hz), broadcast=True, persist=True)
        self.set_dataset("snapshot_amp_mw_pulse", float(self.amp_mw_pulse), broadcast=True, persist=True)
        self.set_dataset("snapshot_amp_RF_2ph", float(self.amp_RF_2ph), broadcast=True, persist=True)
        self.set_dataset("snapshot_atom_threshold", float(self.atom_threshold), broadcast=True, persist=True)
        self.set_dataset("snapshot_nominal_trapdepth", float(self.nominal_trapdepth), broadcast=True, persist=True)
        self.set_dataset("snapshot_trap_ramp_start_amp", float(self.trap_ramp_start_amp), broadcast=True, persist=True)
        self.set_dataset("snapshot_trap_ramp_steps", int(self.trap_ramp_steps), broadcast=True, persist=True)
        self.set_dataset("snapshot_trap_ramp_dt_us", float(self.trap_ramp_dt_us), broadcast=True, persist=True)

        print("RamseyQaxisScan: mode={}, n_qaxis={}, {} free-time points ({} per seg x 4 segments), step={} mu, n_measurements={}".format(
            self._mode_name, self._n_q, len(self.scan_values), n_per_seg, int(step), int(self.n_measurements)))
        print("  field visit order: {} (shuffle={}, seed={})".format(
            self._field_order, bool(self.shuffle_qaxis_order), self._shuffle_seed_used))

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_measurements)


        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)
            for qi in range(self._n_q):
                q = self._field_order[qi]          # shuffled visit order
                self.core.break_realtime()

                # Bind per-field values to typed locals once per field.
                sx = self._q_shimX[q]
                sy = self._q_shimY[q]
                sz1 = self._q_shimZ1[q]
                sz2 = self._q_shimZ2[q]
                f01_hz = self._q_f01_hz[q]
                t01 = self._q_t01_mu[q]
                fm10 = self._q_fm10_2ph_hz[q]
                rf2 = self._q_rf_2ph_ramsey_hz[q]
                th2 = self._q_t_half_2ph_mu[q]
                sdet = self._q_small_detune_khz[q]   # for per-shot logging
                t2ph_full = self._q_t2ph_mu[q]       # for per-shot logging

            # for loop in range(n_loops):
            #     self.core.break_realtime()
            #     delay(100 * us)

                for scan_idx in range(len(self.scan_values)):
                    free_t = self.scan_values[scan_idx]

                    # === Feedback ===
                    prepare_for_feedback(self)
                    amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                    trap_ramp_end_amp = amp3
                    restore_after_feedback(self)

                    # === Background + load + loading image ===
                    count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                    load_atom_deterministic(self)
                    imaging_pgc(self)
                    beams_off_for_imaging(self)
                    count1, count2 = spcm_image_dual(self, self.t_imaging_gate)

                    # === Re-cool after the imaging shot (PGC) -- coherence fix ===
                    imaging_pgc(self)
                    beams_off_for_imaging(self)   # block repump before F=3 prep (matches parity)

                    # === F=3 prep (unified: parity's optical_pumping_F3) -> Q-axis -> trap ramp -> optical pump ===
                    optical_pumping_F3(self)
                    set_qaxis(self, sx, sy, sz1, sz2)
                    ramp_trap_amp(self, self.trap_ramp_start_amp, trap_ramp_end_amp,
                                  self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)
                    optical_pumping(self)

                    # === Ramsey pulse (mode dispatch) ===
                    if self._mode == 0:            # MODE_RAMSEY_00
                        ramsey_00(self, self._f00_ramsey_hz, self._t_half_00_mu, int(free_t))
                    elif self._mode == 1:          # MODE_RAMSEY_2PH
                        ramsey_2ph(self, f01_hz, t01, fm10, rf2, th2, int(free_t))

                    # === Blowaway -> trap ramp back ===
                    blowaway_f4(self, amp2)
                    ramp_trap_amp(self, trap_ramp_end_amp, self.trap_ramp_start_amp,
                                  self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)

                    # === Recool + final readout (unified: parity's final_state_readout) ===
                    count1_post, count2_post = final_state_readout(self)

                    # === Save ===
                    delay(1 * ms)
                    self._save_point(sy, sx, sz1, sz2, f01_hz, t01, sdet, t2ph_full,
                                     free_t, count1_bg, count2_bg,
                                     count1, count2, count1_post, count2_post, scan_idx)

                    self.core.break_realtime()
                    reset_mw_scan(self)

        write_results(self)

    @rpc
    def _save_point(self, shimY, shimX, shimZ1, shimZ2, f01_hz, t01_mu,
                    small_detune_khz, t2ph_mu, free_time, bg_s1, bg_s2,
                    shot1_s1, shot1_s2, shot2_s1, shot2_s2, scan_idx):
        """Append one Ramsey point. Bare (blocking) @rpc: it reads
        self.scan_values and appends in order, so it must not be async.

        The exact per-field VALUES (shimY = q-axis strength, plus the other
        shims and the 2ph calibration) are passed in by value and saved per
        shot, so every data point is self-describing even when the field visit
        order is shuffled."""
        self.append_to_dataset("rqs_shimY", shimY)
        self.append_to_dataset("rqs_shimX", shimX)
        self.append_to_dataset("rqs_shimZ1", shimZ1)
        self.append_to_dataset("rqs_shimZ2", shimZ2)
        self.append_to_dataset("rqs_f01_hz", f01_hz)
        self.append_to_dataset("rqs_t01_mu", t01_mu)
        self.append_to_dataset("rqs_small_detune_khz", small_detune_khz)
        self.append_to_dataset("rqs_t2ph_mu", t2ph_mu)
        self.append_to_dataset("rqs_free_time_mu", free_time)
        self.append_to_dataset("rqs_bg_spcm1", bg_s1)
        self.append_to_dataset("rqs_bg_spcm2", bg_s2)
        self.append_to_dataset("rqs_shot1_spcm1", shot1_s1)
        self.append_to_dataset("rqs_shot1_spcm2", shot1_s2)
        self.append_to_dataset("rqs_shot2_spcm1", shot2_s1)
        self.append_to_dataset("rqs_shot2_spcm2", shot2_s2)

        bg = bg_s1 + bg_s2
        shot1 = shot1_s1 + shot1_s2
        shot2 = shot2_s1 + shot2_s2
        threshold = self.atom_threshold

        loaded = int((shot1 - bg) > threshold)
        retained = int((shot2 - bg) > threshold)
        self.append_to_dataset("rqs_atom_loaded", loaded)
        self.append_to_dataset("rqs_atom_retained", retained)
        ret = float(retained) if loaded else -1.0
        self.append_to_dataset("rqs_retention", ret)

        self.set_dataset("rqs_scan_complete", scan_idx + 1, broadcast=True)

        # Incremental HDF5 save after each complete free-time sweep.
        if not hasattr(self, "_points_saved"):
            self._points_saved = 0
        self._points_saved += 1
        n_pts = max(len(self.scan_values), 1)
        if self._points_saved % n_pts == 0:
            self._incremental_save()

    def _incremental_save(self):
        """Mid-run HDF5 snapshot (plain function, called from _save_point)."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
            print("[incremental save] {} points saved to HDF5".format(self._points_saved))
        except Exception as e:
            print("[incremental save] FAILED: {}".format(e))
