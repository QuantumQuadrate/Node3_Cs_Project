"""Optical-pumping 2-D scan -- mF=0 state-prep fidelity vs. two selectable knobs.

Generalises the old pump-TIME scan into a 2-D scan whose two axes are chosen on the
dashboard from {t_optical_pump, f_pump_repumper, amp_pump_repumper_active} (set the
Y axis to "none" for a 1-D scan). Unselected knobs keep their live value.

Motivation: the no-pi reference showed the pump REPUMPER is imperfect -- a real
fraction of atoms is left in F=4 after pumping (lost_noPi > 0). Scanning the
repumper frequency (f_pump_repumper) x amplitude (amp_pump_repumper_active) finds
the (f, amp) that drives the F=4 leftover down / maximises the clean mF=0 signal.

Probe (unchanged): at the Q-axis B-field the mw00 pi-pulse drives ONLY
|F=3,mF=0> -> |F=4,mF=0>; blow away F=4, then image. Interleaved pi/no-pi at every
grid point: cfg=1 applies the pi, cfg=0 is a matched dark-hold (MW blocked). Among
loaded atoms, lost = 1 - retention; the clean mF=0 signal is
  signal = lost_pi - lost_noPi
(lost_noPi = the F=4-leftover background). See analyze_optical_pump_scan.

Sequence (unchanged): feedback -> load -> confirm (shot1, high trap) ->
optical_pumping_F3 -> set_qaxis -> ramp DOWN -> optical_pumping_timed(t_pump,
f_repump, amp_repump) -> (pi / matched no-pi) -> blowaway_f4 -> ramp UP ->
final_state_readout (shot2). The three scannable knobs are kernel arguments of
optical_pumping_timed (a parallel copy of optical_pumping); optical_pumping() is
untouched.

Optimum is PRINTED, not written -- set the params on the dashboard yourself.
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

# Scannable knobs (the X/Y axes are chosen from these).
_AXES = ["t_optical_pump", "f_pump_repumper", "amp_pump_repumper_active",
         "f_pump_D1", "amp_pump_D1"]


class OpticalPumpScan(EnvExperiment):
    """OpticalPumpScan_Exp"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # --- Scan-axis selection. Pick X and Y; set axis_y="none" for a 1-D scan.
        #     An unselected knob keeps its live (dataset) value. ---
        self.setattr_argument("axis_x", EnumerationValue(_AXES, default="f_pump_repumper"),
                              group="Scan axes")
        self.setattr_argument("axis_y", EnumerationValue(["none"] + _AXES, default="amp_pump_repumper_active"),
                              group="Scan axes")

        # --- Per-knob ranges (each in its NATURAL unit; only the selected axes are
        #     used). Plain NumberValue (no unit= -> no double-scaling). ---
        # Pump-window time (us). prepare() prints the live t_optical_pump.
        self.setattr_argument("t_pump_min_us", NumberValue(0.0, min=0.0, max=100000.0, ndecimals=2, step=1.0), group="Pump-time range (us)")
        self.setattr_argument("t_pump_max_us", NumberValue(2000.0, min=0.0, max=100000.0, ndecimals=2, step=1.0), group="Pump-time range (us)")
        self.setattr_argument("t_pump_step_us", NumberValue(200.0, min=0.01, max=100000.0, ndecimals=2, step=1.0), group="Pump-time range (us)")
        # Repumper frequency (MHz). Live f_pump_repumper ~ 202 MHz.
        self.setattr_argument("f_repump_min_MHz", NumberValue(201.0, min=100.0, max=300.0, ndecimals=4, step=0.1), group="Repump-freq range (MHz)")
        self.setattr_argument("f_repump_max_MHz", NumberValue(203.0, min=100.0, max=300.0, ndecimals=4, step=0.1), group="Repump-freq range (MHz)")
        self.setattr_argument("f_repump_step_MHz", NumberValue(0.5, min=0.0001, max=50.0, ndecimals=4, step=0.1), group="Repump-freq range (MHz)")
        # Repumper amplitude (raw 0-1). Live amp_pump_repumper_active ~ 0.4.
        self.setattr_argument("amp_repump_min", NumberValue(0.3, min=0.0, max=1.0, ndecimals=3, step=0.01), group="Repump-amp range")
        self.setattr_argument("amp_repump_max", NumberValue(0.5, min=0.0, max=1.0, ndecimals=3, step=0.01), group="Repump-amp range")
        self.setattr_argument("amp_repump_step", NumberValue(0.05, min=0.001, max=1.0, ndecimals=3, step=0.01), group="Repump-amp range")
        # D1-pump frequency (MHz). Live f_pump_D1 ~ 145 MHz.
        self.setattr_argument("f_d1_min_MHz", NumberValue(143.0, min=50.0, max=300.0, ndecimals=4, step=0.1), group="D1-pump-freq range (MHz)")
        self.setattr_argument("f_d1_max_MHz", NumberValue(147.0, min=50.0, max=300.0, ndecimals=4, step=0.1), group="D1-pump-freq range (MHz)")
        self.setattr_argument("f_d1_step_MHz", NumberValue(0.5, min=0.0001, max=50.0, ndecimals=4, step=0.1), group="D1-pump-freq range (MHz)")
        # D1-pump amplitude (raw 0-1). Live amp_pump_D1 ~ 0.03.
        self.setattr_argument("amp_d1_min", NumberValue(0.01, min=0.0, max=1.0, ndecimals=3, step=0.005), group="D1-pump-amp range")
        self.setattr_argument("amp_d1_max", NumberValue(0.06, min=0.0, max=1.0, ndecimals=3, step=0.005), group="D1-pump-amp range")
        self.setattr_argument("amp_d1_step", NumberValue(0.005, min=0.001, max=1.0, ndecimals=3, step=0.005), group="D1-pump-amp range")

        # --- Repetitions (each loop runs the full grid x 2 cfg). ---
        self.setattr_argument("n_measurements", NumberValue(50, min=1, max=600000, ndecimals=0, step=1),
                              group="Experiment")

        # All physics (f_mw_00, t_pi_00_mu, amp_mw_pulse, shim_*, trap_ramp_*,
        # atom_threshold, t_imaging_gate, the live pump-knob values) load from
        # datasets via BaseExperiment; the rebalancer is created there.

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # --- Build each scannable knob's value list (NATURAL units) ---
        def _axis_list(vmin, vmax, vstep):
            v, s = float(vmin), float(vstep)
            n = (int(round((float(vmax) - v) / s)) + 1) if s > 0.0 else 1
            return [v + i * s for i in range(max(n, 1))]

        lists_nat = {
            "t_optical_pump": _axis_list(self.t_pump_min_us, self.t_pump_max_us, self.t_pump_step_us),    # us
            "f_pump_repumper": _axis_list(self.f_repump_min_MHz, self.f_repump_max_MHz, self.f_repump_step_MHz),  # MHz
            "amp_pump_repumper_active": _axis_list(self.amp_repump_min, self.amp_repump_max, self.amp_repump_step),  # raw
            "f_pump_D1": _axis_list(self.f_d1_min_MHz, self.f_d1_max_MHz, self.f_d1_step_MHz),             # MHz
            "amp_pump_D1": _axis_list(self.amp_d1_min, self.amp_d1_max, self.amp_d1_step),                # raw
        }
        x_vals = lists_nat[self.axis_x]
        y_vals = lists_nat[self.axis_y] if self.axis_y != "none" else [0.0]   # 0.0 sentinel -> 1-D

        # Live (fixed) SI values used when a knob is NOT an axis.
        live_si = {
            "t_optical_pump": float(self.t_optical_pump),                     # seconds
            "f_pump_repumper": float(self.f_pump_repumper),                   # Hz
            "amp_pump_repumper_active": float(self.amp_pump_repumper_active),  # raw
            "f_pump_D1": float(self.f_pump_D1),                               # Hz
            "amp_pump_D1": float(self.amp_pump_D1),                           # raw
        }

        # natural unit -> SI (us->s, MHz->Hz; amps raw)
        def _to_si(name, nat):
            if name == "t_optical_pump":
                return nat * 1e-6
            if name in ("f_pump_repumper", "f_pump_D1"):
                return nat * 1e6
            return nat

        # Flatten the X x Y grid into per-point kernel arg lists (branch-free kernel).
        self.pt_list = []   # pump time, seconds
        self.fr_list = []   # repump freq, Hz
        self.ar_list = []   # repump amp, raw
        self.fd_list = []   # D1 freq, Hz
        self.ad_list = []   # D1 amp, raw
        self.xv_list = []   # logged X value, natural unit
        self.yv_list = []   # logged Y value, natural unit (0.0 if 1-D)
        for xv in x_vals:
            for yv in y_vals:
                vals = dict(live_si)                     # start from live values
                vals[self.axis_x] = _to_si(self.axis_x, xv)
                if self.axis_y != "none":
                    vals[self.axis_y] = _to_si(self.axis_y, yv)
                self.pt_list.append(vals["t_optical_pump"])
                self.fr_list.append(vals["f_pump_repumper"])
                self.ar_list.append(vals["amp_pump_repumper_active"])
                self.fd_list.append(vals["f_pump_D1"])
                self.ad_list.append(vals["amp_pump_D1"])
                self.xv_list.append(float(xv))
                self.yv_list.append(float(yv))
        n_grid = len(self.pt_list)

        # --- Fixed 00-pi probe (SI, like microwave_scan) ---
        self._f00_hz = self.f_mw_00
        self._t00_mu = int(self.t_pi_00_mu)

        # --- Self-describing snapshot ---
        snapshot_params = {
            "axis_x": self.axis_x, "axis_y": self.axis_y,
            "t_pump_min_us": float(self.t_pump_min_us), "t_pump_max_us": float(self.t_pump_max_us), "t_pump_step_us": float(self.t_pump_step_us),
            "f_repump_min_MHz": float(self.f_repump_min_MHz), "f_repump_max_MHz": float(self.f_repump_max_MHz), "f_repump_step_MHz": float(self.f_repump_step_MHz),
            "amp_repump_min": float(self.amp_repump_min), "amp_repump_max": float(self.amp_repump_max), "amp_repump_step": float(self.amp_repump_step),
            "f_d1_min_MHz": float(self.f_d1_min_MHz), "f_d1_max_MHz": float(self.f_d1_max_MHz), "f_d1_step_MHz": float(self.f_d1_step_MHz),
            "amp_d1_min": float(self.amp_d1_min), "amp_d1_max": float(self.amp_d1_max), "amp_d1_step": float(self.amp_d1_step),
            # live values of the scannable knobs (for reference)
            "t_optical_pump": self.t_optical_pump,
            "f_pump_repumper": self.f_pump_repumper,
            "amp_pump_repumper_active": self.amp_pump_repumper_active,
            "f_pump_D1": self.f_pump_D1, "amp_pump_D1": self.amp_pump_D1,
            # the other (fixed) pump-beam params, for self-description
            "f_pump_excitation": self.f_pump_excitation, "amp_pump_excitation": self.amp_pump_excitation,
            # probe + sequence params
            "f_mw_00": self.f_mw_00, "t_pi_00_mu": self.t_pi_00_mu, "amp_mw_pulse": self.amp_mw_pulse,
            "shim_X_qaxis": self.shim_X_qaxis, "shim_Y_qaxis": self.shim_Y_qaxis,
            "shim_Z1_qaxis": self.shim_Z1_qaxis, "shim_Z2_qaxis": self.shim_Z2_qaxis,
            "shim_X_off": self.shim_X_off, "shim_Y_off": self.shim_Y_off,
            "shim_Z1_off": self.shim_Z1_off, "shim_Z2_off": self.shim_Z2_off,
            "trap_ramp_start_amp": self.trap_ramp_start_amp, "trap_ramp_steps": self.trap_ramp_steps,
            "trap_ramp_dt_us": self.trap_ramp_dt_us, "nominal_trapdepth": self.nominal_trapdepth,
            "atom_threshold": self.atom_threshold, "t_imaging_gate": self.t_imaging_gate,
            "n_measurements": int(self.n_measurements),
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Result datasets (per point) ---
        for k in ("op_axis_x_val", "op_axis_y_val", "op_apply_pi",
                  "op_bg_spcm1", "op_bg_spcm2",
                  "op_shot1_spcm1", "op_shot1_spcm2",
                  "op_shot2_spcm1", "op_shot2_spcm2",
                  "op_atom_loaded", "op_atom_retained", "op_loop_index"):
            self.set_dataset(k, [], broadcast=True, persist=True)
        self.set_dataset("op_scan_complete", 0, broadcast=True)

        # --- Metadata ---
        self.set_dataset("op_axis_x_name", self.axis_x, persist=True, broadcast=True)
        self.set_dataset("op_axis_y_name", self.axis_y, persist=True, broadcast=True)
        self.set_dataset("op_n_x_points", len(x_vals), persist=True, broadcast=True)
        self.set_dataset("op_n_y_points", len(y_vals), persist=True, broadcast=True)
        self.set_dataset("op_n_loops", int(self.n_measurements), persist=True, broadcast=True)
        self.set_dataset("op_best_signal", 0.0, persist=True, broadcast=True)

        self._n_pts_per_round = max(n_grid * 2, 1)
        print(f"OpticalPumpScan(2D): X={self.axis_x} ({len(x_vals)}) x Y={self.axis_y} "
              f"({len(y_vals)}) = {n_grid} grid pts x 2 cfg x {int(self.n_measurements)} loops "
              f"= {n_grid * 2 * int(self.n_measurements)} shots")
        print(f"  live knobs: t_optical_pump={self.t_optical_pump * 1e6:.1f} us, "
              f"f_pump_repumper={self.f_pump_repumper / 1e6:.4f} MHz, "
              f"amp_pump_repumper_active={self.amp_pump_repumper_active:.3f}, "
              f"f_pump_D1={self.f_pump_D1 / 1e6:.4f} MHz, amp_pump_D1={self.amp_pump_D1:.3f}")

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_measurements)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            for gi in range(len(self.pt_list)):
                # Interleave the two MW configs back-to-back at each grid point:
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
                    optical_pumping_timed(self, self.pt_list[gi], self.fr_list[gi], self.ar_list[gi],
                                          self.fd_list[gi], self.ad_list[gi])

                    # === MW: fixed 00 pi (cfg 1) or matched dark-hold (cfg 0) ===
                    if apply_pi:
                        mw_pulse_00(self, self._f00_hz, self._t00_mu)
                    else:
                        delay_mu(4100 + self._t00_mu)

                    # === Blowaway F=4 + trap ramp UP + final readout (shot2) ===
                    blowaway_f4(self, amp2)
                    ramp_trap_amp(self, trap_ramp_end_amp, self.trap_ramp_start_amp,
                                  self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)
                    count1_post, count2_post = final_state_readout(self)

                    # === Save ===
                    delay(1 * ms)
                    self._save_point(gi, cfg,
                                     count1_bg, count2_bg,
                                     count1, count2,
                                     count1_post, count2_post,
                                     loop)

                    # === Reset for next point ===
                    self.core.break_realtime()
                    reset_mw_scan(self)

        self._summarize()
        write_results(self)

    # =====================================================================
    #  HOST-SIDE SAVE / SUMMARY
    # =====================================================================

    @rpc
    def _save_point(self, gi, apply_pi,
                    bg_s1, bg_s2, shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2, loop_idx):
        """Append one point. The axis values are looked up host-side from gi (the
        kernel passes only the grid index). loaded/retained thresholding matches
        microwave_scan; incremental HDF5 save once per full (grid x cfg) round."""
        self.append_to_dataset("op_axis_x_val", self.xv_list[gi])
        self.append_to_dataset("op_axis_y_val", self.yv_list[gi])
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
        """Per (x, y) grid cell: lost-fraction (1 - retention among loaded) for the
        pi and no-pi configs, and the clean mF=0 signal = lost_pi - lost_noPi. Best
        cell = argmax signal. (Cheap table -- the heatmap is in
        analyze_optical_pump_scan.) BLAS-free (sums only)."""
        xs = self.get_dataset("op_axis_x_val")
        ys = self.get_dataset("op_axis_y_val")
        pis = self.get_dataset("op_apply_pi")
        loaded = self.get_dataset("op_atom_loaded")
        retained = self.get_dataset("op_atom_retained")
        if len(xs) == 0:
            print("OpticalPumpScan: no data to summarize.")
            return

        tab = {}        # (x, y, cfg) -> [n_retained, n_loaded]
        for i in range(len(xs)):
            if loaded[i] <= 0:
                continue
            key = (round(float(xs[i]), 4), round(float(ys[i]), 4), int(pis[i]))
            tab.setdefault(key, [0, 0])
            tab[key][0] += int(retained[i])
            tab[key][1] += 1

        cells = sorted({(k[0], k[1]) for k in tab})
        print(f"OpticalPumpScan(2D): lost_pi / lost_noPi / signal per "
              f"({self.axis_x}, {self.axis_y}):")
        best = None
        best_sig = -2.0
        for (x, y) in cells:
            dpi = tab.get((x, y, 1))
            dno = tab.get((x, y, 0))
            lpi = (1.0 - dpi[0] / dpi[1]) if (dpi and dpi[1] > 0) else float("nan")
            lno = (1.0 - dno[0] / dno[1]) if (dno and dno[1] > 0) else float("nan")
            if dpi and dno and dpi[1] > 0 and dno[1] > 0:
                sig = lpi - lno
            else:
                sig = float("nan")
            print(f"  x={x:9.4g}  y={y:9.4g}  lost_pi={lpi:.3f}  lost_noPi={lno:.3f}  signal={sig:.3f}")
            if sig == sig and sig > best_sig:   # sig==sig screens NaN
                best_sig = sig
                best = (x, y)

        if best is not None:
            self.set_dataset("op_best_signal", float(best_sig), persist=True, broadcast=True)
            print(f"OpticalPumpScan(2D): max mF=0 signal = {best_sig:.3f} at "
                  f"({self.axis_x}={best[0]:g}, {self.axis_y}={best[1]:g}). "
                  f"See analyze_optical_pump_scan for the heatmap.")
