"""Excitation survival + temperature scan vs cycles_per_group (LOW-trap recool).

Answers "how many excitation cycles can run back-to-back WITHOUT a PGC recool
before the atom is lost or too hot to re-cool" -- the number that sets parity's
max_excitation_cycles. Replicates parity's excitation+recool sequence (reusing the
SAME subroutines from subroutines.sequences) but:
  - drops the photon-detection branch (the ~0.7% herald success is irrelevant
    here; the excitation pulse fires every cycle regardless, so the average
    per-cycle thermal load is identical to parity); and
  - does the recool + survival image at the LOW (excitation) trap -- the trap
    ramps DOWN once and never ramps up until the loading trap is restored at the
    very end of the point (matches parity's low-trap recovery restructure).

Scan: OUTER = cycles_per_group (excitation cycles run back-to-back with NO recool
between them); INNER = release time (release-and-recapture temperature probe).
n_groups is FIXED (default 1): each group = cycles_per_group cycles + one low-trap
PGC recool; with n_groups=1 the whole point is a single block of cycles_per_group
cycles. The heating probe sits AFTER the recool, so:
  - survival @ release=0  = atom-LOSS probability after cycles_per_group cycles,
  - recapture vs release  = the residual temperature the low-trap recool leaves
    (higher recapture = colder).

Readout convention: shot1 (load confirm) is at the HIGH trap, loaded =
(shot1-bg) > atom_threshold. shot2 (post-release survival) is a LOW-trap image,
retained = RAW shot2 > atom_threshold_lowtrap (no bg subtraction -- the low-trap
convention, matching recool_and_check_survival_lowtrap). Fresh atom per point
(release destroys it). Each (cycles_per_group, release) point is repeated n_loops
times for binomial statistics.

Structure cloned from the previous n_groups scan (BaseExperiment, scan loop,
snapshot self-description, incremental per-round HDF5 save).
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
    excitation_pulse, turn_fiber_aom_on,
    imaging_pgc_low_trap, recool_and_check_survival_lowtrap, image_low_trap,
    reset_mw_scan,
)
from utilities.write_results import write_results


class ExcitationSurvivalScan(EnvExperiment):
    """Excitation_Survival_Scan"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # =============================================================
        #  SCAN-SPECIFIC ARGUMENTS
        # =============================================================

        # --- Excitation-cycle scan (OUTER axis): cycles_per_group = excitation
        # cycles run back-to-back WITHOUT a PGC recool between them (the recool
        # cadence == parity max_excitation_cycles). "How many cycles can we do
        # without PGC cooling before loss/heating." ---
        self.setattr_argument("cycles_per_group_min",
                              NumberValue(2, min=1, max=8000, ndecimals=0, step=1),
                              group="Cycles")
        self.setattr_argument("cycles_per_group_max",
                              NumberValue(32, min=1, max=8000, ndecimals=0, step=1),
                              group="Cycles")
        self.setattr_argument("cycles_per_group_step",
                              NumberValue(2, min=1, max=8000, ndecimals=0, step=1),
                              group="Cycles")
        # Number of (cycles_per_group + low-trap recool) GROUPS, FIXED (not
        # scanned). Default 1 = a single block of cycles_per_group cycles. Raise
        # to study multi-group cumulative survival (one low-trap recool between
        # groups).
        self.setattr_argument("n_groups",
                              NumberValue(1, min=1, max=100000, ndecimals=0, step=1),
                              group="Cycles")

        # --- Release-and-recapture (temperature) axis, in microseconds ---
        self.setattr_argument("release_min_us",
                              NumberValue(0.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Release")
        self.setattr_argument("release_max_us",
                              NumberValue(40.0, min=0.0, max=10000.0, ndecimals=1),
                              group="Release")
        self.setattr_argument("release_step_us",
                              NumberValue(4.0, min=0.1, max=10000.0, ndecimals=1),
                              group="Release")

        # --- Repetitions (binomial statistics: >= 50 per point) ---
        self.setattr_argument("n_loops",
                              NumberValue(50, min=1, max=600000, ndecimals=0, step=1),
                              group="Experiment")

        # --- Recool model (all LOW-trap now) ---
        #   pgc_only            : imaging_pgc_low_trap recool every group, NO survival image (faster candidate)
        #   pgc_plus_image      : also recool_and_check_survival_lowtrap every Check_img_every cycles (current parity)
        #   none                : no PGC recool at all (pure excitation heating -- extreme test)
        self.setattr_argument("recool_mode",
                              EnumerationValue(["pgc_only", "pgc_plus_image", "none"],
                                               default="pgc_only"),
                              group="Recool")
        self.setattr_argument("Check_img_every",
                              NumberValue(12, min=1, max=8000, ndecimals=0, step=1),
                              group="Recool")

        self.setattr_argument("skip_on_no_atom",
                              BooleanValue(True), group="Experiment")

        # --- Excitation timing (strictly follows the parity experiments) ---
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

        # All other physics (f_excitation_pulse, exc_* offsets, shims,
        # atom_threshold, t_imaging_gate, the low-trap imaging/PGC params, ...)
        # load from datasets via BaseExperiment; the rebalancer is created there.

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # Recool model -> kernel-safe int (the kernel branches on this int only).
        if self.recool_mode == "pgc_plus_image":
            self.recool_code = 1
        elif self.recool_mode == "none":
            self.recool_code = 2
        else:  # "pgc_only"
            self.recool_code = 0

        # --- Build scan lists ---
        v1, s1 = int(self.cycles_per_group_min), int(self.cycles_per_group_step)
        n1 = int((int(self.cycles_per_group_max) - v1) // s1) + 1
        self.cpg_list = [float(v1 + i * s1) for i in range(n1)]

        v2, s2 = float(self.release_min_us), float(self.release_step_us)
        n2 = int(round((float(self.release_max_us) - v2) / s2)) + 1
        self.release_list = [v2 + i * s2 for i in range(n2)]

        # =============================================================
        #  PRE-RUN SNAPSHOT: make the data file fully self-describing.
        # =============================================================
        snapshot_params = {
            "cycles_per_group_min": int(self.cycles_per_group_min),
            "cycles_per_group_max": int(self.cycles_per_group_max),
            "cycles_per_group_step": int(self.cycles_per_group_step),
            "n_groups": int(self.n_groups),
            "release_min_us": float(self.release_min_us),
            "release_max_us": float(self.release_max_us),
            "release_step_us": float(self.release_step_us),
            "n_loops": int(self.n_loops),
            "recool_mode": self.recool_mode,
            "Check_img_every": int(self.Check_img_every),
            "skip_on_no_atom": int(self.skip_on_no_atom),
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
            "shim_X_off": self.shim_X_off,
            "shim_Y_off": self.shim_Y_off,
            "shim_Z1_off": self.shim_Z1_off,
            "shim_Z2_off": self.shim_Z2_off,
            "atom_threshold": self.atom_threshold,
            "t_imaging_gate": self.t_imaging_gate,
            "t_PGC_after_img": self.t_PGC_after_img,
            "t_optical_pump": self.t_optical_pump,
            # low-trap recool (imaging_pgc_low_trap) + survival image params
            "f_cooling_PGC": self.f_cooling_PGC,
            "amp_cooling_PGC": self.amp_cooling_PGC,
            "f_cooling_imaging_lowtrap": self.f_cooling_imaging_lowtrap,
            "amp_cooling_imaging_lowtrap": self.amp_cooling_imaging_lowtrap,
            "t_imaging_gate_lowtrap": self.t_imaging_gate_lowtrap,
            "atom_threshold_lowtrap": self.atom_threshold_lowtrap,
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Result datasets (per point) ---
        for k in ("esv_cpg", "esv_n_groups", "esv_total_cycles", "esv_release_us",
                  "esv_bg_spcm1", "esv_bg_spcm2",
                  "esv_shot1_spcm1", "esv_shot1_spcm2",
                  "esv_shot2_spcm1", "esv_shot2_spcm2",
                  "esv_atom_loaded", "esv_atom_retained", "esv_loop_index"):
            self.set_dataset(k, [], broadcast=True, persist=True)
        self.set_dataset("esv_scan_complete", 0, broadcast=True)

        # --- Metadata ---
        self.set_dataset("esv_scan_mode", "cycles_per_group", persist=True, broadcast=True)
        self.set_dataset("esv_n_cpg_points", len(self.cpg_list), persist=True, broadcast=True)
        self.set_dataset("esv_n_release_points", len(self.release_list), persist=True, broadcast=True)
        self.set_dataset("esv_n_loops", int(self.n_loops), persist=True, broadcast=True)

        self._n_pts_per_round = max(len(self.cpg_list) * len(self.release_list), 1)
        print(f"ExcitationSurvivalScan: {len(self.cpg_list)} cycles_per_group-points x "
              f"{len(self.release_list)} release-points x {int(self.n_loops)} loops "
              f"(n_groups={int(self.n_groups)}, recool={self.recool_mode}) | "
              f"cpg={self.cpg_list}")

    # =====================================================================
    #  KERNEL HELPERS
    # =====================================================================

    @kernel
    def _excite_drain(self):
        """One excitation attempt, photon clicks DISCARDED. Reuses the shared
        excitation_pulse (real f/amp_excitation_pulse), reads both SPCM
        timestamps INLINE immediately to drain the gate (an empty-gate
        timestamp_mu blocks to gate-close at ~0 slack; reading inline with
        nothing time-anchored after avoids the channel-39 underflow that bit
        parity), then runs parity's no-detect dead-time tail. Thermally
        identical to a parity no-herald cycle."""
        t_end1, t_end2, t0 = excitation_pulse(self)
        click1 = self.ttl_SPCM1.timestamp_mu(t_end1)   # drain inline (discarded)
        click2 = self.ttl_SPCM2.timestamp_mu(t_end2)   # drain inline (discarded)
        delay(5 * us)
        delay(2 * us)
        self.ttl_MW_switch.on()
        delay(1 * us)

    @kernel
    def _excite_recool_group_lowtrap(self, trap_end, cpg, do_ramp_down, do_image):
        """One excite group at the LOW trap (NO ramp-up). F=3 prep -> Q-axis ON
        -> (only on the FIRST group, do_ramp_down) ramp trap DOWN -> cpg
        excitation attempts back-to-back (no recool between cycles, no photon
        branch) -> low-trap recovery: shim OFF (B~0) -> fiber AOMs on -> repump
        -> imaging_pgc_low_trap (PGC recool at the low trap, NO ramp up). If
        do_image, additionally recool_and_check_survival_lowtrap (low-trap image
        + PGC; return discarded -- parity-faithful overhead)."""
        optical_pumping_F3(self)
        set_qaxis(self, self.shim_X_qaxis, self.shim_Y_qaxis,
                  self.shim_Z1_qaxis, self.shim_Z2_qaxis)
        if do_ramp_down:
            ramp_trap_amp(self, self.trap_ramp_start_amp, trap_end,
                          self.trap_ramp_steps, self.trap_ramp_dt_us)   # DOWN (once)

        for cyc in range(cpg):
            self.core.break_realtime()    # refresh RTIO slack each cycle
            optical_pumping(self)
            self._excite_drain()

        # --- low-trap recovery (mirrors parity.py low-trap recovery; NO ramp up) ---
        self.core.break_realtime()
        set_qaxis(self, self.shim_X_off, self.shim_Y_off,
                  self.shim_Z1_off, self.shim_Z2_off)             # shim OFF (B~0)
        turn_fiber_aom_on(self)              # PGC/imaging path ON
        delay(1 * us)
        self.dds_repump.sw.on()
        delay(1 * us)
        self.ttl_repump_switch.off()         # unblock repump
        if self.recool_code != 2:            # not "none"
            imaging_pgc_low_trap(self)       # PGC recool at the LOW trap (NO image)
        if do_image:                         # recool_code==1 and cadence hit
            delay(1 * ms)
            recool_and_check_survival_lowtrap(self)   # low-trap image + PGC (return discarded)
        beams_off_for_imaging(self)

    @kernel
    def _release_recapture_lowtrap(self, rel_us):
        """Release-and-recapture readout = shot2, AT THE LOW trap (probe AFTER the
        recool). FORT off for rel_us, back on, settle, then a LOW-trap image
        (image_low_trap; raw counts vs atom_threshold_lowtrap). rel_us==0 -> no
        release -> shot2 is the raw post-cycle survival. Sequential sw.off/delay/
        sw.on with break_realtime first so the short release sits in ample slack.
        The trap is NOT ramped up here -- the caller restores the loading trap."""
        self.core.break_realtime()
        if rel_us > 0.0:
            self.dds_FORT.sw.off()
            delay(rel_us * us)
            self.dds_FORT.sw.on()
        delay(1 * ms)                        # settle in the (low) trap before imaging
        set_qaxis(self, self.shim_X_off, self.shim_Y_off,
                  self.shim_Z1_off, self.shim_Z2_off)   # B~0 for imaging (idempotent)
        turn_fiber_aom_on(self)                          # fiber path ON for the image
        c1, c2 = image_low_trap(self, self.f_cooling_imaging_lowtrap,
                                self.amp_cooling_imaging_lowtrap,
                                self.t_imaging_gate_lowtrap)
        beams_off_for_imaging(self)
        return c1, c2

    # =====================================================================
    #  RUN
    # =====================================================================

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_loops)
        n_groups = int(self.n_groups)
        check_every = int(self.Check_img_every)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            for icpg in range(len(self.cpg_list)):
                cpg = int(self.cpg_list[icpg])

                for ir in range(len(self.release_list)):
                    rel_us = self.release_list[ir]

                    # === Feedback ===
                    prepare_for_feedback(self)
                    amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                    trap_end = amp3
                    restore_after_feedback(self)

                    # === Background + load + confirm image (shot1) @ HIGH trap ===
                    count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                    load_atom_deterministic(self)
                    imaging_pgc(self)
                    beams_off_for_imaging(self)
                    count1, count2 = spcm_image_dual(self, self.t_imaging_gate)
                    imaging_pgc(self)
                    beams_off_for_imaging(self)

                    loaded = (count1 + count2 - count1_bg - count2_bg) > self.atom_threshold

                    # === Excitation groups (LOW trap) + release-recapture (shot2, LOW trap) ===
                    count1_post = 0
                    count2_post = 0
                    if loaded or (not self.skip_on_no_atom):
                        for g in range(n_groups):
                            do_image = (self.recool_code == 1) and (((g + 1) * cpg) % check_every == 0)
                            self._excite_recool_group_lowtrap(trap_end, cpg, g == 0, do_image)
                        # probe AFTER recool: release-and-recapture + low-trap image (shot2)
                        count1_post, count2_post = self._release_recapture_lowtrap(rel_us)
                        # restore the loading trap (atom already read): ramp UP once
                        self.core.break_realtime()
                        ramp_trap_amp(self, trap_end, self.trap_ramp_start_amp,
                                      self.trap_ramp_steps, self.trap_ramp_dt_us)   # UP
                        delay(1 * ms)

                    # === Save ===
                    self._save_point(cpg, n_groups, n_groups * cpg, rel_us,
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
    def _save_point(self, cpg, n_groups, total_cycles, rel_us,
                    bg_s1, bg_s2, shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2, loaded, loop_idx):
        """Append one point. loaded = HIGH-trap confirm (shot1-bg) > atom_threshold;
        retained = LOW-trap RAW shot2 > atom_threshold_lowtrap (no bg subtraction --
        the low-trap convention). Incremental HDF5 save once per (cpg x release) round."""
        self.append_to_dataset("esv_cpg", cpg)
        self.append_to_dataset("esv_n_groups", n_groups)
        self.append_to_dataset("esv_total_cycles", total_cycles)
        self.append_to_dataset("esv_release_us", rel_us)
        self.append_to_dataset("esv_bg_spcm1", bg_s1)
        self.append_to_dataset("esv_bg_spcm2", bg_s2)
        self.append_to_dataset("esv_shot1_spcm1", shot1_s1)
        self.append_to_dataset("esv_shot1_spcm2", shot1_s2)
        self.append_to_dataset("esv_shot2_spcm1", shot2_s1)
        self.append_to_dataset("esv_shot2_spcm2", shot2_s2)
        self.append_to_dataset("esv_loop_index", loop_idx)

        # LOW-trap survival: RAW shot2 vs atom_threshold_lowtrap (no bg subtraction).
        shot2 = shot2_s1 + shot2_s2
        retained = int(shot2 > self.atom_threshold_lowtrap) if loaded > 0 else 0
        self.append_to_dataset("esv_atom_loaded", loaded)
        self.append_to_dataset("esv_atom_retained", retained)

        if not hasattr(self, "_points_saved"):
            self._points_saved = 0
        self._points_saved += 1
        self.set_dataset("esv_scan_complete", self._points_saved, broadcast=True)
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
        """Print survival vs cycles_per_group at release=0 (the 'cycles without
        PGC' loss curve). BLAS-free (sums only)."""
        cpg = self.get_dataset("esv_cpg")
        rel = self.get_dataset("esv_release_us")
        loaded = self.get_dataset("esv_atom_loaded")
        retained = self.get_dataset("esv_atom_retained")
        if len(cpg) == 0:
            print("ExcitationSurvivalScan: no data to summarize.")
            return

        totals = {}   # cycles_per_group -> [n_retained, n_loaded]  (release==0 only)
        for i in range(len(cpg)):
            if float(rel[i]) > 0.0 or loaded[i] <= 0:
                continue
            c = int(cpg[i])
            if c not in totals:
                totals[c] = [0, 0]
            totals[c][0] += int(retained[i])
            totals[c][1] += 1

        print("ExcitationSurvivalScan survival vs cycles_per_group (release=0):")
        for c in sorted(totals.keys()):
            nr, nl = totals[c]
            p = nr / nl if nl > 0 else 0.0
            print(f"  cycles_per_group={c:5d}  survival={p:.3f}  ({nr}/{nl})")
