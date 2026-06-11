"""Excitation survival + temperature scan (release-and-recapture).

Standalone optimizer that measures **how many excitation cycles a trapped atom
survives** and **how hot it gets** after the cycles + PGC recool, to speed up the
parity experiment. It replicates parity's excitation+recool sequence (reusing the
SAME subroutines from subroutines.sequences) but:
  - drops the photon-detection branch (the ~0.7% herald success is irrelevant
    here; the excitation pulse fires every cycle regardless, so the average
    per-cycle thermal load is identical to parity); and
  - drops the mid-cycle survival image -- the whole point is to test whether
    PGC-only recool (no image) keeps the atom cold enough that the costly
    survival images (recool_and_check_survival, ~24 ms each) can be dropped.

Figure of merit = RELEASE-AND-RECAPTURE: load atom -> image (shot1) -> run N
excite+recool groups -> turn the FORT off for a variable release time -> FORT
back on -> image (shot2). survival = (shot2-bg)/(shot1-bg); the recapture-vs-
release-time curve at each cycle count is the atom-temperature proxy.

Scan: outer = number of excite+recool GROUPS (total cycles = n_groups *
cycles_per_group); inner = release time. Each (cycles, release) point is repeated
n_loops times for binomial statistics, with a fresh atom per point (release
destroys the atom; shot1 normalizes every shot).

`cycles_per_group` is tunable (default 4 == parity max_excitation_cycles): it is
the excitation cycles between recools, i.e. the recool cadence. At 4, one group ==
one parity retry, so n_groups = number of parity retries survived.

Structure cloned from excitation_scan.py (BaseExperiment, scan loop, snapshot
self-description, incremental per-round HDF5 save). The excitation+recool group is
parity-faithful (parity.py Orange scope minus the photon branch and survival
image).
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
    excitation_pulse, turn_fiber_aom_on, recool_and_check_survival,
    final_state_readout, reset_mw_scan,
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

        # --- Excitation-cycle (recool-cadence) controls ---
        # cycles_per_group = excitation cycles BETWEEN recools (the recool
        # cadence). Default 4 == parity max_excitation_cycles -> one group ==
        # one parity retry. Tune it to explore faster/slower cadences.
        self.setattr_argument("cycles_per_group",
                              NumberValue(4, min=1, max=8000, ndecimals=0, step=1),
                              group="Cycles")
        # Outer scan axis: number of GROUPS. total cycles = n_groups * cycles_per_group.
        self.setattr_argument("n_groups_min",
                              NumberValue(1, min=0, max=100000, ndecimals=0, step=1),
                              group="Cycles")
        self.setattr_argument("n_groups_max",
                              NumberValue(21, min=1, max=100000, ndecimals=0, step=1),
                              group="Cycles")
        self.setattr_argument("n_groups_step",
                              NumberValue(4, min=1, max=100000, ndecimals=0, step=1),
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

        # --- Recool model (the actual speed-up question) ---
        #   pgc_only            : imaging_pgc recool every group, NO survival image (faster candidate)
        #   pgc_plus_image      : also recool_and_check_survival every Check_img_every cycles (current parity)
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
        # atom_threshold, t_imaging_gate, ...) load from datasets via
        # BaseExperiment; the rebalancer is created there too.

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
        v1, s1 = int(self.n_groups_min), int(self.n_groups_step)
        n1 = int((int(self.n_groups_max) - v1) // s1) + 1
        self.n_groups_list = [float(v1 + i * s1) for i in range(n1)]

        v2, s2 = float(self.release_min_us), float(self.release_step_us)
        n2 = int(round((float(self.release_max_us) - v2) / s2)) + 1
        self.release_list = [v2 + i * s2 for i in range(n2)]

        # =============================================================
        #  PRE-RUN SNAPSHOT: make the data file fully self-describing.
        # =============================================================
        snapshot_params = {
            "cycles_per_group": int(self.cycles_per_group),
            "n_groups_min": int(self.n_groups_min),
            "n_groups_max": int(self.n_groups_max),
            "n_groups_step": int(self.n_groups_step),
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
        }
        for key, val in snapshot_params.items():
            self.set_dataset(f"snapshot_{key}", val, persist=True)

        # --- Result datasets (per point) ---
        for k in ("esv_n_groups", "esv_total_cycles", "esv_release_us",
                  "esv_bg_spcm1", "esv_bg_spcm2",
                  "esv_shot1_spcm1", "esv_shot1_spcm2",
                  "esv_shot2_spcm1", "esv_shot2_spcm2",
                  "esv_atom_loaded", "esv_atom_retained", "esv_loop_index"):
            self.set_dataset(k, [], broadcast=True, persist=True)
        self.set_dataset("esv_scan_complete", 0, broadcast=True)

        # --- Metadata ---
        self.set_dataset("esv_scan_mode", "cycles", persist=True, broadcast=True)
        self.set_dataset("esv_n_group_points", len(self.n_groups_list), persist=True, broadcast=True)
        self.set_dataset("esv_n_release_points", len(self.release_list), persist=True, broadcast=True)
        self.set_dataset("esv_n_loops", int(self.n_loops), persist=True, broadcast=True)

        self._n_pts_per_round = max(len(self.n_groups_list) * len(self.release_list), 1)
        print(f"ExcitationSurvivalScan: {len(self.n_groups_list)} cycle-points x "
              f"{len(self.release_list)} release-points x {int(self.n_loops)} loops "
              f"(cycles_per_group={int(self.cycles_per_group)}, recool={self.recool_mode})")

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
    def _excitation_recool_group(self, trap_end, count_bg, do_image):
        """One parity-faithful excite+recool group (= one parity retry at
        cycles_per_group==4): F=3 prep -> Q-axis on -> trap DOWN ->
        cycles_per_group excitation attempts (no photon branch) -> recovery
        (shim OFF -> trap UP -> fiber AOMs on -> repump -> recool). The recool
        is PGC-only (imaging_pgc, NO image) unless do_image, which additionally
        runs recool_and_check_survival (image + PGC) to mirror current parity."""
        cpg = int(self.cycles_per_group)

        optical_pumping_F3(self)
        set_qaxis(self, self.shim_X_qaxis, self.shim_Y_qaxis,
                  self.shim_Z1_qaxis, self.shim_Z2_qaxis)
        ramp_trap_amp(self, self.trap_ramp_start_amp, trap_end,
                      self.trap_ramp_steps, self.trap_ramp_dt_us)   # DOWN

        for cyc in range(cpg):
            self.core.break_realtime()    # refresh RTIO slack each cycle (matches excitation_scan)
            optical_pumping(self)
            self._excite_drain()

        # --- recovery (mirrors parity.py recovery, minus the survival image) ---
        self.core.break_realtime()
        set_qaxis(self, self.shim_X_off, self.shim_Y_off,
                  self.shim_Z1_off, self.shim_Z2_off)             # shim OFF
        ramp_trap_amp(self, trap_end, self.trap_ramp_start_amp,
                      self.trap_ramp_steps, self.trap_ramp_dt_us)   # UP
        turn_fiber_aom_on(self)              # PGC/imaging path ON (also needed for recapture)
        delay(1 * us)
        self.dds_repump.sw.on()
        delay(1 * us)
        self.ttl_repump_switch.off()         # unblock repump
        if self.recool_code != 2:            # not "none"
            imaging_pgc(self)                # recool WITHOUT image
        if do_image:                         # recool_code==1 and cadence hit
            recool_and_check_survival(self, count_bg)   # image + PGC (return discarded)
        beams_off_for_imaging(self)

    @kernel
    def _release_and_recapture(self, rel_us):
        """Release-and-recapture readout = shot2. FORT off for rel_us, back on,
        settle, then image via final_state_readout (opens its own MOT/D1 beams).
        rel_us==0 -> no release (FORT stays on) -> shot2 is the raw post-cycle
        survival. Sequential sw.off/delay/sw.on (NOT at_mu), with break_realtime
        first, so the short release is always inside ample slack."""
        self.core.break_realtime()
        if rel_us > 0.0:
            self.dds_FORT.sw.off()
            delay(rel_us * us)
            self.dds_FORT.sw.on()
        delay(1 * ms)                        # let atom settle in trap before imaging
        c1, c2 = final_state_readout(self)
        return c1, c2

    # =====================================================================
    #  RUN
    # =====================================================================

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        n_loops = int(self.n_loops)
        cpg = int(self.cycles_per_group)
        check_every = int(self.Check_img_every)

        for loop in range(n_loops):
            self.core.break_realtime()
            delay(100 * us)

            for ig in range(len(self.n_groups_list)):
                n_grp = int(self.n_groups_list[ig])

                for ir in range(len(self.release_list)):
                    rel_us = self.release_list[ir]

                    # === Feedback ===
                    prepare_for_feedback(self)
                    amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                    trap_end = amp3
                    restore_after_feedback(self)

                    # === Background + load + confirm image (shot1) ===
                    count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
                    load_atom_deterministic(self)
                    imaging_pgc(self)
                    beams_off_for_imaging(self)
                    count1, count2 = spcm_image_dual(self, self.t_imaging_gate)
                    imaging_pgc(self)
                    beams_off_for_imaging(self)

                    loaded = (count1 + count2 - count1_bg - count2_bg) > self.atom_threshold

                    # === Excitation+recool groups + release-recapture (shot2) ===
                    count1_post = 0
                    count2_post = 0
                    if loaded or (not self.skip_on_no_atom):
                        count_bg = count1_bg + count2_bg
                        n_done = 0
                        for g in range(n_grp):
                            n_done += cpg
                            do_image = (self.recool_code == 1) and (n_done % check_every == 0)
                            self._excitation_recool_group(trap_end, count_bg, do_image)
                        count1_post, count2_post = self._release_and_recapture(rel_us)

                    # === Save ===
                    self._save_point(n_grp, n_grp * cpg, rel_us,
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
    def _save_point(self, n_grp, total_cycles, rel_us,
                    bg_s1, bg_s2, shot1_s1, shot1_s2,
                    shot2_s1, shot2_s2, loaded, loop_idx):
        """Append one point; threshold-based loaded/retained flags; incremental
        HDF5 save once per full (cycles x release) round."""
        self.append_to_dataset("esv_n_groups", n_grp)
        self.append_to_dataset("esv_total_cycles", total_cycles)
        self.append_to_dataset("esv_release_us", rel_us)
        self.append_to_dataset("esv_bg_spcm1", bg_s1)
        self.append_to_dataset("esv_bg_spcm2", bg_s2)
        self.append_to_dataset("esv_shot1_spcm1", shot1_s1)
        self.append_to_dataset("esv_shot1_spcm2", shot1_s2)
        self.append_to_dataset("esv_shot2_spcm1", shot2_s1)
        self.append_to_dataset("esv_shot2_spcm2", shot2_s2)
        self.append_to_dataset("esv_loop_index", loop_idx)

        bg = bg_s1 + bg_s2
        shot2 = shot2_s1 + shot2_s2
        retained = int((shot2 - bg) > self.atom_threshold) if loaded > 0 else 0
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
        """Print survival vs total_cycles at release=0 (the max_excitation_cycles
        ceiling). BLAS-free (sums only)."""
        cycles = self.get_dataset("esv_total_cycles")
        rel = self.get_dataset("esv_release_us")
        loaded = self.get_dataset("esv_atom_loaded")
        retained = self.get_dataset("esv_atom_retained")
        if len(cycles) == 0:
            print("ExcitationSurvivalScan: no data to summarize.")
            return

        totals = {}   # total_cycles -> [n_retained, n_loaded]  (release==0 only)
        for i in range(len(cycles)):
            if float(rel[i]) > 0.0 or loaded[i] <= 0:
                continue
            c = int(cycles[i])
            if c not in totals:
                totals[c] = [0, 0]
            totals[c][0] += int(retained[i])
            totals[c][1] += 1

        print("ExcitationSurvivalScan survival vs total cycles (release=0):")
        for c in sorted(totals.keys()):
            nr, nl = totals[c]
            p = nr / nl if nl > 0 else 0.0
            print(f"  cycles={c:5d}  survival={p:.3f}  ({nr}/{nl})")
