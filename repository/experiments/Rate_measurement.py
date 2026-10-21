"""Photon-click RATE measurement (no waveplates, no MW analysis, no state readout).

A stripped-down sibling of the parity experiment used ONLY to measure throughput:
  - how many heralded photon clicks we get per unit time, and
  - how many excitation attempts we run per second,
so the speed-up from PGC-at-low-trap / imaging-at-low-trap / optimized optical-pump
time can be quantified against this baseline.

It reuses the SAME load + excitation sequence as the parity experiment, with the
LOW-trap recovery: the trap ramps DOWN once per atom and stays low -- the per-retry
PGC recool (imaging_pgc_low_trap) and the periodic survival image
(recool_and_check_survival_lowtrap) BOTH run at the low trap, with no ramp-up. Run
the HIGH-trap variant (deep-PGC recool + ramp-UP + recool_and_check_survival) as
the "before" baseline and compare.

What is REMOVED vs parity (none of which affects the heralding/click rate -- the
click is latched inside excitation_pulse's FORT-off collection gate, before any of
these would run): the HWP/QWP waveplate scan, the microwave analysis pulse, the
F=4 blowaway, and the final atom state-readout.

Data model (one row PER photon click):
  click_times_s     -- click time in seconds since run start (RTIO-based)
  click_detector    -- 1 = SPCM1, 2 = SPCM2
  trials_per_click  -- excitation attempts since the previous click (sum = total attempts)
  how_many_atoms    -- atom-load attempts since the previous click (sum = total loads)
The run stops after n_measurements CLICKS. Rates:
  clicks/s   = n_clicks / duration
  attempts/s = sum(trials_per_click) / duration
  herald p   = n_clicks / sum(trials_per_click)
(duration = HDF5 metadata duration_seconds; _summarize prints an RTIO-based estimate.)

A "click" is a single-detector herald (exactly one SPCM fires), matching the
parity herald definition (XOR); simultaneous double-clicks are not counted.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, imaging_pgc_low_trap, spcm_image_dual, beams_off_for_imaging,
    ramp_trap_amp, excitation_pulse, optical_pumping_F3, optical_pumping,
    recool_and_check_survival_lowtrap, turn_fiber_aom_on, reset_to_mot,
)
from utilities.write_results import write_results


class RateExperiment(EnvExperiment):
    """Rate measurement"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # No basis selector / no waveplate scan -- this is a rate measurement only.

        # --- Experiment-specific arguments ---
        # n_measurements = number of photon CLICKS to collect before stopping.
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

        # Rebalancer created and configured by BaseExperiment.build()

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")

        # Result datasets (one row per photon click; init before append_to_dataset)
        self.set_dataset("click_times_s", [], broadcast=True)
        self.set_dataset("click_detector", [], broadcast=True)
        self.set_dataset("trials_per_click", [], broadcast=True)
        self.set_dataset("how_many_atoms", [], broadcast=True)
        self.set_dataset("rate_progress", 0, broadcast=True)

    @kernel
    def run(self):
        self.base.initialize_hardware()
        # delay(1000 * ms)
        self.core.break_realtime()

        n_meas = int(self.n_measurements)
        n_clicks = 0
        trials = 0           # excitation attempts since the previous click
        how_many_atoms = 0   # atom-load attempts since the previous click
        t_run_start_mu = now_mu()   # time anchor: click times are referenced to here

        while n_clicks < n_meas:
            self.core.break_realtime()

            # ===== YELLOW SCOPE: Loading =====
            trap_ramp_end_amp = self.trap_ramp_start_amp   # safe init; real value (amp3) from the rebalancer below
            count1_bg = 0   # definite-assignment init; real value set in the load loop below
            count2_bg = 0
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

                # Confirm image
                count1_1, count2_1 = spcm_image_dual(self, self.t_imaging_gate)

                # PGC after image
                imaging_pgc(self)
                beams_off_for_imaging(self)

                # Check atom
                if (count1_1 + count2_1 - count1_bg - count2_bg) > self.atom_threshold:
                    break
            # ===== END YELLOW =====

            # ===== ORANGE SCOPE: Excitation (count clicks; keep the atom) =====
            n = 0               # cycle counter for the Check_img_every survival gate
            atom_alive = True
            while True:
                # Per-retry: F=3 prep -> Q-axis ON. The trap ramps DOWN only ONCE
                # per atom (if n == 0) and stays low -- the low-trap recovery never
                # ramps back up, so no per-retry ramp is needed.
                optical_pumping_F3(self)
                self.zotino0.write_dac(0, self.shim_X_qaxis)
                self.zotino0.write_dac(1, self.shim_Y_qaxis)
                self.zotino0.write_dac(2, self.shim_Z1_qaxis)
                self.zotino0.write_dac(3, self.shim_Z2_qaxis)
                self.zotino0.load()
                delay(1 * ms)
                if n == 0:
                    ramp_trap_amp(self, self.trap_ramp_start_amp,   # trap DOWN
                                trap_ramp_end_amp,
                                self.trap_ramp_steps,
                                self.trap_ramp_dt_us)

                # --- Excitation cycles: count EVERY single-detector click; NO
                #     break-on-click (rate mode keeps exciting), no MW, no blowaway. ---
                for exc_cycle in range(int(self.max_excitation_cycles)):
                    n += 1
                    self.core.break_realtime()   # refresh RTIO slack each cycle

                    # Pre-stage MW IF (kept only so the per-cycle timing matches the
                    # real parity cycle; the MW is never pulsed in this experiment).
                    self.dds_mw_IF.set(frequency=self.f_mw_11, amplitude=self.amp_mw_pulse)
                    optical_pumping(self)

                    trials += 1
                    t_end1, t_end2, t0 = excitation_pulse(self)
                    # Read SPCM timestamps inline (drains the gate at ~0 slack; the
                    # empty-SPCM timestamp_mu blocks to gate close -- channel-39
                    # underflow fix). No analysis pulse follows, so nothing is
                    # anchored at click+2000; the next timed op is a break_realtime.
                    click1 = self.ttl_SPCM1.timestamp_mu(t_end1)
                    click2 = self.ttl_SPCM2.timestamp_mu(t_end2)

                    # A "click" = single-detector herald (XOR), matching parity.
                    if click1 > 0 and click2 < 0:
                        n_clicks += 1
                        self._save_click(self.core.mu_to_seconds(click1 - t_run_start_mu),
                                         1, trials, how_many_atoms, n_clicks)
                        trials = 0
                        how_many_atoms = 0
                        self.core.break_realtime()   # restore slack drained by the async save RPC
                    elif click1 < 0 and click2 > 0:
                        n_clicks += 1
                        self._save_click(self.core.mu_to_seconds(click2 - t_run_start_mu),
                                         2, trials, how_many_atoms, n_clicks)
                        trials = 0
                        how_many_atoms = 0
                        self.core.break_realtime()   # restore slack drained by the async save RPC

                    # No-detection dead-time tail (also re-blocks the MW switch that
                    # excitation_pulse unblocked); runs every cycle for uniform timing.
                    delay(5 * us)
                    delay(2 * us)
                    self.ttl_MW_switch.on()
                    delay(1 * us)

                    if n_clicks >= n_meas:
                        break

                if n_clicks >= n_meas:
                    break

                # --- Recovery (LOW trap, no ramp-up): shims OFF -> fiber AOMs +
                #     repump -> imaging_pgc_low_trap recool; low-trap survival image
                #     (recool_and_check_survival_lowtrap) every Check_img_every cycles. ---
                self.core.break_realtime()
                self.zotino0.write_dac(0, self.shim_X_off)
                self.zotino0.write_dac(1, self.shim_Y_off)
                self.zotino0.write_dac(2, self.shim_Z1_off)
                self.zotino0.write_dac(3, self.shim_Z2_off)
                self.zotino0.load()
                delay(1 * ms)


                turn_fiber_aom_on(self)

                delay(1 * us)
                self.dds_repump.sw.on()
                delay(1 * us)
                self.ttl_repump_switch.off()         # unblock repump
                imaging_pgc_low_trap(self)

                if n % int(self.Check_img_every) == 0:
                    delay(1 * ms)
                    # low-trap survival check: RAW counts vs atom_threshold_lowtrap
                    # (no background arg -- it does not subtract a bg).
                    atom_alive = recool_and_check_survival_lowtrap(self)
                    if not atom_alive:
                        break    # atom lost -> back to Yellow (reload)

                beams_off_for_imaging(self)
            # ===== END ORANGE =====

            reset_to_mot(self)

        self._summarize()
        write_results(self)

    @rpc(flags={"async"})
    def _save_click(self, click_time_s, detector, trials, how_many_atoms, n_clicks):
        """Append one photon click. trials / how_many_atoms = excitation attempts /
        atom-load attempts since the previous click (so their column sums are the
        run totals)."""
        self.append_to_dataset("click_times_s", click_time_s)
        self.append_to_dataset("click_detector", detector)
        self.append_to_dataset("trials_per_click", trials)
        self.append_to_dataset("how_many_atoms", how_many_atoms)
        self.set_dataset("rate_progress", n_clicks, broadcast=True)
        # Periodic crash-safe HDF5 write (every 20 clicks, NOT every click): a
        # full-file write per click backs up the async-RPC FIFO, which both stalls
        # the kernel (RTIO underflow) and throttles the rate being measured. The
        # live datasets still update every click via append_to_dataset (broadcast).
        if n_clicks % 20 == 0:
            self._incremental_save()

    def _incremental_save(self):
        """Write current datasets to HDF5 mid-run (mirrors microwave_scan)."""
        try:
            from utilities.write_results import _save_h5_now
            _save_h5_now(self)
        except Exception as e:
            print("[incremental save]", e)

    @rpc
    def _summarize(self):
        """Print the photon-click and excitation-attempt rates. Uses the last click
        time as the run-duration estimate (RTIO-based); the precise wall-clock
        duration is in the HDF5 metadata (duration_seconds)."""
        ts = self.get_dataset("click_times_s")
        trials = self.get_dataset("trials_per_click")
        n_clicks = len(ts)
        if n_clicks == 0:
            print("RateExperiment: no clicks recorded (check loading / excitation).")
            return
        duration = max(ts)                 # last click ~ run end (run stops on a click)
        total_attempts = int(sum(trials))
        print(f"RateExperiment: {n_clicks} clicks, {total_attempts} excitation attempts "
              f"over ~{duration:.1f} s (RTIO-based):")
        if duration > 0.0:
            print(f"  click rate    = {n_clicks / duration:.3f} clicks/s")
            print(f"  attempt rate  = {total_attempts / duration:.1f} attempts/s")
        if total_attempts > 0:
            print(f"  herald prob   = {100.0 * n_clicks / total_attempts:.3f} % (clicks/attempt)")
        print("  (precise wall-clock duration is in the HDF5 metadata: duration_seconds)")
