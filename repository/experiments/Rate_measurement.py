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

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, imaging_pgc_low_trap, spcm_image_dual, beams_off_for_imaging,
    ramp_trap_amp, excitation_pulse, optical_pumping_F3, optical_pumping,
    recool_and_check_survival_lowtrap, image_low_trap, turn_fiber_aom_on, reset_to_mot,
)
from utilities.write_results import write_results


def _fit_decay_host(x, top, n_tau=120, n_w=41):
    """Two-component MLE of the photon arrival profile. HOST side only.

    Model on the window [0, top]: an exponential with onset t0 and decay
    constant tau, plus a flat background of weight w --

        f(t) = (1 - w) * exp(-(t - t0) / tau) / Z  +  w / top    (t >= t0)
               w / top                                           (t <  t0)
        Z    = tau * (1 - exp(-(top - t0) / tau))

    A SINGLE exponential is not enough: the real profile carries a weak slow
    tail (~10 % of clicks, from re-excitation during the remainder of the
    300 ns drive), and a one-component fit absorbs it and biases tau high --
    48 ns instead of 36 ns on RID 3646.

    Returns a dict (t0, tau, tau_lo, tau_hi, w, n, L, flat) or None if there is
    too little data; `flat` means no decay is resolved.

    Only ever called from _summarize (an @rpc, i.e. host context), so the
    kernel compiler never sees it and the local numpy import is safe. KEEP IN
    SYNC with fit_decay() in applets/plot_rate_histograms.py and in
    analysis/analyze_rate_per_atom.py -- same model, same numbers.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0.0) & (x <= top)]
    if x.size < 10 or not np.isfinite(top) or top <= 0.0:
        return None
    # 2nd percentile, not the minimum: robust to a single stray early click.
    t0 = float(np.percentile(x, 2.0)) if x.size >= 50 else float(x.min())
    L = top - t0
    if L <= 0.0:
        return None

    taus = np.exp(np.linspace(np.log(5.0), np.log(max(10.0, 0.6 * L)), n_tau))
    ws = np.linspace(0.0, 0.40, n_w)
    above = x >= t0
    dx = np.where(above, x - t0, 0.0)
    nll = np.empty((n_tau, n_w))
    for i, tau in enumerate(taus):
        z = tau * (1.0 - np.exp(-L / tau))
        sig = np.where(above, np.exp(-dx / tau) / z, 0.0)
        dens = (1.0 - ws[None, :]) * sig[:, None] + ws[None, :] / top
        nll[i] = -np.log(np.maximum(dens, 1e-300)).sum(axis=0)

    i, j = np.unravel_index(np.argmin(nll), nll.shape)
    tau, w, best = float(taus[i]), float(ws[j]), float(nll[i, j])
    prof = nll.min(axis=1)
    ok = np.where(prof - best <= 0.5)[0]          # profile-likelihood 1 sigma
    return dict(t0=float(t0), tau=tau,
                tau_lo=float(taus[ok.min()]), tau_hi=float(taus[ok.max()]),
                w=w, n=int(x.size), L=float(L),
                flat=bool(tau > 0.5 * L or w > 0.6))


class RateExperiment(EnvExperiment):
    """Rate measurement"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        # CCB: lets the experiment auto-create the live histogram applet in the
        # dashboard when it runs. Guarded in prepare() so headless runs don't fail.
        self.setattr_device("ccb")

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
        #
        # The FORT-off window is [exc_trap_off_offset_mu,
        # exc_trap_off_offset_mu + t_trap_off_mu] relative to the excitation
        # reference t0 (sequences.py::excitation_pulse). Both values are already
        # calibrated for the FORT channel's cable/AOM delay so that the OPTICAL
        # trap is off for t_trap_off_mu during the OPTICAL excitation pulse. Do
        # not compare these raw RTIO times with other channels' offsets (e.g.
        # exc_spcm_gate_offset_mu): each channel's delay is folded into its own
        # offset. The position is exposed here so it can be tuned alongside the
        # length.
        #
        # exc_trap_off_offset_mu is normally a dataset-only ExperimentVariables
        # parameter. Declaring it here SHADOWS that dataset, because
        # setattr_argument runs after self.base.build() (which is what loads the
        # datasets as attributes), so this value is the one excitation_pulse
        # reads. sequences.py is untouched -- it reads the attribute either way.
        # Consequence: `archive/` will keep reporting the ExperimentVariables
        # value, since that is what was read during build; `expid` holds what
        # actually ran. The default below is the live value as of 2026-10-22.
        #
        # 1 mu = 1 ns on Kasli, so these are nanoseconds. min/max match the
        # ExperimentVariables declaration so a scan is not capped at 8000.
        self.setattr_argument("exc_trap_off_offset_mu",
                              NumberValue(4215, min=0, max=50000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("t_trap_off_mu",
                              NumberValue(400, min=0, max=50000, ndecimals=0, step=1),
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
        # Photon arrival delay within the collection gate, in ns measured from
        # GATE OPEN (0..t_spcm_collection_mu by construction). An exponential
        # with tau = 30-35 ns means one photon per attempt from a clean pulse; a
        # flat distribution across the gate means multi-photon scattering or
        # background. Independent of detection efficiency -- it is a shape, not
        # a rate -- and independent of the exc_spcm_gate_offset_mu calibration.
        # It also checks that calibration: clicks piled against 0 ns mean the
        # gate opens LATE (photons already arriving), piled at the end mean the
        # photon arrives late in the window and part of the tail is being cut.
        self.set_dataset("click_delay_ns", [], broadcast=True)
        # Two-detector coincidences: rejected by the XOR herald, counted nowhere
        # until now. Reported as a fraction of heralds by _summarize().
        self.set_dataset("n_coincidences", 0, broadcast=True)
        self.set_dataset("trials_per_click", [], broadcast=True)
        self.set_dataset("how_many_atoms", [], broadcast=True)
        self.set_dataset("rate_progress", 0, broadcast=True)

        # Live atom-readout histograms + the thresholds the applet draws as lines.
        #   rate_readout_hightrap : net confirm-image counts (sig - bg) in the load
        #                           loop, compared to atom_threshold (HIGH trap).
        #   rate_readout_lowtrap  : raw survival-image counts in the recovery loop,
        #                           compared to atom_threshold_lowtrap (LOW trap).
        self.set_dataset("rate_readout_hightrap", [], broadcast=True)
        self.set_dataset("rate_readout_lowtrap", [], broadcast=True)
        self.set_dataset("rate_atom_threshold", float(self.atom_threshold),
                         broadcast=True, persist=True)
        self.set_dataset("rate_atom_threshold_lowtrap", float(self.atom_threshold_lowtrap),
                         broadcast=True, persist=True)
        # Nominal collection-gate length, broadcast so the applet can draw the
        # gate-close line on the decay panel. t_spcm_collection_mu is a
        # front-panel argument, so the applet cannot read it any other way.
        # (1 mu = 1 ns on Kasli, so the value is already in ns.)
        self.set_dataset("rate_gate_ns", float(self.t_spcm_collection_mu),
                         broadcast=True)

        # --- PER-CONFIRMED-ATOM diagnostic log (one row per atom that passes the
        # high-trap confirm image). Purpose: find out WHY a large fraction of
        # confirmed atoms never herald. Neither experiment could answer this
        # before -- parity resets `trials` on reload (discarding dud-atom
        # attempts) and the rate experiment merges them into the next click's
        # window. Logging only; no sequence/behaviour change.
        #   atom_confirm_net   net confirm counts (sig - bg) -> a ~2x shoulder on
        #                      duds would indicate TWO-atom loads
        #   atom_attempts      excitation attempts this atom received
        #   atom_heralds       clicks this atom produced (0 = dud)
        #   atom_first_herald  attempt index of its FIRST click (-1 = never);
        #                      a rising trend vs index would indicate a warm-up
        #   atom_end_reason    0 = lost (survival check) -> COMPLETE lifetime,
        #                      1 = run ended -> CENSORED, exclude from dud stats
        # Per-atom herald rate = atom_heralds / atom_attempts; if that is BIMODAL
        # across atoms it means a quenched per-atom property (e.g. excitation-beam
        # overlap), not a per-attempt loss.
        for key in ("atom_confirm_net", "atom_attempts", "atom_heralds",
                    "atom_first_herald", "atom_end_reason"):
            self.set_dataset(key, [], broadcast=True)

        self._spawn_applets()

    def _spawn_applets(self):
        """Auto-create the 3-panel live histogram applet in the dashboard (via CCB)
        so it pops up when the experiment runs. Guarded so a headless run just
        prints a note instead of failing."""
        applet = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "applets", "plot_rate_histograms.py"))
        cmd = ('${python} "' + applet + '" rate_readout_hightrap'
               ' --lowtrap rate_readout_lowtrap'
               ' --delays click_delay_ns'
               ' --gate rate_gate_ns'
               ' --thr_hi rate_atom_threshold'
               ' --thr_lo rate_atom_threshold_lowtrap')
        try:
            self.ccb.issue("create_applet", "rate histograms", cmd, group="rate")
        except Exception as e:
            print("Could not auto-create applet (no dashboard/ccb?):", e)

    @kernel
    def run(self):
        self.base.initialize_hardware()
        # delay(1000 * ms)
        self.core.break_realtime()

        n_meas = int(self.n_measurements)
        n_clicks = 0
        n_coinc = 0          # two-detector coincidences (rejected by the XOR herald)
        trials = 0           # excitation attempts since the previous click
        how_many_atoms = 0   # atom-load attempts since the previous click
        t_run_start_mu = now_mu()   # time anchor: click times are referenced to here

        while n_clicks < n_meas:
            self.core.break_realtime()

            # ===== YELLOW SCOPE: Loading =====
            trap_ramp_end_amp = self.trap_ramp_start_amp   # safe init; real value (amp3) from the rebalancer below
            count1_bg = 0   # definite-assignment init; real value set in the load loop below
            count2_bg = 0
            atom_confirm_net = 0   # confirm counts of the atom this measurement uses
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

                # Check atom (HIGH trap). Log the net readout for the live histogram.
                atom_confirm_net = count1_1 + count2_1 - count1_bg - count2_bg
                self._push_readout(0, atom_confirm_net)
                if atom_confirm_net > self.atom_threshold:
                    break
            # ===== END YELLOW =====

            # ===== ORANGE SCOPE: Excitation (count clicks; keep the atom) =====
            n = 0               # cycle counter for the Check_img_every survival gate
            atom_alive = True
            # Per-atom diagnostic counters (reset per CONFIRMED atom). `n` doubles
            # as this atom's attempt count, so no extra per-cycle work is added.
            atom_heralds = 0     # clicks produced by THIS atom
            atom_first = -1      # attempt index (n) of this atom's FIRST click
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
                    # UNCHANGED. The only additions below are the photon's
                    # arrival delay WITHIN the collection gate, and a counter for
                    # two-detector coincidences (which the XOR discards).
                    #
                    # The delay is referenced to GATE OPEN, computed from the
                    # gate-end time the call already returns:
                    #     delay = click - t_end + t_spcm_collection_mu
                    # so it runs 0..400 ns by construction. Do NOT reference it
                    # to t0: the gate opens at t0 + exc_spcm_gate_offset_mu
                    # (5280 ns) while the excitation RF ends at t0 + 4100, the
                    # ~1.2 us gap being the excitation AOM's acoustic delay. A
                    # t0-referenced delay would carry that whole calibrated
                    # offset and sit near 5500 ns.
                    #   delay histogram ~ exp(-t/tau), tau = 30-35 ns
                    #       -> ONE photon per attempt from a clean pulse
                    #   delay histogram flat across the gate
                    #       -> multi-photon scattering, or background/dark counts
                    # This pins photons-per-attempt WITHOUT needing the detection
                    # efficiency or the imaging scattering rate.
                    if click1 > 0 and click2 < 0:
                        n_clicks += 1
                        atom_heralds += 1
                        if atom_first < 0:
                            atom_first = n
                        self._save_click(self.core.mu_to_seconds(click1 - t_run_start_mu),
                                         1, trials, how_many_atoms, n_clicks,
                                         click1 - t_end1 + int(self.t_spcm_collection_mu))
                        trials = 0
                        how_many_atoms = 0
                        self.core.break_realtime()   # restore slack drained by the async save RPC
                    elif click1 < 0 and click2 > 0:
                        n_clicks += 1
                        atom_heralds += 1
                        if atom_first < 0:
                            atom_first = n
                        self._save_click(self.core.mu_to_seconds(click2 - t_run_start_mu),
                                         2, trials, how_many_atoms, n_clicks,
                                         click2 - t_end2 + int(self.t_spcm_collection_mu))
                        trials = 0
                        how_many_atoms = 0
                        self.core.break_realtime()   # restore slack drained by the async save RPC
                    elif click1 > 0 and click2 > 0:
                        # BOTH detectors fired: not a herald (XOR rejects it), and
                        # until now counted nowhere. Its rate matters: at ~1 photon
                        # per attempt coincidences should be rare, so a large
                        # coincidence fraction is the signature of multi-photon
                        # scattering (or of two atoms). Counted only -- the herald
                        # definition and the timing are untouched.
                        n_coinc += 1

                    # No-detection dead-time tail (also re-blocks the MW switch that
                    # excitation_pulse unblocked); runs every cycle for uniform timing.
                    delay(5 * us)
                    delay(2 * us)
                    self.ttl_MW_switch.on()
                    delay(1 * us)

                    if n_clicks >= n_meas:
                        break

                if n_clicks >= n_meas:
                    # Run finished with this atom still ALIVE -> censored (reason 1).
                    self._push_atom(atom_confirm_net, n, atom_heralds, atom_first, 1)
                    self.core.break_realtime()   # restore slack drained by the async push
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
                    # (no background subtraction). Inlined from
                    # recool_and_check_survival_lowtrap (same image + recool) so we
                    # can LOG the raw counts -- that histogram vs the threshold line
                    # is what reveals whether the check actually separates a trapped
                    # atom from an empty trap, or passes empties (false negatives).
                    count1_lt, count2_lt = image_low_trap(
                        self, self.f_cooling_imaging_lowtrap,
                        self.amp_cooling_imaging_lowtrap, self.t_imaging_gate_lowtrap)
                    delay(10 * us)
                    imaging_pgc_low_trap(self)
                    self._push_readout(1, count1_lt + count2_lt)
                    self.core.break_realtime()   # restore slack drained by the async push
                    if (count1_lt + count2_lt) <= self.atom_threshold_lowtrap:
                        # Atom lost: flush its COMPLETE life history (reason 0).
                        self._push_atom(atom_confirm_net, n, atom_heralds,
                                        atom_first, 0)
                        self.core.break_realtime()   # restore slack drained by the async push
                        break    # atom lost -> back to Yellow (reload)

                beams_off_for_imaging(self)
            # ===== END ORANGE =====

            reset_to_mot(self)

        self._push_coinc(n_coinc)
        self._summarize()
        write_results(self)

    @rpc(flags={"async"})
    def _save_click(self, click_time_s, detector, trials, how_many_atoms, n_clicks,
                    delay_mu):
        """Append one photon click. trials / how_many_atoms = excitation attempts /
        atom-load attempts since the previous click (so their column sums are the
        run totals). delay_mu = arrival time WITHIN the collection gate, measured
        from the excitation-pulse reference t0, in machine units (= ns): the
        excited-state decay curve, which fixes photons-per-attempt."""
        self.append_to_dataset("click_times_s", click_time_s)
        self.append_to_dataset("click_detector", detector)
        self.append_to_dataset("trials_per_click", trials)
        self.append_to_dataset("how_many_atoms", how_many_atoms)
        self.append_to_dataset("click_delay_ns", delay_mu)
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
    def _push_coinc(self, n_coinc):
        """Store the two-detector coincidence count (pushed once, at run end).

        Blocking rather than async so it cannot still be queued when
        _summarize() reads it back.
        """
        self.set_dataset("n_coincidences", n_coinc, broadcast=True)

    @rpc(flags={"async"})
    def _push_atom(self, confirm_net, attempts, heralds, first_attempt, end_reason):
        """One row per CONFIRMED atom: its confirm brightness, how many excitation
        attempts it received, how many heralds it produced, when its first herald
        came, and how its life ended. See prepare() for the diagnostic rationale."""
        self.append_to_dataset("atom_confirm_net", confirm_net)
        self.append_to_dataset("atom_attempts", attempts)
        self.append_to_dataset("atom_heralds", heralds)
        self.append_to_dataset("atom_first_herald", first_attempt)
        self.append_to_dataset("atom_end_reason", end_reason)

    @rpc(flags={"async"})
    def _push_readout(self, channel, value):
        """Append one atom-readout count to its live histogram dataset:
        channel 0 = high-trap confirm image (net), 1 = low-trap survival (raw).
        Incremental append (cheap broadcast), so it is safe to call often."""
        if channel == 0:
            self.append_to_dataset("rate_readout_hightrap", value)
        else:
            self.append_to_dataset("rate_readout_lowtrap", value)

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

        # --- photon arrival delay inside the collection gate ---------------
        # The SHAPE of the arrival time is the single-photon test, and it needs
        # no detection efficiency: an exponential with tau = 30-35 ns (Cs: D2
        # 30.5, D1 34.9) means the atom emits ONE photon and goes dark -- even
        # though exc_beam_duration_mu drives it for ~10 lifetimes (300 ns), it
        # is pumped dark after ~1 scattering event. A profile flat across the
        # gate would mean multi-photon scattering or background instead.
        # Measured on RID 3646: onset 135 ns, tau = 36 ns, flat bg ~10 %.
        #
        # Do NOT judge this by the mean delay against a fraction of the gate
        # (what this printout used to do): the emission starts ~135 ns AFTER
        # gate open, so the mean sits near 177 ns and that test labelled a
        # clean single-photon decay "FLAT".
        try:
            d_all = [float(x) for x in self.get_dataset("click_delay_ns")]
        except Exception:
            d_all = []
        gate_ns = float(self.t_spcm_collection_mu)
        # NEGATIVE delays are not early photons, they are STALE timestamps:
        # timestamp_mu returns only the FIRST edge in a gate, so a second edge
        # (an SPCM afterpulse 20-50 ns later) survives in the input FIFO and is
        # handed to a LATER attempt carrying its old timestamp. Excluded from
        # the shape; reported on their own below.
        stale = [x for x in d_all if x < 0.0]
        d = sorted(x for x in d_all if x >= 0.0)
        if d:
            top = max(gate_ns, d[-1])
            past = sum(1 for x in d if x > gate_ns)
            fit = _fit_decay_host(d, top)
            print(f"  click delay   = median {d[len(d) // 2]:.0f} ns of a "
                  f"{gate_ns:.0f} ns gate  (N={len(d)})")
            if fit is None:
                print("  decay profile = too few clicks to fit")
            elif fit["flat"]:
                print("  decay profile = FLAT across the gate -> multi-photon "
                      "scattering or background, NOT a clean single photon")
            else:
                print(f"  decay profile = onset {fit['t0']:.0f} ns, tau = "
                      f"{fit['tau']:.0f} ns "
                      f"[{fit['tau_lo']:.0f}-{fit['tau_hi']:.0f}], flat bg "
                      f"{100.0 * fit['w']:.0f} %")
                if 20.0 <= fit["tau"] <= 55.0:
                    print("                  -> tau IS the Cs natural lifetime: each "
                          "herald is ONE spontaneously emitted photon")
                else:
                    print("                  -> tau is NOT the natural lifetime: suspect "
                          "multi-photon scattering or background")
            if past:
                print(f"                  {past} clicks past the nominal gate close -- "
                      f"expected: the input gate effective window is offset ~65 ns "
                      f"later in gateware, so the delay axis is shifted, not the shape")
        if stale:
            print(f"  STALE reads   = {len(stale)} of {len(d_all)} clicks "
                  f"({100.0 * len(stale) / len(d_all):.2f} %) carried a timestamp from "
                  f"an EARLIER gate (most negative {min(stale):.0f} ns)")
            print("                  cause: timestamp_mu returns only the FIRST edge, so "
                  "a second edge (SPCM afterpulse) waits in the input FIFO and is read "
                  "by a later attempt")
            print("                  impact here: those heralds are attributed to the "
                  "wrong attempt, inflating p by that fraction. In PARITY the same value "
                  "feeds at_mu(click + 2000) -> MW edge scheduled in the PAST -> "
                  "RTIOUnderflow")

        # --- two-detector coincidences (rejected by the XOR herald) --------
        # These bound the emitted photon number WITHOUT needing the detection
        # efficiency. With per-detector probability q = clicks/(2*attempts),
        # nbar emitted photons give ~(nbar*q)^2 coincidences per attempt, so a
        # small coincidence count caps nbar. RID 3646: 0 of 85,380 attempts ->
        # nbar < 3.2 (95 % CL), which is what excludes "the same-detector
        # doubles are second photons" (that needs nbar ~ 11 -> ~36 expected).
        try:
            n_co = int(self.get_dataset("n_coincidences"))
        except Exception:
            n_co = 0
        q = (n_clicks / (2.0 * total_attempts)) if total_attempts > 0 else 0.0
        if q > 0.0:
            if n_co == 0:
                # 0 observed -> 95 % upper limit of 3 expected events
                print(f"  coincidences  = 0 of {total_attempts} attempts -> nbar < "
                      f"{math.sqrt(3.0 / total_attempts) / q:.1f} emitted photons per "
                      f"attempt (95 % CL)")
            else:
                print(f"  coincidences  = {n_co} "
                      f"({100.0 * n_co / max(n_clicks, 1):.2f} % of heralds, rejected by "
                      f"the XOR) -> nbar ~ "
                      f"{math.sqrt(n_co / total_attempts) / q:.1f} photons per attempt")
            print(f"  => at ~1 photon per attempt the herald probability IS the "
                  f"collection+detection efficiency: eta ~ "
                  f"{100.0 * n_clicks / total_attempts:.2f} %")
        print("  (precise wall-clock duration is in the HDF5 metadata: duration_seconds)")
