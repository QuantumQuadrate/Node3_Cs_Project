"""Atom load + SPCM image + EMCCD trigger (no scanning).

Plain, repeated single-atom cycle -- no parameter scan:
  1. Deterministic atom loading (as usual).
  2. Trapped-atom imaging via the SPCMs (records the counts).
  3. EMCCD exposure: fire a trigger pulse (Zotino DAC channel 6, high for
     `emccd_trigger_ms`, default 1 ms). That trigger opens the EMCCD for its
     camera-side exposure (default 50 ms). We do NOT control the camera timing
     here -- only the trigger -- BUT the imaging beams (cooling at imaging
     frequency + repump) are held ON for the whole exposure so the atom
     fluoresces onto the camera.

This experiment is deliberately SELF-CONTAINED: it only *calls* existing
subroutines (load_atom_deterministic, spcm_image_dual, ...) and inlines the
EMCCD trigger + imaging-beam-on itself. It does NOT modify sequences.py or any
shared code.

EMCCD trigger wiring follows the archive release-and-recapture experiments:
Zotino `zotino0`, DAC channel 6, driven to ~4.1 V for the trigger high, 0 V idle
(channels 0-3 are the shim coils, so channel 6 does not collide).

Data saved per shot (broadcast): img_loaded, img_spcm1, img_spcm2.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, beams_off_for_imaging, spcm_image_dual,
    reset_to_mot,
)
from utilities.write_results import write_results


class AtomImagingEMCCD(EnvExperiment):
    """Atom Load + SPCM image + EMCCD trigger"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("num_runs",
                              NumberValue(50, min=1, max=600000, step=1, ndecimals=0, type="int"),
                              group="Experiment")

        # --- EMCCD trigger ---
        self.setattr_argument("emccd_dac_channel",
                              NumberValue(6, min=0, max=31, step=1, ndecimals=0, type="int"),
                              group="EMCCD")
        self.setattr_argument("emccd_trigger_v",
                              NumberValue(4.1, min=0.0, max=10.0, ndecimals=2),
                              group="EMCCD")
        self.setattr_argument("emccd_trigger_ms",
                              NumberValue(1.0, min=0.001, max=100.0, ndecimals=3),
                              group="EMCCD")
        # Informational: the camera-side exposure the trigger opens. We keep the
        # imaging beams on for this long. The camera's own exposure is set in its
        # software -- match this value to it.
        self.setattr_argument("emccd_exposure_ms",
                              NumberValue(50.0, min=1.0, max=2000.0, ndecimals=1),
                              group="EMCCD")
        # Wait after the trap beam is turned off, before the background frame,
        # so the released atom has left the trap region (frame 2 = no atom).
        self.setattr_argument("t_bg_wait_ms",
                              NumberValue(5.0, min=0.0, max=1000.0, ndecimals=1),
                              group="EMCCD")

        # Rebalancer is created/configured by BaseExperiment.build()

    def prepare(self):
        self.base.prepare(feedback_profile="FORT_loading")

        self.emccd_ch = int(self.emccd_dac_channel)
        self.t_emccd_trig = float(self.emccd_trigger_ms) * 1e-3
        self.t_emccd_expo = float(self.emccd_exposure_ms) * 1e-3
        self.t_bg_wait = float(self.t_bg_wait_ms) * 1e-3

        for key in ("img_loaded", "img_spcm1", "img_spcm2"):
            self.set_dataset(key, [], broadcast=True)
        self.set_dataset("img_progress", 0, broadcast=True)
        self.set_dataset("img_emccd_exposure_ms", float(self.emccd_exposure_ms),
                         broadcast=True, persist=True)

        print(f"AtomImagingEMCCD: {int(self.num_runs)} shots, "
              f"EMCCD trigger = {self.emccd_trigger_ms:.3f} ms on zotino0 DAC{self.emccd_ch} "
              f"@ {self.emccd_trigger_v:.2f} V, imaging beams held ON for "
              f"{self.emccd_exposure_ms:.1f} ms exposure.")

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        # Make sure the EMCCD trigger line starts LOW.
        self.zotino0.write_dac(self.emccd_ch, 0.0)
        self.zotino0.load()
        delay(1 * us)

        # One initial 5-channel feedback to lock FORT / beam balance.
        prepare_for_feedback(self)
        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
        restore_after_feedback(self)
        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            self.core.break_realtime()

            # Background (FORT off, beams on, no atom) for the SPCM threshold.
            count1_bg, count2_bg = background_measurement_dual(self, self.t_imaging_gate)
            bg = count1_bg + count2_bg

            # --- 1. Atom loading as usual ---
            (success, n_tries, n_fb, s1, s2,
             first_unloaded, t_load) = load_atom_deterministic(self)

            # --- 2. Trapped-atom imaging via SPCM ---
            imaging_pgc(self)
            beams_off_for_imaging(self)
            delay(self.t_SPCM_exposure)
            c1, c2 = spcm_image_dual(self, self.t_imaging_gate)
            imaging_pgc(self)
            beams_off_for_imaging(self)
            loaded = ((c1 + c2) - bg) > self.atom_threshold

            # --- 3. EMCCD frame 1: TRAPPED ATOM (trap beam still on) ---
            self._emccd_image()

            # --- 4. Turn OFF the trap beam (release the atom), wait for it to
            #        leave, then EMCCD frame 2 = BACKGROUND (no atom). Comparing
            #        frame 1 vs frame 2 verifies the signal is a real atom. ---
            self.dds_FORT.sw.off()
            delay(self.t_bg_wait)
            self._emccd_image()

            # Log and reset for the next shot (the next load turns the FORT back on).
            self._log(1 if loaded else 0, c1, c2, i)
            reset_to_mot(self)

        write_results(self)

    @kernel
    def _emccd_image(self):
        """One EMCCD frame: hold the imaging beams ON for the whole exposure and
        fire the trigger pulse (zotino0 DAC emccd_ch) to open the camera. Used for
        both the trapped-atom frame and the trap-off background frame."""
        # Imaging beams on (cooling DP at imaging frequency + repump)
        self.dds_cooling_DP.set(frequency=self.f_cooling_imaging,
                                amplitude=self.amp_cooling_imaging)
        self.dds_cooling_DP.sw.on()
        delay(1 * us)
        self.dds_repump.sw.on()
        self.ttl_repump_switch.off()
        delay(1 * us)

        # Fire the EMCCD trigger: rising edge opens the camera exposure.
        self.zotino0.write_dac(self.emccd_ch, self.emccd_trigger_v)
        self.zotino0.load()
        delay(self.t_emccd_trig)                 # trigger high (e.g. 1 ms)
        self.zotino0.write_dac(self.emccd_ch, 0.0)
        self.zotino0.load()

        # Hold the imaging beams on for the full camera exposure (e.g. 50 ms).
        delay(self.t_emccd_expo)

        # Imaging beams off (existing subroutine -- just called, not modified).
        beams_off_for_imaging(self)

    @rpc(flags={"async"})
    def _log(self, loaded, c1, c2, i):
        self.append_to_dataset("img_loaded", loaded)
        self.append_to_dataset("img_spcm1", c1)
        self.append_to_dataset("img_spcm2", c2)
        self.set_dataset("img_progress", i + 1, broadcast=True)
        if i % 10 == 0:
            print(f"  shot {i}: loaded={loaded}  SPCM=({c1},{c2})")
