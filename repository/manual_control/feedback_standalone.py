"""Standalone AOM feedback characterization experiment.

This is the MANUAL class — a full EnvExperiment you run from the dashboard
to characterize beam stability, measure or set reference voltages, and verify
feedback performance independently of any physics experiment.

Per-channel controls:
  - use_stored_ref: True = use stored reference from calibration profile,
                    False = measure current voltage as fresh reference
  - save_ref: True = write this channel's reference to the profile after run,
              False = don't save (overwrite protection)

For the AUTO class called inline within experiments, see subroutines/aom_feedback.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.calibration_store import CalibrationStore, CHANNEL_ORDER


class FeedbackStandalone(EnvExperiment):
    """AOM_Feedback_5Channel"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("sampler0")
        self.setattr_device("urukul2_ch0")   # MOT X
        self.setattr_device("urukul2_ch1")   # MOT Y
        self.setattr_device("urukul2_ch2")   # MOT Z
        self.setattr_device("urukul1_ch1")   # FORT
        self.setattr_device("urukul1_ch2")   # Excitation
        self.setattr_device("urukul1_ch3")   # D1 onoff L
        self.setattr_device("urukul0_ch1")   # Cooling DP
        self.setattr_device("ttl14")
        self.setattr_device("ttl6")
        self.setattr_device("ttl5")

        # --- Calibration profile selection ---
        self.cal_store = CalibrationStore()
        profiles = self.cal_store.list_profiles()
        self.setattr_argument("target_profile",
                              EnumerationValue(profiles, default=profiles[0]),
                              group="Calibration Profile")

        # --- Load defaults from feedback_channels.json ---
        import json
        from pathlib import Path
        json_path = Path(__file__).parent.parent / "config" / "feedback_channels.json"
        try:
            with open(json_path, "r") as f:
                fb_config = json.load(f)
        except Exception:
            fb_config = {"channels": {}}

        chan_keys = ["MOT_X", "MOT_Y", "MOT_Z", "FORT", "excitation"]
        chan_labels = ["MOT_X", "MOT_Y", "MOT_Z", "FORT", "Excitation"]

        for i, (key, label) in enumerate(zip(chan_keys, chan_labels)):
            ch = fb_config["channels"].get(key, {})
            group = f"{label} Settings"

            # Per-channel reference control
            self.setattr_argument(f"use_stored_ref_{i}",
                                  BooleanValue(default=True), group=group)
            self.setattr_argument(f"save_ref_{i}",
                                  BooleanValue(default=False), group=group)

            # Per-channel: use JSON params or override from dashboard?
            self.setattr_argument(f"use_json_params_{i}",
                                  BooleanValue(default=True), group=group)

            # Dashboard overrides (defaults populated from JSON)
            self.setattr_argument(f"initial_amp_{i}",
                                  NumberValue(default=ch.get("initial_amp", 0.1),
                                              step=0.01, min=0.0, max=1.0),
                                  group=group)
            self.setattr_argument(f"frequency_{i}",
                                  NumberValue(default=ch.get("frequency_hz", 80e6),
                                              unit="Hz", step=1e6, min=0.0, max=400e6),
                                  group=group)
            self.setattr_argument(f"p_gain_{i}",
                                  NumberValue(default=ch.get("p_gain", 0.08),
                                              step=0.01, min=0.0, max=1.0, ndecimals=4),
                                  group=group)
            self.setattr_argument(f"tol_{i}",
                                  NumberValue(default=ch.get("tolerance", 0.01),
                                              step=0.005, min=0.0, max=1.0, ndecimals=4),
                                  group=group)
            self.setattr_argument(f"iterations_{i}",
                                  NumberValue(default=ch.get("max_iterations", 20),
                                              step=1, min=1, max=100,
                                              ndecimals=0, type="int"),
                                  group=group)

            # Hard safety cap on the feedback-driven amplitude (JSON only,
            # no dashboard override). Falls back to a conservative 0.5 if the
            # JSON entry is missing.
            setattr(self, f"max_amp_{i}", float(ch.get("max_amp", 0.5)))

        # Store JSON config for prepare() to resolve use_json_params
        self._fb_config = fb_config

    def prepare(self):
        self.stored_refs = self.cal_store.get_refs(self.target_profile)
        info = self.cal_store.get_profile_info(self.target_profile)

        chan_keys = ["MOT_X", "MOT_Y", "MOT_Z", "FORT", "excitation"]
        labels = ["MOT_X", "MOT_Y", "MOT_Z", "FORT", "Excitation"]

        # Resolve use_json_params: override dashboard values with JSON if toggled
        for i, key in enumerate(chan_keys):
            if getattr(self, f"use_json_params_{i}"):
                ch = self._fb_config["channels"].get(key, {})
                setattr(self, f"initial_amp_{i}", ch.get("initial_amp", getattr(self, f"initial_amp_{i}")))
                setattr(self, f"frequency_{i}", ch.get("frequency_hz", getattr(self, f"frequency_{i}")))
                setattr(self, f"p_gain_{i}", ch.get("p_gain", getattr(self, f"p_gain_{i}")))
                setattr(self, f"tol_{i}", ch.get("tolerance", getattr(self, f"tol_{i}")))
                setattr(self, f"iterations_{i}", int(ch.get("max_iterations", getattr(self, f"iterations_{i}"))))
            # Refresh max_amp from JSON regardless of use_json_params toggle —
            # it is a safety cap, never user-overridable.
            ch = self._fb_config["channels"].get(key, {})
            setattr(self, f"max_amp_{i}", float(ch.get("max_amp", getattr(self, f"max_amp_{i}"))))

        print(f"Target profile: '{self.target_profile}'")
        print(f"  Description: {info['description']}")
        print(f"  Last calibrated: {info['calibrated_at'] or 'NEVER'}")
        for i, label in enumerate(labels):
            use_ref = getattr(self, f"use_stored_ref_{i}")
            save = getattr(self, f"save_ref_{i}")
            use_json = getattr(self, f"use_json_params_{i}")
            ref = self.stored_refs[i]
            ref_str = f"stored={ref:.5f}" if use_ref else "MEASURE FRESH"
            params_str = "JSON" if use_json else "DASHBOARD"
            save_str = " [WILL SAVE]" if save else ""
            print(f"  {label}: ref={ref_str}, params={params_str}{save_str}")

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()

        # Set up beam path for feedback measurement
        self.ttl14.on()
        self.ttl6.on()
        self.ttl5.on()
        delay(1000 * us)
        self.urukul1_ch3.set(frequency=250 * MHz, amplitude=0.43)
        self.urukul1_ch3.sw.on()
        delay(1 * us)
        self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3)
        self.urukul0_ch1.sw.on()
        delay(1 * us)
        self.urukul1_ch1.sw.off()
        delay(1 * us)
        self.ttl6.off()
        delay(1 * us)
        self.ttl14.off()
        delay(10 * ms)

        # Turn on each channel
        self.urukul2_ch0.set(frequency=self.frequency_0, amplitude=self.initial_amp_0)
        self.urukul2_ch0.sw.on()
        self.urukul2_ch1.set(frequency=self.frequency_1, amplitude=self.initial_amp_1)
        self.urukul2_ch1.sw.on()
        self.urukul2_ch2.set(frequency=self.frequency_2, amplitude=self.initial_amp_2)
        self.urukul2_ch2.sw.on()
        self.urukul1_ch1.set(frequency=self.frequency_3, amplitude=self.initial_amp_3)
        self.urukul1_ch1.sw.on()
        self.ttl14.off()
        self.ttl6.off()
        self.urukul1_ch2.set(frequency=self.frequency_4, amplitude=self.initial_amp_4)
        self.urukul1_ch2.sw.on()

        self.sampler0.init()
        delay(10 * ms)

        # Capture initial amplitudes and voltages
        init_amps = [0.0] * 5
        init_amps[0] = self.initial_amp_0
        init_amps[1] = self.initial_amp_1
        init_amps[2] = self.initial_amp_2
        init_amps[3] = self.initial_amp_3
        init_amps[4] = self.initial_amp_4

        init_vs = [0.0] * 5
        buf = [0] * 8
        self.sampler0.sample_mu(buf)
        delay(1 * ms)
        for i in range(5):
            init_vs[i] = buf[i] * (20.0 / 65536.0)

        # Per-channel reference determination
        refs = [0.0] * 5

        # Channel 0 (MOT X)
        if self.use_stored_ref_0 and (self.stored_refs[0] >= 0.0):
            refs[0] = self.stored_refs[0]
        else:
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            refs[0] = buf[0] * (20.0 / 65536.0)

        # Channel 1 (MOT Y)
        if self.use_stored_ref_1 and (self.stored_refs[1] >= 0.0):
            refs[1] = self.stored_refs[1]
        else:
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            refs[1] = buf[1] * (20.0 / 65536.0)

        # Channel 2 (MOT Z)
        if self.use_stored_ref_2 and (self.stored_refs[2] >= 0.0):
            refs[2] = self.stored_refs[2]
        else:
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            refs[2] = buf[2] * (20.0 / 65536.0)

        # Channel 3 (FORT)
        if self.use_stored_ref_3 and (self.stored_refs[3] >= 0.0):
            refs[3] = self.stored_refs[3]
        else:
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            refs[3] = buf[3] * (20.0 / 65536.0)

        # Channel 4 (Excitation)
        if self.use_stored_ref_4 and (self.stored_refs[4] >= 0.0):
            refs[4] = self.stored_refs[4]
        else:
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            refs[4] = buf[4] * (20.0 / 65536.0)

        # P-only feedback — channel 0 (MOT X)
        amp = self.initial_amp_0
        for _ in range(self.iterations_0):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[0] * (20.0 / 65536.0)
            if (refs[0] > 0.0) and (abs(meas - refs[0]) / refs[0] < self.tol_0):
                break
            amp = max(0.0, min(self.max_amp_0, amp + self.p_gain_0 * (refs[0] - meas)))
            self.urukul2_ch0.set(frequency=self.frequency_0, amplitude=amp)
            delay(1 * ms)
        a0 = amp

        # Channel 1 (MOT Y)
        amp = self.initial_amp_1
        for _ in range(self.iterations_1):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[1] * (20.0 / 65536.0)
            if (refs[1] > 0.0) and (abs(meas - refs[1]) / refs[1] < self.tol_1):
                break
            amp = max(0.0, min(self.max_amp_1, amp + self.p_gain_1 * (refs[1] - meas)))
            self.urukul2_ch1.set(frequency=self.frequency_1, amplitude=amp)
            delay(1 * ms)
        a1 = amp

        # Channel 2 (MOT Z)
        amp = self.initial_amp_2
        for _ in range(self.iterations_2):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[2] * (20.0 / 65536.0)
            if (refs[2] > 0.0) and (abs(meas - refs[2]) / refs[2] < self.tol_2):
                break
            amp = max(0.0, min(self.max_amp_2, amp + self.p_gain_2 * (refs[2] - meas)))
            self.urukul2_ch2.set(frequency=self.frequency_2, amplitude=amp)
            delay(1 * ms)
        a2 = amp

        # Channel 3 (FORT)
        amp = self.initial_amp_3
        for _ in range(self.iterations_3):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[3] * (20.0 / 65536.0)
            if (refs[3] > 0.0) and (abs(meas - refs[3]) / refs[3] < self.tol_3):
                break
            amp = max(0.0, min(self.max_amp_3, amp + self.p_gain_3 * (refs[3] - meas)))
            self.urukul1_ch1.set(frequency=self.frequency_3, amplitude=amp)
            delay(1 * ms)
        a3 = amp

        # Channel 4 (Excitation)
        amp = self.initial_amp_4
        for _ in range(self.iterations_4):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[4] * (20.0 / 65536.0)
            if (refs[4] > 0.0) and (abs(meas - refs[4]) / refs[4] < self.tol_4):
                break
            amp = max(0.0, min(self.max_amp_4, amp + self.p_gain_4 * (refs[4] - meas)))
            self.urukul1_ch2.set(frequency=self.frequency_4, amplitude=amp)
            delay(1 * ms)
        a4 = amp

        # Final voltages
        final_vs = [0.0] * 5
        self.sampler0.sample_mu(buf)
        delay(1 * ms)
        for i in range(5):
            final_vs[i] = buf[i] * (20.0 / 65536.0)

        self._report(init_amps, init_vs, refs, final_vs, [a0, a1, a2, a3, a4])

    def _report(self, init_amps, init_vs, refs, final_vs, amps):
        """Print results and save per-channel references where enabled."""
        labels = ["MOT_X", "MOT_Y", "MOT_Z", "FORT", "Excitation"]
        use_flags = [self.use_stored_ref_0, self.use_stored_ref_1,
                     self.use_stored_ref_2, self.use_stored_ref_3,
                     self.use_stored_ref_4]
        save_flags = [self.save_ref_0, self.save_ref_1,
                      self.save_ref_2, self.save_ref_3,
                      self.save_ref_4]

        print()
        print(f"Profile: '{self.target_profile}'")
        print("=" * 90)
        print(f"{'Ch':>2} {'Name':>11} | {'Ref Src':>8} | {'Save':>4} | {'Init Amp':>8} | "
              f"{'Init V':>9} | {'Ref V':>8} | {'Final V':>9} | {'Final Amp':>9}")
        print("-" * 90)
        for i in range(5):
            src = "stored" if use_flags[i] else "FRESH"
            save = "YES" if save_flags[i] else "no"
            print(f"{i:2d} {labels[i]:>11} | {src:>8} | {save:>4} | "
                  f" {init_amps[i]:.4f}  | {init_vs[i]:9.5f} | "
                  f"{refs[i]:8.5f} | {final_vs[i]:9.5f} |   {amps[i]:.5f}")
        print("=" * 90)

        # Save only channels where save_ref is enabled
        voltages_to_save = {}
        for i, label in enumerate(labels):
            if save_flags[i]:
                if not use_flags[i]:
                    # Fresh measurement: save the measured reference
                    voltages_to_save[label] = refs[i]
                else:
                    # Used stored ref: save final voltage (after feedback convergence)
                    voltages_to_save[label] = final_vs[i]

        if voltages_to_save:
            self.cal_store.save_refs(voltages_to_save, profile=self.target_profile)
            for label, val in voltages_to_save.items():
                print(f">> SAVED {label} = {val:.6f} to profile '{self.target_profile}'")
        else:
            print(">> No channels saved (all save_ref toggles are OFF)")

        print()
        self.cal_store.print_summary()
