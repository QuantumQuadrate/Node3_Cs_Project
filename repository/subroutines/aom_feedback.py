"""AOM power stabilization for inline use within experiments.

FeedbackRebalancer_5: 5-channel P-only feedback.

Usage:
    # BaseExperiment.build() creates and configures self.rebalancer.
    # BaseExperiment.prepare() loads ref voltages for the active profile.
    # In kernel:
    amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()

All tuning params (gain, tolerance, etc.) are loaded from feedback_channels.json.
Reference voltages are loaded from calibration_references.json in prepare().
rebalance() is a @kernel — no RPC nesting.
"""

from artiq.experiment import *
from utilities.calibration_store import CalibrationStore

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
CHANNEL_ORDER = ["MOT_X", "MOT_Y", "MOT_Z", "FORT", "excitation"]


class FeedbackRebalancer_5:
    """5-channel AOM power stabilizer.

    Devices and params are set by BaseExperiment.
    rebalance() is a @kernel — call directly from experiment kernels.
    """

    def load_params_from_json(self):
        """Load tuning params from feedback_channels.json + init diagnostic attrs."""
        json_path = CONFIG_DIR / "feedback_channels.json"
        with open(json_path, "r") as f:
            config = json.load(f)

        for i, ch_name in enumerate(CHANNEL_ORDER):
            ch = config["channels"].get(ch_name, {})
            setattr(self, f"p{i}",       ch.get("p_gain", 0.08))
            setattr(self, f"tol{i}",     ch.get("tolerance", 0.01))
            setattr(self, f"it{i}",      int(ch.get("max_iterations", 20)))
            setattr(self, f"amp{i}",     ch.get("initial_amp", 0.1))
            setattr(self, f"max_amp{i}", ch.get("max_amp", 0.5))
            setattr(self, f"f{i}",       ch.get("frequency_hz", 80e6))

        # Diagnostic attributes (ARTIQ compiler needs them pre-declared)
        self.init_v0 = 0.0
        self.init_v1 = 0.0
        self.init_v2 = 0.0
        self.init_v3 = 0.0
        self.init_v4 = 0.0
        self.final_v0 = 0.0
        self.final_v1 = 0.0
        self.final_v2 = 0.0
        self.final_v3 = 0.0
        self.final_v4 = 0.0

        # Ref voltages (set by load_profile before kernel runs)
        self.ref0 = 0.0
        self.ref1 = 0.0
        self.ref2 = 0.0
        self.ref3 = 0.0
        self.ref4 = 0.0

        self._cal_store = CalibrationStore()
        self._experiment = None
        self._profile_name = ""

    def load_profile(self, profile=None):
        """Host-side: load ref voltages for a profile onto self.ref0..ref4.

        If profile is None, uses the active_profile from calibration_references.json.
        Call this in prepare() or between kernel runs — NOT from inside a kernel.
        """
        if profile is None:
            profile = self._cal_store.active_profile
        refs = self._cal_store.get_refs(profile)
        self.ref0 = refs[0]
        self.ref1 = refs[1]
        self.ref2 = refs[2]
        self.ref3 = refs[3]
        self.ref4 = refs[4]
        self._profile_name = profile

    def save_diagnostics(self, final_amps):
        """Host-side: write feedback diagnostics to experiment datasets.

        Call this after rebalance() returns — NOT from inside a kernel.
        """
        exp = self._experiment
        if exp is None:
            return

        amp0, amp1, amp2, amp3, amp4 = final_amps

        exp.set_dataset("fb_profile_used", self._profile_name, persist=True, broadcast=True)
        exp.set_dataset("fb_initial_amplitudes",
                        [self.amp0, self.amp1, self.amp2, self.amp3, self.amp4], persist=True)
        exp.set_dataset("fb_final_amplitudes",
                        [amp0, amp1, amp2, amp3, amp4], persist=True, broadcast=True)
        exp.set_dataset("fb_initial_voltages",
                        [self.init_v0, self.init_v1, self.init_v2, self.init_v3, self.init_v4], persist=True)
        exp.set_dataset("fb_final_voltages",
                        [self.final_v0, self.final_v1, self.final_v2, self.final_v3, self.final_v4], persist=True, broadcast=True)
        exp.set_dataset("fb_ref_voltages",
                        [self.ref0, self.ref1, self.ref2, self.ref3, self.ref4], persist=True)
        exp.set_dataset("fb_params_gain",
                        [self.p0, self.p1, self.p2, self.p3, self.p4], persist=True)
        exp.set_dataset("fb_params_tolerance",
                        [self.tol0, self.tol1, self.tol2, self.tol3, self.tol4], persist=True)
        exp.set_dataset("fb_params_iterations",
                        [self.it0, self.it1, self.it2, self.it3, self.it4], persist=True)

    @kernel
    def rebalance(self):
        """Kernel: P-only feedback on all 5 channels.

        Reads ALL params from self.* attributes (set by load_params_from_json
        and load_profile before the kernel started).

        DDS amplitudes are applied to hardware during each iteration.
        Returns (amp0, amp1, amp2, amp3, amp4).
        """
        delay(1 * ms)

        # Initialize each channel
        self.urukul2_ch0.set(frequency=self.f0, amplitude=self.amp0)
        delay(1 * us)
        self.urukul2_ch0.sw.on()
        self.urukul2_ch1.set(frequency=self.f1, amplitude=self.amp1)
        delay(1 * us)
        self.urukul2_ch1.sw.on()
        self.urukul2_ch2.set(frequency=self.f2, amplitude=self.amp2)
        delay(1 * us)
        self.urukul2_ch2.sw.on()
        self.urukul1_ch1.set(frequency=self.f3, amplitude=self.amp3)
        delay(1 * us)
        self.urukul1_ch1.sw.on()
        self.urukul1_ch2.set(frequency=self.f4, amplitude=self.amp4)
        delay(1 * us)
        self.urukul1_ch2.sw.on()
        self.ttl14.off()
        self.ttl6.off()
        delay(10 * ms)

        buf = [0] * 8

        # Capture initial voltages
        self.sampler0.sample_mu(buf)
        delay(1 * ms)
        self.init_v0 = buf[0] * (20.0 / 65536.0)
        self.init_v1 = buf[1] * (20.0 / 65536.0)
        self.init_v2 = buf[2] * (20.0 / 65536.0)
        self.init_v3 = buf[3] * (20.0 / 65536.0)
        self.init_v4 = buf[4] * (20.0 / 65536.0)

        # --- Channel 0: MOT X ---
        target = self.ref0
        amp = self.amp0
        for _ in range(self.it0):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[0] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol0):
                break
            amp = max(0.0, min(self.max_amp0, amp + self.p0 * (target - meas)))
            self.urukul2_ch0.set(frequency=self.f0, amplitude=amp)
            delay(1 * ms)
        amp0 = amp

        # --- Channel 1: MOT Y ---
        target = self.ref1
        amp = self.amp1
        for _ in range(self.it1):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[1] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol1):
                break
            amp = max(0.0, min(self.max_amp1, amp + self.p1 * (target - meas)))
            self.urukul2_ch1.set(frequency=self.f1, amplitude=amp)
            delay(1 * ms)
        amp1 = amp

        # --- Channel 2: MOT Z ---
        target = self.ref2
        amp = self.amp2
        for _ in range(self.it2):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[2] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol2):
                break
            amp = max(0.0, min(self.max_amp2, amp + self.p2 * (target - meas)))
            self.urukul2_ch2.set(frequency=self.f2, amplitude=amp)
            delay(1 * ms)
        amp2 = amp

        # --- Channel 3: FORT ---
        target = self.ref3
        amp = self.amp3
        for _ in range(self.it3):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[3] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol3):
                break
            amp = max(0.0, min(self.max_amp3, amp + self.p3 * (target - meas)))
            self.urukul1_ch1.set(frequency=self.f3, amplitude=amp)
            delay(1 * ms)
        amp3 = amp

        # --- Channel 4: Excitation ---
        target = self.ref4
        amp = self.amp4
        for _ in range(self.it4):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[4] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol4):
                break
            amp = max(0.0, min(self.max_amp4, amp + self.p4 * (target - meas)))
            self.urukul1_ch2.set(frequency=self.f4, amplitude=amp)
            delay(1 * ms)
        amp4 = amp

        # Capture final voltages
        self.sampler0.sample_mu(buf)
        delay(1 * ms)
        self.final_v0 = buf[0] * (20.0 / 65536.0)
        self.final_v1 = buf[1] * (20.0 / 65536.0)
        self.final_v2 = buf[2] * (20.0 / 65536.0)
        self.final_v3 = buf[3] * (20.0 / 65536.0)
        self.final_v4 = buf[4] * (20.0 / 65536.0)

        return amp0, amp1, amp2, amp3, amp4

    @kernel
    def rebalance_3ch(self):
        """Kernel: P-only feedback on MOT X/Y/Z only (channels 0-2).

        Does NOT touch FORT or excitation — leaves them unchanged.
        No TTL toggling (no ttl14/ttl6 changes).
        Matches the original 3-channel FeedbackRebalancer used by atom_load_monitor.

        Returns (amp0, amp1, amp2).
        """
        delay(1 * ms)

        # Initialize MOT channels only
        self.urukul2_ch0.set(frequency=self.f0, amplitude=self.amp0)
        delay(1 * us)
        self.urukul2_ch0.sw.on()
        self.urukul2_ch1.set(frequency=self.f1, amplitude=self.amp1)
        delay(1 * us)
        self.urukul2_ch1.sw.on()
        self.urukul2_ch2.set(frequency=self.f2, amplitude=self.amp2)
        delay(1 * us)
        self.urukul2_ch2.sw.on()
        delay(10 * ms)

        buf = [0] * 8

        # Capture initial voltages (MOT channels only)
        self.sampler0.sample_mu(buf)
        delay(1 * ms)
        self.init_v0 = buf[0] * (20.0 / 65536.0)
        self.init_v1 = buf[1] * (20.0 / 65536.0)
        self.init_v2 = buf[2] * (20.0 / 65536.0)

        # --- Channel 0: MOT X ---
        target = self.ref0
        amp = self.amp0
        for _ in range(self.it0):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[0] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol0):
                break
            amp = max(0.0, min(self.max_amp0, amp + self.p0 * (target - meas)))
            self.urukul2_ch0.set(frequency=self.f0, amplitude=amp)
            delay(1 * ms)
        amp0 = amp

        # --- Channel 1: MOT Y ---
        target = self.ref1
        amp = self.amp1
        for _ in range(self.it1):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[1] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol1):
                break
            amp = max(0.0, min(self.max_amp1, amp + self.p1 * (target - meas)))
            self.urukul2_ch1.set(frequency=self.f1, amplitude=amp)
            delay(1 * ms)
        amp1 = amp

        # --- Channel 2: MOT Z ---
        target = self.ref2
        amp = self.amp2
        for _ in range(self.it2):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[2] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < self.tol2):
                break
            amp = max(0.0, min(self.max_amp2, amp + self.p2 * (target - meas)))
            self.urukul2_ch2.set(frequency=self.f2, amplitude=amp)
            delay(1 * ms)
        amp2 = amp

        # Capture final voltages (MOT channels only)
        self.sampler0.sample_mu(buf)
        delay(1 * ms)
        self.final_v0 = buf[0] * (20.0 / 65536.0)
        self.final_v1 = buf[1] * (20.0 / 65536.0)
        self.final_v2 = buf[2] * (20.0 / 65536.0)

        return amp0, amp1, amp2


class FeedbackRebalancer_exc:
    """Single-channel excitation-only feedback."""

    @kernel
    def rebalance_exc(self, p4, tol4, it4, ref4, amp4, f4):
        delay(1 * ms)
        self.urukul1_ch2.set(frequency=f4, amplitude=amp4)
        delay(1 * us)
        self.urukul1_ch2.sw.on()
        self.ttl14.off()
        self.ttl6.off()
        delay(10 * ms)

        buf = [0] * 8
        target = ref4
        amp = amp4
        for _ in range(it4):
            self.sampler0.sample_mu(buf)
            delay(1 * ms)
            meas = buf[4] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < tol4):
                break
            amp = max(0.0, min(0.5, amp + p4 * (target - meas)))
            self.urukul1_ch2.set(frequency=f4, amplitude=amp)
            delay(1 * ms)
        return amp
