"""Manual shim coil and TTL switch control.

Run from the dashboard to set shim coil voltages and toggle TTL switches.
Provides preset shim configurations and individual device on/off control.

Toggle convention: ON = the device/beam is ON (passing).
The code handles TTL polarity internally (active-low switches).

This is a standalone tool — does NOT use BaseExperiment or ExperimentVariables.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *


class ShimCoilControl(EnvExperiment):
    """Shim Coil and Voltage Control"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("zotino0")
        self.setattr_device("ttl4")   # UV lamp
        self.setattr_device("ttl5")   # Repump switch (active-low: ON=blocked)
        self.setattr_device("ttl6")   # Excitation atten1 (active-low)
        self.setattr_device("ttl7")   # Microwave switch (active-low)
        self.setattr_device("ttl12")  # Excitation atten2 (active-low)
        self.setattr_device("ttl14")  # D1/excitation switch (active-low)

        # --- Shim coil voltages ---
        self.setattr_argument("shim_X",
                              NumberValue(0.0, min=-10.0, max=10.0, step=0.01, ndecimals=3, unit="V"),
                              group="Shim Coils")
        self.setattr_argument("shim_Y",
                              NumberValue(0.0, min=-10.0, max=10.0, step=0.01, ndecimals=3, unit="V"),
                              group="Shim Coils")
        self.setattr_argument("shim_Z1",
                              NumberValue(0.0, min=-10.0, max=10.0, step=0.01, ndecimals=3, unit="V"),
                              group="Shim Coils")
        self.setattr_argument("shim_Z2",
                              NumberValue(0.0, min=-10.0, max=10.0, step=0.01, ndecimals=3, unit="V"),
                              group="Shim Coils")

        # --- TTL device toggles (ON = device/beam is ON) ---
        self.setattr_argument("UV_lamp_on",
                              BooleanValue(default=False),
                              group="TTL Devices (ON = device active)")
        self.setattr_argument("MOT_repump_on",
                              BooleanValue(default=False),
                              group="TTL Devices (ON = device active)")
        self.setattr_argument("microwave_on",
                              BooleanValue(default=False),
                              group="TTL Devices (ON = device active)")
        self.setattr_argument("D1_exc_switch_on",
                              BooleanValue(default=False),
                              group="TTL Devices (ON = device active)")
        self.setattr_argument("excitation_atten1_on",
                              BooleanValue(default=False),
                              group="TTL Devices (ON = device active)")
        self.setattr_argument("excitation_atten2_on",
                              BooleanValue(default=False),
                              group="TTL Devices (ON = device active)")

    def prepare(self):
        pass

    @kernel
    def run(self):
        self.core.reset()
        self.zotino0.init()
        self.core.break_realtime()

        # Set shim coil voltages
        self.zotino0.write_dac(0, self.shim_X)
        self.zotino0.write_dac(1, self.shim_Y)
        self.zotino0.write_dac(2, self.shim_Z1)
        self.zotino0.write_dac(3, self.shim_Z2)
        self.zotino0.load()
        delay(20 * ms)

        # UV lamp (ttl4) — active-high: ON = ttl.on()
        if self.UV_lamp_on:
            self.ttl4.on()
        else:
            self.ttl4.off()
        delay(1 * us)

        # MOT repump (ttl5) — active-low: device ON = ttl.off()
        if self.MOT_repump_on:
            self.ttl5.off()
        else:
            self.ttl5.on()
        delay(1 * us)

        # Microwave (ttl7) — active-low: device ON = ttl.off()
        if self.microwave_on:
            self.ttl7.off()
        else:
            self.ttl7.on()
        delay(1 * us)

        # D1/excitation switch (ttl14) — active-low: device ON = ttl.off()
        if self.D1_exc_switch_on:
            self.ttl14.off()
        else:
            self.ttl14.on()
        delay(1 * us)

        # Excitation attenuator 1 (ttl6) — active-low: device ON = ttl.off()
        if self.excitation_atten1_on:
            self.ttl6.off()
        else:
            self.ttl6.on()
        delay(1 * us)

        # Excitation attenuator 2 (ttl12) — active-low: device ON = ttl.off()
        if self.excitation_atten2_on:
            self.ttl12.off()
        else:
            self.ttl12.on()
        delay(1 * us)
