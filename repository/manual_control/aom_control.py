"""Manual AOM frequency and amplitude control for all 12 DDS channels.

Run from the dashboard to manually set any DDS channel's frequency
and amplitude. Uses human-readable channel names.

This is a standalone tool — it does NOT read from ExperimentVariables
or datasets. Values come entirely from dashboard arguments.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *


class AOMControl(EnvExperiment):
    """AOM_Control."""

    def build(self):
        self.setattr_device("core")

        # DDS devices
        self.setattr_device("urukul0_ch0")
        self.setattr_device("urukul0_ch1")
        self.setattr_device("urukul0_ch2")
        self.setattr_device("urukul0_ch3")
        self.setattr_device("urukul1_ch0")
        self.setattr_device("urukul1_ch1")
        self.setattr_device("urukul1_ch2")
        self.setattr_device("urukul1_ch3")
        self.setattr_device("urukul2_ch0")
        self.setattr_device("urukul2_ch1")
        self.setattr_device("urukul2_ch2")
        self.setattr_device("urukul2_ch3")

        # Each DDS channel gets a frequency, amplitude, and on/off argument
        channels = [
            ("MW IF (urukul0_ch0)", 250e6, 0.7),
            ("Cooling DP (urukul0_ch1)", 120e6, 0.36),
            ("Repump (urukul0_ch2)", 197e6, 0.5),
            ("D1 Pump (urukul0_ch3)", 145e6, 0.3),
            ("RF Antenna (urukul1_ch0)", 2e6, 0.05),
            ("FORT 1064nm (urukul1_ch1)", 150e6, 0.21),
            ("Excitation (urukul1_ch2)", 145e6, 0.1),
            ("D1 onoff L (urukul1_ch3)", 250e6, 0.43),
            ("MOT X (urukul2_ch0)", 80e6, 0.15),
            ("MOT Y (urukul2_ch1)", 80e6, 0.11),
            ("MOT Z (urukul2_ch2)", 80e6, 0.09),
            ("Pump Repumper (urukul2_ch3)", 202e6, 0.3),
        ]

        for i, (name, default_f, default_a) in enumerate(channels):
            self.setattr_argument(f"freq_{i}",
                                  NumberValue(default_f, unit="MHz", min=0, max=500e6,
                                              step=1e6, ndecimals=3),
                                  group=name)
            self.setattr_argument(f"amp_{i}",
                                  NumberValue(default_a, min=0.0, max=1.0,
                                              step=0.01, ndecimals=3),
                                  group=name)
            self.setattr_argument(f"on_{i}",
                                  BooleanValue(default=(i not in [0, 3, 4, 6, 11])),
                                  group=name)

    def prepare(self):
        pass

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()

        # Init CPLDs
        self.urukul0_ch0.cpld.init()
        self.urukul1_ch0.cpld.init()
        self.urukul2_ch0.cpld.init()

        # Init all 12 DDS channels
        self.urukul0_ch0.init()
        self.urukul0_ch1.init()
        self.urukul0_ch2.init()
        self.urukul0_ch3.init()
        self.urukul1_ch0.init()
        self.urukul1_ch1.init()
        self.urukul1_ch2.init()
        self.urukul1_ch3.init()
        self.urukul2_ch0.init()
        self.urukul2_ch1.init()
        self.urukul2_ch2.init()
        self.urukul2_ch3.init()
        self.core.break_realtime()

        # Set each channel from dashboard arguments
        # MW IF (urukul0_ch0)
        self.urukul0_ch0.set(frequency=self.freq_0, amplitude=self.amp_0)
        if self.on_0: self.urukul0_ch0.sw.on()
        else: self.urukul0_ch0.sw.off()
        delay(1*us)
        # Cooling DP (urukul0_ch1)
        self.urukul0_ch1.set(frequency=self.freq_1, amplitude=self.amp_1)
        if self.on_1: self.urukul0_ch1.sw.on()
        else: self.urukul0_ch1.sw.off()
        delay(1*us)
        # Repump (urukul0_ch2)
        self.urukul0_ch2.set(frequency=self.freq_2, amplitude=self.amp_2)
        if self.on_2: self.urukul0_ch2.sw.on()
        else: self.urukul0_ch2.sw.off()
        delay(1*us)
        # D1 Pump (urukul0_ch3)
        self.urukul0_ch3.set(frequency=self.freq_3, amplitude=self.amp_3)
        if self.on_3: self.urukul0_ch3.sw.on()
        else: self.urukul0_ch3.sw.off()
        delay(1*us)
        # RF Antenna (urukul1_ch0)
        self.urukul1_ch0.set(frequency=self.freq_4, amplitude=self.amp_4)
        if self.on_4: self.urukul1_ch0.sw.on()
        else: self.urukul1_ch0.sw.off()
        delay(1*us)
        # FORT 1064nm (urukul1_ch1)
        self.urukul1_ch1.set(frequency=self.freq_5, amplitude=self.amp_5)
        if self.on_5: self.urukul1_ch1.sw.on()
        else: self.urukul1_ch1.sw.off()
        delay(1*us)
        # Excitation (urukul1_ch2)
        self.urukul1_ch2.set(frequency=self.freq_6, amplitude=self.amp_6)
        if self.on_6: self.urukul1_ch2.sw.on()
        else: self.urukul1_ch2.sw.off()
        delay(1*us)
        # D1 onoff L (urukul1_ch3)
        self.urukul1_ch3.set(frequency=self.freq_7, amplitude=self.amp_7)
        if self.on_7: self.urukul1_ch3.sw.on()
        else: self.urukul1_ch3.sw.off()
        delay(1*us)
        # MOT X (urukul2_ch0)
        self.urukul2_ch0.set(frequency=self.freq_8, amplitude=self.amp_8)
        if self.on_8: self.urukul2_ch0.sw.on()
        else: self.urukul2_ch0.sw.off()
        delay(1*us)
        # MOT Y (urukul2_ch1)
        self.urukul2_ch1.set(frequency=self.freq_9, amplitude=self.amp_9)
        if self.on_9: self.urukul2_ch1.sw.on()
        else: self.urukul2_ch1.sw.off()
        delay(1*us)
        # MOT Z (urukul2_ch2)
        self.urukul2_ch2.set(frequency=self.freq_10, amplitude=self.amp_10)
        if self.on_10: self.urukul2_ch2.sw.on()
        else: self.urukul2_ch2.sw.off()
        delay(1*us)
        # Pump Repumper (urukul2_ch3)
        self.urukul2_ch3.set(frequency=self.freq_11, amplitude=self.amp_11)
        if self.on_11: self.urukul2_ch3.sw.on()
        else: self.urukul2_ch3.sw.off()
        delay(1*us)
