from artiq.experiment import *

class AOMurukulcontrolpy(EnvExperiment):
    def build(self):
        self.setattr_device("core")
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

        # Urukul 0 Channel 0
        self.setattr_argument("AOM1_on", BooleanValue(False), group="AOM1 On/Off")
        self.setattr_argument("AOM1_freq", NumberValue(default=90e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM1 Frequency")
        self.setattr_argument("AOM1_amp",  NumberValue(default=0.5, min=0.0, max=1.0), group="AOM1 Amplitude")

        # Urukul 0 Channel 1
        self.setattr_argument("AOM2_on", BooleanValue(False), group="AOM2 On/Off")
        self.setattr_argument("AOM2_freq", NumberValue(default=90e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM2 Frequency")
        self.setattr_argument("AOM2_amp",  NumberValue(default=0.5, min=0.0, max=1.0), group="AOM2 Amplitude")

        # Urukul 0 Channel 2
        self.setattr_argument("AOM3_on", BooleanValue(False), group="AOM3 On/Off")
        self.setattr_argument("AOM3_freq", NumberValue(default=100e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM3 Frequency")
        self.setattr_argument("AOM3_amp",  NumberValue(default=0.5, min=0.0, max=1.0), group="AOM3 Amplitude")

        # Urukul 0 Channel 3
        self.setattr_argument("AOM4_on", BooleanValue(False), group="AOM4 On/Off")
        self.setattr_argument("AOM4_freq", NumberValue(default=110e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM4 Frequency")
        self.setattr_argument("AOM4_amp",  NumberValue(default=0.5, min=0.0, max=1.0), group="AOM4 Amplitude")

        # Urukul 1 Channel 0
        self.setattr_argument("AOM5_on", BooleanValue(False), group="AOM5 On/Off")
        self.setattr_argument("AOM5_freq", NumberValue(default=90e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM5 Frequency")
        self.setattr_argument("AOM5_amp", NumberValue(default=0.5, min=0.0, max=1.0),  group="AOM5 Amplitude")

        # Urukul 1 Channel 1
        self.setattr_argument("AOM6_on", BooleanValue(False), group="AOM6 On/Off")
        self.setattr_argument("AOM6_freq", NumberValue(default=100e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM6 Frequency")
        self.setattr_argument("AOM6_amp", NumberValue(default=0.5, min=0.0, max=1.0), group="AOM6 Amplitude")

        # Urukul 1 Channel 2
        self.setattr_argument("AOM7_on", BooleanValue(False), group="AOM7 On/Off")
        self.setattr_argument("AOM7_freq", NumberValue(default=110e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM7 Frequency")
        self.setattr_argument("AOM7_amp", NumberValue(default=0.5, min=0.0, max=1.0), group="AOM7 Amplitude")

        # Urukul 1 Channel 3
        self.setattr_argument("AOM8_on", BooleanValue(False), group="AOM8 On/Off")
        self.setattr_argument("AOM8_freq", NumberValue(default=80e6, unit="Hz", step=1e6, min=0, max=400e6),  group="AOM8 Frequency")
        self.setattr_argument("AOM8_amp", NumberValue(default=0.5, min=0.0, max=1.0), group="AOM8 Amplitude")

        # Urukul 2 Channel 0
        self.setattr_argument("AOM9_on", BooleanValue(False), group="AOM9 On/Off")
        self.setattr_argument("AOM9_freq", NumberValue(default=90e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM9 Frequency")
        self.setattr_argument("AOM9_amp", NumberValue(default=0.5, min=0.0, max=1.0), group="AOM9 Amplitude")

        # Urukul 2 Channel 1
        self.setattr_argument("AOM10_on", BooleanValue(False), group="AOM10 On/Off")
        self.setattr_argument("AOM10_freq", NumberValue(default=100e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM10 Frequency")
        self.setattr_argument("AOM10_amp", NumberValue(default=0.5, min=0.0, max=1.0), group="AOM10 Amplitude")

        # Urukul 2 Channel 2
        self.setattr_argument("AOM11_on", BooleanValue(False), group="AOM11 On/Off")
        self.setattr_argument("AOM11_freq", NumberValue(default=110e6, unit="Hz", step=1e6, min=0, max=400e6), group="AOM11 Frequency")
        self.setattr_argument("AOM11_amp", NumberValue(default=0.5, min=0.0, max=1.0), group="AOM11 Amplitude")

        # Urukul 2 Channel 3
        self.setattr_argument("AOM12_on", BooleanValue(False), group="AOM12 On/Off")
        self.setattr_argument("AOM12_freq", NumberValue(default=80e6, unit="Hz", step=1e6, min=0, max=400e6),  group="AOM12 Frequency")
        self.setattr_argument("AOM12_amp", NumberValue(default=0.5, min=0.0, max=1.0), group="AOM12 Amplitude")

    @kernel
    def run(self):
        self.core.reset()
        delay(1 * ms)

        self.urukul0_ch0.cpld.init()
        self.urukul0_ch1.cpld.init()
        self.urukul0_ch2.cpld.init()
        self.urukul0_ch3.cpld.init()
        self.urukul1_ch0.cpld.init()
        self.urukul1_ch1.cpld.init()
        self.urukul1_ch2.cpld.init()
        self.urukul1_ch3.cpld.init()
        self.urukul2_ch0.cpld.init()
        self.urukul2_ch1.cpld.init()
        self.urukul2_ch2.cpld.init()
        self.urukul2_ch3.cpld.init()

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
        delay(1 * ms)

        if self.AOM1_on:
            self.urukul0_ch0.set(self.AOM1_freq, amplitude=self.AOM1_amp)
            self.urukul0_ch0.sw.on()
        else:
            self.urukul0_ch0.sw.off()
        delay(1 * ms)

        if self.AOM2_on:
            self.urukul0_ch1.set(self.AOM2_freq, amplitude=self.AOM2_amp)
            self.urukul0_ch1.sw.on()
        else:
            self.urukul0_ch1.sw.off()
        delay(1 * ms)

        if self.AOM3_on:
            self.urukul0_ch2.set(self.AOM3_freq, amplitude=self.AOM3_amp)
            self.urukul0_ch2.sw.on()
        else:
            self.urukul0_ch2.sw.off()
        delay(1 * ms)

        if self.AOM4_on:
            self.urukul0_ch3.set(self.AOM4_freq, amplitude=self.AOM4_amp)
            self.urukul0_ch3.sw.on()
        else:
            self.urukul0_ch3.sw.off()
        delay(1 * ms)

        if self.AOM5_on:
            self.urukul1_ch0.set(self.AOM5_freq, amplitude=self.AOM5_amp)
            self.urukul1_ch0.sw.on()
        else:
            self.urukul1_ch0.sw.off()
        delay(1 * ms)

        if self.AOM6_on:
            self.urukul1_ch1.set(self.AOM6_freq, amplitude=self.AOM6_amp)
            self.urukul1_ch1.sw.on()
        else:
            self.urukul1_ch1.sw.off()
        delay(1 * ms)

        if self.AOM7_on:
            self.urukul1_ch2.set(self.AOM7_freq, amplitude=self.AOM7_amp)
            self.urukul1_ch2.sw.on()
        else:
            self.urukul1_ch2.sw.off()
        delay(1 * ms)

        if self.AOM8_on:
            self.urukul1_ch3.set(self.AOM8_freq, amplitude=self.AOM8_amp)
            self.urukul1_ch3.sw.on()
        else:
            self.urukul1_ch3.sw.off()
        delay(1 * ms)

        if self.AOM9_on:
            self.urukul2_ch0.set(self.AOM9_freq, amplitude=self.AOM9_amp)
            self.urukul2_ch0.sw.on()
        else:
            self.urukul2_ch0.sw.off()
        delay(1 * ms)

        if self.AOM10_on:
            self.urukul2_ch1.set(self.AOM10_freq, amplitude=self.AOM10_amp)
            self.urukul2_ch1.sw.on()
        else:
            self.urukul2_ch1.sw.off()
        delay(1 * ms)

        if self.AOM11_on:
            self.urukul2_ch2.set(self.AOM11_freq, amplitude=self.AOM11_amp)
            self.urukul2_ch2.sw.on()
        else:
            self.urukul2_ch2.sw.off()
        delay(1 * ms)

        if self.AOM12_on:
            self.urukul2_ch3.set(self.AOM12_freq, amplitude=self.AOM12_amp)
            self.urukul2_ch3.sw.on()
        else:
            self.urukul2_ch3.sw.off()
        delay(1 * ms)

        print("AOMs parameters updated")
