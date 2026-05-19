from artiq.experiment import *

class ShimCoilsAndTTLControl(EnvExperiment):

    def build(self):
        self.setattr_device("core")
        self.setattr_device("zotino0")
        self.setattr_device("ttl4")  # UV lamp
        self.setattr_device("ttl5")  # Camera
        self.setattr_device("ttl14") # D1 exc switch
        self.setattr_device("ttl5")  # Repump switch
        self.setattr_device("ttl7")  # Pump repump switch
        self.setattr_device("ttl6")  # Excitation switch

        # Arguments for the Shim Coils
        self.setattr_argument(
            "ShimCoilX_volt",
            NumberValue(default=0.0, unit="V", step=0.1, min=-8, max=8),
            group="ShimCoilX"
        )
        self.setattr_argument(
            "ShimCoilY_volt",
            NumberValue(default=0.0, unit="V", step=0.1, min=-8, max=8),
            group="ShimCoilY"
        )
        self.setattr_argument(
            "ShimCoilZ1_volt",
            NumberValue(default=0.0, unit="V", step=0.1, min=-8, max=8),
            group="ShimCoilZ1"
        )
        self.setattr_argument(
            "ShimCoilZ2_volt",
            NumberValue(default=0.0, unit="V", step=0.1, min=-8, max=8),
            group="ShimCoilZ2"
        )

        # Arguments for TTL control (UVLamp, Camera)
        self.setattr_argument(
            "UVLamp_on",
            BooleanValue(default=False),
            group="UVLamp (TTL4)"
        )
        self.setattr_argument(
            "Camera_on",
            BooleanValue(default=False),
            group="Camera (TTL5)"
        )
        self.setattr_argument(
            "D1_exc_SW_on",
            BooleanValue(default=False),
            group="Exc_SW(TTL14)"
        )

        self.setattr_argument(
            "MOT_Repump_on",
            BooleanValue(default=False),
            group="Exc_SW(TTL5)"
        )

        self.setattr_argument(
            "Microwave_switch_on",
            BooleanValue(default=False),
            group="Exc_SW(TTL7)"
        )

        self.setattr_argument(
            "Excitation_On",
            BooleanValue(default=False),
            group="Exc_SW(TTL6)"
        )

        self.setattr_argument(
            "Cameratester_volt",
            NumberValue(default=0.0, unit="V", step=0.1, min=0, max=4.0),
            group="Cameratester"
        )

    @kernel
    def run(self):
        self.core.reset()
        self.zotino0.init()
        self.core.break_realtime()

        self.zotino0.write_dac(0, self.ShimCoilX_volt)   # ShimCoilX
        self.zotino0.write_dac(1, self.ShimCoilY_volt)   # ShimCoilY
        self.zotino0.write_dac(2, self.ShimCoilZ1_volt)  # ShimCoilZ1
        self.zotino0.write_dac(3, self.ShimCoilZ2_volt)  # ShimCoilZ2
        # self.zotino0.write_dac(5, self.Cameratester_volt)  # Camera tester
        self.zotino0.load()

        if self.UVLamp_on:
            self.ttl4.on()
        else:
            self.ttl4.off()

        if self.Camera_on:
            self.ttl5.on()
        else:
            self.ttl5.off()

        if self.D1_exc_SW_on:
            self.ttl14.off()
        else:
            self.ttl14.on()

        if self.MOT_Repump_on:
            self.ttl5.off()
        else:
            self.ttl5.on()

        if self.Microwave_switch_on:
            self.ttl7.off()
        else:
            self.ttl7.on()

        if self.Excitation_On:
            self.ttl6.off()
        else:
            self.ttl6.on()

        print("  ShimCoilX=", self.ShimCoilX_volt, "V")
        print("  ShimCoilY=", self.ShimCoilY_volt, "V")
        print("  ShimCoilZ1=", self.ShimCoilZ1_volt, "V")
        print("  ShimCoilZ2=", self.ShimCoilZ2_volt, "V")
        print("  UVLamp (ttl4) is", "ON" if self.UVLamp_on else "OFF")
        print("  Camera (ttl5) is", "ON" if self.Camera_on else "OFF")
