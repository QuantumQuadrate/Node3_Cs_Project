from artiq.experiment import *
from artiq.language.core import rpc
from repository.tests.ndsp_tests.thorlabs.with_kasli.Node3_K10CR1.k10cr1_functions import *

# from subroutines.k10cr1_functions import move_by_deg


class HS_20251217_get_current_pos(EnvExperiment):

    def build(self):
        self.setattr_device("core")
        self.setattr_device("led0")

        self.setattr_device("k10cr1_ndsp")
        # try:
        #     self.setattr_device("k10cr1_ndsp")
        # except Exception as e:
        #     print(f"Error connecting to device {e}")

        self.rotators_are_moving = True
        self.deg_to_pos=136533
        # self.deg_to_pos=int(self.deg_to_pos)

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()

        move_to_target_deg(self, name="852_HWP", target_deg = 15)
        move_to_target_deg(self, name="852_QWP", target_deg = 30)

        pos = get_rotator_position(self, name='852_HWP')
        print(pos/self.deg_to_pos)

        pos = get_rotator_position(self, name='852_QWP')
        print(pos/self.deg_to_pos)
