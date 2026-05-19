from artiq.experiment import *
from artiq.language.core import rpc
from repository.tests.ndsp_tests.thorlabs.with_kasli.Node3_K10CR1.k10cr1_functions import *

# from subroutines.k10cr1_functions import move_by_deg


class HS_20251217_rotator_img_ratio(EnvExperiment):

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

        self.setattr_argument("HWP_angle", NumberValue(250, min=0, max=8000, ndecimals=0, step=1),group="HWP angle")
        self.setattr_argument("QWP_angle", NumberValue(30, min=0, max=8000, ndecimals=0, step=1),group="QWP angle")
        # self.t_spcm_collection_mu

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()


        move_by_deg(self, name="852_HWP", target_deg= self.HWP_angle)
        pos = get_rotator_position(self, name='852_HWP')
        print(pos/self.deg_to_pos)

        move_by_deg(self, name="852_QWP", target_deg= self.QWP_angle)
        pos = get_rotator_position(self, name='852_QWP')
        print(pos/self.deg_to_pos)