from artiq.experiment import *
from aom_feedback_final_5channel import FeedbackRebalancer_5, REF_PATH
from artiq.language.core import rpc
from repository.tests.ndsp_tests.thorlabs.with_kasli.Node3_K10CR1.k10cr1_functions import *
from artiq.coredevice.ad9910 import (
    PHASE_MODE_ABSOLUTE,
    PHASE_MODE_CONTINUOUS,
    PHASE_MODE_TRACKING,
    RAM_DEST_ASF,
    RAM_MODE_RAMPUP
)

import os
import time
import csv
import numpy as np


class HS_20260331_Trap_freq_FrenchMethod(EnvExperiment):
    def build(self):
        self.base_dir = r"C:\Networking Experiment\artiq codes\artiq-master\repository\SPCM_CSV"
        ts = time.strftime("%Y%m%d_%H%M")
        self.csv_fname = os.path.join(self.base_dir,
                                      f"Trap_frequency_scan_{ts}.csv")
        # Devices

        self.core = self.get_device("core")
        self.setattr_device("zotino0")
        self.setattr_device("urukul0_ch0")  # Microwave_IF
        self.setattr_device("urukul0_ch1")  # Cooling AOM 1
        self.setattr_device("urukul0_ch2")  # Repump AOM
        self.setattr_device("urukul0_ch3")  # D1 first AOM
        self.setattr_device("urukul1_ch0")  # Exitation
        self.setattr_device("urukul1_ch1")  # Trap AOM @1064 nm
        self.setattr_device("urukul1_ch2")  # D1 onoff aom R
        self.setattr_device("urukul1_ch3")  # D1 onoff aom L
        self.setattr_device("urukul2_ch3")  # Pumping Repumper
        self.setattr_device("urukul2_ch0")
        self.setattr_device("urukul2_ch1")
        self.setattr_device("urukul2_ch2")
        self.sampler0 = self.get_device("sampler0")
        self.setattr_device("ttl0")
        self.setattr_device("ttl1")
        self.setattr_device("ttl4")  # uv lamp
        self.setattr_device("ttl5")  # MOT Repump
        self.setattr_device("ttl7")  # Pump repump switch
        self.setattr_device("ttl12")
        self.setattr_device("ttl14") # Switch for excitation & D1 pump
        self.setattr_device("ttl6")  # Excitation 1st aom currently novatech previously urukul1_ch0
        self.t1 = 1 * us  # first trigger
        self.t2 = 4 * ms  # RF off
        self.t3 = 150 * ms  # RF on + second trigger
        self.exp = 33 * ms  # EMCCD exposure
        self.trig_width = 5 * ms  # Zotino high time
        self.period = 1 * s  # full cycle period
        self.MOT2PGC = 500 * ms
        self.PGC_2_off = 300 * ms
        self.PGC_Af_Img = 10 * ms
        self.wait = 1 * ms
        self.nominal_trapdepth = 0.24

        # Shims
        self.Z1_idle = 4.25  # volts
        self.Z2_idle = 4.2  # volts
        self.X_idle = 0.25
        self.Y_idle = 0.1

        # --- PGC shim values ---
        self.Z1_PGC = 4.25  # volts
        self.Z2_PGC = 4.2  # volts
        self.X_PGC = 0.2
        self.Y_PGC = 0.2

        # ---- off shim values ---
        self.Z1_off = -0.05
        self.Z2_off = 0.05
        self.X_off = 0.0
        self.Y_off = 0.7


        # Rebalancer params
        self.p_gain_0 = 0.08
        self.p_gain_1 = 0.08
        self.p_gain_2 = 0.08
        self.p_gain_3 = 0.08
        self.p_gain_4 = 0.08
        self.tol_0 = 0.01
        self.tol_1 = 0.01
        self.tol_2 = 0.01
        self.tol_3 = 0.001
        self.tol_4 = 0.001
        self.iterations_0 = 20
        self.iterations_1 = 20
        self.iterations_2 = 20
        self.iterations_3 = 60
        self.iterations_4 = 20
        self.initial_amp_0 = 0.13
        self.initial_amp_1 = 0.11
        self.initial_amp_2 = 0.08
        self.initial_amp_3 = 0.093
        self.initial_amp_4 = 0.18
        self.frequency_0 = 80 * MHz
        self.frequency_1 = 80 * MHz
        self.frequency_2 = 80 * MHz
        self.frequency_3 = 150 * MHz
        self.frequency_4 = 145 * MHz
        self.hist_thres = 12       # threshold for atom vs no-atom separation
        self.t_trap_off_mu = 400
        self.length_offset_mu = 12

        # AOM set for rebalancer (we already setattr_device'd in build)
        # self.aoms = [self.urukul2_ch0, self.urukul2_ch1, self.urukul2_ch2]

        # Helper
        self.rebalancer = FeedbackRebalancer_5()
        self.rebalancer.core = self.core
        self.rebalancer.sampler0 = self.sampler0
        self.rebalancer.urukul2_ch0 = self.urukul2_ch0
        self.rebalancer.urukul2_ch1 = self.urukul2_ch1
        self.rebalancer.urukul2_ch2 = self.urukul2_ch2
        self.rebalancer.urukul1_ch1 = self.urukul1_ch1
        self.rebalancer.urukul1_ch2 = self.urukul1_ch2
        self.rebalancer.ttl14 = self.ttl14
        self.rebalancer.ttl6 = self.ttl6
        # UI args
        self.setattr_argument("num_runs", NumberValue(100, min=1, max=600000, step=1))

        # self.setattr_argument("max_excitation_cycles", NumberValue(0.0, min=0.0, max=8000.0), group="Excitation")
        # self.setattr_argument("trap_ramp_start_amp", NumberValue(0.24, min=0.000, max=0.300, ndecimals=3), group="Adiabatic parameter")
        # self.setattr_argument("trap_ramp_steps", NumberValue(200, min=0, max=100000, ndecimals=1), group="Adiabatic parameter")
        # self.setattr_argument("trap_ramp_dt_us", NumberValue(50, min=0.001, max=1000, ndecimals=3), group="Adiabatic parameter")


        self.setattr_argument("Time_scan_min", NumberValue(0.0, min=0.0, max=1000.0))
        self.setattr_argument("Time_scan_max", NumberValue(0.0, min=0.0, max=1000.0))
        self.setattr_argument("Time_scan_step", NumberValue(0.0, min=0.0, max=1000.0))

    @rpc(flags={"async"})
    def write_counts(self, c0, c1, time):
        """
        Host-side RPC: append one row of [shot0, shot1, shot2, background]
        to the timestamped CSV created in build().
        """

        fname = self.csv_fname
        is_new = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["shot1", "shot2", "delay"])
            writer.writerow([c0, c1, time])

    def prepare(self):
        # Timing & controls
        # self.t2        = 50 * ms


        # Reference file
        if REF_PATH.exists():
            vals = [float(x) for x in REF_PATH.read_text().split()]
            self.stored_refs = (vals + [-1, -1, -1, -1, -1])[:5]
        else:
            self.stored_refs = [-1, -1, -1, -1, -1]

        # Datasets for plotting
        # self.set_dataset("x_axis", [], broadcast=True)
        self.set_dataset("spcm_counts1", [], broadcast=True)
        self.set_dataset("spcm_counts2", [], broadcast=True)

        v1, s1 = self.Time_scan_min, self.Time_scan_step
        n1 = int(round((self.Time_scan_max - v1) / s1)) + 1
        self.time_scan = [v1 + i * s1 for i in range(n1)]

    @kernel
    def ramp_trap_amp(self, start_amp, end_amp, ramp_steps, ramp_dt_us):
        """
        Adiabatically ramp urukul1_ch1 amplitude from start_amp to end_amp.
        """
        n_steps = int(ramp_steps)
        dt = ramp_dt_us

        if n_steps <= 0:
            return

        step = (end_amp - start_amp) / ramp_steps

        amp = start_amp
        for i in range(n_steps + 1):
            if amp < 0.0: amp = 0.0
            if amp > 0.3: amp = 0.3

            amp_q = int(amp * 1000.0) / 1000.0
            self.urukul1_ch1.set(frequency=150 * MHz,amplitude=amp_q)

            delay(dt * us)
            amp += step




    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        self.zotino0.init()  # initialise once
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
        self.ttl0.input()
        self.ttl1.input()
        self.ttl5.output()
        self.ttl5.on()
        self.ttl7.output()
        self.ttl7.on()
        self.ttl12.output()
        self.ttl12.on()
        self.ttl14.output()
        self.ttl14.on()
        self.ttl6.output()
        self.ttl6.on()  #ensures excitation 1st aom is off for the duration of this experiment

        delay(1 * ms)

        self.urukul0_ch0.set(frequency=200 * MHz, amplitude=0.7);
        self.urukul0_ch0.sw.off()
        delay(1 * us)
        self.urukul0_ch1.set(frequency=111 * MHz, amplitude=0.45);
        self.urukul0_ch1.sw.on()
        delay(1 * us)
        self.urukul0_ch2.set(frequency=197 * MHz, amplitude=0.5);
        self.urukul0_ch2.sw.on()
        self.ttl5.off()
        delay(1 * us)
        self.urukul0_ch3.set(frequency=145 * MHz, amplitude=0.1);
        self.urukul0_ch3.sw.off()
        delay(1 * us)
        self.urukul1_ch0.set(frequency=220 * MHz, amplitude=0.1);     #replaced with ttl6.on() above
        self.urukul1_ch0.sw.off()
        delay(1 * us)
        self.urukul1_ch1.set(frequency=150 * MHz, amplitude=self.nominal_trapdepth);
        self.urukul1_ch1.sw.on()  # trap aom here
        delay(1 * us)
        self.urukul1_ch2.set(frequency=154 * MHz, amplitude=0.12);
        self.urukul1_ch2.sw.off()  # trap aom here
        delay(1 * us)
        self.urukul1_ch3.set(frequency=250 * MHz, amplitude=0.43);
        self.urukul1_ch3.sw.on()  # trap aom here
        delay(1 * us)
        self.urukul2_ch0.set(frequency=80 * MHz, amplitude=0.12);
        self.urukul2_ch0.sw.on()  # set these to optimal power balanced condition
        delay(1 * us)
        self.urukul2_ch1.set(frequency=80 * MHz, amplitude=0.10);
        self.urukul2_ch1.sw.on()
        delay(1 * us)
        self.urukul2_ch2.set(frequency=80 * MHz, amplitude=0.08);
        self.urukul2_ch2.sw.on()
        delay(1 * us)
        self.urukul2_ch3.set(frequency=202 * MHz, amplitude=0.3);
        self.urukul2_ch3.sw.off()  # trap aom here
        delay(1 * us)

        self.zotino0.write_dac(0, 0.0)  # ShimcoilX back to MOT
        self.zotino0.write_dac(1, 0.0)  # ShimcoilY back to MOT
        self.zotino0.write_dac(2, 0.0)  # ShimcoilZ1 back to MOT
        self.zotino0.write_dac(3, 0.0)  # ShimcoilZ2 back to MOT
        self.zotino0.write_dac(6, 0.0)
        self.zotino0.load()
        delay(20 * ms)
        self.ttl4.on()
        delay(1 * us)
        self.ttl5.on() # Microwave off when ttl on
        delay(1 * us)

        delay(1000 * us)
        self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3);
        self.urukul0_ch1.sw.on()
        delay(100 * us)

        delay(100 * us)
        delay(1 * us)

        final_timestamp1 = 0.0
        final_timestamp2 = 0.0
        measurements = 0
        trials = 0
        count1_bg = 0
        count1_1 = 0
        count1_2 = 0
        count2_bg = 0
        count2_1 = 0
        count2_2 = 0
        # trap_ramp_end_amp = self.trap_ramp_start_amp
        amp2 = self.initial_amp_2

        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            for param in self.time_scan:
                self.urukul0_ch2.sw.off()
                delay(1 * us)
                self.ttl5.on()
                delay(1000 * us)
                self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3);
                self.urukul0_ch1.sw.on()
                delay(100 * us)
                self.urukul1_ch1.sw.off()
                delay(1 * us)
                self.ttl6.off()
                delay(1 * us)
                self.ttl14.off()

                delay(100 * us)

                amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance(
                    self.p_gain_0, self.p_gain_1, self.p_gain_2, self.p_gain_3, self.p_gain_4,
                    self.tol_0, self.tol_1, self.tol_2, self.tol_3, self.tol_4,
                    int(self.iterations_0), int(self.iterations_1), int(self.iterations_2), int(self.iterations_3),
                    int(self.iterations_4),
                    self.stored_refs[0], self.stored_refs[1], self.stored_refs[2], self.stored_refs[3],
                    self.stored_refs[4],
                    self.initial_amp_0, self.initial_amp_1, self.initial_amp_2, self.initial_amp_3, self.initial_amp_4,
                    self.frequency_0, self.frequency_1, self.frequency_2, self.frequency_3, self.frequency_4
                )

                trap_ramp_end_amp = amp3

                delay(1 * us)
                self.ttl14.on()
                delay(1 * us)
                self.urukul1_ch2.sw.off()
                delay(1 * us)
                self.ttl6.on()
                delay(1 * us)

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                self.urukul0_ch1.sw.on()
                delay(100 * us)

                self.urukul0_ch2.sw.on()
                delay(1 * us)
                self.ttl5.off()
                delay(1 * us)

                self.urukul1_ch1.set(frequency=150 * MHz, amplitude=self.nominal_trapdepth);
                self.urukul1_ch1.sw.off()
                delay(100 * us)

                with parallel:
                    tend1_bg = self.ttl0.gate_rising(self.t2)
                    tend2_bg = self.ttl1.gate_rising(self.t2)
                count1_bg = self.ttl0.count(tend1_bg)
                count2_bg = self.ttl1.count(tend2_bg)
                delay(100 * us)

                self.ttl4.on()
                delay(1 * us)
                self.urukul1_ch1.sw.on()

                delay(self.wait)

                # Shim Idle
                self.zotino0.write_dac(0, self.X_idle)
                self.zotino0.write_dac(1, self.Y_idle)
                self.zotino0.write_dac(2, self.Z1_idle)
                self.zotino0.write_dac(3, self.Z2_idle)
                self.zotino0.load()
                delay(20 * ms)
                self.urukul0_ch1.set(frequency=112 * MHz, amplitude=0.3);
                self.urukul0_ch1.sw.on()
                delay(1 * us)

                delay(self.MOT2PGC)

                # Shim PGC
                self.zotino0.write_dac(0, self.X_PGC)
                self.zotino0.write_dac(1, self.Y_PGC)
                self.zotino0.write_dac(2, self.Z1_PGC)
                self.zotino0.write_dac(3, self.Z2_PGC)
                self.zotino0.load()
                delay(1 * us)
                self.ttl4.off()
                delay(1 * us)

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                self.urukul0_ch1.sw.on()
                delay(1 * us)

                delay(self.PGC_2_off)

                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()
                delay(1 * ms)

                # Shim Off
                self.zotino0.write_dac(0, self.X_off)
                self.zotino0.write_dac(1, self.Y_off)
                self.zotino0.write_dac(2, self.Z1_off)
                self.zotino0.write_dac(3, self.Z2_off)
                self.zotino0.load()
                delay(300 * ms)

                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()

                delay(self.PGC_Af_Img)

                self.urukul0_ch1.sw.off()
                delay(1 * us)
                self.urukul0_ch2.sw.off()
                delay(1 * us)
                self.ttl5.on()
                delay(1 * us)

                delay(self.t3)

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                self.urukul0_ch1.sw.on()
                delay(1 * us)
                self.urukul0_ch2.sw.on()
                delay(1 * us)
                self.ttl5.off()
                delay(1 * us)

                with parallel:
                    tend1_1 = self.ttl0.gate_rising(self.t2)
                    tend2_1 = self.ttl1.gate_rising(self.t2)
                count1_1 = self.ttl0.count(tend1_1)
                count2_1 = self.ttl1.count(tend2_1)
                delay(100 * us)

                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()

                delay(self.PGC_Af_Img)
                self.urukul0_ch1.sw.off()
                delay(1 * us)
                self.urukul0_ch2.sw.off()
                delay(1 * us)
                self.ttl5.on()
                delay(1 * us)
                ##PGC after imaging##


                delay(1 * us)

                # dummy
                # self.urukul2_ch0.sw.on()
                # delay(1 * us)
                # self.urukul2_ch1.sw.on()
                # delay(1 * us)
                # self.urukul2_ch2.sw.on()
                # delay(1 * us)
                # self.urukul1_ch3.sw.on()
                # delay(1 * us)
                # dummy

                # before or after
                # self.zotino0.write_dac(0, self.X_off)
                # self.zotino0.write_dac(1, self.Y_off)
                # self.zotino0.write_dac(2, self.Z1_off)
                # self.zotino0.write_dac(3, self.Z2_off)
                # self.zotino0.load()
                # delay(1 * ms)
                # self.ramp_trap_amp(trap_ramp_end_amp, self.trap_ramp_start_amp, self.trap_ramp_steps,
                #                    self.trap_ramp_dt_us)
                # delay(1 * ms)
                # before or after

                ##French Method##
                self.urukul1_ch1.sw.off()
                delay(10 * us)
                self.urukul1_ch1.sw.on()
                delay(int(param) * us)
                self.urukul1_ch1.sw.off()
                delay(60 * us)
                self.urukul1_ch1.sw.on()
                ##French Method ends##

                # Recooling PGC
                delay(1 * us)
                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()
                delay(1 * us)
                self.urukul0_ch2.sw.on()
                delay(1 * us)
                self.ttl5.off()
                delay(1 * us)
                delay(self.PGC_Af_Img)

                # Check Image
                delay(1 * us)
                self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                self.urukul0_ch1.sw.on()
                delay(6 * us)

                with parallel:
                    t_end0 = self.ttl0.gate_rising(self.t2)
                    t_end1 = self.ttl1.gate_rising(self.t2)
                delay(1 * us)
                # delay(600 * us)
                count1_2 = self.ttl0.count(t_end0)
                delay(1 * ms)
                count2_2 = self.ttl1.count(t_end1)
                delay(1 * ms)
                shot2 = count1_2 + count2_2
                delay(1 * ms)
                shot1 = count1_1 + count2_1

                self.write_counts(int(shot1), int(shot2), int(param))




                # self.urukul1_ch1.sw.off()
                # delay(500 * ms)
