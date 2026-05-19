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

LOOPS = 1 # number of repetitions

class HS_20260323_Z_parity_v7_Qoffblow_traprampup_PGC(EnvExperiment):
## This is slightly misnamed, parameters here now based on most recent 1.4 G Ramsey scan
    def build(self):
        self.base_dir = r"C:\Networking Experiment\artiq codes\artiq-master\repository\SPCM_CSV"
        tsD = time.strftime("%Y%m%d_%H%M")
        self.csv_fname = os.path.join(self.base_dir, f"Z_basis_parity_measurement_{tsD}.csv")
        self.core = self.get_device("core")
        self.setattr_device("zotino0")
        self.setattr_device("urukul0_ch0")  # Microwave_IF
        self.setattr_device("urukul0_ch1")  # Cooling AOM 1
        self.setattr_device("urukul0_ch2")  # Repump AOM
        self.setattr_device("urukul0_ch3")  # D1 first AOM
        self.setattr_device("urukul1_ch0")  # RF Antenna
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
        self.setattr_device("ttl7")  # Microwave switch
        self.setattr_device("ttl14") # Switch for excitation & D1 pump
        self.setattr_device("ttl12")  # Excitation 1st aom currently novatech previously urukul1_ch0 now two ttls for more attenuation
        self.setattr_device("ttl6")  # Excitation 1st aom currently novatech previously urukul1_ch0
        try:
            self.setattr_device("k10cr1_ndsp")
        except Exception as e:
            print(f"Error connecting to device {e}")

        self.rotators_are_moving = True
        self.deg_to_pos=136533

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

        # ----- time parameters ------
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



        # ---- PID params ---
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


        self.setattr_argument("n_measurements", NumberValue(10.0, min=0.0, max=600000.0), group="Excitation")
        self.setattr_argument("max_excitation_cycles", NumberValue(1.0, min=0.0, max=8000.0), group="Excitation")
        self.setattr_argument("t_trap_off_mu", NumberValue(30, min=0, max=8000, ndecimals=0, step=1), group="Excitation")
        self.setattr_argument("t_spcm_collection_mu", NumberValue(250, min=0, max=8000, ndecimals=0, step=1),group="Excitation")
        self.setattr_argument("length_offset_mu", NumberValue(30, min=0, max=8000, ndecimals=0, step=1),group="Excitation")

        self.setattr_argument("trap_ramp_start_amp", NumberValue(0.125, min=0.000, max=0.300, ndecimals=3), group="Adiabatic parameter")
        # self.setattr_argument("trap_ramp_end_amp", NumberValue(0.125, min=0.000, max=0.300, ndecimals=3), group="Adiabatic parameter")
        self.setattr_argument("trap_ramp_steps",     NumberValue(1, min=0, max=100000, ndecimals=1), group="Adiabatic parameter")
        self.setattr_argument("trap_ramp_dt_us",        NumberValue(1, min=0.001, max=1000, ndecimals=3), group="Adiabatic parameter")


        # add change it to HWP angle #
        self.setattr_argument("HWP_Angle_min", NumberValue(0.0, min=0.0, max=360.0), group="Angle Scan")
        self.setattr_argument("HWP_Angle_max", NumberValue(360.0, min=0.0, max=3600.0), group="Angle Scan")
        self.setattr_argument("HWP_Angle_step", NumberValue(30.0, min=0.0, max=360.0), group="Angle Scan")
        # add change it to HWP angle #

        # add change it to QWP angle #
        self.setattr_argument("QWP_Angle_min", NumberValue(0.0, min=0.0, max=360.0), group="Angle Scan")
        self.setattr_argument("QWP_Angle_max", NumberValue(360.0, min=0.0, max=360.0), group="Angle Scan")
        self.setattr_argument("QWP_Angle_step", NumberValue(30.0, min=0.0, max=360.0), group="Angle Scan")
        # add change it to QWP angle #


        # add change it to QWP angle #
    @rpc(flags={"async"})
    def write_counts(self, SPCM1_photon, SPCM2_photon, c0, c1, c2, hwp, qwp):
        """Append one row to the CSV (host-side)."""
        fname = self.csv_fname
        is_new = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["SPCM1", "SPCM2", "trials", "counts", "counts_bg", "HWP", "QWP"])
            w.writerow([SPCM1_photon, SPCM2_photon, c0, c1, c2, hwp, qwp])
        # add change it to QWP angle #

    def prepare(self):

        if REF_PATH.exists():
            vals = [float(x) for x in REF_PATH.read_text().split()]
            self.stored_refs = (vals + [-1, -1, -1, -1, -1])[:5]
        else:
            self.stored_refs = [-1, -1, -1, -1, -1]

        v1, s1 = self.HWP_Angle_min, self.HWP_Angle_step
        n1 = int(round((self.HWP_Angle_max - v1) / s1)) + 1
        self.HWP_scan = [v1 + i * s1 for i in range(n1)]

        v2, s2 = self.QWP_Angle_min, self.QWP_Angle_step
        n2 = int(round((self.QWP_Angle_max - v2) / s2)) + 1
        self.QWP_scan = [v2 + i * s2 for i in range(n2)]

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
        # ... (Init sequences omitted for brevity, they remain same as original) ...
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
        self.ttl14.output()
        self.ttl14.on()
        self.ttl6.output()
        self.ttl6.on()
        self.ttl12.output()
        self.ttl12.on()
        delay(2 * us)

        # Initial AOM Setup
        self.urukul0_ch0.set(frequency=200 * MHz, amplitude=0.55);
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
        self.urukul1_ch0.set(frequency=2 * MHz, amplitude=0.1);
        self.urukul1_ch0.sw.off()
        delay(1 * us)
        self.urukul1_ch1.set(frequency=150 * MHz, amplitude=self.nominal_trapdepth);
        self.urukul1_ch1.sw.on()
        delay(1 * us)
        self.urukul1_ch2.set(frequency=154 * MHz, amplitude=0.12);
        self.urukul1_ch2.sw.off()
        delay(1 * us)
        self.urukul1_ch3.set(frequency=250 * MHz, amplitude=0.43);
        self.urukul1_ch3.sw.on()
        delay(1 * us)
        self.urukul2_ch0.set(frequency=80 * MHz, amplitude=0.12);
        self.urukul2_ch0.sw.on()
        delay(1 * us)
        self.urukul2_ch1.set(frequency=80 * MHz, amplitude=0.10);
        self.urukul2_ch1.sw.on()
        delay(1 * us)
        self.urukul2_ch2.set(frequency=80 * MHz, amplitude=0.08);
        self.urukul2_ch2.sw.on()
        delay(1 * us)
        self.urukul2_ch3.set(frequency=202 * MHz, amplitude=0.3);
        self.urukul2_ch3.sw.off()
        delay(1 * us)
        self.urukul0_ch0.set_phase_mode(PHASE_MODE_CONTINUOUS)
        delay(1 * us)
        self.urukul1_ch0.set_phase_mode(PHASE_MODE_ABSOLUTE)
        delay(1 * us)
        self.zotino0.write_dac(0, 0.0)
        self.zotino0.write_dac(1, 0.0)
        self.zotino0.write_dac(2, 0.0)
        self.zotino0.write_dac(3, 0.0)
        self.zotino0.write_dac(6, 0.0)
        self.zotino0.load()
        delay(20 * ms)
        self.ttl4.off()
        delay(1 * us)
        self.ttl5.on()
        delay(1 * us)
        self.urukul1_ch0.sw.off()
        delay(1 * us)
        micro_freq = 250.0 - 0.492 - 0.125
        RF_freq = 0.492 + 0.125 - 0.00887 + 0.1
        delay(1000 * ms)


        qwp_first = self.QWP_scan[0]  # 0
        qwp_last = self.QWP_scan[-1]  # 330
        hwp_first = self.HWP_scan[0]
        hwp_last = self.HWP_scan[-1]


        for qwp in self.QWP_scan:
            delay(100 * ms)
            # wrap check: previous was 330 and new is 0  -> do +30 using move_by_deg
            move_to_target_deg(self, name="852_QWP", target_deg=qwp)
            delay(15000 * ms)
            self.core.break_realtime()

            for hwp in self.HWP_scan:
                delay(100 * ms)

                # THIS is the one that will trigger every time you restart HWP at 0 after ending at 330
                if (hwp == hwp_first) :
                    move_to_target_deg(self, name="852_HWP", target_deg=hwp)
                    delay(15000 * ms)
                else:
                    move_to_target_deg(self, name="852_HWP", target_deg=hwp)
                    delay(5000 * ms)
                self.core.break_realtime()

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
                trap_ramp_end_amp = self.trap_ramp_start_amp
                amp2 = self.initial_amp_2

                # --- GREEN SCOPE: MEASUREMENT LOOP ---
                while measurements < self.n_measurements:

                    # Variables for tracking
                    self.core.break_realtime()

                    # --- YELLOW SCOPE: LOADING & VERIFICATION ---
                    while True:
                        delay(1000 * us)
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
                            int(self.iterations_0), int(self.iterations_1), int(self.iterations_2), int(self.iterations_3), int(self.iterations_4),
                            self.stored_refs[0], self.stored_refs[1], self.stored_refs[2], self.stored_refs[3], self.stored_refs[4],
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

                        # --- LOGIC: Check Loading ---
                        if (count1_1 + count2_1 - count1_bg - count2_bg) > self.hist_thres:
                            break # Atom loaded, move to Orange Scope
                        else:
                            continue # Reload (Yellow Scope)


                    # --- ORANGE SCOPE: EXCITATION RETRY LOOP ---
                    photon_detected = False
                    while True:
                        # F=3 state preparation
                        self.urukul0_ch1.sw.on()
                        delay(1 * ms)
                        self.urukul0_ch1.sw.off()
                        delay(1 * us)
                        self.urukul1_ch3.sw.off()
                        delay(1 * us)


                        time_stamp1 = [-2.0] * int(self.max_excitation_cycles)
                        time_stamp2 = [-2.0] * int(self.max_excitation_cycles)
                        final_timestamp1 = 0.0
                        final_timestamp2 = 0.0

                        # Excitation Cycles
                        for excitation_cycle in range(int(self.max_excitation_cycles)):

                            # Q-axis on
                            self.zotino0.write_dac(0, -0.013)
                            self.zotino0.write_dac(1, 2.72)
                            self.zotino0.write_dac(2, -0.06)  # ShimcoilZ1 back to MOT
                            self.zotino0.write_dac(3, 0.06)  # ShimcoilZ2 back to MOT
                            self.zotino0.load()
                            delay(1 * ms)

                            # Trap adiabatic down
                            self.ramp_trap_amp(self.trap_ramp_start_amp, trap_ramp_end_amp, self.trap_ramp_steps,
                                               self.trap_ramp_dt_us)
                            delay(1 * ms)

                            self.urukul2_ch0.sw.on()
                            delay(1 * us)
                            self.urukul2_ch1.sw.on()
                            delay(1 * us)
                            self.urukul2_ch2.sw.on()
                            delay(1 * us)

                            trials += 1
                            SPCM1_click_counter = 0
                            SPCM2_click_counter = 0

                            # slack = now_mu() - self.core.get_rtio_counter_mu()
                            # if slack < 1e5:
                            #     # self.print_async("slack added in measurement:", self.measurement)
                            #     self.core.break_realtime()

                            self.urukul0_ch0.set(frequency = 250.978 * MHz, amplitude=0.7);
                            delay(6 * us)

                            # Pump begin
                            # self.urukul0_ch3.set(frequency=145 * MHz, amplitude=0.02);
                            self.urukul0_ch3.set(frequency=145 * MHz, amplitude=0.03);
                            self.urukul0_ch3.sw.on()
                            delay(6 * us)
                            self.urukul1_ch2.set(frequency=150 * MHz, amplitude=0.04);
                            self.urukul1_ch2.sw.on()
                            delay(6 * us)
                            self.urukul2_ch3.set(frequency=202 * MHz, amplitude=0.4);
                            self.urukul2_ch3.sw.on()  # trap aom here
                            delay(6 * us)
                            self.ttl14.off()

                            delay(1 * ms)

                            t_optend = now_mu()
                            at_mu(t_optend + 1000)
                            with parallel:
                                self.ttl14.on()  # Pump off
                                self.urukul0_ch3.sw.off()
                                self.urukul1_ch2.sw.off()
                                self.urukul2_ch3.sw.off()  # Pump repump off
                                self.urukul2_ch0.sw.off()
                                self.urukul2_ch1.sw.off()
                                self.urukul2_ch2.sw.off()

                            delay(3 * us)

                            # Excitation experiment
                            # self.urukul1_ch2.sw.on()
                            # delay(1 * us)


                            # self.urukul1_ch1.set(frequency=150 * MHz, amplitude=trap_ramp_end_amp);
                            self.urukul1_ch2.set(frequency=145 * MHz, amplitude=0.13);

                            delay(5 * us)

                            t0 = now_mu()

                            # packing
                            # at_mu(t0)
                            # self.ttl7.off()
                            # packing

                            at_mu(t0 + 4215)
                            self.urukul1_ch1.sw.off()

                            at_mu(t0 + 4215 + int(self.t_trap_off_mu))
                            self.urukul1_ch1.sw.on()

                            at_mu(t0 + 4028)
                            self.ttl14.off()

                            at_mu(t0 + 4028 + int(self.length_offset_mu))
                            self.ttl14.on()

                            at_mu(t0 + 4540 + 100)
                            with parallel:
                                self.ttl12.off()
                                self.ttl6.off()

                            at_mu(t0 + 4540 + 100 + 100)
                            with parallel:
                                self.ttl12.on()
                                self.ttl6.on()

                            at_mu(t0 + 3800)
                            self.urukul1_ch2.sw.on()

                            at_mu(t0 + 3800 + 300)
                            self.urukul1_ch2.sw.off()

                            at_mu(t0 + 4500 + 780)
                            with parallel:
                                t_end = self.ttl0.gate_rising(self.t_spcm_collection_mu * ns)
                                t_end_2 = self.ttl1.gate_rising(self.t_spcm_collection_mu * ns)

                            SPCM1_click_time = self.ttl0.timestamp_mu(t_end)
                            SPCM2_click_time = self.ttl1.timestamp_mu(t_end_2)

                            delay(5 * us)

                            if SPCM1_click_time > 0 and SPCM2_click_time < 0:
                                # Microwave pulse
                                at_mu(SPCM1_click_time + 1100 + 12000)
                                self.ttl7.off()  # Microwave off when ttl on
                                at_mu(SPCM1_click_time + 20000)
                                self.urukul0_ch0.sw.on()
                                at_mu(SPCM1_click_time + 5120+ 20000)
                                self.urukul0_ch0.sw.off()
                                at_mu(SPCM1_click_time + 100 + 5120 + 20000 + 3300)
                                self.ttl7.on()  # Microwave off when ttl on

                                # at_mu(SPCM1_click_time + 100 + 4770 + 5000 + 3300)
                                # self.urukul0_ch0.set(frequency=micro_freq * MHz, amplitude=0.7)
                                # at_mu(SPCM1_click_time + 100 + 4770 + 15000 + 3300)
                                # self.urukul1_ch0.set(frequency=RF_freq * MHz, amplitude=0.6)
                                #
                                # at_mu(SPCM1_click_time + 100 + 4770 + 25000 + 3300)
                                # self.ttl7.on()  # Microwave off when ttl on
                                # at_mu(SPCM1_click_time + 100 + 4770 + 25000 + 4000 + 3300)
                                # with parallel:
                                #     self.urukul0_ch0.sw.off()
                                #     self.urukul1_ch0.sw.off()
                                # at_mu(SPCM1_click_time + 100 + 4770 + 25000 + 4000 + 117900+ 3300)
                                # with parallel:
                                #     self.urukul0_ch0.sw.off()
                                #     self.urukul1_ch0.sw.off()
                                # at_mu(SPCM1_click_time + 100 + 4770 + 25000 + 4100 + 117900+ 3300)
                                # self.ttl7.on()  # Microwave off when ttl on

                                # Blowaway
                                delay(10 * us)
                                self.zotino0.write_dac(0, self.X_off)
                                self.zotino0.write_dac(1, self.Y_off)
                                self.zotino0.write_dac(2, self.Z1_off)
                                self.zotino0.write_dac(3, self.Z2_off)
                                self.zotino0.load()
                                delay(9 * ms)

                                delay(10 * us)
                                self.urukul2_ch2.set(frequency=80 * MHz, amplitude=0.11);
                                self.urukul0_ch1.set(frequency=99 * MHz, amplitude=0.4);
                                delay(10 * us)

                                self.urukul0_ch1.sw.on()
                                delay(1 * us)
                                self.urukul1_ch3.sw.on()
                                delay(1 * us)
                                self.urukul2_ch2.sw.on()
                                delay(500 * us)

                                self.urukul0_ch1.sw.off()
                                delay(1 * us)
                                self.urukul2_ch0.sw.on()
                                delay(1 * us)
                                self.urukul2_ch1.sw.on()
                                delay(1 * us)

                                self.urukul0_ch1.set(frequency=130 * MHz, amplitude=0.35);
                                self.urukul2_ch2.set(frequency=80 * MHz, amplitude=amp2);
                                delay(10 * us)

                            if SPCM1_click_time < 0 and SPCM2_click_time > 0:
                                # Microwave pulse
                                at_mu(SPCM2_click_time + 1100 + 12000)
                                self.ttl7.off()  # Microwave off when ttl on
                                at_mu(SPCM2_click_time + 20000)
                                self.urukul0_ch0.sw.on()
                                at_mu(SPCM2_click_time + 5120 + 20000)
                                self.urukul0_ch0.sw.off()
                                at_mu(SPCM2_click_time + 100 + 5120 + 20000+ 3300)
                                self.ttl7.on()  # Microwave off when ttl on

                                # at_mu(SPCM2_click_time + 100 + 4770 + 5000+ 3300)
                                # self.urukul0_ch0.set(frequency=micro_freq * MHz, amplitude=0.7)
                                # at_mu(SPCM2_click_time + 100 + 4770 + 15000+ 3300)
                                # self.urukul1_ch0.set(frequency=RF_freq * MHz, amplitude=0.6)
                                #
                                # at_mu(SPCM2_click_time + 100 + 4770 + 25000+ 3300)
                                # self.ttl7.on()  # Microwave off when ttl on
                                # at_mu(SPCM2_click_time + 100 + 4770 + 25000 + 4000+ 3300)
                                # with parallel:
                                #     self.urukul0_ch0.sw.off()
                                #     self.urukul1_ch0.sw.off()
                                # at_mu(SPCM2_click_time + 100 + 4770 + 25000 + 4000 + 117900+ 3300)
                                # with parallel:
                                #     self.urukul0_ch0.sw.off()
                                #     self.urukul1_ch0.sw.off()
                                # at_mu(SPCM2_click_time + 100 + 4770 + 25000 + 4100 + 117900+ 3300)
                                # self.ttl7.on()  # Microwave off when ttl on

                                #

                                # Blowaway
                                delay(10 * us)
                                self.zotino0.write_dac(0, self.X_off)
                                self.zotino0.write_dac(1, self.Y_off)
                                self.zotino0.write_dac(2, self.Z1_off)
                                self.zotino0.write_dac(3, self.Z2_off)
                                self.zotino0.load()
                                delay(9 * ms)

                                delay(10 * us)
                                self.urukul2_ch2.set(frequency=80 * MHz, amplitude=0.11);
                                self.urukul0_ch1.set(frequency=99 * MHz, amplitude=0.4);
                                delay(10 * us)

                                self.urukul0_ch1.sw.on()
                                delay(1 * us)
                                self.urukul1_ch3.sw.on()
                                delay(1 * us)
                                self.urukul2_ch2.sw.on()
                                delay(500 * us)

                                self.urukul0_ch1.sw.off()
                                delay(1 * us)
                                self.urukul2_ch0.sw.on()
                                delay(1 * us)
                                self.urukul2_ch1.sw.on()
                                delay(1 * us)

                                self.urukul0_ch1.set(frequency=130 * MHz, amplitude=0.35);
                                self.urukul2_ch2.set(frequency=80 * MHz, amplitude=amp2);
                                delay(10 * us)


                            delay(5 * us)
                            while SPCM1_click_counter < 1:
                                if SPCM1_click_time == -1.0:
                                    break
                                time_stamp1[excitation_cycle] = self.core.mu_to_seconds(SPCM1_click_time - t0)
                                SPCM1_click_counter += 1

                            while SPCM2_click_counter < 1:
                                if SPCM2_click_time == -1.0:
                                    break
                                time_stamp2[excitation_cycle] = self.core.mu_to_seconds(SPCM2_click_time - t0)
                                SPCM2_click_counter += 1

                            delay(2 * us)
                            self.ttl7.on()
                            delay(1 * us)

                            if (SPCM1_click_counter != 0) or (SPCM2_click_counter != 0):
                                photon_detected = True
                                final_timestamp1 = time_stamp1[excitation_cycle]
                                final_timestamp2 = time_stamp2[excitation_cycle]
                                break  # Break the cycle loop (Success)

                            # PGC after each excitation cycle start#

                            delay(1 * us)
                            self.zotino0.write_dac(0, self.X_off)
                            self.zotino0.write_dac(1, self.Y_off)
                            self.zotino0.write_dac(2, self.Z1_off)
                            self.zotino0.write_dac(3, self.Z2_off)
                            self.zotino0.load()
                            delay(1 * ms)
                            # Trap back up
                            self.ramp_trap_amp(trap_ramp_end_amp, self.trap_ramp_start_amp, self.trap_ramp_steps,
                                               self.trap_ramp_dt_us)
                            delay(1 * ms)

                            self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);

                            self.urukul2_ch0.sw.on()            #fiber aoms back on since they were off for opt pump and coolinG AOM
                            self.urukul2_ch1.sw.on()
                            self.urukul2_ch2.sw.on()
                            delay(1 * us)
                            self.urukul1_ch3.sw.on()

                            self.urukul0_ch1.sw.on()
                            delay(1 * us)
                            self.urukul0_ch2.sw.on()
                            delay(1 * us)
                            self.ttl5.off()
                            delay(self.PGC_Af_Img)
                            delay(1 * us)
                            self.urukul0_ch1.sw.off()
                            delay(1 * us)
                            self.urukul0_ch2.sw.off()
                            delay(1 * us)
                            self.ttl5.on()
                            self.urukul1_ch3.sw.off()


                            # PGC after each excitation cycle end#






                        # --- End of Cycle Loop ---

                        if photon_detected:
                            break # Break the "Retry" loop (Proceed to Success)

                        # --- Case 1 Recovery (Max Cycles Reached, No Photon) ---
                        # Recooling



                        # PGC
                        # self.urukul1_ch3.sw.on()
                        # delay(1 * us)
                        # self.urukul2_ch0.sw.on()
                        # delay(1 * us)
                        # self.urukul2_ch1.sw.on()
                        # delay(1 * us)
                        # self.urukul2_ch2.sw.on()
                        # delay(1 * us)
                        # self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                        # self.urukul0_ch1.sw.on()
                        # delay(1 * us)
                        # self.urukul0_ch2.sw.on()
                        # delay(1 * us)
                        # self.ttl5.off()
                        # delay(1 * us)
                        # delay(self.PGC_Af_Img)

                        self.urukul1_ch3.sw.on()
                        delay(1 * us)
                        self.urukul0_ch2.sw.on()
                        delay(1 * us)
                        self.ttl5.off()

                        # Check Image (Survival Check)
                        self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                        self.urukul0_ch1.sw.on()

                        with parallel:
                            tend1_recov = self.ttl0.gate_rising(self.t2)
                            tend2_recov = self.ttl1.gate_rising(self.t2)
                        count1_recov = self.ttl0.count(tend1_recov)
                        count2_recov = self.ttl1.count(tend2_recov)
                        delay(100 * us)

                        self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                        self.urukul0_ch1.sw.on()
                        delay(10 * us)
                        delay(self.PGC_Af_Img)
                        self.urukul0_ch1.sw.off()
                        delay(1 * us)
                        self.urukul0_ch2.sw.off()
                        delay(1 * us)
                        self.ttl5.on()
                        delay(1 * us)

                        if (count1_recov + count2_recov - count1_bg - count2_bg) > self.hist_thres:
                            # Atom exists: Retry Excitation (stay in Orange while loop)
                            continue
                        else:
                            # Atom lost: Break Orange loop, will loop back to Yellow (Loading)
                            break

                    # --- End of Excitation Retry Loop ---


                    # --- CASE 2: SUCCESS (Blue Scope) ---
                    if photon_detected:

                        delay(1 * us)

                        # Trap back
                        self.ramp_trap_amp(trap_ramp_end_amp, self.trap_ramp_start_amp, self.trap_ramp_steps, self.trap_ramp_dt_us)
                        delay(1 * ms)


                        # PGC
                        self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                        self.urukul0_ch1.sw.on()
                        delay(1 * us)
                        self.urukul0_ch2.sw.on()
                        delay(1 * us)
                        self.ttl5.off()
                        delay(1 * us)
                        delay(self.PGC_Af_Img)

                        # Image 2 (Final State Detection)
                        self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                        self.urukul0_ch1.sw.on()

                        with parallel:
                            tend1_2 = self.ttl0.gate_rising(self.t2)
                            tend2_2 = self.ttl1.gate_rising(self.t2)
                        count1_2 = self.ttl0.count(tend1_2)
                        count2_2 = self.ttl1.count(tend2_2)
                        delay(100 * us)

                        self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                        self.urukul0_ch1.sw.on()
                        delay(10 * us)
                        delay(self.PGC_Af_Img)
                        self.urukul0_ch1.sw.off()
                        delay(1 * us)
                        self.urukul0_ch2.sw.off()
                        delay(1 * us)
                        self.ttl5.on()
                        delay(1 * us)

                        # Write Counts (Fixed arguments)
                        # self.write_counts(final_timestamp1, final_timestamp2, trials, (count1_2 + count2_2), (count1_2 + count2_2 - count1_bg - count2_bg), hwp, qwp)
                        self.write_counts(final_timestamp1, final_timestamp2, trials, (count1_2 + count2_2), (count1_bg + count2_bg), hwp, qwp)
                        trials = 0
                        delay(1 * us)
                        measurements += 1

                    # Reset Zotino for next loop
                    delay(1 * ms)
                    delay(1 * us)
                    self.zotino0.write_dac(0, 0.0)
                    self.zotino0.write_dac(1, 0.0)
                    self.zotino0.write_dac(2, 0.0)
                    self.zotino0.write_dac(3, 0.0)
                    self.zotino0.load()
                    delay(1 * us)
                    self.urukul0_ch1.set(frequency=112 * MHz, amplitude=0.35);
                    self.urukul0_ch1.sw.on()
                    delay(1 * ms)

                    self.urukul1_ch3.sw.on()
                    delay(5 * us)
                    self.urukul2_ch2.sw.on()
                    delay(5 * us)
                    self.urukul2_ch0.sw.on()
                    delay(5 * us)
                    self.urukul2_ch1.sw.on()
                    delay(5 * us)