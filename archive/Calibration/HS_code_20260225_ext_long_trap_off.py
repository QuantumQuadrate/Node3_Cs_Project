from artiq.experiment import *
from aom_feedback_final_5channel import FeedbackRebalancer_5, REF_PATH
from artiq.language.core import rpc
import os
import time
import csv
import numpy as np

LOOPS = 1 # number of repetitions

class HS_20260225_ext_long_trap_off(EnvExperiment):

    def build(self):
        self.base_dir = r"C:\Networking Experiment\artiq codes\artiq-master\repository\SPCM_CSV"
        tsD = time.strftime("%Y%m%d_%H%M")
        self.csv_fname = os.path.join(self.base_dir,
                                      f"Opt_rate_scan_SPCM_{tsD}.csv")
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
        self.setattr_device("ttl14") # Switch for excitation & D1 pump
        self.setattr_device("ttl12")  # Excitation 1st aom currently novatech previously urukul1_ch0 now two ttls for more attenuation
        self.setattr_device("ttl6")  # Excitation 1st aom currently novatech previously urukul1_ch0


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
        self.t2 = 10 * ms  # RF off
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
        self.X_PGC = 0.28
        self.Y_PGC = 0.1

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
        self.p_gain_4 = 0.0
        self.tol_0 = 0.01
        self.tol_1 = 0.01
        self.tol_2 = 0.01
        self.tol_3 = 0.001
        self.tol_4 = 1.0
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

        self.setattr_argument("Trap_amp_min",  NumberValue(0.150, min=0.000, max=0.500, ndecimals=3), group="Excitation power")
        self.setattr_argument("Trap_amp_max",  NumberValue(0.250, min=0.000, max=0.600, ndecimals=3), group="Excitation power")
        self.setattr_argument("Trap_amp_step", NumberValue(0.010, min=0.000, max=0.500, ndecimals=3), group="Excitation power")

        self.setattr_argument("back_ground_only_measure_on", BooleanValue(default=False), "Microwave Scans - choose one of the following")

    @rpc(flags={"async"})
    def write_counts(self, array1, array2):
        """
        Host-side RPC: append one row of [shot0, shot1, shot2, background]
        to the timestamped CSV created in build().
        """
        row_to_write = array1 + array2
        fname = self.csv_fname
        is_new = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            writer = csv.writer(f)
            # if is_new:
            #     writer.writerow(["shot0", "shot1", "shot2", "background"])
            writer.writerow(row_to_write)


    def prepare(self):

        if REF_PATH.exists():
            vals = [float(x) for x in REF_PATH.read_text().split()]
            self.stored_refs = (vals + [-1, -1, -1, -1, -1])[:5]
        else:
            self.stored_refs = [-1, -1, -1, -1, -1]

        td0X, dtdX = self.Trap_amp_min, self.Trap_amp_step
        ntdX       = int(round((self.Trap_amp_max - td0X) / dtdX)) + 1
        self.Trap_amps = [td0X + i * dtdX for i in range(ntdX)]

        self.amp_indexcount = 0

    @kernel
    def ramp_trap_amp(self, start_amp, end_amp, ramp_steps, ramp_dt_us):
        """
        Adiabatically ramp urukul1_ch1 amplitude from start_amp to end_amp.

        ramp_steps   : GUI argument (NumberValue) -> total number of steps
        ramp_dt_us   : GUI argument (NumberValue) -> step time in microseconds
        """
        # --- make step count an *integer* for range() ---
        n_steps = int(ramp_steps)
        dt = ramp_dt_us

        if n_steps <= 0:
            return  # nothing to do, avoid division by zero

        # --- amplitude increment per step (float is fine) ---
        step = (end_amp - start_amp) / ramp_steps  # ramp_steps is float

        amp = start_amp
        for i in range(n_steps + 1):
            if amp < 0.0: amp = 0.0
            if amp > 0.3: amp = 0.3

            amp_q = int(amp * 1000.0) / 1000.0  # 0.001 truncation
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
        self.ttl14.output()
        self.ttl14.on()
        self.ttl6.output()
        self.ttl6.on()  #ensures excitation 1st aom is off for the duration of this experiment
        self.ttl12.output()
        self.ttl12.on()
        delay(2 * us)

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
        self.urukul1_ch0.set(frequency=2 * MHz, amplitude=0.1);     #replaced with ttl6.on() above
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
        self.ttl4.off()
        delay(1 * us)
        self.ttl5.on()
        delay(1 * us)

        delay(1000 * ms)

        for ampT_noI2V in self.Trap_amps:

            Ref_amp = 1.0
            delay(1 * us)
            self.amp_indexcount += 1
            measurements = 0

            while measurements < self.n_measurements:
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
                    # always use these refs:
                    self.stored_refs[0], self.stored_refs[1], self.stored_refs[2], self.stored_refs[3], Ref_amp,
                    # last-known amps & freqs:
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

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()
                delay(100 * us)

                self.urukul0_ch2.sw.on()
                delay(1 * us)
                self.ttl5.off()
                delay(1 * us)

                self.urukul1_ch1.set(frequency=150 * MHz, amplitude=self.nominal_trapdepth);
                self.urukul1_ch1.sw.on()
                delay(100 * us)

                tend_bg = self.ttl0.gate_rising(self.t2)  # SPCM Background
                count_bg = self.ttl0.count(tend_bg)
                delay(10 * us)

                self.ttl4.on()

                delay(self.wait)

                self.zotino0.write_dac(0, self.X_idle)  # ShimcoilX
                self.zotino0.write_dac(1, self.Y_idle)  # ShimcoilY
                self.zotino0.write_dac(2, self.Z1_idle)  # ShimcoilZ1
                self.zotino0.write_dac(3, self.Z2_idle)  # ShimcoilZ2
                self.zotino0.load()
                delay(20 * ms)
                self.urukul0_ch1.set(frequency=112 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()
                delay(1 * us)

                delay(self.MOT2PGC)

                self.zotino0.write_dac(0, self.X_PGC)  # ShimcoilX
                self.zotino0.write_dac(1, self.Y_PGC)  # ShimcoilY
                self.zotino0.write_dac(2, self.Z1_PGC)  # ShimcoilZ1
                self.zotino0.write_dac(3, self.Z2_PGC)  # ShimcoilZ2
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

                self.zotino0.write_dac(0, self.X_off)  # ShimcoilX
                self.zotino0.write_dac(1, self.Y_off)  # ShimcoilY
                self.zotino0.write_dac(2, self.Z1_off)  # ShimcoilZ1
                self.zotino0.write_dac(3, self.Z2_off)  # ShimcoilZ2        #and replaced it with this
                self.zotino0.load()
                delay(300 * ms)

                # self.urukul0_ch1.set(frequency=126 * MHz, amplitude=0.3);
                # self.urukul0_ch1.sw.on()
                # delay(self.PGC_Af_Img)

                self.urukul0_ch1.sw.off()
                delay(1 * us)
                self.urukul0_ch2.sw.off()
                delay(1 * us)
                self.ttl5.on()
                delay(1 * us)

                delay(self.t3)

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude= 0.36);
                self.urukul0_ch1.sw.on()
                delay(1 * us)
                self.urukul0_ch2.sw.on()
                delay(1 * us)
                self.ttl5.off()
                delay(1 * us)

                delay(1 * us)
                tend1 = self.ttl0.gate_rising(self.t2)  # changed delay to SPCM counting for delay length
                count1 = self.ttl0.count(tend1)
                count1_bg = count1 - count_bg
                delay(10 * us)

                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()

                delay(self.PGC_Af_Img)

                self.urukul0_ch1.sw.off()
                delay(1 * us)
                self.urukul0_ch2.sw.off()
                delay(1 * us)
                self.ttl5.on()
                delay(1 * us)

                count2 = self.hist_thres + 1
                count2_bg = self.hist_thres + 1

                if self.back_ground_only_measure_on :
                    count1_bg = self.hist_thres + 1
                    self.urukul1_ch1.sw.off()
                    delay(10 * ms)
                    self.urukul1_ch1.sw.on()
                    delay(1 * us)

                while (measurements < self.n_measurements) and (count1_bg > self.hist_thres) and (count2_bg > self.hist_thres):
                    # F=3 state preparation
                    self.urukul0_ch1.sw.on()
                    delay(10 * ms)
                    self.urukul0_ch1.sw.off()
                    delay(1 * us)
                    # F=3 state preparation

                    # Q-axis on #
                    self.zotino0.write_dac(0, -0.013)
                    self.zotino0.write_dac(1, 5.93)
                    # self.zotino0.write_dac(1, -4.53)
                    self.zotino0.write_dac(2, -0.06)  # ShimcoilZ1 back to MOT
                    self.zotino0.write_dac(3, 0.06)  # ShimcoilZ2 back to MOT
                    self.zotino0.load()
                    delay(1 * ms)
                    # Q-axis on #

                    # Trap adiabatic down #
                    self.ramp_trap_amp(self.trap_ramp_start_amp, trap_ramp_end_amp, self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)
                    # Trap adiabatic down #

                    time_stamp1 = [-1.0] * int(self.max_excitation_cycles)
                    time_stamp2 = [-1.0] * int(self.max_excitation_cycles)

                    for excitation_cycle in range(int(self.max_excitation_cycles)):
                        delay(10 * us)

                        # Pump begin
                        # self.urukul0_ch3.set(frequency=145 * MHz, amplitude=0.02);
                        self.urukul0_ch3.set(frequency=145 * MHz, amplitude=0.03);
                        self.urukul0_ch3.sw.on()
                        delay(1 * us)
                        self.urukul1_ch2.set(frequency=150 * MHz, amplitude=0.04);
                        self.urukul1_ch2.sw.on()
                        delay(1 * us)
                        self.urukul2_ch3.set(frequency=202 * MHz, amplitude=0.4);
                        self.urukul2_ch3.sw.on()  # trap aom here
                        delay(1 * us)
                        self.ttl14.off()

                        delay(1 * ms)

                        t_optend = now_mu()
                        at_mu(t_optend + 1000)
                        with parallel:
                            self.ttl14.on()  # Pump off
                            self.urukul0_ch3.sw.off()
                            self.urukul1_ch2.sw.off()
                            self.urukul2_ch3.sw.off()  # Pump repump off
                        delay(1 * us)
                        # pump end

                        # Set excitation experiment AOM values before now_mu
                        self.urukul1_ch2.sw.on()  # excitation on
                        self.ttl6.off()
                        delay(1 * us)
                        # self.ttl14.off()  # Pump off
                        delay(1 * us)

                        self.urukul1_ch1.set(frequency=150 * MHz, amplitude=trap_ramp_end_amp);  # trap off
                        self.urukul1_ch2.set(frequency=145 * MHz, amplitude=ampT_noI2V);
                        delay(10 * us)

                        t0 = now_mu()

                        at_mu(t0 + 4215)
                        self.urukul1_ch1.sw.off()

                        at_mu(t0 + 4215 + int(self.t_trap_off_mu))
                        self.urukul1_ch1.sw.on()

                        at_mu(t0 + 4028)
                        self.ttl14.off()

                        at_mu(t0 + 4028 + int(self.length_offset_mu))
                        self.ttl14.on()

                        at_mu(t0 + 4540)
                        with parallel:
                            self.ttl12.off()
                            self.ttl6.off()

                        at_mu(t0 + 4540 + 300)
                        with parallel:
                            self.ttl12.on()
                            self.ttl6.on()

                        at_mu(t0 + 3800)
                        self.urukul1_ch2.sw.on()

                        at_mu(t0 + 3800 + 275)
                        self.urukul1_ch2.sw.off()

                        at_mu(t0 + 4500 + 780)
                        with parallel:
                            t_end = self.ttl0.gate_rising(self.t_spcm_collection_mu * ns)
                            t_end_2 = self.ttl1.gate_rising(self.t_spcm_collection_mu * ns)

                        SPCM1_click_counter = 0
                        delay(5 * us)
                        # t = -1.0
                        while SPCM1_click_counter < 1:
                            SPCM1_click_time = self.ttl0.timestamp_mu(t_end)
                            if SPCM1_click_time == -1.0:
                                break
                            time_stamp1[excitation_cycle] = self.core.mu_to_seconds(SPCM1_click_time - t0)
                            SPCM1_click_counter += 1


                        SPCM2_click_counter = 0
                        # j = -1.0
                        while SPCM2_click_counter < 1:
                            SPCM2_click_time = self.ttl1.timestamp_mu(t_end_2)
                            if SPCM2_click_time == -1.0:
                                break
                            time_stamp2[excitation_cycle] = self.core.mu_to_seconds(SPCM2_click_time - t0)
                            SPCM2_click_counter += 1

                        delay(10 * us)
                        #spcm_singlephoton_array[excitation_cycle] = t  # Store time tagging instead of count
                        delay(1 * us)

                    # Recooling Shim coil set
                    self.zotino0.write_dac(0, self.X_off)  # ShimcoilX
                    self.zotino0.write_dac(1, self.Y_off)  # ShimcoilY
                    self.zotino0.write_dac(2, self.Z1_off)  # ShimcoilZ1
                    self.zotino0.write_dac(3, self.Z2_off)  # ShimcoilZ2        #and replaced it with this
                    self.zotino0.load()
                    delay(1 * ms)  # coil relaxation time


                    # trap back #
                    self.ramp_trap_amp(trap_ramp_end_amp, self.trap_ramp_start_amp, self.trap_ramp_steps, self.trap_ramp_dt_us)
                    delay(1 * ms)
                    # trap back #



                    # PGC before 2nd shot image
                    self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                    self.urukul0_ch1.sw.on()
                    delay(1 * us)
                    self.urukul0_ch2.sw.on()
                    delay(1 * us)
                    self.ttl5.off()
                    delay(1 * us)

                    delay(self.PGC_Af_Img)

                    # 2 shot imaging start
                    self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                    self.urukul0_ch1.sw.on()

                    delay(1 * us)
                    tend2 = self.ttl0.gate_rising(self.t2)  # changed delay to SPCM counting for delay length
                    count2 = self.ttl0.count(tend2)
                    count2_bg = count2 - count_bg
                    delay(10 * us)

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

                    if self.back_ground_only_measure_on:
                        count2_bg = self.hist_thres + 1

                    if count2_bg > self.hist_thres:
                        measurements += 1
                        self.write_counts(time_stamp1, time_stamp2)

                    delay(1 * us)

                # print("Start bg measure again")
                delay(1 * ms)
                delay(1 * us)
                self.zotino0.write_dac(0, 0.0)  # ShimcoilX back to MOT
                self.zotino0.write_dac(1, 0.0)  # ShimcoilY back to MOT
                self.zotino0.write_dac(2, 0.0)  # ShimcoilZ1 back to MOT
                self.zotino0.write_dac(3, 0.0)  # ShimcoilZ2 back to MOT
                self.zotino0.load()
                delay(1 * us)
                self.urukul0_ch1.set(frequency=112 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()
                # wait until 1 s after t_start before next loop
                # self.ttl4.on()
                delay(1 * ms)