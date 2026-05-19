from artiq.experiment import *
from aom_feedback_final_5channel import FeedbackRebalancer_5, REF_PATH
from artiq.coredevice.ad9910 import (
    PHASE_MODE_ABSOLUTE,
    PHASE_MODE_CONTINUOUS,
    PHASE_MODE_TRACKING,
    RAM_DEST_ASF,
    RAM_MODE_RAMPUP
)
from artiq.language.core import rpc
import os
import time
import csv

# LOOPS = 300 # number of repetitions
LOOPS = 100

class Node3_Microwave_variables_scan_v7_packing_20260317(EnvExperiment):

    def build(self):

        self.core = self.get_device("core")
        self.setattr_device("zotino0")
        self.setattr_device("urukul0_ch0")  # Microwave_IF
        self.setattr_device("urukul0_ch1")  # Cooling AOM 1
        self.setattr_device("urukul0_ch2")  # Repump AOM
        self.setattr_device("urukul0_ch3")  # D1 first AOM
        self.setattr_device("urukul1_ch0")  # RF
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
        # self.t1 = 1 * us  # first trigger
        # self.t2 = 10 * ms  # RF off
        # self.t3 = 150 * ms  # RF on + second trigger
        # self.exp = 33 * ms  # EMCCD exposure
        # self.trig_width = 5 * ms  # Zotino high time
        # self.period = 1 * s  # full cycle period
        # self.MOT2PGC = 500 * ms
        # self.PGC_2_off = 300 * ms
        # self.PGC_Af_Img = 20 * ms
        # self.wait = 1 * ms
        # self.nominal_trapdepth = 0.24

        self.t1 = 1 * us  # first trigger
        self.t2 = 10 * ms  # RF off
        self.t3 = 50 * ms  # RF on + second trigger
        self.exp = 33 * ms  # EMCCD exposure
        self.trig_width = 5 * ms  # Zotino high time
        self.period = 1 * s  # full cycle period
        self.MOT2PGC = 400 * ms
        self.PGC_2_off = 50 * ms
        self.PGC_Af_Img = 20 * ms
        self.wait = 1 * ms
        self.nominal_trapdepth = 0.24



        # Shims
        self.Z1_idle = 4.25  # volts
        self.Z2_idle = 4.2  # volts
        self.X_idle = 0.25
        self.Y_idle = 0.1

        # --- PGC shim values ---
        self.Z1_PGC = 4.26  # volts
        self.Z2_PGC = 4.2  # volts
        self.X_PGC = 0.22
        self.Y_PGC = 0.3


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

        self.setattr_argument("trap_ramp_start_amp", NumberValue(0.24, min=0.000, max=0.300, ndecimals=3), group="Adiabatic parameter")
        # self.setattr_argument("trap_ramp_end_amp", NumberValue(0.125, min=0.000, max=0.300, ndecimals=3), group="Adiabatic parameter")
        self.setattr_argument("trap_ramp_steps",     NumberValue(200, min=0, max=100000, ndecimals=1), group="Adiabatic parameter")
        self.setattr_argument("trap_ramp_dt_us",        NumberValue(50, min=0.001, max=1000, ndecimals=3), group="Adiabatic parameter")

        self.setattr_argument("X_Qaxis", NumberValue(-0.013, min=-3.0, max=8.0, ndecimals=3), group="Q_Axis_Parameter")
        self.setattr_argument("Y_Qaxis",     NumberValue(2.72, min=-3.0, max=8.0, ndecimals=3), group="Q_Axis_Parameter")
        self.setattr_argument("Z1_Qaxis",    NumberValue(-0.06, min=-3.0, max=8.0, ndecimals=3), group="Q_Axis_Parameter")
        self.setattr_argument("Z2_Qaxis",    NumberValue(0.06, min=-3.0, max=8.0, ndecimals=3), group="Q_Axis_Parameter")

        self.setattr_argument("Frequency_00_Scan", BooleanValue(default=False),"Microwave Scans - choose one of the following")
        self.setattr_argument("Frequency_01_Scan", BooleanValue(default=False),"Microwave Scans - choose one of the following")
        self.setattr_argument("Frequency_11_Scan", BooleanValue(default=False),"Microwave Scans - choose one of the following")
        self.setattr_argument("Frequency_2ph_Scan", BooleanValue(default=False),"Microwave Scans - choose one of the following")


        self.setattr_argument("Time_00_Scan", BooleanValue(default=False), "Microwave Scans - choose one of the following")
        self.setattr_argument("Time_01_Scan", BooleanValue(default=False), "Microwave Scans - choose one of the following")
        self.setattr_argument("Time_11_Scan", BooleanValue(default=False), "Microwave Scans - choose one of the following")
        self.setattr_argument("Time_2ph_Scan", BooleanValue(default=False), "Microwave Scans - choose one of the following")

        self.setattr_argument("Ramsey_00_Scan", BooleanValue(default=False), "Ramsey Scans - choose one of the following")
        self.setattr_argument("Ramsey_11_Scan", BooleanValue(default=False), "Ramsey Scans - choose one of the following")
        self.setattr_argument("Ramsey_2ph_Scan", BooleanValue(default=False), "Ramsey Scans - choose one of the following")


        self.setattr_argument("Micro_Fix_00_freq", NumberValue(250.000, min=190.000000, max=260.000000, step=0.000100, ndecimals=6),group="Freq_fix_param")
        self.setattr_argument("Micro_Fix_01_freq", NumberValue(250.468, min=190.000000, max=260.000000, step=0.000100, ndecimals=6),group="Freq_fix_param")
        self.setattr_argument("Micro_Fix_11_freq", NumberValue(250.972, min=190.000000, max=260.000000, step=0.000100, ndecimals=6),group="Freq_fix_param")
        self.setattr_argument("Micro_RF_Fix_detune_freq_kHz", NumberValue(-125, min=-1000.000, max=1000.000000, step=0.50, ndecimals=3),group="Freq_fix_param")
        self.setattr_argument("RF_Fix_small_detune_freq_kHz", NumberValue(-9.33, min=-100.0, max=100.0, step=0.0100, ndecimals=3),group="Freq_fix_param")


        self.setattr_argument("Microwave_freq_min", NumberValue(250.400000, min=190.000000, max=260.000000, step=0.000100, ndecimals=6), group="Freq_scan_param")
        self.setattr_argument("Microwave_freq_max", NumberValue(250.600000, min=190.000000, max=260.000000, step=0.000100, ndecimals=6), group="Freq_scan_param")
        self.setattr_argument("Microwave_freq_step", NumberValue( 0.010000, min=  0.000001, max= 10.000000, step=0.000100, ndecimals=6), group="Freq_scan_param")
        self.setattr_argument("RF_freq_detun_min_kHz", NumberValue(-15.0, min=-50.0, max=50.0, step=0.0100, ndecimals=6), group="Freq_scan_param")
        self.setattr_argument("RF_freq_detun_max_kHz", NumberValue(-5.0, min=-50.0, max=50.0, step=0.0100, ndecimals=6), group="Freq_scan_param")
        self.setattr_argument("RF_freq_detun_step_kHz", NumberValue(0.5, min=-50.0, max=50.0, step=0.0100, ndecimals=6), group="Freq_scan_param")



        self.setattr_argument("Time_Fix_00", NumberValue(4530, min=0.0, max=10000000.0), group="time_fix_param")
        self.setattr_argument("Time_Fix_01", NumberValue(31876, min=0.0, max=10000000.0), group="time_fix_param")
        self.setattr_argument("Time_Fix_11", NumberValue(5033, min=0.0, max=10000000.0), group="time_fix_param")
        self.setattr_argument("Time_Fix_2ph", NumberValue(2.43e5, min=0.0, max=10000000.0), group="time_fix_param")
        self.setattr_argument("time_min", NumberValue(0.0, min=0.0, max=10000000.0), group="time_scan_param")
        self.setattr_argument("time_max", NumberValue(1e6, min=0.0, max=10000000.0), group="time_scan_param")
        self.setattr_argument("time_step", NumberValue(50000.0, min=0.0, max=10000000.0), group="time_scan_param")

        self.base_dir = r"C:\Networking Experiment\artiq codes\artiq-master\repository\SPCM_CSV"
        ts = time.strftime("%Y%m%d_%H%M")
        scan = "0"
        if self.Frequency_00_Scan :
            scan = f"MW_scan_F00_{self.Microwave_freq_min}_{self.Microwave_freq_max}_{self.Microwave_freq_step}"
        elif self.Frequency_01_Scan :
            scan = f"MW_scan_F01_{self.Microwave_freq_min}_{self.Microwave_freq_max}_{self.Microwave_freq_step}"
        elif self.Frequency_11_Scan :
            scan = f"MW_scan_F11_{self.Microwave_freq_min}_{self.Microwave_freq_max}_{self.Microwave_freq_step}"
        elif self.Frequency_2ph_Scan:
            scan = f"MW_scan_F2ph_{self.RF_freq_detun_min_kHz}_{self.RF_freq_detun_max_kHz}_{self.RF_freq_detun_step_kHz}"
        elif self.Time_00_Scan :
            scan = f"MW_scan_T00_{self.time_min}_{self.time_max}_{self.time_step}"
        elif self.Time_01_Scan :
            scan = f"MW_scan_T01_{self.time_min}_{self.time_max}_{self.time_step}"
        elif self.Time_11_Scan :
            scan = f"MW_scan_T11_{self.time_min}_{self.time_max}_{self.time_step}"
        elif self.Time_2ph_Scan :
            scan = f"MW_scan_T2ph_{self.time_min}_{self.time_max}_{self.time_step}"
        elif self.Ramsey_00_Scan :
            scan = f"Ramsey_scan_00"
        elif self.Ramsey_11_Scan :
            scan = f"Ramsey_scan_11"
        elif self.Ramsey_2ph_Scan :
            scan = f"Ramsey_scan_2ph_V{self.Y_Qaxis}"
        # elif self.Ramsey_00_Scan :
        #     scan = f"Ramsey_scan_00_{self.time_min}_{self.time_max}_{self.time_step}"
        # elif self.Ramsey_11_Scan :
        #     scan = f"Ramsey_scan_11_{self.time_min}_{self.time_max}_{self.time_step}"
        # elif self.Ramsey_2ph_Scan :
        #     scan = f"Ramsey_scan_2ph_{self.time_min}_{self.time_max}_{self.time_step}"

        self.csv_fname = os.path.join(self.base_dir,
                                      f"{scan}_{ts}.csv")
    @rpc(flags={"async"})
    def write_counts(self, c0, c1, c2, c3):
        """Append one row to the CSV (host-side)."""
        fname = self.csv_fname
        is_new = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["shot0", "shot1", "shot2", "background"])
            w.writerow([c0, c1, c2, c3])


    def prepare(self):

        if REF_PATH.exists():
            vals = [float(x) for x in REF_PATH.read_text().split()]
            self.stored_refs = (vals + [-1, -1, -1, -1, -1])[:5]
        else:
            self.stored_refs = [-1, -1, -1, -1, -1]


        self.f00 = self.Micro_Fix_00_freq
        self.f01 = self.Micro_Fix_01_freq
        self.f11 = self.Micro_Fix_11_freq
        self.fm10 = 2 * self.f00 - self.f01
        self.micro_rf_detun = self.Micro_RF_Fix_detune_freq_kHz / 1000.0 # Convert to MHz
        self.rf_detun =  self.RF_Fix_small_detune_freq_kHz / 1000.0 # Convert to MHz

        self.fm10_2ph = self.fm10 + self.micro_rf_detun
        self.rf_2ph = self.f01 - self.f00 - self.micro_rf_detun + self.rf_detun

        MF_min, MF_step = self.Microwave_freq_min, self.Microwave_freq_step
        MF_num       = int(round((self.Microwave_freq_max - MF_min) / MF_step)) + 1
        self.Microwave_freq_targets = [MF_min + i * MF_step for i in range(MF_num)]

        RF_min, RF_step = self.RF_freq_detun_min_kHz / 1000.0, self.RF_freq_detun_step_kHz / 1000.0 # Now everything in MHz
        RF_num       = int(round((self.RF_freq_detun_max_kHz / 1000.0 - RF_min) / RF_step)) + 1
        self.RF_freq_targets = [self.f01 - self.f00 - self.micro_rf_detun + ((RF_min + i * RF_step)) for i in range(RF_num)]

        self.t00 = int(self.Time_Fix_00)
        self.t01 = int(self.Time_Fix_01)
        self.t11 = int(self.Time_Fix_11)
        self.t2ph = int(self.Time_Fix_2ph)

        v1, s1 = self.time_min, self.time_step
        n1 = int(round((self.time_max - v1) / s1)) + 1
        self.time_scan = [int(v1 + i * s1) for i in range(n1)]

        # self.time_scan_ramsey = (
        #         [0 + i * 10000 for i in range(21)] +
        #         [8000000 + i * 10000 for i in range(21)] +
        #         [16000000 + i * 10000 for i in range(21)]
        # )

        self.time_scan_ramsey = (
                [0 + i * 6000 for i in range(21)] +
                [8000000 + i * 6000 for i in range(21)] +
                [16000000 + i * 6000 for i in range(21)] +
                [24000000 + i * 6000 for i in range(21)]
        )

        self.time_scan_ramsey_2photon = (
                [0 + i * 25000 for i in range(21)] +
                [8000000 + i * 25000 for i in range(21)] +
                [16000000 + i * 25000 for i in range(21)] +
                [24000000 + i * 25000 for i in range(21)]
        )
        # if (self.Frequency_00_Scan or self.Frequency_01_Scan or self.Frequency_11_Scan):
        #     self.scan_param = self.Microwave_freq_targets
        # elif self.Frequency_2ph_Scan :
        #     self.scan_param = self.RF_freq_targets
        # elif (self.Time_00_Scan or self.Time_01_Scan or self.Time_11_Scan or self.Time_2ph_Scan) :
        #     self.scan_param = self.time_scan
        # elif (self.Ramsey_00_Scan or self.Ramsey_11_Scan or self.Ramsey_2ph_Scan) :
        #     self.scan_param = self.time_scan_ramsey
        if (self.Frequency_00_Scan or self.Frequency_01_Scan or self.Frequency_11_Scan):
            self.scan_param = self.Microwave_freq_targets
        elif self.Frequency_2ph_Scan :
            self.scan_param = self.RF_freq_targets
        elif (self.Time_00_Scan or self.Time_01_Scan or self.Time_11_Scan or self.Time_2ph_Scan or self.Ramsey_11_Scan):
            self.scan_param = self.time_scan
        elif (self.Ramsey_00_Scan) :
            self.scan_param = self.time_scan_ramsey
        elif (self.Ramsey_2ph_Scan):
            self.scan_param = self.time_scan_ramsey_2photon


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
        # self.urukul1_ch0.set(frequency=220 * MHz, amplitude=0.1);     #replaced with ttl6.on() above
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

        self.urukul0_ch0.set_phase_mode(PHASE_MODE_CONTINUOUS)
        delay(1 * us)
        self.urukul1_ch0.set_phase_mode(PHASE_MODE_ABSOLUTE)

        self.zotino0.write_dac(0, 0.0)  # ShimcoilX back to MOT
        self.zotino0.write_dac(1, 0.0)  # ShimcoilY back to MOT
        self.zotino0.write_dac(2, 0.0)  # ShimcoilZ1 back to MOT
        self.zotino0.write_dac(3, 0.0)  # ShimcoilZ2 back to MOT
        self.zotino0.write_dac(6, 0.0)
        self.zotino0.load()
        delay(17 * ms)
        self.ttl4.on()
        delay(1 * us)
        self.ttl5.on()
        delay(1 * us)

        delay(1000 * ms)


        for _ in range(LOOPS):
            self.core.break_realtime()
            delay(100 * us)
            self.urukul0_ch1.sw.off()
            delay(100 * us)
            self.ttl5.on()
            delay(100 * us)

            self.urukul0_ch1.set(frequency=111 * MHz, amplitude=0.47);
            self.urukul0_ch1.sw.on()
            delay(10 * us)
            self.urukul0_ch2.set(frequency=197 * MHz, amplitude=0.5);
            self.ttl5.off()
            delay(10 * us)

            delay(100 * us)
            # print(_)
            delay(100 * us)

            # for RaR in self.RAR_scan:
            for param in self.scan_param:
                delay(1000 * us)
                self.urukul0_ch2.sw.off()
                delay(1 * us)
                self.ttl5.on()
                delay(1000 * us)
                self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3);
                self.urukul0_ch1.sw.on()
                delay(100 * us)
                self.ttl6.off()
                delay(1 * us)
                self.ttl14.off()
                delay(100 * us)

                amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance(
                    self.p_gain_0, self.p_gain_1, self.p_gain_2, self.p_gain_3, self.p_gain_4,
                    self.tol_0, self.tol_1, self.tol_2, self.tol_3, self.tol_4,
                    int(self.iterations_0), int(self.iterations_1), int(self.iterations_2), int(self.iterations_3), int(self.iterations_4),
                    # always use these refs:
                    self.stored_refs[0], self.stored_refs[1], self.stored_refs[2], self.stored_refs[3], self.stored_refs[4],
                    # last-known amps & freqs:
                    self.initial_amp_0, self.initial_amp_1, self.initial_amp_2, self.initial_amp_3, self.initial_amp_4,
                    self.frequency_0, self.frequency_1, self.frequency_2, self.frequency_3, self.frequency_4
                )

                trap_ramp_end_amp = amp3
                # trap_ramp_end_amp = 0.06

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
                self.ttl5.off()
                delay(10 * us)

                self.urukul1_ch1.set(frequency=150 * MHz, amplitude=self.nominal_trapdepth);
                self.urukul1_ch1.sw.on()
                delay(100 * us)

                delay(self.wait)

                self.zotino0.write_dac(0, self.X_idle)  # ShimcoilX
                self.zotino0.write_dac(1, self.Y_idle)  # ShimcoilY
                self.zotino0.write_dac(2, self.Z1_idle)  # ShimcoilZ1
                self.zotino0.write_dac(3, self.Z2_idle)  # ShimcoilZ2
                self.zotino0.load()
                delay(1 * us)
                self.urukul0_ch1.set(frequency=112 * MHz, amplitude=0.3);
                self.urukul0_ch1.sw.on()
                delay(1 * us)

                delay(self.MOT2PGC)

                # SPCM 0
                delay(1 * us)
                tend0 = self.ttl0.gate_rising(self.t2)
                delay(1 * us)
                count0 = self.ttl0.count(tend0)
                delay(10 * us)

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

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude= 0.36);
                self.urukul0_ch1.sw.on()
                delay(1 * us)
                self.zotino0.write_dac(6, 4.1)  # 0 shot
                self.zotino0.load()
                delay(self.trig_width)
                self.zotino0.write_dac(6, 0.0)
                self.zotino0.load()
                delay(self.t2)

                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()

                delay(self.PGC_Af_Img)

                self.urukul0_ch1.set(frequency=124 * MHz, amplitude=0.4);
                self.urukul0_ch1.sw.off()
                delay(1 * us)
                self.urukul0_ch2.set(frequency=197 * MHz, amplitude=0.5);
                self.ttl5.on()
                delay(1 * us)

                delay(self.t3)

                delay(self.t3)

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude= 0.36);
                self.urukul0_ch1.sw.on()
                delay(1 * us)
                self.urukul0_ch2.set(frequency=197 * MHz, amplitude=0.5);
                self.ttl5.off()
                delay(1 * us)

                delay(1 * us)
                self.zotino0.write_dac(6, 4.1)  # 1 shot
                self.zotino0.load()
                delay(self.trig_width)
                self.zotino0.write_dac(6, 0.0)
                self.zotino0.load()

                delay(1 * us)
                tend1 = self.ttl0.gate_rising(self.t2)
                count1 = self.ttl0.count(tend1)
                delay(10 * us)

                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()

                delay(self.PGC_Af_Img)


                # F3 preparation#
                self.ttl5.on()
                delay(1 * ms)
                self.urukul0_ch1.sw.off()
                delay(1 * us)
                # F3 preparation#

                # completely turn off #
                self.urukul0_ch2.sw.off()
                delay(1 * us)
                # completely turn off #


                # # old Q-axis on#
                # delay(1 * ms)
                # self.zotino0.write_dac(0, -0.007)
                # self.zotino0.write_dac(1, 5.93)
                # self.zotino0.write_dac(2, -0.054)  # ShimcoilZ1 back to MOT
                # self.zotino0.write_dac(3, 0.054)  # ShimcoilZ2 back to MOT
                # self.zotino0.load()
                # delay(20 * ms)
                # # old Q-axis on#

                # Q-axis on# get back to this value after measurement
                delay(1 * ms)
                self.zotino0.write_dac(0, self.X_Qaxis)
                self.zotino0.write_dac(1, self.Y_Qaxis)
                self.zotino0.write_dac(2, self.Z1_Qaxis)  # ShimcoilZ1 back to MOT
                self.zotino0.write_dac(3, self.Z2_Qaxis)  # ShimcoilZ2 back to MOT
                self.zotino0.load()
                delay(1 * ms)
                # Q-axis on#

                self.ramp_trap_amp(self.trap_ramp_start_amp, trap_ramp_end_amp, self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)

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

                if self.Frequency_00_Scan :
                    self.urukul0_ch0.set(frequency=param * MHz, amplitude=0.7);
                    t_RF = now_mu()
                    at_mu(t_RF + 100)
                    self.ttl7.off()  # Microwave off when ttl on
                    at_mu(t_RF + 4000)
                    self.urukul0_ch0.sw.on()
                    at_mu(t_RF + 4000 + self.t00)
                    self.urukul0_ch0.sw.off()
                    at_mu(t_RF + 4100 + self.t00)
                    self.ttl7.on()  # Microwave off when ttl on

                elif self.Frequency_01_Scan :
                    self.urukul0_ch0.set(frequency=param * MHz, amplitude=0.7);
                    delay(5 * us)
                    self.ttl7.off()  # Microwave off when ttl on
                    delay(5 * us)
                    t_RF = now_mu()
                    if t_RF > 0:
                        at_mu(t_RF)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + self.t01)
                        self.urukul0_ch0.sw.off()
                    delay(1 * us)
                    self.ttl7.on()  # Microwave off when ttl on


                elif self.Frequency_11_Scan :
                    self.urukul0_ch0.set(frequency= self.f01 * MHz, amplitude=0.7);
                    delay(5 * us)
                    self.ttl7.off()  # Microwave off when ttl on
                    delay(5 * us)
                    t_RF = now_mu()
                    if t_RF > 0:
                        at_mu(t_RF)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + self.t01)
                        self.urukul0_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01)
                        self.ttl7.on()  # Microwave off when ttl on

                        at_mu(t_RF + 100 + self.t01 + 1000)
                        self.urukul0_ch0.set(frequency=param * MHz, amplitude=0.7);

                        at_mu(t_RF + 100 + self.t01 + 11000)
                        self.ttl7.off()  # Microwave off when ttl on
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000 + self.t11)
                        self.urukul0_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4100 + self.t11)
                        self.ttl7.on()  # Microwave off when ttl on

                elif self.Frequency_2ph_Scan :
                    # mf0to1 transition start
                    self.urukul0_ch0.set(frequency= self.f01 * MHz, amplitude=0.7);
                    delay(5 * us)
                    self.ttl7.off()  # Microwave off when ttl on
                    delay(5 * us)
                    t_RF = now_mu()
                    if t_RF > 0:
                        at_mu(t_RF)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + self.t01)
                        self.urukul0_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01)
                        self.ttl7.on()  # Microwave off when ttl on
                    # mf0to1 transition end

                        # 2ph frequency scan start
                        at_mu(t_RF + 100 + self.t01 + 1000)
                        with parallel:
                            self.urukul0_ch0.set(frequency = self.fm10_2ph * MHz, amplitude=0.7)
                            self.urukul1_ch0.set(frequency= param * MHz, amplitude=0.6); # Scanning the frequency
                        # self.urukul1_ch0.set(frequency=param * MHz, amplitude=1.0);
                        at_mu(t_RF + 100 + self.t01 + 11000)
                        self.ttl7.off()  # Microwave off when ttl on
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000)
                        with parallel :
                            self.urukul0_ch0.sw.on()
                            self.urukul1_ch0.sw.on()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000 + self.t2ph)
                        with parallel :
                            self.urukul0_ch0.sw.off()
                            self.urukul1_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4100 + self.t2ph)
                        self.ttl7.on()  # Microwave off when ttl on
                        # 2ph frequency scan end

                elif self.Time_00_Scan :
                    self.urukul0_ch0.set(frequency=self.f00 * MHz, amplitude=0.7);
                    t_RF = now_mu()
                    at_mu(t_RF + 100)
                    self.ttl7.off()  # Microwave off when ttl on
                    at_mu(t_RF + 4000)
                    self.urukul0_ch0.sw.on()
                    at_mu(t_RF + 4000 + int(param))
                    self.urukul0_ch0.sw.off()
                    at_mu(t_RF + 4100 + int(param))
                    self.ttl7.on()  # Microwave off when ttl on

                elif self.Time_01_Scan:
                    self.urukul0_ch0.set(frequency=self.f01 * MHz, amplitude=0.7);
                    delay(5 * us)
                    self.ttl7.off()  # Microwave off when ttl on
                    delay(5 * us)
                    t_RF = now_mu()
                    if t_RF > 0:
                        at_mu(t_RF)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + int(param))
                        self.urukul0_ch0.sw.off()
                    delay(1 * us)
                    self.ttl7.on()  # Microwave off when ttl on

                elif self.Time_11_Scan:
                    self.urukul0_ch0.set(frequency= self.f01 * MHz, amplitude=0.7);
                    delay(5 * us)
                    self.ttl7.off()  # Microwave off when ttl on
                    delay(5 * us)
                    t_RF = now_mu()
                    if t_RF > 0:
                        at_mu(t_RF)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + self.t01)
                        self.urukul0_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01)
                        self.ttl7.on()  # Microwave off when ttl on

                        at_mu(t_RF + 100 + self.t01 + 1000)
                        self.urukul0_ch0.set(frequency= self.f11 * MHz, amplitude=0.7);

                        at_mu(t_RF + 100 + self.t01 + 11000)
                        self.ttl7.off()  # Microwave off when ttl on
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000 + int(param))
                        self.urukul0_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4100 + int(param))
                        self.ttl7.on()  # Microwave off when ttl on

                elif self.Time_2ph_Scan :
                    # mf0to1 transition start
                    self.urukul0_ch0.set(frequency= self.f01 * MHz, amplitude=0.7);
                    delay(5 * us)
                    self.ttl7.off()  # Microwave off when ttl on
                    delay(5 * us)
                    t_RF = now_mu()
                    if t_RF > 0:
                        at_mu(t_RF)
                        self.urukul0_ch0.sw.on()
                        at_mu(t_RF + self.t01)
                        self.urukul0_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01)
                        self.ttl7.on()  # Microwave off when ttl on
                    # mf0to1 transition end
                        # 2ph frequency scan start
                        at_mu(t_RF + 100 + self.t01 + 1000)
                        with parallel:
                            self.urukul0_ch0.set(frequency = self.fm10_2ph * MHz, amplitude=0.7)
                            self.urukul1_ch0.set(frequency= self.rf_2ph * MHz, amplitude=0.6); # Scanning the frequency
                        # self.urukul1_ch0.set(frequency=param * MHz, amplitude=1.0);
                        at_mu(t_RF + 100 + self.t01 + 11000)
                        self.ttl7.off()  # Microwave off when ttl on
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000)
                        with parallel :
                            self.urukul0_ch0.sw.on()
                            self.urukul1_ch0.sw.on()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4000 + int(param))
                        with parallel :
                            self.urukul0_ch0.sw.off()
                            self.urukul1_ch0.sw.off()
                        at_mu(t_RF + 100 + self.t01 + 11000 + 4100 + int(param))
                        self.ttl7.on()  # Microwave off when ttl on
                        # 2ph frequency scan end


                elif self.Ramsey_00_Scan :

                    # self.urukul0_ch0.set(frequency= (self.f00 - 0.01 + 0.012) * MHz, amplitude=0.7);
                    self.urukul0_ch0.set(frequency=(self.f00 + 0.03) * MHz, amplitude=0.7);
                    t_RF = now_mu()
                    at_mu(t_RF + 100)
                    self.ttl7.off()  # Microwave on when ttl off
                    at_mu(t_RF + 4000)
                    self.urukul0_ch0.sw.on()
                    at_mu(t_RF + 4000 + int(self.t00 / 2))  #pi/2 pulse
                    self.urukul0_ch0.sw.off()
                    # at_mu(t_RF + 4100 + 24600)
                    # self.ttl7.on()  # Microwave off when ttl on
                    at_mu(t_RF + 4100 + int(self.t00 / 2) + int(param))    #Ramsey Delay
                    self.urukul0_ch0.sw.on()
                    at_mu(t_RF + 4100 + int(self.t00 / 2) + int(param) + int(self.t00 / 2)) #second pi/2 pulse
                    self.urukul0_ch0.sw.off()
                    at_mu(t_RF + 4100 + int(self.t00 / 2) + int(param) + int(self.t00 / 2) + 100)
                    self.ttl7.on()

                elif self.Ramsey_11_Scan :

                    self.urukul0_ch0.set(frequency=self.f01 * MHz, amplitude=0.7);
                    t_RF = now_mu()
                    at_mu(t_RF + 100)
                    self.ttl7.off()  # Microwave off when ttl on
                    at_mu(t_RF + 4000)
                    self.urukul0_ch0.sw.on()
                    at_mu(t_RF + 4000 + int(self.t01))
                    self.urukul0_ch0.sw.off()
                    at_mu(t_RF + 4100 + int(self.t01))
                    self.ttl7.on()  # Microwave off when ttl on

                    at_mu(t_RF + 4100 + int(self.t01) + 1000)
                    self.urukul0_ch0.set(frequency=(self.f11 - 0.03) * MHz, amplitude=0.7);
                    at_mu(t_RF + 4100 + int(self.t01) + 2000)
                    self.ttl7.off()  # Microwave off when ttl on
                    at_mu(t_RF + 4100 + int(self.t01) + 2000 + 1000)
                    self.urukul0_ch0.sw.on()  # pi/2 pulse
                    at_mu(t_RF + 4100 + int(self.t01) + 2000 + 1000 + int(self.t11 / 2))
                    self.urukul0_ch0.sw.off()  # delay for ramsey fringe
                    at_mu(t_RF + 4100 + int(self.t01) + 2000 + 1000 + int(self.t11 / 2) + int(param))
                    self.urukul0_ch0.sw.on()  # second pi/2 pulse
                    at_mu(t_RF + 4100 + int(self.t01) + 2000 + 1000 + int(self.t11 / 2) + int(param) + int(self.t11 / 2))
                    self.urukul0_ch0.sw.off()
                    at_mu(t_RF + 4100 + int(self.t01) + 2000 + 1000 + int(self.t11 / 2) + int(param) + int(self.t11 / 2))
                    self.ttl7.on()  # Microwave off when ttl on
                    # ## Microwave scan mF=1 to mF=1 #

                elif self.Ramsey_2ph_Scan :

                    self.urukul0_ch0.set(frequency=self.f01 * MHz, amplitude=0.7);
                    t_RF = now_mu()
                    at_mu(t_RF + 100)
                    self.ttl7.off()  # Microwave off when ttl on
                    at_mu(t_RF + 4000)
                    self.urukul0_ch0.sw.on()
                    at_mu(t_RF + 4000 + int(self.t01))
                    self.urukul0_ch0.sw.off()
                    at_mu(t_RF + 4100 + int(self.t01))
                    self.ttl7.on()  # Microwave off when ttl on
                    # map mf1 to mf1 #

                    # two-photon ramsey sequence #
                    at_mu(t_RF + 4100 + int(self.t01) + 1000)
                    self.urukul0_ch0.set(frequency=self.fm10_2ph * MHz, amplitude=0.7)
                    at_mu(t_RF + 4100 + int(self.t01) + 2000)
                    self.urukul1_ch0.set(frequency=(self.rf_2ph - 0.0002) * MHz, amplitude=0.6)

                    at_mu(t_RF + 4100 + int(self.t01) + 3000)
                    self.ttl7.off()  # Microwave off when ttl on
                    at_mu(t_RF + 4100 + int(self.t01) + 4000)
                    with parallel:
                        self.urukul0_ch0.sw.on()
                        self.urukul1_ch0.sw.on()
                    at_mu(t_RF + 4100 + int(self.t01) + 4000 + int(self.t2ph / 2))
                    with parallel:
                        self.urukul0_ch0.sw.off()
                        self.urukul1_ch0.sw.off()

                    at_mu(t_RF + 4100 + int(self.t01) + 4000 + int(self.t2ph / 2) + int(param))
                    with parallel:
                        self.urukul0_ch0.sw.on()
                        self.urukul1_ch0.sw.on()
                    at_mu(t_RF + 4100 + int(self.t01) + 4000 + int(self.t2ph / 2) + int(param) + int(self.t2ph / 2))
                    with parallel:
                        self.urukul0_ch0.sw.off()
                        self.urukul1_ch0.sw.off()

                    at_mu(t_RF + 4200 + int(self.t01) + 4000 + int(self.t2ph / 2) + int(param) + int(self.t2ph / 2))
                    self.ttl7.on()

                #Blowaway start#
                # self.zotino0.write_dac(0, self.X_off)
                # self.zotino0.write_dac(1, self.Y_off)
                # self.zotino0.write_dac(2, self.Z1_off)
                # self.zotino0.write_dac(3, self.Z2_off)
                # self.zotino0.load()
                # delay(9 * ms)

                # Blowaway one retro-beam
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

                # t_blow = now_mu()
                #
                # at_mu(t_blow + 1000)
                # with parallel:
                #     self.urukul0_ch1.sw.on()
                #     self.urukul1_ch3.sw.on()
                #     self.urukul2_ch0.sw.off()
                #     self.urukul2_ch1.sw.off()
                #     self.urukul2_ch2.sw.on()
                #
                # at_mu(t_blow + 1000 + 500000)
                # with parallel:
                #     self.urukul0_ch1.sw.off()
                #     self.urukul2_ch0.sw.on()
                #     self.urukul2_ch1.sw.on()

                self.urukul0_ch1.set(frequency=130 * MHz, amplitude=0.35);
                self.urukul2_ch2.set(frequency=80 * MHz, amplitude=amp2);
                delay(10 * us)

                delay(1 * us)
                self.zotino0.write_dac(0, self.X_off)
                self.zotino0.write_dac(1, self.Y_off)
                self.zotino0.write_dac(2, self.Z1_off)
                self.zotino0.write_dac(3, self.Z2_off)
                self.zotino0.load()
                delay(1 * ms)

                # trap back
                self.ramp_trap_amp(trap_ramp_end_amp, self.trap_ramp_start_amp, self.trap_ramp_steps, self.trap_ramp_dt_us)
                delay(1 * ms)


                # PGC cooling after blowaway start
                self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                self.urukul0_ch1.sw.on()
                delay(1 * us)
                self.ttl5.off()
                delay(1 * us)
                self.urukul0_ch2.sw.on()
                delay(10 * ms)

                self.urukul0_ch1.sw.off()
                delay(1 * us)
                self.ttl5.on()
                delay(1 * us)
                # PGC cooling after blowaway end

                self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                self.urukul0_ch1.sw.on()
                delay(1 * us)
                self.urukul0_ch2.set(frequency=197 * MHz, amplitude=0.5);
                self.ttl5.off()

                delay(1 * us)

                self.zotino0.write_dac(6, 4.1)  # 2 shot
                self.zotino0.load()
                delay(self.trig_width)
                self.zotino0.write_dac(6, 0.0)
                self.zotino0.load()
                delay(1 * us)
                tend2 = self.ttl0.gate_rising(self.t2)
                count2 = self.ttl0.count(tend2)
                delay(10 * us)

                self.zotino0.write_dac(0, 0.0)  # ShimcoilX back to MOT
                self.zotino0.write_dac(1, 0.0)  # ShimcoilY back to MOT
                self.zotino0.write_dac(2, 0.0)  # ShimcoilZ1 back to MOT
                self.zotino0.write_dac(3, 0.0)  # ShimcoilZ2 back to MOT
                self.zotino0.write_dac(6, 0.0)
                self.zotino0.load()
                delay(1 * ms)

                delay(1 * us); self.urukul1_ch1.sw.off()
                delay(10000 * us); self.urukul1_ch1.sw.on(); delay(1 * us)
                tend_bg = self.ttl0.gate_rising(self.t2)
                count_bg = self.ttl0.count(tend_bg)
                delay(10 * us)

                delay(1 * us)

                # host logging
                delay(1 * ms)
                self.write_counts(count0, count1, count2, count_bg)
                delay(10 * ms)
                delay(2000 * us)
                self.urukul0_ch1.set(frequency=112 * MHz, amplitude=0.35);
                delay(2000 * us)
                self.urukul0_ch1.sw.on()
                delay(1000 * us)
                # wait until 1 s after t_start before next loop
                self.ttl4.on()
                delay(1000 * us)

                self.urukul0_ch0.sw.off()
                delay(1 * us)
                self.urukul1_ch0.sw.off()
                delay(1 * us)
                self.ttl7.on()
                delay(1 * us)