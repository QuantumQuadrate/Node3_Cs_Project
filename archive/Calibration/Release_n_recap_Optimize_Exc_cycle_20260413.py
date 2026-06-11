from artiq.experiment import *
from artiq.language.core import kernel, rpc
from artiq.language.types import TInt32
#from aom_feedback_final import FeedbackRebalancer, REF_PATH
from aom_feedback_final_5channel import FeedbackRebalancer_5, REF_PATH
import os, time, csv

class Release_n_recap_Optim_Exc_cycle_scan(EnvExperiment):
    def build(self):
        self.base_dir = r"C:\Networking Experiment\artiq codes\artiq-master\repository\SPCM_CSV"
        ts = time.strftime("%Y%m%d_%H%M")
        self.csv_fname = os.path.join(self.base_dir,
                                      f"Release_recap_exp_{ts}.csv")
        # Devices

        self.core = self.get_device("core")
        self.setattr_device("zotino0")
        self.setattr_device("urukul0_ch0")  # Microwave_IF
        self.setattr_device("urukul0_ch1")  # Cooling AOM 1
        self.setattr_device("urukul0_ch2")  # Repump AOM
        self.setattr_device("urukul0_ch3")  # D1 first AOM
        # self.setattr_device("urukul1_ch0")  # Exitation
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
        self.setattr_device("ttl6")  # Excitation 1st aom currently novatech previously urukul1_ch0
        self.setattr_device("ttl12")

        # UI args
        self.setattr_argument("num_runs", NumberValue(100, min=1, max=600000, step=1))

        self.setattr_argument("Time_scan_min", NumberValue(0.0, min=0.0, max=1000.0))
        self.setattr_argument("Time_scan_max", NumberValue(150.0, min=0.0, max=1000.0))
        self.setattr_argument("Time_scan_step", NumberValue(150.0, min=0.0, max=1000.0))

        self.setattr_argument("Exc_cyc_min", NumberValue(30.0, min=0.0, max=10000.0))
        self.setattr_argument("Exc_cyc_max", NumberValue(300.0, min=0.0, max=10000.0))
        self.setattr_argument("Exc_cyc_step", NumberValue(30.0, min=0.0, max=10000.0))

        self.setattr_argument("trap_ramp_start_amp", NumberValue(0.24, min=0.000, max=0.300, ndecimals=3),
                              group="Adiabatic parameter")
        # self.setattr_argument("trap_ramp_end_amp", NumberValue(0.125, min=0.000, max=0.300, ndecimals=3), group="Adiabatic parameter")
        self.setattr_argument("trap_ramp_steps", NumberValue(200, min=0, max=100000, ndecimals=1),
                              group="Adiabatic parameter")
        self.setattr_argument("trap_ramp_dt_us", NumberValue(50, min=0.001, max=1000, ndecimals=3),
                              group="Adiabatic parameter")

        self.setattr_argument("t_trap_off_mu", NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("t_spcm_collection_mu", NumberValue(400, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")
        self.setattr_argument("length_offset_mu", NumberValue(12, min=0, max=8000, ndecimals=0, step=1),
                              group="Excitation")

    @rpc(flags={"async"})
    def write_counts(self, c0, c1, time, PARAM):
        """
        Host-side RPC: append one row of [shot0, shot1, shot2, background]
        to the timestamped CSV created in build().
        """

        fname = self.csv_fname
        is_new = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["shot1", "shot2", "delay", "MOT_2_PGC(ms)"])
            writer.writerow([c0, c1, time, PARAM])


    def prepare(self):
        # Timing & controls
        # self.t2        = 50 * ms
        self.t1 = 1 * us  # first trigger
        self.t2 = 4 * ms  # RF off
        self.t3 = 1 * ms  # RF on + second trigger
        self.exp = 33 * ms  # EMCCD exposure
        self.trig_width = 5 * ms  # Zotino high time
        self.period = 1 * s  # full cycle period
        self.MOT2PGC = 350 * ms
        self.PGC_2_off = 1 * ms
        self.PGC_Af_Img = 1 * ms
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
        self.hist_thres = 12  # threshold for atom vs no-atom separation

        # AOM set for rebalancer (we already setattr_device'd in build)
        # self.aoms = [self.urukul2_ch0, self.urukul2_ch1, self.urukul2_ch2]

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

        # Reference file
        # if REF_PATH.exists():
        #     vals = [float(x) for x in REF_PATH.read_text().split()]
        #     self.stored_refs = (vals + [-1, -1, -1])[:3]
        # else:
        #     self.stored_refs = [-1, -1, -1]
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

        v2, s2 = self.Exc_cyc_min, self.Exc_cyc_step
        n2 = int(round((self.Exc_cyc_max - v2) / s2)) + 1
        self.Exc_cyc_scan = [v2 + i * s2 for i in range(n2)]

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
            self.urukul1_ch1.set(frequency=150 * MHz, amplitude=amp_q)

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
        # self.urukul1_ch0.cpld.init()
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
        # self.urukul1_ch0.init()
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
        # self.urukul1_ch0.set(frequency=220 * MHz, amplitude=0.1);     #replaced with ttl6.on() above
        # self.urukul1_ch0.sw.off()
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

       # #faisal added:
       #  delay(1 * us)
       #  self.urukul1_ch0.sw.off()
       #  delay(1 * us)
       #  #faisal added end

        # delay(100 * us)
        # Rebalance (RPC inside kernel is fine if the method is @rpc)
        # self.rebalancer.rebalance(
        #     self.p_gain_0, self.p_gain_1, self.p_gain_2,
        #     self.tol_0, self.tol_1, self.tol_2,
        #     int(self.iterations_0),
        #     int(self.iterations_1),
        #     int(self.iterations_2),
        #     # always use these refs:
        #     self.stored_refs[0],
        #     self.stored_refs[1],
        #     self.stored_refs[2],
        #     # last-known amps & freqs:
        #     self.initial_amp_0,
        #     self.initial_amp_1,
        #     self.initial_amp_2,
        #     self.frequency_0,
        #     self.frequency_1,
        #     self.frequency_2,
        # )
        # delay(1 * us)


        # self.core.break_realtime()

        for i in range(int(self.num_runs)):
            for exc in self.Exc_cyc_scan :
                for param in self.time_scan:

                    # self.ttl4.on()
                    # delay(1 * us)
                    # self.ttl5.on()  # Microwave off when ttl on
                    # delay(1 * us)
                    #
                    # delay(10 * us)
                    # self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3);
                    # self.urukul0_ch1.sw.on()
                    # delay(10 * us)

                    # delay(100 * us)
                    # Rebalance (RPC inside kernel is fine if the method is @rpc)
                    # self.rebalancer.rebalance(
                    #     self.p_gain_0, self.p_gain_1, self.p_gain_2,
                    #     self.tol_0, self.tol_1, self.tol_2,
                    #     int(self.iterations_0),
                    #     int(self.iterations_1),
                    #     int(self.iterations_2),
                    #     # always use these refs:
                    #     self.stored_refs[0],
                    #     self.stored_refs[1],
                    #     self.stored_refs[2],
                    #     # last-known amps & freqs:
                    #     self.initial_amp_0,
                    #     self.initial_amp_1,
                    #     self.initial_amp_2,
                    #     self.frequency_0,
                    #     self.frequency_1,
                    #     self.frequency_2,
                    # )
                    # delay(1 * us)
                    #
                    # self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                    # self.urukul0_ch1.sw.on()
                    # delay(100 * us)
                    #
                    # self.urukul0_ch2.set(frequency=197 * MHz, amplitude=0.5);
                    # self.ttl5.off()
                    # delay(10 * us)

                    # faisal added:
                    self.core.break_realtime()
                    delay(100 * us)
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
                        self.initial_amp_0, self.initial_amp_1, self.initial_amp_2, self.initial_amp_3,
                        self.initial_amp_4,
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
                    #faisal added end

                    self.urukul1_ch1.set(frequency=150 * MHz, amplitude=self.nominal_trapdepth);
                    self.urukul1_ch1.sw.on()
                    delay(100 * us)

                    delay(self.wait)

                    self.zotino0.write_dac(0, self.X_idle)
                    self.zotino0.write_dac(1, self.Y_idle)
                    self.zotino0.write_dac(2, self.Z1_idle)
                    self.zotino0.write_dac(3, self.Z2_idle)
                    self.zotino0.load()

                    self.urukul0_ch1.set(frequency=112 * MHz, amplitude=0.3);
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
                    delay(10 * ms)

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

                    # delay(1 * us)
                    # self.zotino0.write_dac(6, 4.1)  # 1 shot
                    # self.zotino0.load()
                    # delay(self.trig_width)
                    # self.zotino0.write_dac(6, 0.0)
                    # self.zotino0.load()
                    delay(1 * us)
                    with parallel:
                        t_end0 = self.ttl0.gate_rising(self.t2)
                        t_end1 = self.ttl1.gate_rising(self.t2)
                    delay(1 * us)
                    # delay(600 * us)
                    c0 = self.ttl0.count(t_end0)
                    delay(1 * ms)
                    c1 = self.ttl1.count(t_end1)
                    delay(1 * ms)
                    shot1 = c0 + c1

                    ##PGC after imaging##
                    delay(1 * us)
                    self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                    self.urukul0_ch1.sw.on()
                    delay(self.PGC_Af_Img)
                    self.urukul0_ch1.sw.off()
                    self.ttl5.on()
                    delay(1*us)
                    # Q-axis on
                    self.zotino0.write_dac(0, -0.013)
                    self.zotino0.write_dac(1, 2.72)
                    self.zotino0.write_dac(2, -0.06)  # ShimcoilZ1 back to MOT
                    self.zotino0.write_dac(3, 0.06)  # ShimcoilZ2 back to MOT
                    self.zotino0.load()
                    delay(1 * ms)

                    # Trap adiabatic down
                    self.ramp_trap_amp(self.trap_ramp_start_amp, trap_ramp_end_amp, self.trap_ramp_steps,
                                       self.trap_ramp_dt_us)  # Takes ~ 10 ms
                    # delay(1 * ms)
                    ##PGC after imaging##

                    # Pumping and excitation cycle

                    for i in range(int(exc)) :

                        self.urukul2_ch0.sw.on()
                        delay(1 * us)
                        self.urukul2_ch1.sw.on()
                        delay(1 * us)
                        self.urukul2_ch2.sw.on()
                        delay(1 * us)

                        # trials += 1
                        # SPCM1_click_counter = 0
                        # SPCM2_click_counter = 0

                        # slack = now_mu() - self.core.get_rtio_counter_mu()
                        # if slack < 1e5:
                        #     # self.print_async("slack added in measurement:", self.measurement)
                        #     self.core.break_realtime()

                        self.urukul0_ch0.set(frequency=250.966 * MHz, amplitude=0.7);  ##UPDATE 11 RES HERE!!
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

                        self.urukul1_ch2.set(frequency=145 * MHz, amplitude=0.13);

                        delay(15 * us)

                        t0 = now_mu()

                        # packing
                        at_mu(t0)
                        self.ttl7.off()
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
                            self.ttl7.on()

                        at_mu(t0 + 3800)
                        self.urukul1_ch2.sw.on()

                        at_mu(t0 + 3800 + 300)
                        self.urukul1_ch2.sw.off()
                        #
                        # at_mu(t0 + 4500 + 780)
                        # with parallel:
                        #     t_end = self.ttl0.gate_rising(self.t_spcm_collection_mu * ns)
                        #     t_end_2 = self.ttl1.gate_rising(self.t_spcm_collection_mu * ns)

                        # SPCM1_click_time = self.ttl0.timestamp_mu(t_end)
                        # SPCM2_click_time = self.ttl1.timestamp_mu(t_end_2)
                    # Recooling
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

                    # PGC
                    self.urukul1_ch3.sw.on()
                    delay(1 * us)
                    self.urukul2_ch0.sw.on()
                    delay(1 * us)
                    self.urukul2_ch1.sw.on()
                    delay(1 * us)
                    self.urukul2_ch2.sw.on()
                    delay(1 * us)
                    self.urukul0_ch1.set(frequency=128 * MHz, amplitude=0.35);
                    self.urukul0_ch1.sw.on()
                    delay(1 * us)
                    self.urukul0_ch2.sw.on()
                    delay(1 * us)
                    self.ttl5.off()
                    delay(1 * us)
                    delay(self.PGC_Af_Img)
                    # delay(2 * ms)
                    self.urukul0_ch1.sw.off()
                    self.ttl5.on()
                    delay(1 * us)
                    ##PGC after imaging

                    # Check Image (Survival Check)

                    ##release and recapture##
                    self.urukul1_ch1.sw.off()
                    delay(int(param) * us)
                    self.urukul1_ch1.sw.on()
                    ##release and recapture##


                    delay(1 * us)
                    self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                    self.urukul0_ch1.sw.on()
                    self.ttl5.off()
                    # delay(1 * us)
                    # self.zotino0.write_dac(6, 4.1)  # 2 shot
                    # self.zotino0.load()
                    # delay(self.trig_width)
                    # self.zotino0.write_dac(6, 0.0)
                    # self.zotino0.load()
                    with parallel:
                        t_end0 = self.ttl0.gate_rising(self.t2)
                        t_end1 = self.ttl1.gate_rising(self.t2)
                    delay(1 * us)
                    # delay(600 * us)
                    c0 = self.ttl0.count(t_end0)
                    delay(1 * ms)
                    c1 = self.ttl1.count(t_end1)
                    delay(1 * ms)
                    shot2 = c0 + c1
                    # self.write_counts(int(shot1), int(shot2), int(param), int(param2))
                    self.write_counts(int(shot1), int(shot2), int(param), int(exc))




                # self.urukul1_ch1.sw.off()
                # delay(500 * ms)
