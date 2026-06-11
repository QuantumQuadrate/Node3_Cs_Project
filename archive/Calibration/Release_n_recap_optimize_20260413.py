from artiq.experiment import *
from artiq.language.core import kernel, rpc
from artiq.language.types import TInt32
from aom_feedback_final import FeedbackRebalancer, REF_PATH
import os, time, csv

class Release_n_recap_Optimize_scan(EnvExperiment):
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

        # UI args
        self.setattr_argument("num_runs", NumberValue(100, min=1, max=600000, step=1))

        self.setattr_argument("Time_scan_min", NumberValue(0.0, min=0.0, max=1000.0))
        self.setattr_argument("Time_scan_max", NumberValue(150.0, min=0.0, max=1000.0))
        self.setattr_argument("Time_scan_step", NumberValue(150.0, min=0.0, max=1000.0))

        self.setattr_argument("PGC_2_off_min", NumberValue(30.0, min=0.0, max=10000.0))
        self.setattr_argument("PGC_2_off_max", NumberValue(300.0, min=0.0, max=10000.0))
        self.setattr_argument("PGC_2_off_step", NumberValue(30.0, min=0.0, max=10000.0))




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


        # Rebalancer params
        self.p_gain_0 = 0.08
        self.p_gain_1 = 0.08
        self.p_gain_2 = 0.08
        self.tol_0 = 0.01
        self.tol_1 = 0.01
        self.tol_2 = 0.01
        self.iterations_0 = 20
        self.iterations_1 = 20
        self.iterations_2 = 20
        self.initial_amp_0 = 0.13
        self.initial_amp_1 = 0.11
        self.initial_amp_2 = 0.08
        self.frequency_0 = 80 * MHz
        self.frequency_1 = 80 * MHz
        self.frequency_2 = 80 * MHz

        # AOM set for rebalancer (we already setattr_device'd in build)
        # self.aoms = [self.urukul2_ch0, self.urukul2_ch1, self.urukul2_ch2]

        # Helper
        self.rebalancer = FeedbackRebalancer()
        self.rebalancer.core = self.core
        self.rebalancer.sampler0 = self.sampler0
        self.rebalancer.urukul2_ch0 = self.urukul2_ch0
        self.rebalancer.urukul2_ch1 = self.urukul2_ch1
        self.rebalancer.urukul2_ch2 = self.urukul2_ch2

        # Reference file
        if REF_PATH.exists():
            vals = [float(x) for x in REF_PATH.read_text().split()]
            self.stored_refs = (vals + [-1, -1, -1])[:3]
        else:
            self.stored_refs = [-1, -1, -1]

        # Datasets for plotting
        # self.set_dataset("x_axis", [], broadcast=True)
        self.set_dataset("spcm_counts1", [], broadcast=True)
        self.set_dataset("spcm_counts2", [], broadcast=True)

        v1, s1 = self.Time_scan_min, self.Time_scan_step
        n1 = int(round((self.Time_scan_max - v1) / s1)) + 1
        self.time_scan = [v1 + i * s1 for i in range(n1)]

        v2, s2 = self.PGC_2_off_min, self.PGC_2_off_step
        n2 = int(round((self.PGC_2_off_max - v2) / s2)) + 1
        self.PGC_2_off_scan = [v2 + i * s2 for i in range(n2)]




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
        delay(1 * us)

        delay(1000 * us)
        self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3);
        self.urukul0_ch1.sw.on()
        delay(100 * us)

        delay(100 * us)
        # Rebalance (RPC inside kernel is fine if the method is @rpc)
        self.rebalancer.rebalance(
            self.p_gain_0, self.p_gain_1, self.p_gain_2,
            self.tol_0, self.tol_1, self.tol_2,
            int(self.iterations_0),
            int(self.iterations_1),
            int(self.iterations_2),
            # always use these refs:
            self.stored_refs[0],
            self.stored_refs[1],
            self.stored_refs[2],
            # last-known amps & freqs:
            self.initial_amp_0,
            self.initial_amp_1,
            self.initial_amp_2,
            self.frequency_0,
            self.frequency_1,
            self.frequency_2,
        )
        delay(1 * us)

        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            for param2 in self.PGC_2_off_scan :
                self.t3 = int(param2) * ms
                for param in self.time_scan:
                    self.ttl4.on()
                    delay(1 * us)
                    self.ttl5.on()  # Microwave off when ttl on
                    delay(1 * us)

                    delay(10 * us)
                    self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3);
                    self.urukul0_ch1.sw.on()
                    delay(10 * us)

                    # delay(100 * us)
                    # Rebalance (RPC inside kernel is fine if the method is @rpc)
                    self.rebalancer.rebalance(
                        self.p_gain_0, self.p_gain_1, self.p_gain_2,
                        self.tol_0, self.tol_1, self.tol_2,
                        int(self.iterations_0),
                        int(self.iterations_1),
                        int(self.iterations_2),
                        # always use these refs:
                        self.stored_refs[0],
                        self.stored_refs[1],
                        self.stored_refs[2],
                        # last-known amps & freqs:
                        self.initial_amp_0,
                        self.initial_amp_1,
                        self.initial_amp_2,
                        self.frequency_0,
                        self.frequency_1,
                        self.frequency_2,
                    )
                    delay(1 * us)

                    self.urukul0_ch1.set(frequency=120 * MHz, amplitude=0.36);
                    self.urukul0_ch1.sw.on()
                    delay(100 * us)

                    self.urukul0_ch2.set(frequency=197 * MHz, amplitude=0.5);
                    self.ttl5.off()
                    delay(10 * us)

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
                    ##PGC after imaging##


                    delay(1 * us)


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
                    self.write_counts(int(shot1), int(shot2), int(param), int(param2))




                # self.urukul1_ch1.sw.off()
                # delay(500 * ms)
