# feedback_pid_experiment.py
# Standalone ARTIQ-8 experiment: three-channel AOM power stabilizer without JSON.
# Persists reference voltages across runs by reading/writing a local file,
# updates file only on fresh measurements, and prints initial amplitude and voltages.

from pathlib import Path
from artiq.experiment import *

# Path to store five reference voltages (space-separated floats)
REF_PATH = Path(__file__).parent / "ref_voltages_5.txt"


class FeedbackPIDExperiment_5channel_revisedJan26(EnvExperiment):
    def build(self):
        # devices for kernels
        self.setattr_device("core")
        self.setattr_device("sampler0")
        self.setattr_device("urukul2_ch0")
        self.setattr_device("urukul2_ch1")
        self.setattr_device("urukul2_ch2")
        self.setattr_device("urukul1_ch1")  # Trap
        self.setattr_device("urukul1_ch2")  # Excitation
        self.setattr_device("urukul1_ch3")
        self.setattr_device("urukul0_ch1")
        self.setattr_device("ttl14")
        self.setattr_device("ttl6")
        self.setattr_device("ttl5")
        # Human-readable labels for channels
        self._chan_labels = ["AOM0", "AOM1", "AOM2", "Trap", "Excitation"]

        # ------------- AOM / Trap / Excitation settings -------------
        for i in range(5):
            if i < 3:
                group_name = f"AOM{i} Settings"
            elif i == 3:
                group_name = "Trap Settings"
            else:
                group_name = "Excitation Settings"

            self.setattr_argument(
                f"initial_amp_{i}",
                NumberValue(default=0.5, step=0.01, min=0.0, max=1.0),
                group=group_name
            )
            self.setattr_argument(
                f"frequency_{i}",
                NumberValue(default=80e6, unit="Hz", step=1e6,
                            min=0.0, max=400e6),
                group=group_name

            )

            self.setattr_argument(
                f"p_gain_{i}",
                NumberValue(default=0.1, step=0.01, min=0.0, max=1.0, ndecimals=4),
                group=group_name
            )
            self.setattr_argument(
                f"tol_{i}",
                NumberValue(default=0.01, step=0.005, min=0.0, max=1.0, ndecimals=4),
                group=group_name
            )
            # Force this to be an integer in ARTIQ-7
            self.setattr_argument(
                f"iterations_{i}",
                NumberValue(
                    default=10,
                    step=1,
                    min=1,
                    max=100,
                    ndecimals=0,
                    type="int"
                ),
                group=group_name
            )

            self.setattr_argument(
                f"use_stored_ref_{i}",
                BooleanValue(default=True),
                group=group_name
            )


    def prepare(self):
        # Host-side file I/O
        if REF_PATH.exists():
            try:
                vals = [float(x) for x in REF_PATH.read_text().strip().split()]
                # pad/truncate to 5 entries to support old 3-value files as well
                self.stored_refs = (vals + [-1.0, -1.0, -1.0, -1.0, -1.0])[:5]
            except Exception:
                self.stored_refs = [-1.0, -1.0, -1.0, -1.0, -1.0]
        else:
            self.stored_refs = [-1.0, -1.0, -1.0, -1.0, -1.0]

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        self.ttl14.on()
        self.ttl6.on()
        self.ttl5.on()
        delay(1000 * us)
        self.urukul1_ch3.set(frequency=250 * MHz, amplitude=0.43);
        self.urukul1_ch3.sw.on()
        delay(1 * us)
        self.urukul0_ch1.set(frequency=122 * MHz, amplitude=0.3);
        self.urukul0_ch1.sw.on()
        delay(1 * us)
        self.urukul1_ch1.sw.off()
        delay(1 * us)
        self.ttl6.off()
        delay(1 * us)
        self.ttl14.off()
        delay(10*ms)



        # ---------- Turn on each channel ----------
        # 0–2: generic AOMs on urukul2
        self.urukul2_ch0.set(frequency=self.frequency_0,
                             amplitude=self.initial_amp_0)
        self.urukul2_ch0.sw.on()

        self.urukul2_ch1.set(frequency=self.frequency_1,
                             amplitude=self.initial_amp_1)
        self.urukul2_ch1.sw.on()

        self.urukul2_ch2.set(frequency=self.frequency_2,
                             amplitude=self.initial_amp_2)
        self.urukul2_ch2.sw.on()

        # 3: Trap (urukul1_ch1)
        self.urukul1_ch1.set(frequency=self.frequency_3,
                             amplitude=self.initial_amp_3)
        self.urukul1_ch1.sw.on()

        # 4: Excitation (urukul1_ch2)
        self.ttl14.off()
        self.ttl6.off()
        self.urukul1_ch2.set(frequency=self.frequency_4,
                             amplitude=self.initial_amp_4)
        self.urukul1_ch2.sw.on()

        # Init sampler (recommended on 7)
        self.sampler0.init()
        delay(10*ms)

        # ---------- capture initial amps & voltages ----------
        init_amps = [0.0]*5
        init_amps[0] = self.initial_amp_0
        init_amps[1] = self.initial_amp_1
        init_amps[2] = self.initial_amp_2
        init_amps[3] = self.initial_amp_3
        init_amps[4] = self.initial_amp_4

        init_vs = [0.0]*5
        buf = [0]*8

        self.sampler0.sample_mu(buf)
        delay(1*ms)
        init_vs[0] = buf[0] * (20.0/65536.0)
        init_vs[1] = buf[1] * (20.0/65536.0)
        init_vs[2] = buf[2] * (20.0/65536.0)
        init_vs[3] = buf[3] * (20.0/65536.0)
        init_vs[4] = buf[4] * (20.0/65536.0)

        # ---------- references ----------
        refs = [0.0]*5
        fresh_any = False

        # ch0 reference (sampler ch0)
        if self.use_stored_ref_0 and (self.stored_refs[0] >= 0.0):
            refs[0] = self.stored_refs[0]
        else:
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            refs[0] = buf[0] * (20.0/65536.0)
            fresh_any = True

        # ch1 reference (sampler ch1)
        if self.use_stored_ref_1 and (self.stored_refs[1] >= 0.0):
            refs[1] = self.stored_refs[1]
        else:
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            refs[1] = buf[1] * (20.0/65536.0)
            fresh_any = True

        # ch2 reference (sampler ch2)
        if self.use_stored_ref_2 and (self.stored_refs[2] >= 0.0):
            refs[2] = self.stored_refs[2]
        else:
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            refs[2] = buf[2] * (20.0/65536.0)
            fresh_any = True

        # ch3 reference (Trap, sampler ch3)
        if self.use_stored_ref_3 and (self.stored_refs[3] >= 0.0):
            refs[3] = self.stored_refs[3]
        else:
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            refs[3] = buf[3] * (20.0/65536.0)
            fresh_any = True

        # ch4 reference (Excitation, sampler ch4)
        if self.use_stored_ref_4 and (self.stored_refs[4] >= 0.0):
            refs[4] = self.stored_refs[4]
            fresh_any = fresh_any or False
        else:
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            refs[4] = buf[4] * (20.0/65536.0)
            fresh_any = True

        # ---------- P-only feedback for each channel ----------
        final_vs = [0.0]*5

        # channel 0 – AOM0 (urukul2_ch0, sampler0 ch0)
        amp = self.initial_amp_0
        ref_v = refs[0]
        meas_v = 0.0
        for _ in range(self.iterations_0):
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            meas_v = buf[0] * (20.0/65536.0)
            if (ref_v > 0.0) and (abs(meas_v - ref_v)/ref_v < self.tol_0):
                break
            amp = max(0.0, min(0.17, amp + self.p_gain_0*(ref_v - meas_v)))
            self.urukul2_ch0.set(frequency=self.frequency_0,
                                 amplitude=amp)
            delay(1*ms)
        final_vs[0] = meas_v
        a0 = amp

        # channel 1 – AOM1 (urukul2_ch1, sampler0 ch1)
        amp = self.initial_amp_1
        ref_v = refs[1]
        meas_v = 0.0
        for _ in range(self.iterations_1):
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            meas_v = buf[1] * (20.0/65536.0)
            if (ref_v > 0.0) and (abs(meas_v - ref_v)/ref_v < self.tol_1):
                break
            amp = max(0.0, min(0.17, amp + self.p_gain_1*(ref_v - meas_v)))
            self.urukul2_ch1.set(frequency=self.frequency_1,
                                 amplitude=amp)
            delay(1*ms)
        final_vs[1] = meas_v
        a1 = amp

        # channel 2 – AOM2 (urukul2_ch2, sampler0 ch2)
        amp = self.initial_amp_2
        ref_v = refs[2]
        meas_v = 0.0
        for _ in range(self.iterations_2):
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            meas_v = buf[2] * (20.0/65536.0)
            if (ref_v > 0.0) and (abs(meas_v - ref_v)/ref_v < self.tol_2):
                break
            amp = max(0.0, min(0.17, amp + self.p_gain_2*(ref_v - meas_v)))
            self.urukul2_ch2.set(frequency=self.frequency_2,
                                 amplitude=amp)
            delay(1*ms)
        final_vs[2] = meas_v
        a2 = amp

        # channel 3 – Trap (urukul1_ch1, sampler0 ch3)
        amp = self.initial_amp_3
        ref_v = refs[3]
        meas_v = 0.0
        for _ in range(self.iterations_3):
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            meas_v = buf[3] * (20.0/65536.0)
            if (ref_v > 0.0) and (abs(meas_v - ref_v)/ref_v < self.tol_3):
                break
            amp = max(0.0, min(0.25, amp + self.p_gain_3*(ref_v - meas_v)))
            self.urukul1_ch1.set(frequency=self.frequency_3,
                                 amplitude=amp)
            delay(1*ms)
        final_vs[3] = meas_v
        a3 = amp

        # channel 4 – Excitation (urukul1_ch2, sampler0 ch4)
        amp = self.initial_amp_4
        ref_v = refs[4]
        meas_v = 0.0
        for _ in range(self.iterations_4):
            self.sampler0.sample_mu(buf)
            delay(1*ms)
            meas_v = buf[4] * (20.0/65536.0)
            if (ref_v > 0.0) and (abs(meas_v - ref_v)/ref_v < self.tol_4):
                break
            amp = max(0.0, min(0.5, amp + self.p_gain_4*(ref_v - meas_v)))
            self.urukul1_ch2.set(frequency=self.frequency_4,
                                 amplitude=amp)
            delay(1*ms)
        final_vs[4] = meas_v
        a4 = amp

        # self.ttl14.on()
        # self.ttl6.on()
        # self.urukul1_ch1.off()
        # self.urukul1_ch2.off()

        # host-side report/save via implicit RPC
        self._report(
            init_amps, init_vs, refs, final_vs,
            [a0, a1, a2, a3, a4],
            fresh_any
        )

    # Host-only function (implicit RPC when called from kernel)
    def _report(self, init_amps, init_vs, refs, meas, amps, fresh_any):
        labels = ["AOM0", "AOM1", "AOM2", "Trap", "Excitation"]
        print("Ch Name        | Init Amp | Init Meas (V) | Ref (V) | Final Meas (V) | Final Amp")
        for i in range(5):
            name = labels[i]
            print(
                f"{i:2d} {name:11s} |  {init_amps[i]:.3f}   | {init_vs[i]:7.4f} | "
                f"{refs[i]:7.4f} | {meas[i]:7.4f} |   {amps[i]:.3f}"
            )

        if fresh_any:
            try:
                REF_PATH.write_text(" ".join(str(r) for r in refs))
            except Exception as e:
                print("ERROR saving reference:", e)

class FeedbackRebalancer_5_revisedJan26:
    """
    ARTIQ-7 compatible: explicit device attributes, no list indexing in kernels.
    Expect the caller to assign:
        self.core, self.sampler0,
        self.urukul2_ch0, self.urukul2_ch1, self.urukul2_ch2
    """

    # NOTE: no build(); caller sets attributes from its own build()

    @kernel
    def rebalance(self,
                  p0, p1, p2, p3, p4,
                  tol0, tol1, tol2, tol3, tol4,
                  it0, it1, it2, it3, it4,
                  ref0, ref1, ref2, ref3, ref4,
                  amp0, amp1, amp2, amp3, amp4,
                  f0, f1, f2, f3, f4):

        delay(1 * ms)

        # Initialize each channel explicitly
        self.urukul2_ch0.set(frequency=f0, amplitude=amp0); delay(1*us); self.urukul2_ch0.sw.on()   #mot_beam_x
        self.urukul2_ch1.set(frequency=f1, amplitude=amp1); delay(1*us); self.urukul2_ch1.sw.on()   #mot_beam_y
        self.urukul2_ch2.set(frequency=f2, amplitude=amp2); delay(1*us); self.urukul2_ch2.sw.on()   #mot_beam_z
        self.urukul1_ch1.set(frequency = f3, amplitude=amp3); delay(1*us); self.urukul1_ch1.sw.on() #trap_beam
        self.urukul1_ch2.set(frequency = f4, amplitude=amp4); delay(1*us); self.urukul1_ch2.sw.on(); self.ttl14.off(); self.ttl6.off() #excitation_beam
        delay(10 * ms)

        buf = [0]*8
        meas = -1.0
        # ch 0 mot_x
        target = ref0
        amp = amp0
        for _ in range(it0):
            self.sampler0.sample_mu(buf); delay(1*ms)
            meas = buf[0]*(20.0/65536.0)
            if (target > 0.0) and (abs(meas-target)/target < tol0):
                break
            amp = max(0.0, min(0.17, amp + p0*(target - meas)))
            self.urukul2_ch0.set(frequency=f0, amplitude=amp); delay(1*ms)
        amp0 = amp

        # ch 1 mot_y
        target = ref1
        amp = amp1
        for _ in range(it1):
            self.sampler0.sample_mu(buf); delay(1*ms)
            meas = buf[1]*(20.0/65536.0)
            if (target > 0.0) and (abs(meas-target)/target < tol1):
                break
            amp = max(0.0, min(0.17, amp + p1*(target - meas)))
            self.urukul2_ch1.set(frequency=f1, amplitude=amp); delay(1*ms)

        amp1 = amp
        # ch 2 mot_z
        target = ref2
        amp = amp2
        for _ in range(it2):
            self.sampler0.sample_mu(buf); delay(1*ms)
            meas = buf[2]*(20.0/65536.0)
            if (target > 0.0) and (abs(meas-target)/target < tol2):
                break
            amp = max(0.0, min(0.17, amp + p2*(target - meas)))
            self.urukul2_ch2.set(frequency=f2, amplitude=amp); delay(1*ms)

        amp2 = amp
        # ch 3 trap
        target = ref3
        amp = amp3
        for _ in range(it3):
            self.sampler0.sample_mu(buf);
            delay(1 * ms)
            meas = buf[3] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < tol3):
                break
            amp = max(0.0, min(0.25, amp + p3 * (target - meas)))
            self.urukul1_ch1.set(frequency=f3, amplitude=amp);
            delay(1 * ms)
        amp3 = amp

        # ch 4 excitation
        target = ref4
        amp = amp4
        for _ in range(it4):
            self.sampler0.sample_mu(buf);
            delay(1 * ms)
            meas = buf[4] * (20.0 / 65536.0)
            if (target > 0.0) and (abs(meas - target) / target < tol4):
                break
            amp = max(0.0, min(0.5, amp + p4 * (target - meas)))
            self.urukul1_ch2.set(frequency=f4, amplitude=amp);

            delay(1 * ms)
        amp4 = amp


        return amp0, amp1, amp2, amp3, amp4

    class FeedbackRebalancer_exc_revisedJan26:
        @kernel
        def rebalance_exc(self, p4, tol4, it4, ref4, amp4, f4):
            delay(1 * ms)
            self.urukul1_ch2.set(frequency=f4, amplitude=amp4);delay(1 * us);self.urukul1_ch2.sw.on();self.ttl14.off();self.ttl6.off()  # excitation_beam
            delay(10 * ms)

            buf = [0] * 8
            # ch 4 excitation
            target = ref4
            amp = amp4
            for _ in range(it4):
                self.sampler0.sample_mu(buf);
                delay(1 * ms)
                meas = buf[4] * (20.0 / 65536.0)
                if (target > 0.0) and (abs(meas - target) / target < tol4):
                    break
                amp = max(0.0, min(0.5, amp + p4 * (target - meas)))
                self.urukul1_ch2.set(frequency=f4, amplitude=amp);

                delay(1 * ms)
            amp4 = amp

            return amp4

        # done — each AOM remains at its final amplitude










