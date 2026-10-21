"""Read back the digital step attenuator settings of all three Urukul boards.

Read-only diagnostic: CPLD.get_att_mu() shifts the attenuator register out
and latches the same value back in, so attenuation is unchanged. Does NOT
call cpld.init() / ad9910.init() and touches no DDS frequency/amplitude/switch.

Per board the 32-bit register holds one byte per channel (ch0 = bits 0-7).
att_mu 255 = 0 dB, 0 = 31.5 dB; dB = (255 - att_mu) / 8.
Results are printed and stored as datasets urukul{N}_att_mu / urukul{N}_att_dB.
"""

from artiq.experiment import *


class ReadUrukulAtt(EnvExperiment):
    """Read_Urukul_Attenuators."""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("urukul0_cpld")
        self.setattr_device("urukul1_cpld")
        self.setattr_device("urukul2_cpld")

    def _store(self, board: TInt32, reg: TInt32):
        self.regs[board] = reg

    @kernel
    def read_att(self):
        self.core.break_realtime()
        self._store(0, self.urukul0_cpld.get_att_mu())
        self.core.break_realtime()
        self._store(1, self.urukul1_cpld.get_att_mu())
        self.core.break_realtime()
        self._store(2, self.urukul2_cpld.get_att_mu())

    def run(self):
        try:
            self._run()
        except Exception:
            import traceback
            self.set_dataset("error", traceback.format_exc(), archive=True)
            raise

    def _run(self):
        self.regs = [0, 0, 0]
        self.core.reset()
        self.read_att()
        regs = self.regs
        for board, reg in enumerate(regs):
            mu = [(reg >> (8 * ch)) & 0xff for ch in range(4)]
            db = [(255 - m) / 8 for m in mu]
            print("urukul{} reg=0x{:08x}  att_mu={}  att_dB={}".format(
                board, reg & 0xffffffff, mu, db))
            self.set_dataset("urukul{}_att_reg".format(board), reg, archive=True)
            self.set_dataset("urukul{}_att_mu".format(board), mu, archive=True)
            self.set_dataset("urukul{}_att_dB".format(board), db, archive=True)
