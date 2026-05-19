"""Continuous atom loading monitor with dual-SPCM readout.

Matches the original Atom_load_applet_faster_release_double_spcm_251202.py
exactly — same frequencies, amplitudes, timings, and sequence order.
Uses device aliases for readability.

Applet command:
  ${python} -m applets.plot_spcm_counts spcm_counts1 --y2 spcm_counts2 --pts 300
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artiq.experiment import *
from utilities.base_experiment import BaseExperiment
from utilities.write_results import write_results


class AtomLoadMonitor(EnvExperiment):
    """Real-time atom loading monitor with dual SPCM."""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument("num_runs",
                              NumberValue(100, min=1, max=600000, step=1),
                              group="Experiment")

    def prepare(self):
        self.base.prepare()

        # Override with original atom_load_monitor values (Dec 2025 optimized)
        self.shim_Y_idle = 0.2
        self.shim_Z1_idle = 4.27
        self.shim_X_compression = 0.25
        self.shim_Y_compression = 0.0
        self.shim_Z1_compression = 4.27
        self.nominal_trapdepth = 0.22

        self.set_dataset("spcm_counts1", [], broadcast=True)
        self.set_dataset("spcm_counts2", [], broadcast=True)

    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        # Initial feedback before loop (matches original lines 239-266)
        delay(1000 * us)
        self.dds_cooling_DP.set(frequency=122 * MHz, amplitude=0.3)
        self.dds_cooling_DP.sw.on()
        delay(100 * us)
        delay(100 * us)
        self.rebalancer.rebalance_3ch()
        delay(1 * us)
        self.core.break_realtime()

        for i in range(int(self.num_runs)):
            # UV on + block repump at start of each cycle (matches original lines 270-272)
            self.ttl_UV.on()
            delay(1 * us)
            self.ttl_repump_switch.on()
            delay(1 * us)

            # Periodic 3-channel feedback (MOT X/Y/Z only, matches original)
            if i % 10 == 1:
                delay(1000 * us)
                self.dds_cooling_DP.set(frequency=122 * MHz, amplitude=0.3)
                self.dds_cooling_DP.sw.on()
                delay(100 * us)
                delay(100 * us)
                self.rebalancer.rebalance_3ch()
                delay(1 * us)

            # Cooling + repump on (original lines 302-308)
            self.dds_cooling_DP.set(frequency=120 * MHz, amplitude=0.35)
            self.dds_cooling_DP.sw.on()
            delay(100 * us)

            self.dds_repump.set(frequency=197 * MHz, amplitude=0.5)
            self.ttl_repump_switch.off()
            delay(10 * us)

            # FORT on (original lines 310-312)
            self.dds_FORT.set(frequency=150 * MHz, amplitude=self.nominal_trapdepth)
            self.dds_FORT.sw.on()
            delay(100 * us)

            delay(self.t_wait)

            # Idle shims (original lines 316-320)
            self.zotino0.write_dac(0, self.shim_X_idle)
            self.zotino0.write_dac(1, self.shim_Y_idle)
            self.zotino0.write_dac(2, self.shim_Z1_idle)
            self.zotino0.write_dac(3, self.shim_Z2_idle)
            self.zotino0.load()

            # MOT cooling (original lines 322-325)
            self.dds_cooling_DP.set(frequency=112 * MHz, amplitude=0.35)
            self.dds_cooling_DP.sw.on()
            delay(1 * us)
            delay(self.t_MOT2PGC)

            # Compression shims (original lines 327-334)
            self.zotino0.write_dac(0, self.shim_X_compression)
            self.zotino0.write_dac(1, self.shim_Y_compression)
            self.zotino0.write_dac(2, self.shim_Z1_compression)
            self.zotino0.write_dac(3, self.shim_Z2_compression)
            self.zotino0.load()
            delay(1 * us)
            self.ttl_UV.off()
            delay(1 * us)

            # Compression cooling (original line 336: 122 MHz, 0.4)
            self.dds_cooling_DP.set(frequency=122 * MHz, amplitude=0.4)
            self.dds_cooling_DP.sw.on()
            delay(1 * us)

            delay(self.t_PGC_2_off)

            # Deep PGC (original line 342: 129 MHz, 0.22)
            self.dds_cooling_DP.set(frequency=129 * MHz, amplitude=0.22)
            self.dds_cooling_DP.sw.on()
            delay(1 * ms)

            # Off shims + field settle (original lines 346-351)
            self.zotino0.write_dac(0, self.shim_X_off)
            self.zotino0.write_dac(1, self.shim_Y_off)
            self.zotino0.write_dac(2, self.shim_Z1_off)
            self.zotino0.write_dac(3, self.shim_Z2_off)
            self.zotino0.load()
            delay(300 * ms)

            # Imaging PGC (original line 353: 130 MHz, 0.22)
            self.dds_cooling_DP.set(frequency=130 * MHz, amplitude=0.22)
            self.dds_cooling_DP.sw.on()
            delay(20 * ms)  # PGC_Af_Img = 20 ms in original

            # Wait for atom (original lines 357-358: t3 + t3 = 300ms)
            delay(150 * ms)
            delay(150 * ms)

            # Beams off before readout (original lines 360-365)
            self.dds_cooling_DP.set(frequency=124 * MHz, amplitude=0.4)
            self.dds_cooling_DP.sw.off()
            delay(1 * us)
            self.dds_repump.set(frequency=197 * MHz, amplitude=0.5)
            self.ttl_repump_switch.on()
            delay(1 * us)

            # Readout loop (original lines 366-411)
            c0 = 900
            while c0 > 60:
                # Imaging beams on (original lines 371-376)
                self.dds_cooling_DP.set(frequency=120 * MHz, amplitude=0.36)
                self.dds_cooling_DP.sw.on()
                delay(1 * us)
                self.dds_repump.set(frequency=197 * MHz, amplitude=0.5)
                self.ttl_repump_switch.off()
                delay(1 * us)

                # DAC6 trigger pulse (original lines 378-384)
                delay(1 * us)
                self.zotino0.write_dac(6, 4.1)
                self.zotino0.load()
                delay(5 * ms)  # trig_width
                self.zotino0.write_dac(6, 0.0)
                self.zotino0.load()
                delay(1 * us)

                # Dual SPCM gate (original lines 385-391: t2 = 100ms)
                t_end0 = self.ttl_SPCM1.gate_rising(100 * ms)
                t_end1 = self.ttl_SPCM2.gate_rising(100 * ms)
                delay(1 * us)
                c0 = self.ttl_SPCM1.count(t_end0)
                delay(1 * ms)
                c1 = self.ttl_SPCM2.count(t_end1)
                delay(1 * ms)

                # Save to datasets (original lines 394-397)
                self.append_to_dataset("spcm_counts1", c0)
                delay(1 * ms)
                self.append_to_dataset("spcm_counts2", c1)
                delay(1 * ms)

                # Between-shot PGC (original lines 401-411)
                delay(1 * us)
                self.dds_cooling_DP.set(frequency=130 * MHz, amplitude=0.22)
                self.dds_cooling_DP.sw.on()
                delay(20 * ms)  # PGC_Af_Img

                self.dds_cooling_DP.set(frequency=124 * MHz, amplitude=0.4)
                self.dds_cooling_DP.sw.off()
                delay(1 * us)
                self.dds_repump.set(frequency=197 * MHz, amplitude=0.5)
                self.ttl_repump_switch.on()
                delay(1 * us)

            self.set_dataset("measurement_progress", i, broadcast=True)

        write_results(self)
