"""Parametric heating: modulate the FORT depth and look for atom loss vs frequency.

Measures the trap frequency. Modulating the trap depth at f_mod parametrically
heats the atom, and loss is resonant at

        f_mod = 2 x f_trap          (and weaker features at f_trap, 2f_trap/n)

so the survival-vs-frequency curve shows a DIP whose centre gives f_trap. Radial
FORT frequencies are typically tens of kHz, axial a few kHz.

Sequence per shot (one modulation frequency):
    feedback (every feedback_every_n shots, between atoms only)
    background window           (FORT off, no atom)        -> bg1, bg2
    load_atom_deterministic     (waits until an atom is caught)
    PGC -> image 1 -> PGC       (confirm + recool)         -> shot1
    MODULATE the trap for t_mod at f_mod
    PGC -> image 2 -> PGC       (survival + recool)        -> shot2
    save one row
The frequency is scanned in the inner loop and the whole scan repeated num_runs
times, so slow drifts average over the scan instead of tilting it.

=============================================================================
HOW THE MODULATION IS GENERATED (AD9910 RAM)
=============================================================================
The FORT DDS (dds_FORT = urukul1_ch1) plays a sine from its RAM into the
AMPLITUDE (ASF) while the frequency stays at f_FORT. A 1024-word table holds
`k` whole periods, each word held for `step` x t_DDS (t_DDS = 4/SYSCLK = 4 ns at
our 1 GHz SYSCLK), so

        f_mod = k / (1024 * step * t_DDS) = k * 244140.625 / step   [Hz]

`k` is fixed for a run (k = 1024 / samples_per_period) and only `step` changes
per scan point, so the RAM is written ONCE at startup and each point costs just
one profile-register write.

PROFILE PINS -- the reason this is safe. The AD9910 PROFILE pins are COMMON to
all four channels of a Urukul board, so `cpld.set_profile()` would move the
profile of dds_RF, dds_excitation and dds_D1_onoff_L too (they share urukul1),
and they have no valid data in any profile but 7. This experiment therefore uses
profile 7 (= ARTIQ's DEFAULT_PROFILE, where the pins already sit) as the RAM
profile and NEVER touches the pins. Nothing on the other three channels moves.
dds_D1_onoff_L matters here: it is loading-critical and is only ever switched
(never re-`set()`), so leaving its profile alone is exactly right.

GLITCH-FREE SWITCHING. The AD9910 latches the profile and CFR1 registers on
IO_UPDATE, so both are written over SPI first and a SINGLE io_update makes the
mode change atomic -- the chip never interprets RAM parameters as single-tone
data (which would throw the trap to a garbage amplitude/frequency with the atom
in it). Entry lands on table address 0, where sin = 0 and the amplitude equals
the unmodulated value, so entry is continuous. `t_mod` is snapped to a whole
number of modulation periods, so playback also ENDS at a phase-0 sample and the
exit is continuous too.

=============================================================================
CALIBRATION / HONEST CAVEATS -- read before trusting a number
=============================================================================
* `mod_depth` is a fraction of the DDS AMPLITUDE, not of the trap depth. DDS
  amplitude -> RF power -> AOM efficiency -> optical power is nonlinear, so the
  actual trap-depth modulation must be calibrated with a photodiode if you want
  a heating RATE. The resonance POSITION (hence f_trap) does not care.
* The table is a sampled sine: `samples_per_period` = 32 by default, whose
  staircase carries harmonics that can drive the 3f and 5f features. Use a
  larger value to clean the waveform, a smaller one for a finer frequency grid
  (see below) -- the trade is explicit.
* FREQUENCY GRID. `step` is an integer, so the achievable frequencies are
  discrete and get sparser as f rises (spacing ~ f/step). At the default k = 32
  the error is under ~0.2% anywhere above 1 kHz, but the realised frequency is
  NOT exactly what you asked for: `ph_f_mod_actual` is what was played and is
  what the analysis must plot against. Same for `ph_t_mod_actual`.
* The table is centred on `nominal_trapdepth`, which is what
  load_atom_deterministic programs the FORT to. If the trap amplitude is changed
  elsewhere (ramping, a different loader), the centre must be changed to match or
  the modulation will be offset.
* RAM mode is entered only for the modulation window and explicitly left before
  any shared subroutine touches dds_FORT again. This matters because the default
  phase_mode is CONTINUOUS, so ARTIQ's `set()` does NOT clear CFR1 -- a stale
  ram_enable would make the next `dds_FORT.set()` write single-tone data that
  the chip reads as RAM parameters.

FIRST TIME: run with mod_depth = 0. The table is then a constant equal to the
normal trap amplitude, so the trap must look completely unchanged and atoms must
survive as usual. If they do not, the RAM word layout or the centre amplitude is
wrong -- fix that before believing any spectrum. Then set mod_depth ~ 0.1 and
look for the dip.

Datasets (one row per shot, appended, never truncated):
    ph_f_mod_set / ph_f_mod_actual   requested vs PLAYED modulation freq [Hz]
    ph_ram_step / ph_n_periods       the RAM step, and periods played
    ph_t_mod_set / ph_t_mod_actual   requested vs PLAYED modulation time [s]
    ph_mod_depth / ph_center_amp     modulation fraction, centre DDS amplitude
    ph_bg1 / ph_bg2                  background window, per SPCM
    ph_shot1_spcm1 / ph_shot1_spcm2  image 1 (after loading)
    ph_shot2_spcm1 / ph_shot2_spcm2  image 2 (after modulation)
    ph_loaded / ph_survived          0/1 against atom_threshold
    ph_loop_index                    which repeat of the scan
To analyse: group the rows by ph_f_mod_actual (NOT ph_f_mod_set) and plot
survived/loaded per frequency with a binomial error bar; the dip centre is
2 x f_trap. No analysis module is written for this yet.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

from artiq.experiment import *
from artiq.coredevice import ad9910
from artiq.coredevice import urukul
from artiq.coredevice import spi2 as spi
from utilities.base_experiment import BaseExperiment
from subroutines.sequences import (
    prepare_for_feedback, restore_after_feedback,
    background_measurement_dual, load_atom_deterministic,
    imaging_pgc, spcm_image_dual, beams_off_for_imaging,
)
from utilities.write_results import write_results

# RAM table length in words. The AD9910 RAM is 1024 x 32 bit; using all of it
# maximises the number of whole periods we can store.
RAM_N = 1024

# RAM profile index. Deliberately ARTIQ's DEFAULT_PROFILE (7) -- the index the
# CPLD already drives on the shared PROFILE pins -- so the other three channels
# of urukul1 are never disturbed. See the module docstring.
RAM_PROFILE = 7

# Profile-register base address, used to rewrite the single-tone profile WITHOUT
# an io_update (ARTIQ's set_mu() always pulses one, which would un-atomise the
# RAM -> single-tone switch). Taken from ARTIQ so it cannot drift out of sync.
try:
    from artiq.coredevice.ad9910 import _AD9910_REG_PROFILE0 as _REG_PROFILE0
except ImportError:                      # pragma: no cover - AD9910 datasheet
    _REG_PROFILE0 = 0x0e

# RAM word layout for ram_destination = RAM_DEST_ASF: the 14-bit amplitude scale
# factor occupies the MOST SIGNIFICANT bits, 31:18 (AD9910 datasheet, RAM data
# format -- every destination is MSB-aligned).
_ASF_SHIFT = 18

# AD9910 RAM register, for the padded RAM write below (see _write_ram_padded).
try:
    from artiq.coredevice.ad9910 import _AD9910_REG_RAM as _REG_RAM
except ImportError:                      # pragma: no cover - AD9910 datasheet
    _REG_RAM = 0x16

# Extra timeline per RAM word, in machine units (ns).
#
# WHY ARTIQ's write_ram() CANNOT BE USED HERE. It issues one SPI transfer per
# word back to back. Each bus.write() advances the TIMELINE by only
# xfer_duration_mu = ((32+1)*2+1) = 67 ns, but costs the kernel CPU ~1.05 us to
# execute, so the timeline falls behind wall clock at ~0.98 us per word --
# ~1.0 ms over 1024 words. That CANNOT be covered by pre-loading slack: the CPU
# can only run ~fifo_depth (128) events ahead of the hardware, i.e. 128*67 ns =
# 8.6 us of lookahead. Beyond that the output FIFO fills, the CPU throttles to
# the timeline rate, and slack pins near zero -- then the deficit drives it
# negative within a few words. Observed exactly that: RTIOUnderflow on channel
# 43 (spi_urukul1) at slack -1920 mu with a 1 ms cushion AND at -1152 mu with a
# 20 ms cushion. More slack changes nothing.
#
# The fix is to make the timeline advance FASTER than the CPU, by padding each
# word. 3 us per word comfortably exceeds the ~1.05 us CPU cost, so slack now
# GROWS through the burst. Cost: 1024 * 3 us = ~3 ms, once, at startup.
_RAM_WRITE_PAD_MU = 3000


class ParametricHeating(EnvExperiment):
    """Parametric Heating (FORT RAM modulation)"""

    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()

        self.setattr_argument(
            "num_runs",
            NumberValue(20, min=1, max=100000, step=1, ndecimals=0, type="int"),
            group="Experiment")
        self.setattr_argument(
            "feedback_every_n",
            NumberValue(10, min=1, max=10000, step=1, ndecimals=0, type="int"),
            group="Experiment")
        # Crash-safe HDF5 snapshot every N shots (a full-file write every shot
        # backs up the async-RPC queue and can stall the kernel).
        self.setattr_argument(
            "save_every",
            NumberValue(20, min=1, max=100000, step=1, ndecimals=0, type="int"),
            group="Experiment")

        # --- modulation scan. NumberValue defaults are in SI BASE units; the
        # `unit` only sets how they are DISPLAYED. So 150e3 with unit="kHz"
        # shows as 150 kHz. (Writing 150.0 here would mean 150 Hz.)
        self.setattr_argument(
            "f_mod_min",
            NumberValue(5e3, min=0.0, max=2e6, step=1e3, unit="kHz", ndecimals=3),
            group="Modulation")
        self.setattr_argument(
            "f_mod_max",
            NumberValue(150e3, min=0.0, max=2e6, step=1e3, unit="kHz", ndecimals=3),
            group="Modulation")
        self.setattr_argument(
            "f_mod_step",
            NumberValue(2e3, min=1.0, max=1e6, step=1e3, unit="kHz", ndecimals=3),
            group="Modulation")
        # Fraction of the DDS amplitude, NOT of the trap depth (see docstring).
        # Set 0 for the commissioning check / a no-modulation control.
        self.setattr_argument(
            "mod_depth",
            NumberValue(0.1, min=0.0, max=0.9, step=0.01, ndecimals=3),
            group="Modulation")
        # Modulation duration. Snapped up/down to a whole number of modulation
        # periods per point; the played value is logged. Must exceed ~50 us so
        # the exit registers can be pre-written inside the window.
        self.setattr_argument(
            "t_mod",
            NumberValue(10e-3, min=100e-6, max=1.0, step=1e-3, unit="ms", ndecimals=3),
            group="Modulation")
        # Samples per modulation period = waveform quality. k = 1024/this is the
        # number of periods in the table; a LARGER k (smaller value here) gives a
        # FINER frequency grid but a coarser sine. Powers of two only, so the
        # table always ends on a phase-0 sample.
        self.setattr_argument(
            "samples_per_period",
            EnumerationValue(["8", "16", "32", "64", "128"], default="32"),
            group="Modulation")
        # Prepend a 0 Hz point: trap held steady for t_mod. The survival control
        # -- anything below it is modulation-induced loss.
        self.setattr_argument(
            "include_zero_point", BooleanValue(True), group="Modulation")

    # =====================================================================
    #  HOST SETUP
    # =====================================================================
    def prepare(self):
        # Shims, gates, thresholds, f_FORT, nominal_trapdepth: all from
        # ExperimentVariables. No overrides.
        self.base.prepare()

        spp = int(self.samples_per_period)
        self.k_periods = RAM_N // spp               # whole periods in the table

        # t_DDS, the RAM address dwell quantum: 4 SYSCLK periods (4 ns @ 1 GHz).
        # Read from the device so this follows the real clock chain.
        self.t_dds = 4.0 / float(self.dds_FORT.sysclk)

        # The trap amplitude the modulation is centred on. This is what
        # load_atom_deterministic programs the FORT to, so it is also what the
        # atom is held in when modulation starts.
        self.center_amp = float(self.nominal_trapdepth)

        # --- frequency list -> achievable (step, f_actual, t_mod_actual) ---
        f_list = []
        if self.include_zero_point:
            f_list.append(0.0)
        f = float(self.f_mod_min)
        f_hi = float(self.f_mod_max)
        df = float(self.f_mod_step)
        while f <= f_hi + 1e-9:
            if f > 0.0:
                f_list.append(f)
            f += df

        # f_mod = k / (RAM_N * step * t_dds); step integer 1..65535.
        f_unit = self.k_periods / (RAM_N * self.t_dds)      # f at step = 1
        self.f_set_list = []
        self.f_act_list = []
        self.step_list = []
        self.t_mod_list = []
        self.nper_list = []
        for f_t in f_list:
            if f_t <= 0.0:                      # control point: no modulation
                self.f_set_list.append(0.0)
                self.f_act_list.append(0.0)
                self.step_list.append(0)        # 0 = marker for "do not play"
                self.t_mod_list.append(float(self.t_mod))
                self.nper_list.append(0)
                continue
            step = int(round(f_unit / f_t))
            step = max(1, min(65535, step))
            f_act = f_unit / step
            # Snap t_mod to a whole number of modulation periods so playback
            # ends on a phase-0 sample (amplitude back at centre).
            period = 1.0 / f_act
            nper = max(1, int(round(float(self.t_mod) / period)))
            self.f_set_list.append(f_t)
            self.f_act_list.append(f_act)
            self.step_list.append(step)
            self.t_mod_list.append(nper * period)
            self.nper_list.append(nper)

        self.n_points = len(self.step_list)
        if self.n_points == 0:
            raise ValueError("Empty frequency list: check f_mod_min/max/step.")

        # --- build the RAM table on the HOST (shape is fixed for the run) ---
        # Word = asf << 18. asf <= 0x3fff so the result can exceed int32; store
        # the equivalent SIGNED 32-bit value, since only the bit pattern is sent.
        asf0 = int(round(self.center_amp * 0x3fff))
        if not 0 <= asf0 <= 0x3fff:
            raise ValueError("nominal_trapdepth out of range for an ASF.")
        depth = float(self.mod_depth)
        self._ram = [0] * RAM_N
        asf_min, asf_max = 0x3fff, 0
        for i in range(RAM_N):
            s = math.sin(2.0 * math.pi * self.k_periods * i / RAM_N)
            asf = int(round(asf0 * (1.0 + depth * s)))
            asf = max(0, min(0x3fff, asf))
            asf_min = min(asf_min, asf)
            asf_max = max(asf_max, asf)
            w = (asf & 0x3fff) << _ASF_SHIFT
            if w >= 0x80000000:                 # -> signed int32 bit pattern
                w -= 0x100000000
            self._ram[i] = w
        self._asf_center = asf0
        if asf_max >= 0x3fff and depth > 0.0:
            print("WARNING: modulation clips at full scale "
                  "(centre asf %d, peak asf %d). Reduce mod_depth or "
                  "nominal_trapdepth." % (asf0, asf_max))

        # --- report the realised scan, so the quantisation is visible up front
        print("Parametric heating: %d points x %d runs" % (self.n_points, int(self.num_runs)))
        print("  table %d words, %d periods (%d samples/period), t_DDS %.2f ns"
              % (RAM_N, self.k_periods, spp, 1e9 * self.t_dds))
        print("  centre amplitude %.4f (asf %d), modulation asf %d..%d"
              % (self.center_amp, asf0, asf_min, asf_max))
        worst = 0.0
        for i in range(self.n_points):
            if self.f_set_list[i] > 0.0:
                err = abs(self.f_act_list[i] - self.f_set_list[i]) / self.f_set_list[i]
                worst = max(worst, err)
        print("  frequency grid: worst |actual-set|/set = %.3f %%" % (100 * worst))

        # --- result datasets: one row per shot, appended, never trimmed ---
        for key in ("ph_f_mod_set", "ph_f_mod_actual", "ph_ram_step",
                    "ph_n_periods", "ph_t_mod_set", "ph_t_mod_actual",
                    "ph_mod_depth", "ph_center_amp",
                    "ph_bg1", "ph_bg2",
                    "ph_shot1_spcm1", "ph_shot1_spcm2",
                    "ph_shot2_spcm1", "ph_shot2_spcm2",
                    "ph_loaded", "ph_survived", "ph_loop_index"):
            self.set_dataset(key, [], broadcast=True)
        self.set_dataset("ph_n_shots", 0, broadcast=True)
        self.set_dataset("ph_survival_live", 0.0, broadcast=True)
        self.set_dataset("measurement_progress", 0, broadcast=True)

        self._n = 0
        self._n_loaded = 0
        self._n_survived = 0

    # =====================================================================
    #  RAM SETUP / MODULATION
    # =====================================================================
    @kernel
    def _ram_write(self, ftw_fort: TInt32):
        """Load the sine table into the FORT DDS's RAM. Once, at startup.

        Must run with the trap OFF: while the RAM profile occupies profile 7 and
        RAM is still disabled, the chip reads those RAM parameters as single-tone
        data, i.e. a garbage frequency/amplitude. Harmless with no light and no
        atom; it is cleaned up at the end of this method.
        """
        f = self.dds_FORT

        f.set_cfr1(ram_enable=0)
        f.cpld.io_update.pulse_mu(8)
        delay(10 * us)

        # Address window for WRITING: the whole table.
        f.set_profile_ram(start=0, end=RAM_N - 1, step=1,
                          profile=RAM_PROFILE,
                          mode=ad9910.RAM_MODE_CONT_RAMPUP)
        f.cpld.io_update.pulse_mu(8)

        self.core.break_realtime()
        delay(1 * ms)

        # Padded RAM write -- NOT f.write_ram(), which underflows here however
        # much slack it is given. See _RAM_WRITE_PAD_MU and _write_ram_padded.
        self._write_ram_padded()
        delay(1 * ms)

        # The FTW register supplies the frequency while RAM drives the ASF
        # (in single-tone mode the frequency comes from the profile instead, so
        # writing it now is harmless). phase_mode is CONTINUOUS by default, so
        # set_mu does not touch CFR1 here.
        f.set_mu(ftw=ftw_fort, ram_destination=ad9910.RAM_DEST_ASF)
        delay(100 * us)

        # Leave profile 7 as a normal single-tone profile, so every shared
        # subroutine that calls dds_FORT.set() behaves exactly as usual.
        f.set(frequency=self.f_FORT, amplitude=self.center_amp)
        delay(100 * us)

    @kernel
    def _write_ram_padded(self):
        """write_ram() with an explicit per-word delay, so the timeline keeps up.

        Byte-for-byte the same SPI framing as artiq.coredevice.ad9910.write_ram
        (RAM register address, then one 32-bit transfer per word, SPI_END on the
        last) with `delay_mu(_RAM_WRITE_PAD_MU)` added between words. That pad is
        the whole point: ARTIQ's version advances the timeline 67 ns per word
        while costing the CPU ~1.05 us, which no amount of pre-loaded slack can
        cover (see _RAM_WRITE_PAD_MU). Padding makes the timeline outrun the CPU,
        so slack grows instead of draining.

        The pause between words keeps CS asserted with SCLK idle, which the
        AD9910 serial port allows -- it simply waits for the next transfer.
        """
        f = self.dds_FORT
        n = len(self._ram)

        # Address phase: 8 bits, RAM register, transaction left open.
        f.bus.set_config_mu(urukul.SPI_CONFIG, 8, urukul.SPIT_DDS_WR,
                            f.chip_select)
        f.bus.write(_REG_RAM << 24)

        # Data phase: 32-bit words, all but the last with the transaction open.
        f.bus.set_config_mu(urukul.SPI_CONFIG, 32, urukul.SPIT_DDS_WR,
                            f.chip_select)
        for i in range(n - 1):
            f.bus.write(self._ram[i])
            delay_mu(_RAM_WRITE_PAD_MU)

        # Last word closes the transaction (SPI_END deasserts CS).
        f.bus.set_config_mu(urukul.SPI_CONFIG | spi.SPI_END, 32,
                            urukul.SPIT_DDS_WR, f.chip_select)
        f.bus.write(self._ram[n - 1])

    @kernel
    def _modulate(self, step: TInt32, t_mod_mu: TInt64,
                  ftw_fort: TInt32, asf_center: TInt32):
        """Play `step`-rate RAM modulation for exactly t_mod_mu, then restore.

        Both the entry and the exit are a pair of SPI register writes followed by
        ONE io_update, so the chip switches mode atomically and the trap
        amplitude never passes through a garbage value. The exit registers are
        written while the modulation is still playing, so the window is timed by
        the io_update pulses alone -- t_mod_mu is exact.
        """
        f = self.dds_FORT

        # ---- enter RAM mode, playing (latched together) ----
        f.set_profile_ram(start=0, end=RAM_N - 1, step=step,
                          profile=RAM_PROFILE,
                          mode=ad9910.RAM_MODE_CONT_RAMPUP)
        f.set_cfr1(ram_enable=1, ram_destination=ad9910.RAM_DEST_ASF)
        t0 = now_mu()
        f.cpld.io_update.pulse_mu(8)           # playback starts at address 0

        # ---- pre-arm the exit: single-tone profile 7 + RAM off, NOT latched ----
        f.set_cfr1(ram_enable=0)
        f.write64(_REG_PROFILE0 + RAM_PROFILE,
                  (asf_center << 16), ftw_fort)

        # ---- latch the exit exactly t_mod after the start ----
        at_mu(t0 + t_mod_mu)
        f.cpld.io_update.pulse_mu(8)

    # =====================================================================
    #  RUN
    # =====================================================================
    @kernel
    def run(self):
        self.base.initialize_hardware()
        delay(1000 * ms)

        ftw_fort = self.dds_FORT.frequency_to_ftw(self.f_FORT)
        asf_center = self.dds_FORT.amplitude_to_asf(self.center_amp)

        # Load the RAM with the trap off, before any atom exists.
        self.dds_FORT.sw.off()
        delay(1 * ms)
        self._ram_write(ftw_fort)
        self.core.break_realtime()

        n_shot = 0
        for loop in range(int(self.num_runs)):
            for ip in range(self.n_points):
                self.core.break_realtime()

                # --- feedback, between atoms only ---
                if n_shot % int(self.feedback_every_n) == 0:
                    prepare_for_feedback(self)
                    amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
                    restore_after_feedback(self)
                    self.core.break_realtime()

                # --- background: FORT off, no atom ---
                bg1, bg2 = background_measurement_dual(self, self.t_imaging_gate)

                # --- load an atom, confirm it, recool ---
                load_atom_deterministic(self)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                s1a, s2a = spcm_image_dual(self, self.t_imaging_gate)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                loaded = (s1a + s2a - bg1 - bg2) > self.atom_threshold

                # --- modulate the trap (atom in the dark) ---
                delay(100 * us)                       # let the beams settle
                step = self.step_list[ip]
                t_mod_mu = self.core.seconds_to_mu(self.t_mod_list[ip])
                if step > 0:
                    self._modulate(step, t_mod_mu, ftw_fort, asf_center)
                else:
                    delay_mu(t_mod_mu)                # 0 Hz control: steady trap

                # --- image again: did it survive? ---
                imaging_pgc(self)
                beams_off_for_imaging(self)
                s1b, s2b = spcm_image_dual(self, self.t_imaging_gate)
                imaging_pgc(self)
                beams_off_for_imaging(self)
                survived = (s1b + s2b - bg1 - bg2) > self.atom_threshold

                self._save_point(ip, bg1, bg2, s1a, s2a, s1b, s2b,
                                 1 if loaded else 0, 1 if survived else 0, loop)
                n_shot += 1

            self.set_dataset("measurement_progress", loop, broadcast=True)

        write_results(self)

    # =====================================================================
    #  HOST-SIDE SAVE
    # =====================================================================
    @rpc
    def _save_point(self, ip, bg1, bg2, s1a, s2a, s1b, s2b,
                    loaded, survived, loop):
        """Append one shot. Logs the PLAYED frequency and time, not the request.

        Survival is only meaningful on shots that actually loaded, so
        ph_survival_live counts survivors among loaded atoms.
        """
        self.append_to_dataset("ph_f_mod_set", self.f_set_list[ip])
        self.append_to_dataset("ph_f_mod_actual", self.f_act_list[ip])
        self.append_to_dataset("ph_ram_step", self.step_list[ip])
        self.append_to_dataset("ph_n_periods", self.nper_list[ip])
        self.append_to_dataset("ph_t_mod_set", float(self.t_mod))
        self.append_to_dataset("ph_t_mod_actual", self.t_mod_list[ip])
        self.append_to_dataset("ph_mod_depth", float(self.mod_depth))
        self.append_to_dataset("ph_center_amp", self.center_amp)
        self.append_to_dataset("ph_bg1", bg1)
        self.append_to_dataset("ph_bg2", bg2)
        self.append_to_dataset("ph_shot1_spcm1", s1a)
        self.append_to_dataset("ph_shot1_spcm2", s2a)
        self.append_to_dataset("ph_shot2_spcm1", s1b)
        self.append_to_dataset("ph_shot2_spcm2", s2b)
        self.append_to_dataset("ph_loaded", loaded)
        self.append_to_dataset("ph_survived", survived)
        self.append_to_dataset("ph_loop_index", loop)

        self._n += 1
        if loaded:
            self._n_loaded += 1
            if survived:
                self._n_survived += 1
        self.set_dataset("ph_n_shots", self._n, broadcast=True)
        self.set_dataset("ph_survival_live",
                         (self._n_survived / self._n_loaded) if self._n_loaded
                         else 0.0,
                         broadcast=True)

        if self._n % int(self.save_every) == 0:
            try:
                from utilities.write_results import _save_h5_now
                _save_h5_now(self)
            except Exception as e:
                print("[incremental save]", e)
