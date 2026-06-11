"""Reusable @kernel physics subroutines for SNAQ-Node-3 Cs experiments.

All functions take the experiment instance (``exp``) as the first argument
so they can access device aliases and datasets set up by BaseExperiment.

Device aliases used throughout (see config/device_aliases.json):
  DDS: dds_cooling_DP, dds_repump, dds_D1_pump, dds_mw_IF, dds_RF,
       dds_FORT, dds_excitation, dds_D1_onoff_L, dds_MOT_X/Y/Z, dds_pump_repumper
  TTL: ttl_SPCM1/2, ttl_UV, ttl_repump_switch, ttl_MW_switch,
       ttl_exc_atten1/2, ttl_D1_exc_switch

Extracted from the latest Z/X parity experiments (April 2026).
"""

from artiq.experiment import *


# =========================================================================
#  FEEDBACK PREPARATION
# =========================================================================

@kernel
def prepare_for_feedback(exp):
    """Configure beams for feedback measurement.

    Turns off repump RF, sets cooling to 122 MHz for feedback beam path,
    turns off trap, and unblocks excitation path on sampler.
    """
    delay(1000 * us)
    exp.dds_repump.sw.off()
    delay(1 * us)
    exp.ttl_repump_switch.on()          # block repump RF
    delay(1000 * us)
    exp.dds_cooling_DP.set(frequency=122 * MHz, amplitude=0.3)
    exp.dds_cooling_DP.sw.on()
    delay(100 * us)
    exp.dds_FORT.sw.off()               # trap off for feedback
    delay(1 * us)
    exp.ttl_exc_atten1.off()            # unblock excitation atten1
    delay(1 * us)
    exp.ttl_D1_exc_switch.off()         # unblock D1/excitation switch
    delay(100 * us)


@kernel
def restore_after_feedback(exp):
    """Restore beams after feedback: block excitation, set cooling to MOT freq."""
    delay(1 * us)
    exp.ttl_D1_exc_switch.on()          # block D1/excitation
    delay(1 * us)
    exp.dds_excitation.sw.off()
    delay(1 * us)
    exp.ttl_exc_atten1.on()             # block excitation atten1
    delay(1 * us)
    exp.ttl_exc_atten2.on()             # block excitation atten1
    delay(1 * us)
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_MOT, amplitude=exp.amp_cooling_MOT)
    exp.dds_cooling_DP.sw.on()
    delay(100 * us)


@kernel
def imaging_pgc(exp):
    """Short PGC before SPCM readout."""
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_deep_PGC, amplitude=exp.amp_cooling_deep_PGC)
    exp.dds_cooling_DP.sw.on()
    delay(exp.t_PGC_after_img)


# =========================================================================
#  SPCM IMAGING
# =========================================================================

@kernel
def spcm_image_dual(exp, gate_time):
    """Dual-SPCM fluorescence imaging (parallel gating) — the default imaging.

    Both SPCMs gate simultaneously. Higher timing precision but the
    RTIO FIFO handles fewer events (overflow at lower count rates).
    Use for short gates (parity, MW scan). For high count rates / long
    gates (atom loading monitor) use spcm_image_dual_sequential.

    Returns (count1, count2).
    """
    # Imaging beams on (cooling + repump)
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_imaging, amplitude=exp.amp_cooling_imaging)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_repump.sw.on()
    exp.ttl_repump_switch.off()
    delay(1 * us)

    with parallel:
        tend1 = exp.ttl_SPCM1.gate_rising(gate_time)
        tend2 = exp.ttl_SPCM2.gate_rising(gate_time)
    count1 = exp.ttl_SPCM1.count(tend1)
    count2 = exp.ttl_SPCM2.count(tend2)
    delay(100 * us)

    return count1, count2


@kernel
def spcm_image_dual_sequential(exp, gate_time):
    """Dual-SPCM fluorescence imaging with sequential gating.

    Gates SPCMs one after the other (not parallel). Handles much higher
    count rates without RTIO overflow. Use for long gates (atom loading monitor).
    Matches the original atom_load_applet readout pattern.

    Returns (count1, count2).
    """
    # Imaging beams on (cooling + repump)
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_imaging, amplitude=exp.amp_cooling_imaging)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_repump.sw.on()
    exp.ttl_repump_switch.off()
    delay(1 * us)

    t_end0 = exp.ttl_SPCM1.gate_rising(gate_time)
    t_end1 = exp.ttl_SPCM2.gate_rising(gate_time)
    delay(1 * us)
    count1 = exp.ttl_SPCM1.count(t_end0)
    delay(1 * ms)
    count2 = exp.ttl_SPCM2.count(t_end1)
    delay(1 * ms)

    return count1, count2


@kernel
def beams_off_for_imaging(exp):
    """Turn off cooling and repump beams, block repump RF."""
    exp.dds_cooling_DP.sw.off()
    delay(1 * us)
    exp.dds_repump.sw.off()
    delay(1 * us)
    exp.ttl_repump_switch.on()          # block repump
    delay(1 * us)


@kernel
def beams_on_for_pgc(exp):
    """Re-enable cooling and repump for PGC/imaging."""
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_repump.sw.on()
    delay(1 * us)
    exp.ttl_repump_switch.off()         # unblock repump
    delay(1 * us)


@kernel
def check_atom_loaded(exp, count1_bg, count2_bg):
    """Full atom readout cycle: PGC → image → PGC → beams off → threshold check.

    Performs imaging PGC, dual-SPCM readout, post-image PGC, then beams off.
    Returns (atom_present, count1, count2) where atom_present is True if
    (count1 + count2 - count1_bg - count2_bg) > atom_threshold.

    Matches the parity Yellow scope atom check exactly.
    """
    imaging_pgc(exp)
    beams_off_for_imaging(exp)
    delay(exp.t_SPCM_exposure)

    count1, count2 = spcm_image_dual(exp, exp.t_imaging_gate)

    imaging_pgc(exp)
    beams_off_for_imaging(exp)

    atom_present = (count1 + count2 - count1_bg - count2_bg) > exp.atom_threshold
    return atom_present, count1, count2


# =========================================================================
#  OPTICAL PUMPING
# =========================================================================

@kernel
def optical_pumping_F3(exp):
    """F=3 state preparation: brief cooling then beams off."""
    exp.dds_cooling_DP.sw.on()
    delay(1 * ms)
    exp.dds_cooling_DP.sw.off()
    delay(1 * us)
    exp.dds_D1_onoff_L.sw.off()
    delay(1 * us)


@kernel
def optical_pumping(exp):
    """Single global optical pumping (recool-during-pump).

    Caller contract: set the Q-axis (set_qaxis / the qaxis shim writes) BEFORE
    calling. If a downstream MW/analysis pulse relies on a pre-staged dds_mw_IF
    frequency (e.g. z_parity's microwave_pi_pulse_z), set dds_mw_IF BEFORE calling.
    """
    # MOT fiber beams on for recooling during pump
    exp.dds_MOT_X.sw.on()
    delay(1 * us)
    exp.dds_MOT_Y.sw.on()
    delay(1 * us)
    exp.dds_MOT_Z.sw.on()
    delay(1 * us)
    # D1 pump
    exp.dds_D1_pump.set(frequency=exp.f_pump_D1, amplitude=exp.amp_pump_D1)
    exp.dds_D1_pump.sw.on()
    delay(6 * us)
    # Excitation for pumping
    exp.dds_excitation.set(frequency=exp.f_pump_excitation, amplitude=exp.amp_pump_excitation)
    exp.dds_excitation.sw.on()
    delay(6 * us)
    # Pump repumper
    exp.dds_pump_repumper.set(frequency=exp.f_pump_repumper, amplitude=exp.amp_pump_repumper_active)
    exp.dds_pump_repumper.sw.on()
    delay(6 * us)
    # Unblock D1/excitation path + pump window
    exp.ttl_D1_exc_switch.off()
    delay(exp.t_optical_pump)
    # Shut everything off simultaneously (incl MOT fiber beams)
    t_end = now_mu()
    at_mu(t_end + 1000)
    with parallel:
        exp.ttl_D1_exc_switch.on()
        exp.dds_D1_pump.sw.off()
        exp.dds_excitation.sw.off()
        exp.dds_pump_repumper.sw.off()
        exp.dds_MOT_X.sw.off()
        exp.dds_MOT_Y.sw.off()
        exp.dds_MOT_Z.sw.off()
    delay(3 * us)

@kernel
def turn_fiber_aom_on(exp):
    """Turn on the MOT X/Y/Z fiber AOMs for the PGC/imaging path."""
    exp.dds_D1_onoff_L.sw.on()
    delay(1 * us)
    exp.dds_MOT_X.sw.on()
    delay(1 * us)
    exp.dds_MOT_Y.sw.on()
    delay(1 * us)
    exp.dds_MOT_Z.sw.on()
    delay(1 * us)
# =========================================================================
#  TRAP RAMP
# =========================================================================

@kernel
def ramp_trap_amp(exp, start_amp, end_amp, n_steps, dt_us):
    """Adiabatically ramp FORT amplitude.

    Clamps amplitude to [0.0, 0.3].
    """
    steps = int(n_steps)
    if steps <= 0:
        return

    step_size = (end_amp - start_amp) / n_steps
    amp = start_amp

    for i in range(steps + 1):
        if amp < 0.0:
            amp = 0.0
        if amp > 0.3:
            amp = 0.3
        amp_q = int(amp * 1000.0) / 1000.0
        exp.dds_FORT.set(frequency=exp.f_FORT, amplitude=amp_q)
        delay(dt_us * us)
        amp += step_size


# =========================================================================
#  EXCITATION PULSE WITH TIMESTAMPS
# =========================================================================

@kernel
def excitation_pulse(exp):
    """Single excitation attempt with trap-off/on; opens the SPCM collection
    gate and returns (t_end1, t_end2, t0) WITHOUT reading the timestamps.

    The caller must read the SPCM timestamps itself, immediately, so the
    post-detection analysis pulse (anchored at click + 2000 mu) fires with
    minimal CPU between the gate-close read and the first MW edge. An empty
    SPCM timestamp_mu blocks until the gate closes, pinning the wall clock
    with ~0 slack; if a function return + dispatch sit in that window the
    click+2000 anchor underflows.

    Returns (t_end1, t_end2, t0): int64 machine-unit SPCM1/SPCM2 gate-end
    times (pass to ttl_SPCMx.timestamp_mu) and the excitation reference t0.
    """
    exp.dds_excitation.set(frequency=exp.f_excitation_pulse, amplitude=exp.amp_excitation_pulse)
    delay(5 * us)

    t0 = now_mu()

    # Unblock MW switch once at t0 (v2:604-605 "packing"); photon-branch
    # analysis pulses rely on this and do not re-unblock for their pi pulse.
    at_mu(t0)
    exp.ttl_MW_switch.off()

    # Trap off
    at_mu(t0 + int(exp.exc_trap_off_offset_mu))
    exp.dds_FORT.sw.off()
    at_mu(t0 + int(exp.exc_trap_off_offset_mu) + int(exp.t_trap_off_mu))
    exp.dds_FORT.sw.on()

    # D1/excitation switch pulse
    at_mu(t0 + int(exp.exc_d1_switch_offset_mu))
    exp.ttl_D1_exc_switch.off()
    at_mu(t0 + int(exp.exc_d1_switch_offset_mu) + int(exp.length_offset_mu))
    exp.ttl_D1_exc_switch.on()

    # Excitation attenuator pulse
    at_mu(t0 + int(exp.exc_atten_offset_mu))
    with parallel:
        exp.ttl_exc_atten2.off()
        exp.ttl_exc_atten1.off()
    at_mu(t0 + int(exp.exc_atten_offset_mu) + 100)
    with parallel:
        exp.ttl_exc_atten2.on()
        exp.ttl_exc_atten1.on()

    # Excitation beam pulse
    at_mu(t0 + int(exp.exc_beam_on_offset_mu))
    exp.dds_excitation.sw.on()
    at_mu(t0 + int(exp.exc_beam_on_offset_mu) + int(exp.exc_beam_duration_mu))
    exp.dds_excitation.sw.off()

    # SPCM gate
    at_mu(t0 + int(exp.exc_spcm_gate_offset_mu))
    with parallel:
        t_end1 = exp.ttl_SPCM1.gate_rising(exp.t_spcm_collection_mu * ns)
        t_end2 = exp.ttl_SPCM2.gate_rising(exp.t_spcm_collection_mu * ns)

    # Reads moved to the caller (see docstring): returning the gate-end times
    # keeps the function-return boundary OUT of the slack-critical window
    # between the gate-close read and the post-detection MW pulse.
    return t_end1, t_end2, t0


# =========================================================================
#  POST-DETECTION MICROWAVE SEQUENCES
# =========================================================================

@kernel
def microwave_pi_pulse_z(exp, click_time_mu):
    """Z-basis: single MW pi pulse on 11 transition after photon detection.

    Uses t_pi_11_mu (calibrated live via the ARTIQ dashboard), the 11
    transition (F=3,mF=1 -> F=4,mF=1).
    click_time_mu is the SPCM timestamp used as timing reference.
    """
    at_mu(click_time_mu + 2000)
    exp.dds_mw_IF.sw.on()
    at_mu(click_time_mu + int(exp.t_pi_11_mu) + 2000)
    exp.dds_mw_IF.sw.off()
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 2000 + 3300)
    exp.ttl_MW_switch.on()              # block MW


@kernel
def microwave_two_photon_x(exp, click_time_mu):
    """X-basis: MW pi pulse + two-photon basis rotation.

    Sequence:
    1. Single pi pulse on 11 transition (t_pi_11_mu)
    2. Set MW IF to two-photon frequency, set RF antenna frequency
    3. Simultaneous MW + RF pulse for basis rotation (t_pi_2ph_mu // 2)

    click_time_mu is the SPCM timestamp used as timing reference.
    """
    # Step 1: pi pulse (11 transition)
    at_mu(click_time_mu + 2000)
    exp.dds_mw_IF.sw.on()
    at_mu(click_time_mu + int(exp.t_pi_11_mu) + 2000)
    exp.dds_mw_IF.sw.off()
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 2000 + 3300)
    exp.ttl_MW_switch.on()              # block MW briefly

    # Step 2: set two-photon frequencies
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 5000 + 3300)
    exp.dds_mw_IF.set(frequency=exp.f_mw_2ph, amplitude=exp.amp_mw_pulse)
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 15000 + 3300)
    exp.dds_RF.set(frequency=exp.f_RF_2ph, amplitude=exp.amp_RF_2ph)

    # Step 3: simultaneous MW + RF pulse
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 3300)
    exp.ttl_MW_switch.off()             # unblock MW
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 4000 + 3300)
    with parallel:
        exp.dds_mw_IF.sw.on()
        exp.dds_RF.sw.on()
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 4000 + (int(exp.t_pi_2ph_mu) // 2) + 3300)
    with parallel:
        exp.dds_mw_IF.sw.off()
        exp.dds_RF.sw.off()
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 4100 + (int(exp.t_pi_2ph_mu) // 2) + 3300)
    exp.ttl_MW_switch.on()              # block MW


@kernel
def microwave_two_photon_y(exp, click_time_mu):
    """Y-basis: MW pi pulse + two-photon basis rotation with RF phase = 0.25.

    Identical to microwave_two_photon_x except the RF antenna is set
    with phase=0.25 (a 90 deg phase shift), which rotates the basis from
    X to Y on the Bloch sphere. See archive Y_Parity_Optimized_v2 line 658.

    Sequence:
    1. Single pi pulse on 11 transition (t_pi_11_mu)
    2. Set MW IF to two-photon frequency; set RF antenna with phase=0.25
    3. Simultaneous MW + RF pulse for basis rotation (t_pi_2ph_mu // 2)

    click_time_mu is the SPCM timestamp used as timing reference.
    """
    # Step 1: pi pulse (11 transition)
    at_mu(click_time_mu + 2000)
    exp.dds_mw_IF.sw.on()
    at_mu(click_time_mu + int(exp.t_pi_11_mu) + 2000)
    exp.dds_mw_IF.sw.off()
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 2000 + 3300)
    exp.ttl_MW_switch.on()              # block MW briefly

    # Step 2: set two-photon frequencies (RF phase = 0.25 selects Y basis)
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 5000 + 3300)
    exp.dds_mw_IF.set(frequency=exp.f_mw_2ph, amplitude=exp.amp_mw_pulse)
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 15000 + 3300)
    exp.dds_RF.set(frequency=exp.f_RF_2ph, phase=0.25, amplitude=exp.amp_RF_2ph)

    # Step 3: simultaneous MW + RF pulse 2 photon transition
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 3300)
    exp.ttl_MW_switch.off()             # unblock MW
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 4000 + 3300)
    with parallel:
        exp.dds_mw_IF.sw.on()
        exp.dds_RF.sw.on()
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 4000 + (int(exp.t_pi_2ph_mu) // 2) + 3300)
    with parallel:
        exp.dds_mw_IF.sw.off()
        exp.dds_RF.sw.off()
    at_mu(click_time_mu + 100 + int(exp.t_pi_11_mu) + 25000 + 4100 + (int(exp.t_pi_2ph_mu) // 2) + 3300)
    exp.ttl_MW_switch.on()              # block MW


# =========================================================================
#  BLOWAWAY
# =========================================================================

@kernel
def blowaway_f4(exp, amp_MOT_Z_current):
    """Resonant blowaway of F=4 population.

    Uses off shims, cooling at blowaway frequency (99 MHz),
    D1_onoff_L and MOT_Z beams for 500us.

    Parameters
    ----------
    amp_MOT_Z_current : float
        Current MOT Z amplitude to restore after blowaway.
    """
    delay(10 * us)
    exp.zotino0.write_dac(0, exp.shim_X_off)
    exp.zotino0.write_dac(1, exp.shim_Y_off)
    exp.zotino0.write_dac(2, exp.shim_Z1_off)
    exp.zotino0.write_dac(3, exp.shim_Z2_off)
    exp.zotino0.load()
    delay(9 * ms)

    delay(10 * us)
    exp.dds_MOT_Z.set(frequency=exp.f_MOT_Z, amplitude=exp.amp_cooling_blowaway_Z)
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_blowaway, amplitude=exp.amp_cooling_blowaway)
    delay(10 * us)

    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_D1_onoff_L.sw.on()
    delay(1 * us)
    exp.dds_MOT_Z.sw.on()
    delay(exp.t_blowaway)

    # Restore
    exp.dds_cooling_DP.sw.off()
    delay(1 * us)
    exp.dds_MOT_X.sw.on()
    delay(1 * us)
    exp.dds_MOT_Y.sw.on()
    delay(1 * us)

    exp.dds_cooling_DP.set(frequency=exp.f_cooling_restore, amplitude=exp.amp_cooling_restore)
    exp.dds_MOT_Z.set(frequency=exp.f_MOT_Z, amplitude=amp_MOT_Z_current)
    delay(10 * us)


# =========================================================================
#  RECOOLING & SURVIVAL CHECK
# =========================================================================

@kernel
def recool_and_check_survival(exp, count_bg: TInt32) -> TBool:
    """PGC recooling + dual SPCM imaging for atom survival check.

    Returns True if atom is still present (counts above threshold).
    """
    # Imaging
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_imaging, amplitude=exp.amp_cooling_imaging)
    exp.dds_cooling_DP.sw.on()

    with parallel:
        tend1 = exp.ttl_SPCM1.gate_rising(exp.t_imaging_gate)
        tend2 = exp.ttl_SPCM2.gate_rising(exp.t_imaging_gate)
    count1 = exp.ttl_SPCM1.count(tend1)
    count2 = exp.ttl_SPCM2.count(tend2)
    delay(100 * us)

    # PGC
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_deep_PGC, amplitude=exp.amp_cooling_deep_PGC)
    exp.dds_cooling_DP.sw.on()
    delay(10 * us)
    delay(exp.t_PGC_after_img)

    

    return (count1 + count2 - count_bg) > exp.atom_threshold


# =========================================================================
#  PGC BETWEEN EXCITATION CYCLES
# =========================================================================

@kernel
def pgc_between_excitations(exp):
    """Brief PGC recooling between excitation attempts (Orange scope)."""
    delay(1 * us)
    exp.zotino0.write_dac(0, exp.shim_X_off)
    exp.zotino0.write_dac(1, exp.shim_Y_off)
    exp.zotino0.write_dac(2, exp.shim_Z1_off)
    exp.zotino0.write_dac(3, exp.shim_Z2_off)
    exp.zotino0.load()
    delay(1 * ms)

    # Trap back up
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_deep_PGC, amplitude=exp.amp_cooling_deep_PGC)

    exp.dds_MOT_X.sw.on()
    exp.dds_MOT_Y.sw.on()
    exp.dds_MOT_Z.sw.on()
    delay(1 * us)
    exp.dds_D1_onoff_L.sw.on()

    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_repump.sw.on()
    delay(1 * us)
    exp.ttl_repump_switch.off()         # unblock repump
    delay(exp.t_PGC_after_img)
    delay(1 * us)
    exp.dds_cooling_DP.sw.off()
    delay(1 * us)
    exp.dds_repump.sw.off()
    delay(1 * us)
    exp.ttl_repump_switch.on()
    exp.dds_D1_onoff_L.sw.off()


# =========================================================================
#  RESET TO MOT
# =========================================================================

@kernel
def reset_to_mot(exp):
    """Reset all beams and shims to MOT loading configuration."""
    delay(1 * ms)
    delay(1 * us)
    exp.zotino0.write_dac(0, 0.0)
    exp.zotino0.write_dac(1, 0.0)
    exp.zotino0.write_dac(2, 0.0)
    exp.zotino0.write_dac(3, 0.0)
    exp.zotino0.load()
    delay(1 * us)

    exp.dds_cooling_DP.set(frequency=112 * MHz, amplitude=0.35)
    exp.dds_cooling_DP.sw.on()
    delay(1 * ms)

    exp.dds_D1_onoff_L.sw.on()
    delay(5 * us)
    exp.dds_MOT_Z.sw.on()
    delay(5 * us)
    exp.dds_MOT_X.sw.on()
    delay(5 * us)
    exp.dds_MOT_Y.sw.on()
    delay(5 * us)


# =========================================================================
#  FINAL STATE READOUT (Blue scope ending)
# =========================================================================

@kernel
def final_state_readout(exp):
    """PGC + dual SPCM imaging for final-state / survival detection.

    Self-sufficient: turns on its own imaging beams (cooling DP, the MOT X/Y/Z
    fiber AOMs the imaging light reaches the atom through, the D1 on/off PGC
    beam, and repump) so it works whether or not the caller ran blowaway_f4.
    optical_pumping / optical_pumping_F3 leave MOT X/Y/Z and D1_onoff_L OFF;
    without reopening them here, a caller that skips blowaway (e.g.
    excitation_scan) would image into a closed fiber AOM -> shot2~bg -> a false
    0% retention.

    Returns (count1, count2) for the final readout.
    """
    # PGC -- open every beam the PGC + imaging needs (cooling DP, the MOT fiber
    # AOMs, D1_onoff_L, repump) so the readout owns its own beam setup.
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_deep_PGC, amplitude=exp.amp_cooling_deep_PGC)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_MOT_X.sw.on()
    exp.dds_MOT_Y.sw.on()
    exp.dds_MOT_Z.sw.on()
    exp.dds_D1_onoff_L.sw.on()
    delay(1 * us)
    exp.dds_repump.sw.on()
    delay(1 * us)
    exp.ttl_repump_switch.off()
    delay(1 * us)
    delay(exp.t_PGC_after_img)

    # Final imaging
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_imaging, amplitude=exp.amp_cooling_imaging)
    exp.dds_cooling_DP.sw.on()

    with parallel:
        tend1 = exp.ttl_SPCM1.gate_rising(exp.t_imaging_gate)
        tend2 = exp.ttl_SPCM2.gate_rising(exp.t_imaging_gate)
    count1 = exp.ttl_SPCM1.count(tend1)
    count2 = exp.ttl_SPCM2.count(tend2)
    delay(100 * us)

    # PGC after final readout
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_deep_PGC, amplitude=exp.amp_cooling_deep_PGC)
    exp.dds_cooling_DP.sw.on()
    delay(10 * us)
    delay(exp.t_PGC_after_img)

    beams_off_for_imaging(exp)

    return count1, count2


# =========================================================================
#  COMPOSITE SEQUENCES (high-level, reusable across experiments)
# =========================================================================

@kernel
def background_measurement_dual(exp, gate_time):
    """Dual-SPCM background measurement: cooling + repump ON, FORT OFF.

    Measures fluorescence background from scattered cooling light with
    no trapped atom present. This is the correct background for atom
    detection thresholding: (signal - background) > threshold.

    Matches the parity experiment's background measurement exactly:
      1. Cooling beam ON at imaging frequency
      2. Repump ON, repump switch unblocked
      3. FORT set to nominal depth but switch OFF (no atom)
      4. Dual SPCM gate
      5. Returns (count1_bg, count2_bg)

    Call BEFORE load_atom().
    """
    # Cooling on at imaging frequency
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_imaging, amplitude=exp.amp_cooling_imaging)
    exp.dds_cooling_DP.sw.on()
    delay(100 * us)

    # Repump on
    exp.dds_repump.sw.on()
    delay(1 * us)
    exp.ttl_repump_switch.off()
    delay(1 * us)

    # FORT set but OFF (no trap = no atom)
    exp.dds_FORT.set(frequency=exp.f_FORT, amplitude=exp.nominal_trapdepth)
    exp.dds_FORT.sw.off()
    delay(100 * us)

    # Dual SPCM background gate
    with parallel:
        tend1 = exp.ttl_SPCM1.gate_rising(gate_time)
        tend2 = exp.ttl_SPCM2.gate_rising(gate_time)
    count1_bg = exp.ttl_SPCM1.count(tend1)
    count2_bg = exp.ttl_SPCM2.count(tend2)
    delay(100 * us)

    return count1_bg, count2_bg


@kernel
def load_atom(exp):
    """Complete atom loading cycle: MOT → compression → PGC → field settle.

    Performs the full sequence:
      1. Set idle (MOT) shims, cooling at MOT frequency, wait t_MOT2PGC
      2. Set compression shims, cooling at compress frequency, wait t_PGC_2_off
      3. Deep PGC at max detuning, 1ms
      4. Set off (B~0) shims, wait t_field_settle for field to decay

    Does NOT do imaging or SPCM readout — call spcm_readout after this.
    """
    # Phase 1: MOT loading
    # UV ON: increases atom density in MOT
    exp.ttl_UV.on()
    delay(1 * us)
    exp.dds_repump.sw.on()
    delay(1 * us)
    exp.ttl_repump_switch.off()
    delay(1 * us)
    exp.dds_FORT.set(frequency=exp.f_FORT, amplitude=exp.nominal_trapdepth)
    exp.dds_FORT.sw.on()
    delay(100 * us)
    delay(exp.t_wait)

    exp.zotino0.write_dac(0, exp.shim_X_idle)
    exp.zotino0.write_dac(1, exp.shim_Y_idle)
    exp.zotino0.write_dac(2, exp.shim_Z1_idle)
    exp.zotino0.write_dac(3, exp.shim_Z2_idle)
    exp.zotino0.load()
    delay(1 * us)

    exp.dds_cooling_DP.set(frequency=exp.f_cooling_MOT, amplitude=exp.amp_cooling_MOT)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    delay(exp.t_MOT2PGC)

    # Phase 2: MOT compression
    exp.zotino0.write_dac(0, exp.shim_X_compression)
    exp.zotino0.write_dac(1, exp.shim_Y_compression)
    exp.zotino0.write_dac(2, exp.shim_Z1_compression)
    exp.zotino0.write_dac(3, exp.shim_Z2_compression)
    exp.zotino0.load()
    delay(1 * us)
    exp.ttl_UV.off()
    delay(1 * us)

    exp.dds_cooling_DP.set(frequency=exp.f_cooling_compress, amplitude=exp.amp_cooling_compress)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    delay(exp.t_PGC_2_off)

    # Phase 3: Deep sub-Doppler PGC
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_deep_PGC, amplitude=exp.amp_cooling_deep_PGC)
    exp.dds_cooling_DP.sw.on()
    delay(1 * ms)

    # Phase 4: Field settle (B~0)
    exp.zotino0.write_dac(0, exp.shim_X_off)
    exp.zotino0.write_dac(1, exp.shim_Y_off)
    exp.zotino0.write_dac(2, exp.shim_Z1_off)
    exp.zotino0.write_dac(3, exp.shim_Z2_off)
    exp.zotino0.load()
    delay(exp.t_field_settle)


@kernel
def load_atom_deterministic(exp):
    """Deterministic atom loader v2: MOT load -> compress -> image+poll
    -> PGC on captured atom.

    Adapted from Node-1's `load_MOT_and_FORT_until_atom`
    (archive/Node1_Experiment/experiment_functions_NODE1.py:638-817) but
    restructured for Node-3: polling happens during a dedicated post-
    compression imaging phase (cooling DP at f_cooling_imaging) for SNR,
    then a brief PGC cools the captured atom before returning.

    Phases:
      A.  MOT loading       idle shims, cooling at f_cooling_MOT, wait
                            t_MOT2PGC. Same as load_atom Phase 1.
      B1. Compression       compression shims, cooling at f_cooling_compress,
                            wait t_PGC_2_off. Same as load_atom Phase 2,
                            ending without doing deep PGC.
      B2. Image + polling   cooling at f_cooling_imaging, MOT XYZ + repump
                            stay on so light-assisted collisions keep
                            occupancy <= 1 atom in the FORT. Poll both
                            SPCMs in parallel. On rate above threshold,
                            break and proceed to Phase C. Polls up to
                            max_polling_tries * loading_timeout_rounds total
                            looks. No feedback runs inside the loop -- the
                            caller stabilizes beams/FORT with one rebalance
                            right before this call (just before MOT
                            formation), matching the load_atom contract.
      C.  Detected          shims -> off (B~0); cooling DP / repump / UV
                            off (MOT XYZ fiber AOMs stay on -- they're
                            shutters, no light passes when cooling DP is
                            off); wait t_MOT_dissipation; PGC at
                            f_cooling_deep_PGC for t_PGC_after_img;
                            cooling/repump off again; return success.
      D.  Timeout           cooling DP / repump / UV off; shims -> off;
                            return atom_loaded=False.

    Post-loading state contract (drop-in replacement for load_atom):
      - FORT on at nominal_trapdepth
      - cooling DP off, dds_repump off, ttl_repump_switch ON (block RF)
      - MOT_X/Y/Z ON (matches load_atom; required for downstream
        spcm_image_dual to actually deliver light to the atom)
      - shims = "off" (B~0)
      - UV off
      - atom is PGC-cooled (new vs v1; v1 left the atom at MOT temperature)

    Returns 7-tuple:
      (atom_loaded, n_tries_total, n_feedback_rounds,
       spcm1_at_load, spcm2_at_load, counts_first_unloaded, atom_loading_time_s)
    n_feedback_rounds now counts polling rounds elapsed without detection (no
    feedback runs in-loop); it is purely a "how many rounds did it take"
    diagnostic. Position in the tuple is unchanged for the applet/test.

    On timeout (atom_loaded=False): spcm1_at_load and spcm2_at_load are 0;
    counts_first_unloaded carries the first polling try's (SPCM1+SPCM2) sum.
    On success: counts_first_unloaded is also populated (first try of this
    shot, before the atom was detected) for histogram calibration.
    """
    # =========================================================
    #  Phase A: MOT loading
    # =========================================================
    exp.ttl_UV.on()
    delay(1 * us)

    # Idle shims for MOT
    exp.zotino0.write_dac(0, exp.shim_X_idle)
    exp.zotino0.write_dac(1, exp.shim_Y_idle)
    exp.zotino0.write_dac(2, exp.shim_Z1_idle)
    exp.zotino0.write_dac(3, exp.shim_Z2_idle)
    exp.zotino0.load()
    delay(1 * ms)                                 # Zotino settle

    # Cooling DP at MOT frequency, repump on, MOT XYZ on, D1_onoff_L on, FORT on
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_MOT, amplitude=exp.amp_cooling_MOT)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_repump.sw.on()
    exp.ttl_repump_switch.off()
    delay(1 * us)
    exp.dds_MOT_X.sw.on()
    exp.dds_MOT_Y.sw.on()
    exp.dds_MOT_Z.sw.on()
    delay(1 * us)
    # D1_onoff_L / PGC beam is "always on during MOT/PGC". optical_pumping_F3
    # switches it OFF and historically only blowaway_f4 switched it back ON, so
    # an experiment that skips blowaway (e.g. excitation_scan) would otherwise
    # load with this beam stuck off -> no MOT/PGC -> no atom. Guarantee it on
    # here so the loader never depends on the prior step having restored it.
    exp.dds_D1_onoff_L.sw.on()
    delay(1 * us)
    exp.dds_FORT.set(frequency=exp.f_FORT, amplitude=exp.nominal_trapdepth)
    exp.dds_FORT.sw.on()
    delay(100 * us)

    # MOT formation
    # delay(exp.t_MOT2PGC)
    delay(100 *ms)

    # =========================================================
    #  Phase B1: MOT compression
    # =========================================================
    # Compression shims
    exp.zotino0.write_dac(0, exp.shim_X_compression)
    exp.zotino0.write_dac(1, exp.shim_Y_compression)
    exp.zotino0.write_dac(2, exp.shim_Z1_compression)
    exp.zotino0.write_dac(3, exp.shim_Z2_compression)
    exp.zotino0.load()
    delay(1 * us)
    exp.ttl_UV.off()                              # UV not needed past MOT load
    delay(1 * us)

    # Cooling DP at compression frequency
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_compress, amplitude=exp.amp_cooling_compress)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    delay(exp.t_PGC_2_off)

    # =========================================================
    #  Phase B2: Switch cooling to imaging freq, then poll
    # =========================================================
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_imaging, amplitude=exp.amp_cooling_imaging)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)

    # Initialize diagnostics
    t_before_atom_mu = now_mu()
    atom_loaded = False
    total_tries = 0
    fb_rounds = 0
    c0_at_load = 0
    c1_at_load = 0
    counts_first_unloaded = 0
    captured_first_unloaded = False

    # Outer round loop. No feedback runs between rounds: the caller does one
    # rebalance right before this call (just before MOT formation), matching
    # load_atom. The outer loop only bounds the total poll budget and refreshes
    # RTIO slack between rounds. The FORT stays at nominal_trapdepth (set in
    # Phase A) for the whole poll.
    while (not atom_loaded) and (fb_rounds < int(exp.loading_timeout_rounds)):
        try_n = 0
        while (not atom_loaded) and (try_n < int(exp.max_polling_tries)):
            delay(50 * us)                        # RTIO slack
            with parallel:
                exp.ttl0_counter.gate_rising(exp.t_atom_check_time)
                exp.ttl1_counter.gate_rising(exp.t_atom_check_time)
            c0 = exp.ttl0_counter.fetch_count()
            c1 = exp.ttl1_counter.fetch_count()
            try_n += 1
            total_tries += 1

            # Capture exactly one MOT-only sample per call for histogram
            if not captured_first_unloaded:
                counts_first_unloaded = c0 + c1
                captured_first_unloaded = True

            # Rate-based threshold (counts/s) — gate-width invariant
            rate = 0.5 * float(c0 + c1) / exp.t_atom_check_time
            if rate > exp.single_atom_threshold_for_loading:
                atom_loaded = True
                c0_at_load = c0
                c1_at_load = c1
                # break out of inner loop via the while-condition

        if atom_loaded:
            break

        # Inner round exhausted without detection. No feedback here -- the
        # caller stabilizes beams/FORT once before the call. The Phase B2
        # beams are still on, so just count the round and refresh RTIO slack
        # (fetch_count() in the inner loop drifts the timeline) before
        # continuing to poll.
        fb_rounds += 1
        exp.core.break_realtime()

    # =========================================================
    #  Recover RTIO slack from fetch_count() drift in the polling loop.
    # =========================================================
    exp.core.break_realtime()

    # =========================================================
    #  Phase D: Timeout branch
    # =========================================================
    if not atom_loaded:
        # Kill cooling light + repump (LAC stops, MOT dissipates),
        # but keep MOT_X/Y/Z ON to match the load_atom contract so that
        # the caller's spcm_image_dual gets light through the fibers.
        exp.dds_cooling_DP.sw.off()
        delay(1 * us)
        exp.ttl_repump_switch.on()                # block repump RF
        exp.dds_repump.sw.off()
        delay(1 * us)
        exp.ttl_UV.off()
        delay(1 * us)
        # Shims to off (B~0)
        exp.zotino0.write_dac(0, exp.shim_X_off)
        exp.zotino0.write_dac(1, exp.shim_Y_off)
        exp.zotino0.write_dac(2, exp.shim_Z1_off)
        exp.zotino0.write_dac(3, exp.shim_Z2_off)
        exp.zotino0.load()
        delay(1 * us)
        return (False, total_tries, fb_rounds, 0, 0, counts_first_unloaded, 0.0)

    # =========================================================
    #  Phase C: Detected -- kill light, dissipate, PGC, return
    # =========================================================
    t_after_atom_mu = now_mu()

    # Shims to off (B~0)
    exp.zotino0.write_dac(0, exp.shim_X_off)
    exp.zotino0.write_dac(1, exp.shim_Y_off)
    exp.zotino0.write_dac(2, exp.shim_Z1_off)
    exp.zotino0.write_dac(3, exp.shim_Z2_off)
    exp.zotino0.load()
    delay(1 * us)

    # Kill cooling light. MOT_X/Y/Z fiber AOMs stay on (they're shutters --
    # no light passes when cooling DP source is off). This stops LAC and
    # lets the surrounding MOT cloud dissipate while the FORT keeps the
    # captured atom.
    exp.dds_cooling_DP.sw.off()
    delay(1 * us)
    exp.ttl_repump_switch.on()                    # block repump RF
    exp.dds_repump.sw.off()
    delay(1 * us)
    exp.ttl_UV.off()
    delay(1 * us)

    # Wait for residual MOT cloud to fall away
    delay(exp.t_MOT_dissipation)

    # PGC on the captured atom: cooling DP at deep PGC freq, repump on
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_deep_PGC, amplitude=exp.amp_cooling_deep_PGC)
    exp.dds_repump.sw.on()
    exp.ttl_repump_switch.off()
    delay(1 * us)
    exp.dds_cooling_DP.sw.on()
    delay(exp.t_PGC_after_img)

    # End PGC: cooling off, repump off, RF blocked. MOT_X/Y/Z stay on
    # (load_atom contract) so the caller's confirmation imaging shot
    # has light reaching the atom.
    exp.dds_cooling_DP.sw.off()
    delay(1 * us)
    exp.dds_repump.sw.off()
    exp.ttl_repump_switch.on()                    # block repump RF
    delay(1 * us)

    # Return success
    t_load = exp.core.mu_to_seconds(t_after_atom_mu - t_before_atom_mu)
    return (True, total_tries, fb_rounds, c0_at_load, c1_at_load,
            counts_first_unloaded, t_load)


@kernel
def spcm_readout_single(exp, gate_time):
    """Bare single-SPCM gate on SPCM1 only. Returns count.

    Does NOT set up beams — call imaging_readout_single for full readout.
    """
    tend = exp.ttl_SPCM1.gate_rising(gate_time)
    count = exp.ttl_SPCM1.count(tend)
    delay(10 * us)
    return count


@kernel
def imaging_readout_single(exp, gate_time):
    """Full single-SPCM imaging readout: cooling + repump ON, SPCM gate.

    Sets cooling to imaging frequency, turns on repump, then takes
    single-SPCM readout with the specified gate time. Returns count.
    """
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_imaging, amplitude=exp.amp_cooling_imaging)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_repump.set(frequency=exp.f_repump, amplitude=exp.amp_repump)
    exp.ttl_repump_switch.off()
    delay(1 * us)
    return spcm_readout_single(exp, gate_time)


# imaging_readout_dual removed: scan experiments now use final_state_readout
# (parity's), which does its own recool + dual-SPCM image + post-PGC + beams-off.


# f3_state_prep removed: F=3 prep unified to optical_pumping_F3 (parity's version).
# All experiments now use optical_pumping_F3, which requires the repump to be
# blocked on entry (callers run beams_off_for_imaging just before it).


@kernel
def set_qaxis(exp, x_q, y_q, z1_q, z2_q):
    """Set quantization axis shim coils and wait for field to settle.

    Call before trap ramp and optical pumping to establish the B-field
    direction for MW/excitation operations.
    """
    delay(1 * ms)
    exp.zotino0.write_dac(0, x_q)
    exp.zotino0.write_dac(1, y_q)
    exp.zotino0.write_dac(2, z1_q)
    exp.zotino0.write_dac(3, z2_q)
    exp.zotino0.load()
    delay(1 * ms)


@kernel
def blowaway_no_coil_relax(exp, amp_MOT_Z_current):
    """Immediate blowaway — does NOT wait for the shim coils/field to relax
    before firing; off shims are set AFTER the blow.

    KEPT FOR LATER USE (not currently called). The experiments now use
    blowaway_f4, which settles B~0 for 9 ms before the blow. This variant
    instead blows within ~20us of the MW pulse to minimize the gap, because
    any gap allows F=4 atoms to depump back to F=3 via stray light. Swap this
    back in if the 9 ms settle in blowaway_f4 degrades blowaway effectiveness.

    Matches v7_packing working code:
      - NO off shims before blowaway; MOT X/Y stay ON; sequential switching;
        off shims set AFTER blowaway.
    """
    # Configure blowaway beams (immediate, no shim pre-settle)
    delay(10 * us)
    exp.dds_MOT_Z.set(frequency=exp.f_MOT_Z, amplitude=exp.amp_cooling_blowaway_Z)
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_blowaway, amplitude=exp.amp_cooling_blowaway)
    delay(10 * us)

    # Blowaway pulse: sequential switching (matches working original)
    exp.dds_cooling_DP.sw.on()
    delay(1 * us)
    exp.dds_D1_onoff_L.sw.on()
    delay(1 * us)
    exp.dds_MOT_Z.sw.on()
    delay(exp.t_blowaway)

    # End blowaway
    exp.dds_cooling_DP.sw.off()
    delay(1 * us)
    exp.dds_MOT_X.sw.on()
    delay(1 * us)
    exp.dds_MOT_Y.sw.on()
    delay(1 * us)

    # Restore cooling and MOT Z frequencies
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_restore, amplitude=exp.amp_cooling_restore)
    exp.dds_MOT_Z.set(frequency=exp.f_MOT_Z, amplitude=amp_MOT_Z_current)
    delay(10 * us)

    # Off shims AFTER blowaway
    delay(1 * us)
    exp.zotino0.write_dac(0, exp.shim_X_off)
    exp.zotino0.write_dac(1, exp.shim_Y_off)
    exp.zotino0.write_dac(2, exp.shim_Z1_off)
    exp.zotino0.write_dac(3, exp.shim_Z2_off)
    exp.zotino0.load()
    delay(1 * ms)


# pgc_long removed: scan experiments now use final_state_readout (parity's),
# whose pre-image PGC provides the post-blowaway recool.


@kernel
def background_measurement(exp):
    """FORT-off background dual-SPCM measurement.

    Zeros shims, turns FORT off, waits 10ms for atoms to fall,
    turns FORT on, then takes dual-SPCM readout.
    Returns (count1_bg, count2_bg).
    """
    exp.zotino0.write_dac(0, 0.0)
    exp.zotino0.write_dac(1, 0.0)
    exp.zotino0.write_dac(2, 0.0)
    exp.zotino0.write_dac(3, 0.0)
    exp.zotino0.write_dac(6, 0.0)
    exp.zotino0.load()
    delay(1 * ms)
    delay(1 * us)
    exp.dds_FORT.sw.off()
    delay(10000 * us)
    exp.dds_FORT.sw.on()
    delay(1 * us)
    count1_bg, count2_bg = spcm_image_dual(exp, exp.t_imaging_gate)
    return count1_bg, count2_bg


@kernel
def reset_mw_scan(exp):
    """Reset beams and switches for next MW scan cycle."""
    delay(10 * ms)
    delay(2000 * us)
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_MOT, amplitude=exp.amp_cooling_deep_PGC)
    delay(2000 * us)
    exp.dds_cooling_DP.sw.on()
    delay(1000 * us)
    exp.ttl_UV.on()
    delay(1000 * us)
    exp.dds_mw_IF.sw.off()
    delay(1 * us)
    exp.dds_RF.sw.off()
    delay(1 * us)
    exp.ttl_MW_switch.on()
    delay(1 * us)


@kernel
def reload_beams(exp):
    """Per-loop beam reload for MW scan: set cooling and repump for fast MOT reload."""
    exp.dds_cooling_DP.sw.off()
    delay(100 * us)
    exp.ttl_repump_switch.on()
    delay(100 * us)
    exp.dds_cooling_DP.set(frequency=exp.f_cooling_reload, amplitude=exp.amp_cooling_reload)
    exp.dds_cooling_DP.sw.on()
    delay(10 * us)
    exp.dds_repump.set(frequency=exp.f_repump, amplitude=exp.amp_repump)
    exp.ttl_repump_switch.off()
    delay(10 * us)
    delay(200 * us)


# =========================================================================
#  MICROWAVE SPECTROSCOPY PULSE SUBROUTINES
# =========================================================================

@kernel
def mw_pulse_00(exp, freq_hz, duration_mu):
    """Single MW pulse on 00 transition.

    Used for Frequency_00 and Time_00 scans.
    freq_hz: MW IF frequency in Hz (scanned or fixed).
    duration_mu: pulse duration in machine units (fixed or scanned).
    """
    exp.dds_mw_IF.set(frequency=freq_hz, amplitude=exp.amp_mw_pulse)
    t_rf = now_mu()
    at_mu(t_rf + 100)
    exp.ttl_MW_switch.off()             # unblock MW
    at_mu(t_rf + 4000)
    exp.dds_mw_IF.sw.on()
    at_mu(t_rf + 4000 + int(duration_mu))
    exp.dds_mw_IF.sw.off()
    at_mu(t_rf + 4100 + int(duration_mu))
    exp.ttl_MW_switch.on()              # block MW


@kernel
def mw_pulse_01(exp, freq_hz, duration_mu):
    """Single MW pulse on 01 transition.

    Used for Frequency_01 and Time_01 scans.
    01 transition has different timing: two 5us delays before pulse.
    """
    exp.dds_mw_IF.set(frequency=freq_hz, amplitude=exp.amp_mw_pulse)
    delay(5 * us)
    exp.ttl_MW_switch.off()             # unblock MW
    delay(5 * us)
    t_rf = now_mu()
    if t_rf > 0:
        at_mu(t_rf)
        exp.dds_mw_IF.sw.on()
        at_mu(t_rf + int(duration_mu))
        exp.dds_mw_IF.sw.off()
    delay(1 * us)
    exp.ttl_MW_switch.on()              # block MW


@kernel
def mw_two_step_11(exp, f01_hz, t01_mu, f11_hz, t11_mu):
    """Fixed 01 pi pulse followed by scanned 11 pulse.

    Used for Frequency_11 and Time_11 scans.
    Step 1: pi pulse on 01 transition (maps population to mF=1).
    Step 2: probe pulse on 11 transition (scanned freq or duration).
    """
    # Step 1: fixed 01 pi pulse
    exp.dds_mw_IF.set(frequency=f01_hz, amplitude=exp.amp_mw_pulse)
    delay(5 * us)
    exp.ttl_MW_switch.off()
    delay(5 * us)
    t_rf = now_mu()
    if t_rf > 0:
        at_mu(t_rf)
        exp.dds_mw_IF.sw.on()
        at_mu(t_rf + int(t01_mu))
        exp.dds_mw_IF.sw.off()
        at_mu(t_rf + 100 + int(t01_mu))
        exp.ttl_MW_switch.on()

        # Step 2: scanned 11 pulse
        at_mu(t_rf + 100 + int(t01_mu) + 1000)
        exp.dds_mw_IF.set(frequency=f11_hz, amplitude=exp.amp_mw_pulse)
        at_mu(t_rf + 100 + int(t01_mu) + 11000)
        exp.ttl_MW_switch.off()
        at_mu(t_rf + 100 + int(t01_mu) + 11000 + 4000)
        exp.dds_mw_IF.sw.on()
        at_mu(t_rf + 100 + int(t01_mu) + 11000 + 4000 + int(t11_mu))
        exp.dds_mw_IF.sw.off()
        at_mu(t_rf + 100 + int(t01_mu) + 11000 + 4100 + int(t11_mu))
        exp.ttl_MW_switch.on()


@kernel
def mw_two_step_2ph(exp, f01_hz, t01_mu, f_mw_2ph_hz, f_rf_hz, t2ph_mu):
    """Fixed 01 pi pulse followed by simultaneous MW+RF two-photon pulse.

    Used for Frequency_2ph and Time_2ph scans.
    Step 1: pi pulse on 01 (maps to mF=1).
    Step 2: two-photon drive (mF=-1 <-> mF=+1) with MW IF + RF antenna.
    """
    # Step 1: fixed 01 pi pulse
    exp.dds_mw_IF.set(frequency=f01_hz, amplitude=exp.amp_mw_pulse)
    delay(5 * us)
    exp.ttl_MW_switch.off()
    delay(5 * us)
    t_rf = now_mu()
    if t_rf > 0:
        at_mu(t_rf)
        exp.dds_mw_IF.sw.on()
        at_mu(t_rf + int(t01_mu))
        exp.dds_mw_IF.sw.off()
        at_mu(t_rf + 100 + int(t01_mu))
        exp.ttl_MW_switch.on()

        # Step 2: two-photon pulse (MW + RF simultaneously)
        at_mu(t_rf + 100 + int(t01_mu) + 1000)
        with parallel:
            exp.dds_mw_IF.set(frequency=f_mw_2ph_hz, amplitude=exp.amp_mw_pulse)
            exp.dds_RF.set(frequency=f_rf_hz, amplitude=exp.amp_RF_2ph)
        at_mu(t_rf + 100 + int(t01_mu) + 11000)
        exp.ttl_MW_switch.off()
        at_mu(t_rf + 100 + int(t01_mu) + 11000 + 4000)
        with parallel:
            exp.dds_mw_IF.sw.on()
            exp.dds_RF.sw.on()
        at_mu(t_rf + 100 + int(t01_mu) + 11000 + 4000 + int(t2ph_mu))
        with parallel:
            exp.dds_mw_IF.sw.off()
            exp.dds_RF.sw.off()
        at_mu(t_rf + 100 + int(t01_mu) + 11000 + 4100 + int(t2ph_mu))
        exp.ttl_MW_switch.on()


@kernel
def mw_two_step_m10(exp, f00_hz, t00_mu, fm10_hz, tm10_mu):
    """Fixed 00 pi pulse followed by scanned m10 pulse.

    Used for Frequency_m10 and Time_m10 scans.
    State starts in |F=3,mF=0> (optical pumping state).
    Step 1: pi pulse on 00 transition |3,0> -> |4,0> (prep).
    Step 2: probe pulse on m10 transition |4,0> <-> |3,-1> (scanned freq or duration).

    Timing structure is identical to mw_two_step_11 (proven), with the 00
    transition used as the fixed prep step instead of 01.
    """
    # Step 1: fixed 00 pi pulse
    exp.dds_mw_IF.set(frequency=f00_hz, amplitude=exp.amp_mw_pulse)
    delay(5 * us)
    exp.ttl_MW_switch.off()
    delay(5 * us)
    t_rf = now_mu()
    if t_rf > 0:
        at_mu(t_rf)
        exp.dds_mw_IF.sw.on()
        at_mu(t_rf + int(t00_mu))
        exp.dds_mw_IF.sw.off()
        at_mu(t_rf + 100 + int(t00_mu))
        exp.ttl_MW_switch.on()

        # Step 2: scanned m10 pulse
        at_mu(t_rf + 100 + int(t00_mu) + 1000)
        exp.dds_mw_IF.set(frequency=fm10_hz, amplitude=exp.amp_mw_pulse)
        at_mu(t_rf + 100 + int(t00_mu) + 11000)
        exp.ttl_MW_switch.off()
        at_mu(t_rf + 100 + int(t00_mu) + 11000 + 4000)
        exp.dds_mw_IF.sw.on()
        at_mu(t_rf + 100 + int(t00_mu) + 11000 + 4000 + int(tm10_mu))
        exp.dds_mw_IF.sw.off()
        at_mu(t_rf + 100 + int(t00_mu) + 11000 + 4100 + int(tm10_mu))
        exp.ttl_MW_switch.on()


@kernel
def ramsey_00(exp, freq_hz, t_half_pi_mu, t_free_mu):
    """Ramsey sequence on 00 transition: pi/2 - free evolution - pi/2.

    freq_hz: MW IF frequency (slightly detuned from resonance).
    t_half_pi_mu: half-pi pulse duration (= t_pi_00 / 2).
    t_free_mu: free evolution time (scanned parameter).
    """
    exp.dds_mw_IF.set(frequency=freq_hz, amplitude=exp.amp_mw_pulse)
    t_rf = now_mu()
    at_mu(t_rf + 100)
    exp.ttl_MW_switch.off()
    # First pi/2
    at_mu(t_rf + 4000)
    exp.dds_mw_IF.sw.on()
    at_mu(t_rf + 4000 + int(t_half_pi_mu))
    exp.dds_mw_IF.sw.off()
    # Free evolution
    at_mu(t_rf + 4100 + int(t_half_pi_mu) + int(t_free_mu))
    # Second pi/2
    exp.dds_mw_IF.sw.on()
    at_mu(t_rf + 4100 + int(t_half_pi_mu) + int(t_free_mu) + int(t_half_pi_mu))
    exp.dds_mw_IF.sw.off()
    at_mu(t_rf + 4200 + int(t_half_pi_mu) + int(t_free_mu) + int(t_half_pi_mu))
    exp.ttl_MW_switch.on()


@kernel
def ramsey_11(exp, f01_hz, t01_mu, f11_hz, t_half_pi_mu, t_free_mu):
    """01 prep followed by Ramsey on 11 transition.

    Step 1: pi pulse on 01 (maps to mF=1).
    Step 2: Ramsey pi/2 - free evolution - pi/2 on 11.
    """
    # Step 1: fixed 01 pi pulse
    exp.dds_mw_IF.set(frequency=f01_hz, amplitude=exp.amp_mw_pulse)
    t_rf = now_mu()
    at_mu(t_rf + 100)
    exp.ttl_MW_switch.off()
    at_mu(t_rf + 4000)
    exp.dds_mw_IF.sw.on()
    at_mu(t_rf + 4000 + int(t01_mu))
    exp.dds_mw_IF.sw.off()
    at_mu(t_rf + 4100 + int(t01_mu))
    exp.ttl_MW_switch.on()

    # Step 2: Ramsey on 11
    at_mu(t_rf + 4100 + int(t01_mu) + 1000)
    exp.dds_mw_IF.set(frequency=f11_hz, amplitude=exp.amp_mw_pulse)
    at_mu(t_rf + 4100 + int(t01_mu) + 2000)
    exp.ttl_MW_switch.off()
    # First pi/2
    at_mu(t_rf + 4100 + int(t01_mu) + 2000 + 1000)
    exp.dds_mw_IF.sw.on()
    at_mu(t_rf + 4100 + int(t01_mu) + 2000 + 1000 + int(t_half_pi_mu))
    exp.dds_mw_IF.sw.off()
    # Free evolution + second pi/2
    at_mu(t_rf + 4100 + int(t01_mu) + 2000 + 1000 + int(t_half_pi_mu) + int(t_free_mu))
    exp.dds_mw_IF.sw.on()
    at_mu(t_rf + 4100 + int(t01_mu) + 2000 + 1000 + int(t_half_pi_mu) + int(t_free_mu) + int(t_half_pi_mu))
    exp.dds_mw_IF.sw.off()
    at_mu(t_rf + 4200 + int(t01_mu) + 2000 + 1000 + int(t_half_pi_mu) + int(t_free_mu) + int(t_half_pi_mu))
    exp.ttl_MW_switch.on()


@kernel
def ramsey_2ph(exp, f01_hz, t01_mu, f_mw_2ph_hz, f_rf_hz, t_half_pi_mu, t_free_mu):
    """01 prep followed by two-photon Ramsey.

    Step 1: pi pulse on 01 (maps to mF=1).
    Step 2: two-photon Ramsey pi/2 - free evolution - pi/2 (MW + RF simultaneously).
    """
    # Step 1: fixed 01 pi pulse
    exp.dds_mw_IF.set(frequency=f01_hz, amplitude=exp.amp_mw_pulse)
    t_rf = now_mu()
    at_mu(t_rf + 100)
    exp.ttl_MW_switch.off()
    at_mu(t_rf + 4000)
    exp.dds_mw_IF.sw.on()
    at_mu(t_rf + 4000 + int(t01_mu))
    exp.dds_mw_IF.sw.off()
    at_mu(t_rf + 4100 + int(t01_mu))
    exp.ttl_MW_switch.on()

    # Step 2: two-photon Ramsey
    at_mu(t_rf + 4100 + int(t01_mu) + 1000)
    exp.dds_mw_IF.set(frequency=f_mw_2ph_hz, amplitude=exp.amp_mw_pulse)
    at_mu(t_rf + 4100 + int(t01_mu) + 2000)
    exp.dds_RF.set(frequency=f_rf_hz, amplitude=exp.amp_RF_2ph)
    at_mu(t_rf + 4100 + int(t01_mu) + 3000)
    exp.ttl_MW_switch.off()
    # First pi/2
    at_mu(t_rf + 4100 + int(t01_mu) + 4000)
    with parallel:
        exp.dds_mw_IF.sw.on()
        exp.dds_RF.sw.on()
    at_mu(t_rf + 4100 + int(t01_mu) + 4000 + int(t_half_pi_mu))
    with parallel:
        exp.dds_mw_IF.sw.off()
        exp.dds_RF.sw.off()
    # Free evolution + second pi/2
    at_mu(t_rf + 4100 + int(t01_mu) + 4000 + int(t_half_pi_mu) + int(t_free_mu))
    with parallel:
        exp.dds_mw_IF.sw.on()
        exp.dds_RF.sw.on()
    at_mu(t_rf + 4100 + int(t01_mu) + 4000 + int(t_half_pi_mu) + int(t_free_mu) + int(t_half_pi_mu))
    with parallel:
        exp.dds_mw_IF.sw.off()
        exp.dds_RF.sw.off()
    at_mu(t_rf + 4200 + int(t01_mu) + 4000 + int(t_half_pi_mu) + int(t_free_mu) + int(t_half_pi_mu))
    exp.ttl_MW_switch.on()
