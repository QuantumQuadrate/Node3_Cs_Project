"""Centralized experiment parameter definitions for SNAQ-Node-3 Cs experiment.

Run this experiment from the ARTIQ dashboard to create or update all
persistent datasets in dataset_db.pyon. Every parameter is a dashboard
argument — edit values in the GUI before running, then they persist for
all subsequent experiments.

Organized by experimental phase, with shim coils and DDS settings paired
per phase for physics clarity.

Units convention:
  - All times displayed in microseconds (us), stored internally in seconds (SI)
  - All frequencies displayed in MHz, stored internally in Hz (SI)
  - IMPORTANT: NumberValue default/min/max must be in SI base units.
    The unit= parameter only controls dashboard display scaling.
  - Machine-unit times (mu) are unitless integers (1 mu ~ 1 ns on this core)
  - Amplitudes and voltages are dimensionless (0-1 or volts)
"""

from artiq.experiment import *


class ExperimentVariables(EnvExperiment):
    """Experiment_Variables"""

    def build(self):
        # =================================================================
        #  TIMING (defaults in seconds, displayed in us)
        # =================================================================
        self.setattr_argument("t_MOT_loading",   NumberValue(0.5,     min=0.0, max=10.0,  unit="us", ndecimals=0), "Timing")
        # Defaults below match Microwave_variables_scan_v7_packing_20260317.py:
        # MOT2PGC=400ms, PGC_2_off=50ms, PGC_Af_Img=20ms. Z parity v2 uses
        # much shorter values (350/1/1 ms) — override per-experiment in
        # prepare() if running parity, or edit in the dashboard.
        self.setattr_argument("t_MOT2PGC",       NumberValue(0.4,     min=0.0, max=10.0,  unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_PGC_2_off",     NumberValue(0.05,    min=0.0, max=10.0,  unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_PGC_after_img", NumberValue(0.02,    min=0.0, max=1.0,   unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_field_settle",  NumberValue(0.3,     min=0.0, max=10.0,  unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_imaging_gate",  NumberValue(4e-3,    min=0.0, max=1.0,   unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_readout_monitor", NumberValue(100e-3, min=0.0, max=1.0,  unit="ms", ndecimals=0), "Timing")
        self.setattr_argument("t_wait",          NumberValue(1e-3,    min=0.0, max=1.0,   unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_SPCM_exposure", NumberValue(0.15,    min=0.0, max=10.0,  unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_blowaway",      NumberValue(500e-6,  min=0.0, max=0.1,   unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_optical_pump",  NumberValue(1e-3,    min=0.0, max=0.1,   unit="us", ndecimals=0), "Timing")
        self.setattr_argument("t_PGC_after_blowaway", NumberValue(10e-3, min=0.0, max=1.0, unit="us", ndecimals=0), "Timing")

        # =================================================================
        #  TRAP
        # =================================================================
        self.setattr_argument("nominal_trapdepth",   NumberValue(0.24,  min=0, max=1, ndecimals=3), "Trap")
        self.setattr_argument("trap_ramp_start_amp", NumberValue(0.24, min=0, max=0.3, ndecimals=3), "Trap")
        self.setattr_argument("trap_ramp_steps",     NumberValue(200,     min=0, max=100000, ndecimals=0), "Trap")
        self.setattr_argument("trap_ramp_dt_us",     NumberValue(50.0,   min=0.001, max=1000, ndecimals=3), "Trap")

        # =================================================================
        #  PHASE 1: MOT LOADING ("idle")
        #  Large Z-gradient for MOT, low detuning (112 MHz) = bright dense MOT.
        # =================================================================
        self.setattr_argument("shim_X_idle",     NumberValue(0.25,  min=-10, max=10, ndecimals=3), "Phase 1 - MOT Loading")
        self.setattr_argument("shim_Y_idle",     NumberValue(0.1,   min=-10, max=10, ndecimals=3), "Phase 1 - MOT Loading")
        self.setattr_argument("shim_Z1_idle",    NumberValue(4.25,  min=-10, max=10, ndecimals=3), "Phase 1 - MOT Loading")
        self.setattr_argument("shim_Z2_idle",    NumberValue(4.2,   min=-10, max=10, ndecimals=3), "Phase 1 - MOT Loading")
        self.setattr_argument("f_cooling_MOT",   NumberValue(112e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Phase 1 - MOT Loading")
        self.setattr_argument("amp_cooling_MOT", NumberValue(0.3,   min=0, max=1, ndecimals=3), "Phase 1 - MOT Loading")

        # =================================================================
        #  PHASE 2: MOT COMPRESSION (historically mislabeled "PGC")
        #  Compress MOT onto FORT tweezer, moderate detuning (120 MHz).
        # =================================================================
        # Compression shim values from HS_code_20260416_Z_Parity_Optimized_v2.py
        # (newest production calibration). MW scan v7 used slightly different
        # values (X=0.22, Y=0.3, Z1=4.26) — set manually in dashboard if running
        # MW scan and the difference matters.
        self.setattr_argument("shim_X_compression",  NumberValue(0.2,   min=-10, max=10, ndecimals=3), "Phase 2 - MOT Compression")
        self.setattr_argument("shim_Y_compression",  NumberValue(0.2,   min=-10, max=10, ndecimals=3), "Phase 2 - MOT Compression")
        self.setattr_argument("shim_Z1_compression", NumberValue(4.25,  min=-10, max=10, ndecimals=3), "Phase 2 - MOT Compression")
        self.setattr_argument("shim_Z2_compression", NumberValue(4.2,   min=-10, max=10, ndecimals=3), "Phase 2 - MOT Compression")
        self.setattr_argument("f_cooling_compress",  NumberValue(120e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Phase 2 - MOT Compression")
        self.setattr_argument("amp_cooling_compress", NumberValue(0.36, min=0, max=1, ndecimals=3), "Phase 2 - MOT Compression")

        # =================================================================
        #  PHASE 3: SUB-DOPPLER PGC (historically mislabeled "off")
        #  True PGC: B~0 (compensate ambient), max detuning (128 MHz).
        #  X/Z values nearly match Q-axis (both compensate background field).
        # =================================================================
        self.setattr_argument("shim_X_off",      NumberValue(0.0,   min=-10, max=10, ndecimals=3), "Phase 3 - Sub-Doppler PGC (B~0)")
        self.setattr_argument("shim_Y_off",      NumberValue(0.7,   min=-10, max=10, ndecimals=3), "Phase 3 - Sub-Doppler PGC (B~0)")
        self.setattr_argument("shim_Z1_off",     NumberValue(-0.05, min=-10, max=10, ndecimals=3), "Phase 3 - Sub-Doppler PGC (B~0)")
        self.setattr_argument("shim_Z2_off",     NumberValue(0.05,  min=-10, max=10, ndecimals=3), "Phase 3 - Sub-Doppler PGC (B~0)")
        self.setattr_argument("f_cooling_deep_PGC",  NumberValue(128e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Phase 3 - Sub-Doppler PGC (B~0)")
        self.setattr_argument("amp_cooling_deep_PGC", NumberValue(0.35, min=0, max=1, ndecimals=3), "Phase 3 - Sub-Doppler PGC (B~0)")

        # =================================================================
        #  PHASE 4: QUANTIZATION AXIS (MW and excitation operations)
        #  Like PGC (background compensation) but large Y for Q-axis.
        # =================================================================
        self.setattr_argument("shim_X_qaxis",    NumberValue(-0.013, min=-10, max=10, ndecimals=3), "Phase 4 - Quantization Axis")
        self.setattr_argument("shim_Y_qaxis",    NumberValue(2.72,   min=-10, max=10, ndecimals=3), "Phase 4 - Quantization Axis")
        self.setattr_argument("shim_Z1_qaxis",   NumberValue(-0.06,  min=-10, max=10, ndecimals=3), "Phase 4 - Quantization Axis")
        self.setattr_argument("shim_Z2_qaxis",   NumberValue(0.06,   min=-10, max=10, ndecimals=3), "Phase 4 - Quantization Axis")

        # =================================================================
        #  DDS IDLE DEFAULTS (defaults in Hz, displayed in MHz)
        # =================================================================
        # Urukul0
        self.setattr_argument("f_mw_IF",            NumberValue(200e6,  min=0, max=500e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul0")
        self.setattr_argument("amp_mw_IF",          NumberValue(0.55,   min=0, max=1, ndecimals=3), "DDS Idle - Urukul0")
        self.setattr_argument("f_cooling_DP_idle",  NumberValue(111e6,  min=50e6, max=200e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul0")
        self.setattr_argument("amp_cooling_DP_idle", NumberValue(0.45,  min=0, max=1, ndecimals=3), "DDS Idle - Urukul0")
        self.setattr_argument("f_repump",           NumberValue(197e6,  min=100e6, max=300e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul0")
        self.setattr_argument("amp_repump",         NumberValue(0.5,    min=0, max=1, ndecimals=3), "DDS Idle - Urukul0")
        self.setattr_argument("f_D1_pump",          NumberValue(145e6,  min=50e6, max=300e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul0")
        self.setattr_argument("amp_D1_pump",        NumberValue(0.1,    min=0, max=1, ndecimals=3), "DDS Idle - Urukul0")
        # Urukul1
        self.setattr_argument("f_RF",               NumberValue(2e6,    min=0, max=50e6, unit="MHz", ndecimals=6), "DDS Idle - Urukul1")
        self.setattr_argument("amp_RF",             NumberValue(0.1,    min=0, max=1, ndecimals=3), "DDS Idle - Urukul1")
        self.setattr_argument("f_FORT",             NumberValue(150e6,  min=50e6, max=300e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul1")
        self.setattr_argument("amp_FORT",           NumberValue(0.24,   min=0, max=1, ndecimals=3), "DDS Idle - Urukul1")
        self.setattr_argument("f_excitation_idle",  NumberValue(154e6,  min=50e6, max=300e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul1")
        self.setattr_argument("amp_excitation_idle", NumberValue(0.12,  min=0, max=1, ndecimals=3), "DDS Idle - Urukul1")
        self.setattr_argument("f_D1_onoff_L",      NumberValue(250e6,  min=100e6, max=400e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul1")
        self.setattr_argument("amp_D1_onoff_L",    NumberValue(0.43,   min=0, max=1, ndecimals=3), "DDS Idle - Urukul1")
        # Urukul2
        self.setattr_argument("f_MOT_X",            NumberValue(80e6,  min=0, max=200e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul2")
        self.setattr_argument("amp_MOT_X",          NumberValue(0.12,  min=0, max=0.5, ndecimals=3), "DDS Idle - Urukul2")
        self.setattr_argument("f_MOT_Y",            NumberValue(80e6,  min=0, max=200e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul2")
        self.setattr_argument("amp_MOT_Y",          NumberValue(0.10,  min=0, max=0.5, ndecimals=3), "DDS Idle - Urukul2")
        self.setattr_argument("f_MOT_Z",            NumberValue(80e6,  min=0, max=200e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul2")
        self.setattr_argument("amp_MOT_Z",          NumberValue(0.08,  min=0, max=0.5, ndecimals=3), "DDS Idle - Urukul2")
        self.setattr_argument("f_pump_repumper",    NumberValue(202e6, min=100e6, max=300e6, unit="MHz", ndecimals=3), "DDS Idle - Urukul2")
        self.setattr_argument("amp_pump_repumper",  NumberValue(0.3,   min=0, max=1, ndecimals=3), "DDS Idle - Urukul2")

        # =================================================================
        #  COOLING DP SETTINGS PER STAGE (defaults in Hz, displayed in MHz)
        # =================================================================
        self.setattr_argument("f_cooling_PGC",        NumberValue(120e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_PGC",      NumberValue(0.36,  min=0, max=1, ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("f_cooling_imaging",    NumberValue(120e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_imaging",  NumberValue(0.36,  min=0, max=1, ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("f_cooling_blowaway",   NumberValue(99e6,  min=50e6, max=200e6, unit="MHz", ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_blowaway", NumberValue(0.4,   min=0, max=1, ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_blowaway_Z", NumberValue(0.11, min=0, max=0.5, ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("f_cooling_restore",    NumberValue(130e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_restore",  NumberValue(0.35,  min=0, max=1, ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("f_cooling_pre_wait",   NumberValue(124e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_pre_wait", NumberValue(0.4,   min=0, max=1, ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("f_cooling_feedback",   NumberValue(122e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_feedback", NumberValue(0.3,   min=0, max=1, ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("f_cooling_reload",     NumberValue(111e6, min=50e6, max=200e6, unit="MHz", ndecimals=3), "Cooling DP per stage")
        self.setattr_argument("amp_cooling_reload",   NumberValue(0.47,  min=0, max=1, ndecimals=3), "Cooling DP per stage")

        # =================================================================
        #  OPTICAL PUMPING DDS SETTINGS (defaults in Hz, displayed in MHz)
        # =================================================================
        self.setattr_argument("f_pump_D1",              NumberValue(145e6, min=50e6, max=300e6, unit="MHz", ndecimals=3), "Optical Pumping")
        self.setattr_argument("amp_pump_D1",            NumberValue(0.03,  min=0, max=1, ndecimals=3), "Optical Pumping")
        self.setattr_argument("f_pump_excitation",      NumberValue(150e6, min=50e6, max=300e6, unit="MHz", ndecimals=3), "Optical Pumping")
        self.setattr_argument("amp_pump_excitation",    NumberValue(0.04,  min=0, max=1, ndecimals=3), "Optical Pumping")
        self.setattr_argument("amp_pump_repumper_active", NumberValue(0.4, min=0, max=1, ndecimals=3), "Optical Pumping")

        # =================================================================
        #  MICROWAVE TRANSITIONS (defaults in Hz, displayed in MHz)
        # =================================================================
        self.setattr_argument("f_mw_00",           NumberValue(250.000e6, min=190e6, max=260e6, unit="MHz", ndecimals=6), "Microwave Transitions")
        self.setattr_argument("f_mw_01",           NumberValue(250.468e6, min=190e6, max=260e6, unit="MHz", ndecimals=6), "Microwave Transitions")
        # f_mw_11 default from HS_code_20260416_Z_Parity_Optimized_v2.py
        # ("UPDATE 11 RES HERE!!" line 560). MW scan v7 used 250.972.
        self.setattr_argument("f_mw_11",           NumberValue(250.9688e6, min=190e6, max=260e6, unit="MHz", ndecimals=6), "Microwave Transitions")
        # m10 transition |F=3,mF=-1> <-> |F=4,mF=0> (sigma+, near the 01 line).
        # Reached via a 00 pi-pulse prep (|3,0>->|4,0>) then this probe pulse.
        # Default is a PLACEHOLDER seeded from f_mw_01 — calibrate via Frequency_m10 scan.
        self.setattr_argument("f_mw_m10",          NumberValue(250.468e6, min=190e6, max=260e6, unit="MHz", ndecimals=6), "Microwave Transitions")
        self.setattr_argument("amp_mw_pulse",      NumberValue(0.7,  min=0, max=1, ndecimals=3), "Microwave Transitions")
        # Pi-pulse times (machine units, unitless integers)
        self.setattr_argument("t_pi_00_mu",        NumberValue(4530,    min=0, max=1e7, ndecimals=0), "Microwave Pi-Pulse Times (mu)")
        self.setattr_argument("t_pi_01_mu",        NumberValue(31876,   min=0, max=1e7, ndecimals=0), "Microwave Pi-Pulse Times (mu)")
        # t_pi_11_mu: pi rotation on the 11 transition. Also used as the
        # parity pi pulse (parity.py, all of X/Y/Z). Default 4771 from
        # HS_code_20260416_Z_Parity_Optimized_v2.py (newest calibration).
        # MW scan v7 measured 5033 — recalibrate via Time_11 scan if needed.
        self.setattr_argument("t_pi_11_mu",        NumberValue(4771,    min=0, max=1e7, ndecimals=0), "Microwave Pi-Pulse Times (mu)")
        self.setattr_argument("t_pi_2ph_mu",       NumberValue(243000,  min=0, max=1e7, ndecimals=0), "Microwave Pi-Pulse Times (mu)")
        # t_pi_m10_mu: pi time for the m10 probe pulse. PLACEHOLDER seeded from
        # t_pi_01_mu — calibrate via Time_m10 (Rabi) scan.
        self.setattr_argument("t_pi_m10_mu",       NumberValue(31876,   min=0, max=1e7, ndecimals=0), "Microwave Pi-Pulse Times (mu)")
        # Two-photon (defaults in Hz, displayed in MHz).
        # f_mw_2ph / f_RF_2ph are now DERIVED in the experiment from f_mw_00,
        # f_mw_01, rf_detune_khz, rf_small_detune_khz (parity.py prepare /
        # microwave_scan.py). Kept commented as an artifact in case we want to
        # revert to independent inputs.
        # self.setattr_argument("f_mw_2ph",          NumberValue(249.383e6, min=190e6, max=260e6, unit="MHz", ndecimals=6), "Microwave Two-Photon")
        # self.setattr_argument("f_RF_2ph",          NumberValue(0.6081e6,  min=0, max=10e6, unit="MHz", ndecimals=6), "Microwave Two-Photon")
        self.setattr_argument("amp_RF_2ph",        NumberValue(0.6,   min=0, max=1, ndecimals=3), "Microwave Two-Photon")
        self.setattr_argument("rf_detune_khz",     NumberValue(-125.0, min=-1000, max=1000, ndecimals=3), "Microwave Two-Photon")
        self.setattr_argument("rf_small_detune_khz", NumberValue(-9.33, min=-100, max=100, ndecimals=3), "Microwave Two-Photon")

        # =================================================================
        #  EXCITATION PARAMETERS (frequency in Hz displayed as MHz, timings in mu)
        # =================================================================
        self.setattr_argument("max_excitation_cycles", NumberValue(3,    min=1, max=8000, ndecimals=0), "Excitation")
        self.setattr_argument("t_trap_off_mu",         NumberValue(400,   min=0, max=8000, ndecimals=0), "Excitation")
        self.setattr_argument("t_spcm_collection_mu",  NumberValue(400,  min=0, max=8000, ndecimals=0), "Excitation")
        self.setattr_argument("length_offset_mu",      NumberValue(12,   min=0, max=8000, ndecimals=0), "Excitation")
        self.setattr_argument("f_excitation_pulse",    NumberValue(145e6, min=50e6, max=300e6, unit="MHz", ndecimals=3), "Excitation")
        self.setattr_argument("amp_excitation_pulse",  NumberValue(0.13, min=0, max=1, ndecimals=3), "Excitation")
        self.setattr_argument("exc_trap_off_offset_mu",  NumberValue(4215, min=0, max=50000, ndecimals=0), "Excitation Timing (mu)")
        self.setattr_argument("exc_d1_switch_offset_mu", NumberValue(4028, min=0, max=50000, ndecimals=0), "Excitation Timing (mu)")
        self.setattr_argument("exc_atten_offset_mu",     NumberValue(4640, min=0, max=50000, ndecimals=0), "Excitation Timing (mu)")
        self.setattr_argument("exc_beam_on_offset_mu",   NumberValue(3800, min=0, max=50000, ndecimals=0), "Excitation Timing (mu)")
        self.setattr_argument("exc_beam_duration_mu",    NumberValue(300,  min=0, max=50000, ndecimals=0), "Excitation Timing (mu)")
        self.setattr_argument("exc_spcm_gate_offset_mu", NumberValue(5280, min=0, max=50000, ndecimals=0), "Excitation Timing (mu)")

        # =================================================================
        #  DETECTION
        # =================================================================
        self.setattr_argument("atom_threshold",  NumberValue(12, min=0, max=1000, ndecimals=0), "Detection")
        self.setattr_argument("n_measurements",  NumberValue(10, min=1, max=600000, ndecimals=0), "Detection")

        # =================================================================
        #  DETERMINISTIC LOADING
        #  Used by load_atom_deterministic() in subroutines/sequences.py.
        #  See archive/Node1_Experiment/experiment_functions_NODE1.py
        #  function load_MOT_and_FORT_until_atom (Node-1 reference).
        # =================================================================
        # Gate width per polling try. 100 us collects ~10 photons at single-atom
        # rate (~100 kcps) and lets 100 polls fit in ~10 ms (≪ MOT lifetime).
        # Floor at 10 us prevents RTIO underflow.
        self.setattr_argument("t_atom_check_time",
                              NumberValue(100e-6, min=10e-6, max=10e-3, unit="us", ndecimals=1),
                              "Deterministic Loading")
        # Atom-presence threshold as count rate (counts/second), gate-width
        # invariant. Default ~1e5 cps = 10 counts per 100 us gate. Calibrate
        # via histogram (see plan Step 4) before relying on it.
        self.setattr_argument("single_atom_threshold_for_loading",
                              NumberValue(1.0e5, min=1.0, max=1.0e8, ndecimals=1),
                              "Deterministic Loading")
        # Wait for residual MOT cloud to dissipate after MOT-off.
        self.setattr_argument("t_MOT_dissipation",
                              NumberValue(5e-3, min=0.0, max=1.0, unit="ms", ndecimals=2),
                              "Deterministic Loading")
        # Polling gates per round before triggering MOT-only feedback.
        self.setattr_argument("max_polling_tries",
                              NumberValue(100, min=1, max=10000, ndecimals=0),
                              "Deterministic Loading")
        # Failed rounds before returning atom_loaded=False (caller decides).
        # Total budget: max_polling_tries × loading_timeout_rounds × t_atom_check_time
        # ≈ 100 × 30 × 100 us = 300 ms polling + feedback overhead per shot.
        self.setattr_argument("loading_timeout_rounds",
                              NumberValue(30, min=1, max=1000, ndecimals=0),
                              "Deterministic Loading")

        # =================================================================
        #  FEEDBACK parameters are stored in config/feedback_channels.json
        #  (loaded by BaseExperiment._load_feedback_params)
        #  NOT duplicated here — edit the JSON to change feedback tuning.
        # =================================================================

        # =================================================================
        #  WAVEPLATES
        # =================================================================
        self.setattr_argument("HWP_angle_min",  NumberValue(0.0,   min=0, max=360, ndecimals=1), "Waveplates")
        self.setattr_argument("HWP_angle_max",  NumberValue(360.0, min=0, max=3600, ndecimals=1), "Waveplates")
        self.setattr_argument("HWP_angle_step", NumberValue(30.0,  min=0.1, max=360, ndecimals=1), "Waveplates")
        self.setattr_argument("QWP_angle_min",  NumberValue(0.0,   min=0, max=360, ndecimals=1), "Waveplates")
        self.setattr_argument("QWP_angle_max",  NumberValue(360.0, min=0, max=360, ndecimals=1), "Waveplates")
        self.setattr_argument("QWP_angle_step", NumberValue(30.0,  min=0.1, max=360, ndecimals=1), "Waveplates")
        self.setattr_argument("deg_to_pos",     NumberValue(136533, min=1, max=1e6, ndecimals=0), "Waveplates")

    def run(self):
        """Write all argument values to persistent datasets."""
        for key in sorted(self.__dict__):
            if key.startswith("_") or key in ("core", "scheduler", "ccb"):
                continue
            try:
                val = getattr(self, key)
                if isinstance(val, (int, float, bool)):
                    self.set_dataset(key, val, persist=True, broadcast=True)
            except Exception:
                pass

        # Monitoring datasets (broadcast-only)
        self.set_dataset("SPCM1_counts",        0, broadcast=True)
        self.set_dataset("SPCM2_counts",        0, broadcast=True)
        self.set_dataset("measurement_progress", 0, broadcast=True)
        self.set_dataset("atom_loaded",         False, broadcast=True)
        self.set_dataset("feedback_MOT_X",      0.0, broadcast=True)
        self.set_dataset("feedback_MOT_Y",      0.0, broadcast=True)
        self.set_dataset("feedback_MOT_Z",      0.0, broadcast=True)
        self.set_dataset("feedback_FORT",       0.0, broadcast=True)
        self.set_dataset("feedback_excitation", 0.0, broadcast=True)

        # Path config
        self.set_dataset("results_dir", "results", persist=True, broadcast=True)

        print("ExperimentVariables: persisted all datasets.")
