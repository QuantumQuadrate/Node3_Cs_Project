"""BaseExperiment: composition-based foundation for all Cs experiments.

Usage in any experiment::

    class MyExperiment(EnvExperiment):
        def build(self):
            self.base = BaseExperiment(self)
            self.base.build()
            # add experiment-specific arguments here

        def prepare(self):
            self.base.prepare()

        @kernel
        def run(self):
            self.base.initialize_hardware()
            # physics here
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path

from artiq.experiment import *
from artiq.coredevice.ad9910 import PHASE_MODE_ABSOLUTE, PHASE_MODE_CONTINUOUS

from utilities.device_aliases import register_devices, load_device_aliases
from utilities.calibration_store import CalibrationStore
from utilities.write_results import record_start_time


class BaseExperiment:
    """Composition-based base for all SNAQ-Node-3 Cs experiments.

    Handles device registration, parameter loading, and hardware
    initialization so that experiment classes stay focused on physics.
    """

    def __init__(self, experiment):
        self.experiment = experiment

    # -----------------------------------------------------------------
    #  BUILD PHASE (host side, called once)
    # -----------------------------------------------------------------

    def build(self):
        """Register all devices and load persistent datasets as attributes."""
        exp = self.experiment

        # Register all hardware devices and create aliases
        register_devices(exp)

        # Scheduler (provides RID for write_results and provenance)
        exp.setattr_device("scheduler")

        # Load all persistent datasets as experiment attributes
        self._load_datasets()

        # Initialize calibration store and load feedback references
        self.cal_store = CalibrationStore()
        self._load_feedback_refs()

        # Create and configure rebalancer (devices + params from JSON)
        self._setup_rebalancer()

    def _load_datasets(self):
        """Load every persistent dataset from dataset_db as an attribute."""
        exp = self.experiment
        try:
            dataset_mgr = exp._HasEnvironment__dataset_mgr
            # Read from archive (persistent datasets)
            for key in list(dataset_mgr.local.keys()):
                try:
                    val = exp.get_dataset(key)
                    setattr(exp, key, val)
                except Exception:
                    pass
        except Exception:
            pass

        # Load all datasets we know about by trying to get them
        # This covers the case where dataset_db.pyon already exists
        dataset_names = self._get_known_dataset_names()
        for name in dataset_names:
            if not hasattr(exp, name):
                try:
                    val = exp.get_dataset(name)
                    setattr(exp, name, val)
                except KeyError:
                    pass  # dataset not yet created

    def _get_known_dataset_names(self):
        """Return list of all dataset names we expect to exist."""
        return [
            # Timing
            "t_MOT_loading", "t_MOT2PGC", "t_PGC_2_off", "t_PGC_after_img",
            "t_field_settle", "t_imaging_gate", "t_readout_monitor", "t_wait", "t_SPCM_exposure",
            "t_blowaway", "t_optical_pump",
            # Trap
            "nominal_trapdepth", "trap_ramp_start_amp", "trap_ramp_steps",
            "trap_ramp_dt_us",
            # Shim coils
            "shim_X_idle", "shim_Y_idle", "shim_Z1_idle", "shim_Z2_idle",
            "shim_X_compression", "shim_Y_compression", "shim_Z1_compression", "shim_Z2_compression",
            "shim_X_off", "shim_Y_off", "shim_Z1_off", "shim_Z2_off",
            "shim_X_qaxis", "shim_Y_qaxis", "shim_Z1_qaxis", "shim_Z2_qaxis",
            # DDS defaults
            "f_mw_IF", "amp_mw_IF",
            "f_cooling_DP_idle", "amp_cooling_DP_idle",
            "f_repump", "amp_repump",
            "f_D1_pump", "amp_D1_pump",
            "f_RF", "amp_RF",
            "f_FORT", "amp_FORT",
            "f_excitation_idle", "amp_excitation_idle",
            "f_D1_onoff_L", "amp_D1_onoff_L",
            "f_MOT_X", "amp_MOT_X",
            "f_MOT_Y", "amp_MOT_Y",
            "f_MOT_Z", "amp_MOT_Z",
            "f_pump_repumper", "amp_pump_repumper",
            # Cooling phases
            "f_cooling_MOT", "amp_cooling_MOT",
            "f_cooling_PGC", "amp_cooling_PGC",
            "f_cooling_deep_PGC", "amp_cooling_deep_PGC",
            "f_cooling_imaging", "amp_cooling_imaging",
            "f_cooling_blowaway", "amp_cooling_blowaway",
            "amp_cooling_blowaway_Z",
            "f_cooling_restore", "amp_cooling_restore",
            "f_cooling_pre_wait", "amp_cooling_pre_wait",
            "f_cooling_compress", "amp_cooling_compress",
            "f_cooling_feedback", "amp_cooling_feedback",
            "f_cooling_reload", "amp_cooling_reload",
            "t_PGC_after_blowaway",
            # Microwave
            "f_mw_00", "f_mw_01", "f_mw_11", "f_mw_m10",
            "t_pi_00_mu", "t_pi_01_mu", "t_pi_11_mu", "t_pi_m10_mu", "t_pi_2ph_mu",
            "amp_mw_pulse",
            # f_mw_2ph / f_RF_2ph are derived in parity.prepare() (artifact,
            # see experiment_variables.py) -- not loaded as datasets:
            # "f_mw_2ph", "f_RF_2ph",
            "amp_RF_2ph",
            "rf_detune_khz", "rf_small_detune_khz",
            # Optical pumping
            "f_pump_D1", "amp_pump_D1",
            "f_pump_excitation", "amp_pump_excitation",
            "amp_pump_repumper_active",
            # Excitation
            "max_excitation_cycles", "t_trap_off_mu", "t_spcm_collection_mu",
            "length_offset_mu", "f_excitation_pulse", "amp_excitation_pulse",
            "exc_trap_off_offset_mu", "exc_d1_switch_offset_mu",
            "exc_atten_offset_mu", "exc_beam_on_offset_mu",
            "exc_beam_duration_mu", "exc_spcm_gate_offset_mu",
            # Detection
            "atom_threshold", "n_measurements",
            # Deterministic loading (see subroutines/sequences.py::load_atom_deterministic)
            "t_atom_check_time", "single_atom_threshold_for_loading",
            "t_MOT_dissipation", "max_polling_tries", "loading_timeout_rounds",
            # Feedback params now loaded from feedback_channels.json
            # via _load_feedback_params(), not from datasets
            # Waveplate
            "HWP_angle_min", "HWP_angle_max", "HWP_angle_step",
            "QWP_angle_min", "QWP_angle_max", "QWP_angle_step",
            "deg_to_pos",
            # Paths
            "results_dir",
        ]

    def _load_feedback_refs(self, profile=None):
        """Load feedback reference voltages from the calibration store.

        Parameters
        ----------
        profile : str or None
            Calibration profile name. If None, uses the active profile.
        """
        exp = self.experiment
        exp.stored_refs = self.cal_store.get_refs(profile)
        exp._feedback_profile = profile or self.cal_store.active_profile

        # Warn if stale
        if self.cal_store.is_stale(exp._feedback_profile, max_hours=24):
            age = self.cal_store.get_ref_age(exp._feedback_profile)
            if age is None:
                print(f"WARNING: Feedback profile '{exp._feedback_profile}' has NEVER been calibrated.")
            else:
                hours = age.total_seconds() / 3600
                print(f"WARNING: Feedback profile '{exp._feedback_profile}' is {hours:.1f}h old.")

    def _setup_rebalancer(self):
        """Create FeedbackRebalancer_5, assign devices, load params from JSON.

        After this, experiments just call self.rebalancer.rebalance(profile).
        """
        from subroutines.aom_feedback import FeedbackRebalancer_5

        exp = self.experiment
        exp.rebalancer = FeedbackRebalancer_5()

        # Assign devices
        exp.rebalancer.core = exp.core
        exp.rebalancer.sampler0 = exp.sampler0
        exp.rebalancer.urukul2_ch0 = exp.dds_MOT_X
        exp.rebalancer.urukul2_ch1 = exp.dds_MOT_Y
        exp.rebalancer.urukul2_ch2 = exp.dds_MOT_Z
        exp.rebalancer.urukul1_ch1 = exp.dds_FORT
        exp.rebalancer.urukul1_ch2 = exp.dds_excitation
        exp.rebalancer.ttl14 = exp.ttl_D1_exc_switch
        exp.rebalancer.ttl6 = exp.ttl_exc_atten1

        # Load tuning params from feedback_channels.json
        exp.rebalancer.load_params_from_json()

        # Give rebalancer access to experiment for saving diagnostics to datasets
        exp.rebalancer._experiment = exp

    # -----------------------------------------------------------------
    #  PREPARE PHASE (host side, called before each run)
    # -----------------------------------------------------------------

    def prepare(self, feedback_profile=None):
        """Compute derived values and build scan sequences.

        Parameters
        ----------
        feedback_profile : str or None
            Override the calibration profile for this run.
        """
        exp = self.experiment

        # Refresh feedback refs (may have been updated since build)
        self._load_feedback_refs(feedback_profile)

        # Load ref voltages onto rebalancer for the active profile
        profile = feedback_profile or self.cal_store.active_profile
        exp.rebalancer.load_profile(profile)

    # -----------------------------------------------------------------
    #  HARDWARE INITIALIZATION (kernel side)
    # -----------------------------------------------------------------

    @kernel
    def initialize_hardware(self):
        """Initialize all hardware: core, CPLDs, DDS, TTLs, DAC, ADC.

        Call this at the start of every experiment's run() method.
        """
        exp = self.experiment

        exp.core.reset()
        exp.core.break_realtime()

        # Stamp wall-clock run() start (async RPC; first call wins).
        record_start_time(exp)

        # --- DAC ---
        exp.zotino0.init()
        exp.zotino0.write_dac(0, 0.0)
        exp.zotino0.write_dac(1, 0.0)
        exp.zotino0.write_dac(2, 0.0)
        exp.zotino0.write_dac(3, 0.0)
        exp.zotino0.write_dac(6, 0.0)
        exp.zotino0.load()
        delay(20 * ms)

        # --- CPLDs (one per Urukul board) ---
        exp.urukul0_cpld.init()
        exp.urukul1_cpld.init()
        exp.urukul2_cpld.init()

        # --- DDS channels (all 12) ---
        exp.urukul0_ch0.init()
        exp.urukul0_ch1.init()
        exp.urukul0_ch2.init()
        exp.urukul0_ch3.init()
        exp.urukul1_ch0.init()
        exp.urukul1_ch1.init()
        exp.urukul1_ch2.init()
        exp.urukul1_ch3.init()
        exp.urukul2_ch0.init()
        exp.urukul2_ch1.init()
        exp.urukul2_ch2.init()
        exp.urukul2_ch3.init()
        exp.core.break_realtime()

        # --- TTL directions ---
        exp.ttl0.input()   # SPCM1
        exp.ttl1.input()   # SPCM2
        exp.ttl5.output()
        exp.ttl7.output()
        exp.ttl14.output()
        exp.ttl6.output()
        exp.ttl12.output()

        # --- TTL default states (ON = blocked for switches) ---
        exp.ttl5.on()    # repump RF blocked
        exp.ttl7.on()    # MW blocked
        exp.ttl14.on()   # D1/excitation blocked
        exp.ttl6.on()    # excitation atten1 blocked
        exp.ttl12.on()   # excitation atten2 blocked
        delay(1 * ms)  # TTL settle time (original uses 1ms, not 2us)

        # --- DDS defaults ---
        # Urukul0
        exp.urukul0_ch0.set(frequency=exp.f_mw_IF, amplitude=exp.amp_mw_IF)
        exp.urukul0_ch0.sw.off()
        delay(1 * us)
        exp.urukul0_ch1.set(frequency=exp.f_cooling_DP_idle, amplitude=exp.amp_cooling_DP_idle)
        exp.urukul0_ch1.sw.on()
        delay(1 * us)
        exp.urukul0_ch2.set(frequency=exp.f_repump, amplitude=exp.amp_repump)
        exp.urukul0_ch2.sw.on()
        exp.ttl5.off()  # unblock repump
        delay(1 * us)
        exp.urukul0_ch3.set(frequency=exp.f_D1_pump, amplitude=exp.amp_D1_pump)
        exp.urukul0_ch3.sw.off()
        delay(1 * us)

        # Urukul1
        # urukul1_ch0 (RF antenna) NOT touched — original deliberately skips it
        exp.urukul1_ch0.set(frequency=2e6, amplitude=0.01)  # dummy values to avoid errors; will be reconfigured by rebalancer
        exp.urukul1_ch0.sw.off()
        
        delay(1 * us)
        exp.urukul1_ch1.set(frequency=exp.f_FORT, amplitude=exp.nominal_trapdepth)
        exp.urukul1_ch1.sw.on()
        delay(1 * us)
        exp.urukul1_ch2.set(frequency=exp.f_excitation_idle, amplitude=exp.amp_excitation_idle)
        exp.urukul1_ch2.sw.off()
        delay(1 * us)
        exp.urukul1_ch3.set(frequency=exp.f_D1_onoff_L, amplitude=exp.amp_D1_onoff_L)
        exp.urukul1_ch3.sw.on()
        delay(1 * us)

        # Urukul2
        exp.urukul2_ch0.set(frequency=exp.f_MOT_X, amplitude=exp.amp_MOT_X)
        exp.urukul2_ch0.sw.on()
        delay(1 * us)
        exp.urukul2_ch1.set(frequency=exp.f_MOT_Y, amplitude=exp.amp_MOT_Y)
        exp.urukul2_ch1.sw.on()
        delay(1 * us)
        exp.urukul2_ch2.set(frequency=exp.f_MOT_Z, amplitude=exp.amp_MOT_Z)
        exp.urukul2_ch2.sw.on()
        delay(1 * us)
        exp.urukul2_ch3.set(frequency=exp.f_pump_repumper, amplitude=exp.amp_pump_repumper)
        exp.urukul2_ch3.sw.off()
        delay(1 * us)

        # --- Phase modes (only urukul0_ch0, NOT urukul1_ch0) ---
        exp.urukul0_ch0.set_phase_mode(PHASE_MODE_CONTINUOUS)
        delay(1 * us)
        exp.urukul1_ch0.set_phase_mode(PHASE_MODE_ABSOLUTE)
        delay(1 * us)

        # --- UV ON at init (increases MOT atom density) ---
        exp.ttl_UV.on()
        delay(1 * us)

        # --- Repump switch: on then settle ---
        exp.ttl5.on()
        delay(1 * us)

        # --- ADC ---
        exp.sampler0.init()
        delay(10 * ms)

    # -----------------------------------------------------------------
    #  UTILITY KERNEL METHODS
    # -----------------------------------------------------------------

    @kernel
    def set_shims(self, config_x, config_y, config_z1, config_z2):
        """Set shim coil voltages on Zotino DAC channels 0-3."""
        exp = self.experiment
        exp.zotino0.write_dac(0, config_x)
        exp.zotino0.write_dac(1, config_y)
        exp.zotino0.write_dac(2, config_z1)
        exp.zotino0.write_dac(3, config_z2)
        exp.zotino0.load()

    @kernel
    def set_shims_idle(self):
        exp = self.experiment
        self.set_shims(exp.shim_X_idle, exp.shim_Y_idle,
                       exp.shim_Z1_idle, exp.shim_Z2_idle)

    @kernel
    def set_shims_compression(self):
        exp = self.experiment
        self.set_shims(exp.shim_X_compression, exp.shim_Y_compression,
                       exp.shim_Z1_compression, exp.shim_Z2_compression)

    @kernel
    def set_shims_off(self):
        exp = self.experiment
        self.set_shims(exp.shim_X_off, exp.shim_Y_off,
                       exp.shim_Z1_off, exp.shim_Z2_off)

    @kernel
    def set_shims_qaxis(self):
        exp = self.experiment
        self.set_shims(exp.shim_X_qaxis, exp.shim_Y_qaxis,
                       exp.shim_Z1_qaxis, exp.shim_Z2_qaxis)

    @kernel
    def zero_shims(self):
        self.set_shims(0.0, 0.0, 0.0, 0.0)
