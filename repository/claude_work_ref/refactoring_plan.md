# Plan: Refactor useful_codes into Production-Quality ARTIQ-7 Cs Experiment Framework

## Context

The `useful_codes/` folder contains 17 working Python files for SNAQ-Node-3, a Cesium neutral atom quantum networking experiment running ARTIQ-7 on a Kasli FPGA (192.168.1.130). The code is functional but messy: no `device_db.py`, parameters hardcoded in every file, device declarations copy-pasted across experiments, data saved only to CSV, no monitoring, and no analysis tools. A previous Claude attempt (`experiments_another_claude_attempt_incomplete/`) made good base classes but left feedback unintegrated, HDF5 incomplete, and had a wrong Q-axis in calibration.yaml.

The goal is to restructure into a clean, modular codebase modeled after the senior labmates' Rb reference repo (`qn_artiq_routines-main`), with proper HDF5 data saving, dashboard-settable parameters, continuous monitoring, calibration management, and good wrappers so each experiment is easily readable and parametrized.

---

## Verified Hardware Channel Map (from schematic + code)

### DDS Channels (3x Urukul, 12x AD9910, 125 MHz ext clk, PLL n=32)

| Alias | Device | Function | Default Freq | Default Amp | Notes |
|-------|--------|----------|-------------|-------------|-------|
| `dds_mw_IF` | urukul0_ch0 | Microwave IF | 200-250.978 MHz | 0.55-0.7 | PHASE_MODE_CONTINUOUS |
| `dds_cooling_DP` | urukul0_ch1 | Cooling AOM double-pass | 111-130 MHz | 0.3-0.45 | Freq varies by phase: MOT=112, PGC=120/128, imaging=120, blowaway=99 |
| `dds_repump` | urukul0_ch2 | Repump AOM | 197 MHz | 0.5 | |
| `dds_D1_pump` | urukul0_ch3 | D1 pump first AOM | 145 MHz | 0.02-0.1 | Low power optical pumping |
| `dds_RF` | urukul1_ch0 | RF antenna | 0.49-0.61 MHz | 0.1-0.6 | PHASE_MODE_ABSOLUTE, two-photon transitions |
| `dds_FORT` | urukul1_ch1 | Trap AOM @1064nm | 150 MHz | 0.093-0.24 | Feedback ch3, max_amp=0.25 |
| `dds_excitation` | urukul1_ch2 | D1 on/off R / Excitation | 145-154 MHz | 0.04-0.18 | Feedback ch4, max_amp=0.5 |
| `dds_D1_onoff_L` | urukul1_ch3 | D1 on/off L / PGC beam | 250 MHz | 0.43 | Always on during MOT/PGC |
| `dds_MOT_X` | urukul2_ch0 | MOT cooling X fiber AOM | 80 MHz | 0.12-0.13 | Feedback ch0, max_amp=0.17 |
| `dds_MOT_Y` | urukul2_ch1 | MOT cooling Y fiber AOM | 80 MHz | 0.10-0.11 | Feedback ch1, max_amp=0.17 |
| `dds_MOT_Z` | urukul2_ch2 | MOT cooling Z fiber AOM | 80 MHz | 0.08-0.11 | Feedback ch2, max_amp=0.17 |
| `dds_pump_repumper` | urukul2_ch3 | Pump Repumper | 202 MHz | 0.3-0.4 | Used during optical pumping |

### TTL Channels (active-low switch logic: ON = beam blocked)

| Alias | Device | Type | Function | Logic |
|-------|--------|------|----------|-------|
| `ttl_SPCM1` | ttl0 | TTLInOut | Single-photon counter 1 | Input, gate_rising for counting |
| `ttl_SPCM2` | ttl1 | TTLInOut | Single-photon counter 2 | Input, gate_rising for counting |
| `ttl_UV_camera` | ttl4 | TTLOut | UV lamp / camera trigger | ON = UV/trigger active |
| `ttl_repump_switch` | ttl5 | TTLOut | MOT repump RF switch | ON = repump RF blocked |
| `ttl_exc_atten1` | ttl6 | TTLOut | Excitation attenuator 1 | ON = excitation blocked |
| `ttl_MW_switch` | ttl7 | TTLOut | Microwave switch | ON = MW blocked |
| `ttl_exc_atten2` | ttl12 | TTLOut | Excitation attenuator 2 | ON = excitation blocked |
| `ttl_D1_exc_switch` | ttl14 | TTLOut | D1 pump / excitation switch | ON = D1/exc blocked |

### DAC Channels (Zotino0)

| Channel | Function | Typical Range |
|---------|----------|---------------|
| DAC 0 | Shim coil X | -0.175 to 0.25 V |
| DAC 1 | Shim coil Y | -0.291 to 5.0 V |
| DAC 2 | Shim coil Z1 | -0.22 to 4.25 V |
| DAC 3 | Shim coil Z2 | -0.05 to 4.2 V |
| DAC 6 | (zeroed at init) | 0 V |

### Shim Coil Configurations (from latest Z parity experiment)

| Config | X | Y | Z1 | Z2 | Use |
|--------|------|------|------|------|-----|
| idle | 0.25 | 0.1 | 4.25 | 4.2 | MOT loading |
| PGC | 0.2 | 0.2 | 4.25 | 4.2 | Polarization gradient cooling |
| off | 0.0 | 0.7 | -0.05 | 0.05 | Free-space (field settling) |
| Q-axis | -0.013 | 2.72 | -0.06 | 0.06 | Quantization axis for excitation |

### Feedback Channels (sampler0, P-only control)

| Ch | Sampler | DDS | P gain | Tolerance | Iterations | Init amp | Max amp |
|----|---------|-----|--------|-----------|------------|----------|---------|
| 0 | buf[0] | urukul2_ch0 (MOT X) | 0.08 | 0.01 | 20 | 0.13 | 0.17 |
| 1 | buf[1] | urukul2_ch1 (MOT Y) | 0.08 | 0.01 | 20 | 0.11 | 0.17 |
| 2 | buf[2] | urukul2_ch2 (MOT Z) | 0.08 | 0.01 | 20 | 0.08 | 0.17 |
| 3 | buf[3] | urukul1_ch1 (FORT) | 0.08 | 0.001 | 60 | 0.093 | 0.25 |
| 4 | buf[4] | urukul1_ch2 (Excitation) | 0.08 | 0.001 | 20 | 0.18 | 0.5 |

### Microwave Transitions (from MW scan experiment)

| Transition | IF Frequency (MHz) | Pi-pulse (mu) | Notes |
|------------|-------------------|---------------|-------|
| F=3,mF=0 → F=4,mF=0 (00) | 250.000 (varies) | 4530 | Clock transition |
| F=3,mF=0 → F=4,mF=1 (01) | 250.468 | 31876 | |
| F=3,mF=1 → F=4,mF=1 (11) | 250.972 | 5033 | |
| Two-photon (2ph) | MW: 249.383, RF: 0.608 MHz | 243000 (=106000 mu pulse) | X-basis rotation |

### K10CR1 Waveplate Rotators (NDSP at localhost:8083)

| Nickname | Serial | Function |
|----------|--------|----------|
| 852_QWP | 55000617 | Quarter-wave plate for 852nm |
| 852_HWP | 55422734 | Half-wave plate for 852nm |

Conversion: `deg_to_pos = 136533`

---

## Experiment Sequence Architecture (from Z/X parity code)

The experiments follow a nested loop structure with color-coded scopes:

```
FOR each QWP angle:
  FOR each HWP angle:
    GREEN SCOPE: Measurement loop (n_measurements repetitions)
    │
    ├── YELLOW SCOPE: Loading & Verification (retry until atom loaded)
    │   ├── Feedback rebalance (5-channel AOM stabilization)
    │   ├── MOT loading (idle shims, cooling at 112 MHz)
    │   ├── PGC (PGC shims, cooling at 120→128 MHz)
    │   ├── Field settling (off shims, 300ms wait)
    │   ├── Imaging PGC (128 MHz, 10ms)
    │   ├── SPCM dual imaging (4ms gate)
    │   └── Atom check: (count1+count2-bg) > threshold(12) → proceed
    │
    ├── ORANGE SCOPE: Excitation retry (retry until photon or atom lost)
    │   ├── State prep (F=3 optical pumping with D1 pump + repumper)
    │   ├── Q-axis coils on
    │   ├── Trap ramp down (adiabatic)
    │   ├── Excitation pulse (trap off/on, D1 switch, SPCM timestamps)
    │   ├── Photon check → if detected: break to Blue
    │   ├── PGC recooling between cycles
    │   └── Survival check → if atom lost: break to Yellow
    │
    └── BLUE SCOPE: Post-photon processing (Z vs X basis differs here)
        ├── Trap ramp back up
        ├── [Z-basis] Single MW pi pulse on 00 transition (5120 mu)
        ├── [X-basis] MW pi pulse + two-photon rotation (MW IF + RF antenna, 106000 mu)
        ├── Blowaway (off shims, cooling at 99 MHz, 500us)
        ├── Final SPCM imaging (state readout)
        └── Write data (CSV currently → will be HDF5)
```

---

## Target Directory Structure

```
useful_codes/
  device_db.py                          # Copied verbatim from Artiq7_experiment/device_db.py
  dataset_db.pyon                       # Generated by ExperimentVariables on first run
  results/                              # HDF5 output: results/{date}/{hour}/{RID}-{name}.h5

  archive/                              # Original files preserved for reference
    Calibration/
    Characterization/
    Devices_and_config/
    Experiments/
    Feedback_and_Manual_Control/

  config/
    __init__.py
    device_aliases.json                 # DDS/TTL human name → hardware channel mapping
    feedback_channels.json              # 5-channel feedback config (sampler, gains, limits)

  utilities/
    __init__.py
    base_experiment.py                  # Composition-based base: build/prepare/initialize_hardware
    experiment_variables.py             # ~150 persistent dashboard-settable parameters
    device_aliases.py                   # Reads JSON, creates device alias attributes
    write_h5.py                         # @rpc write_results() for HDF5 with full metadata
    conversions.py                      # dB_to_V, amplitude unit helpers

  subroutines/
    __init__.py
    sequences.py                        # @kernel physics subroutines (all reusable sequences)
    aom_feedback.py                     # AOMPowerStabilizer auto-class (called from experiments)
    k10cr1_functions.py                 # Waveplate motor control (relocated, fixed imports)

  experiments/
    __init__.py
    z_parity.py                         # Z-basis parity oscillation
    x_parity.py                         # X-basis parity oscillation
    microwave_scan.py                   # MW frequency/time/Ramsey scanning
    trap_frequency.py                   # French method trap frequency measurement
    excitation_scan.py                  # Excitation rate with long trap-off
    atom_load_monitor.py                # Continuous atom loading display
    general_variable_scan.py            # 1D/2D parameter sweep framework

  manual_control/
    __init__.py
    aom_control.py                      # Manual AOM freq/amp tuning (12 DDS channels)
    shim_coil_control.py                # Manual Zotino shim coil + TTL control
    feedback_standalone.py              # MANUAL feedback class: standalone EnvExperiment for
                                        # characterizing & setting reference voltages from dashboard

  applets/
    plot_spcm_counts.py                 # Rolling SPCM1/SPCM2 count rates (line plot)
    plot_feedback_powers.py             # 5-channel AOM power bar chart (normalized to setpoint)
    plot_atom_loading.py                # Loading rate and retention scatter
    plot_parity.py                      # Parity oscillation vs waveplate angle
    plot_histogram.py                   # SPCM count histogram with threshold marker

  fitting/
    __init__.py
    fit_base.py                         # scipy curve_fit wrapper (from reference repo)
    lorentzian.py                       # Microwave resonance peaks
    sinusoid.py                         # Ramsey fringes, parity oscillations
    gaussian.py                         # SPCM count histograms, beam profiles
    exponential_decay.py                # Trap lifetime

  analysis/
    __init__.py
    h5_loader.py                        # Load HDF5 by RID or date range, batch loading
    parity_analysis.py                  # Fidelity/visibility from Z+X basis measurements
```

---

## Implementation Phases

### Phase 1: Foundation (device_db, config, base classes)

**1.0** Move all existing folders into `useful_codes/archive/`:
- `Calibration/` → `archive/Calibration/`
- `Characterization/` → `archive/Characterization/`
- `Devices and config (I think)/` → `archive/Devices_and_config/`
- `Experiments/` → `archive/Experiments/`
- `Feedback and Monitoring and Manual Control/` → `archive/Feedback_and_Manual_Control/`
- Keep `Csnode_schematic.PNG` at root level for reference.

**1.1** Copy `Artiq7_experiment/device_db.py` → `useful_codes/device_db.py` verbatim (756 lines, verified correct for SNAQ-Node-3 Kasli).

**1.2** Create `config/device_aliases.json`:
Complete mapping of all 12 DDS channels + TTLs + DAC channels to human-readable names with default frequencies/amplitudes. Includes per-channel metadata (max_amp, feedback_sampler_ch where applicable). All values verified against the latest Z parity experiment (April 2026).

**1.3** Create `config/feedback_channels.json`:
5-channel configuration with sampler channel index, setpoint, P gain, tolerance, max iterations, initial amplitude, max amplitude, and frequency for each. Values from the verified table above.

**1.4** Create `utilities/device_aliases.py`:
- Class that reads `device_aliases.json` and calls `setattr_device()` for each hardware device
- Creates alias attributes on the experiment: `self.dds_FORT` = `self.urukul1_ch1`, etc.
- `@kernel initialize()`: CPLD init (3 boards), DDS init (12 channels), set default freq/amp, TTL direction setup, Zotino init, Sampler init
- Based on reference repo's `DeviceAliases.py` pattern

**1.5** Create `utilities/experiment_variables.py`:
- Standalone `EnvExperiment` run once from dashboard to create all persistent datasets
- **Clean Cs naming** throughout (no Rb-style AOM_A1/A2/A3)
- ~150 parameters organized by group:
  - **Timing**: `t_MOT_loading`=500ms, `t_MOT2PGC`=500ms, `t_PGC_2_off`=300ms, `t_PGC_after_img`=10ms, `t_field_settle`=300ms, `t_imaging_gate`=4ms, `t_wait`=1ms, `t_SPCM_exposure`=150ms
  - **Trap**: `nominal_trapdepth`=0.24, `trap_ramp_start_amp`=0.125, `trap_ramp_steps`=1, `trap_ramp_dt_us`=1
  - **Shim Coils** (4 configs x 4 channels = 16 params): idle, PGC, off, Q-axis values from table above
  - **DDS Defaults** (12 channels x 2 = 24 params): frequency and amplitude for each channel
  - **Cooling Phases**: `f_cooling_MOT`=112MHz, `f_cooling_PGC`=120MHz, `f_cooling_deep_PGC`=128MHz, `f_cooling_imaging`=120MHz, `f_cooling_blowaway`=99MHz, corresponding amplitudes
  - **Microwave**: 4 transition IF frequencies, 4 pi-pulse durations (mu), RF detune parameters
  - **Feedback**: 5 channels x 6 params = 30 params (gain, tolerance, iterations, init_amp, frequency, setpoint)
  - **Excitation**: `max_excitation_cycles`=1, `t_trap_off_mu`=30, `t_spcm_collection_mu`=250, `length_offset_mu`=30, excitation timing offsets
  - **Detection**: `atom_threshold`=12, `n_measurements`=10
  - **Waveplate**: HWP/QWP scan ranges, step sizes, `deg_to_pos`=136533
  - **Paths**: `results_dir`="results/"
- All parameters created with `persist=True` and `broadcast=True`

**1.6** Create `utilities/base_experiment.py`:
- **Composition pattern**: experiments use `self.base = BaseExperiment(experiment=self)`
- `build()`:
  - Loads all datasets from `dataset_db.pyon` as experiment attributes (using clean Cs names)
  - Creates DeviceAliases instance → registers all devices and aliases
  - Sets up TTL alias attributes
  - Sets up Zotino channel map
  - Creates sampler references
- `prepare()`:
  - Computes derived values (unit conversions: ms→s for delays, MHz→Hz for DDS)
  - Loads feedback reference voltages from file
  - Builds scan sequences from dashboard arguments
- `set_datasets_from_gui_args()`: syncs experiment-specific dashboard arguments back to persistent datasets
- `@kernel initialize_hardware()`:
  - `core.reset()`, `core.break_realtime()`
  - Zotino init, zero all DACs
  - CPLD init (3 boards), DDS init (12 channels)
  - Set all DDS to default idle frequencies/amplitudes
  - TTL direction setup (ttl0/1 input, all others output)
  - Default TTL states (ttl5/7/14/6/12 ON = all switches blocking)
  - Phase mode setup (urukul0_ch0 CONTINUOUS, urukul1_ch0 ABSOLUTE)
  - Sampler init

**1.7** Create `utilities/write_h5.py`:
- `@rpc(flags={"async"}) write_results(experiment, name=None)`:
  - Gets RID from `experiment.scheduler.rid`
  - Creates `results/{date}/{hour}/{RID:09d}-{ExperimentName}.h5`
  - Saves ALL current datasets (auto-captures every ExperimentVariable)
  - Adds metadata group: RID, timestamp, ARTIQ version, experiment class name, hostname
  - File organized as:
    ```
    /{dataset_name}          → dataset values
    /archive/{param_name}    → all ExperimentVariables at time of run
    /metadata/rid            → run ID
    /metadata/start_time     → ISO timestamp
    /metadata/experiment     → class name
    ```

**1.8** Create `utilities/conversions.py`:
- `dB_to_V(dB)`: RF power dBm → DDS amplitude voltage
- `sampler_mu_to_V(mu_val)`: sampler raw → voltage (`mu * 20.0 / 65536.0`)

**Critical source files:**
- `Artiq7_experiment/device_db.py` → copy verbatim
- `qn_artiq_routines-main/.../utilities/BaseExperiment.py` → composition pattern model
- `qn_artiq_routines-main/.../utilities/DeviceAliases.py` → alias pattern model
- `qn_artiq_routines-main/.../ExperimentVariables.py` → parameter management model
- `qn_artiq_routines-main/.../utilities/write_h5.py` → HDF5 saving model

---

### Phase 2: Subroutines (physics sequences, feedback split, motor control)

**2.1** Create `subroutines/sequences.py`:
Extract all repeated kernel sequences from the Z/X parity experiments (970+ lines each) into reusable `@kernel` functions. Each takes the experiment instance as first argument.

Functions to extract (with source line references from Z parity):

```python
# --- Loading ---
@kernel
def set_shims(exp, config):
    """Set shim coils. config: 'idle', 'pgc', 'off', 'qaxis'"""
    # Lines 421-426 (idle), 434-438 (PGC), 454-458 (off), 530-534 (Q-axis)

@kernel
def mot_loading_sequence(exp):
    """MOT loading: idle shims, cooling at 112 MHz, wait MOT2PGC"""
    # Lines 420-431

@kernel
def pgc_sequence(exp):
    """PGC: PGC shims, cooling 120→128 MHz, deep PGC"""
    # Lines 433-451

@kernel
def field_settle(exp):
    """Off shims, wait 300ms for field settling"""
    # Lines 453-459

@kernel
def imaging_pgc(exp):
    """Short PGC before SPCM readout: 128 MHz, PGC_Af_Img duration"""
    # Lines 461-471

@kernel
def spcm_image_dual(exp, gate_time) -> TTuple([TInt32, TInt32]):
    """Dual-SPCM fluorescence imaging, returns (count1, count2)"""
    # Lines 483-488

@kernel
def check_atom_loaded(exp, count_signal, count_bg, threshold) -> TBool:
    """Check if atom present: (signal - bg) > threshold"""
    # Line 503

# --- State Preparation ---
@kernel
def optical_pumping_F3(exp):
    """F=3 state prep: D1 pump + excitation + pump repumper, 1ms duration"""
    # Lines 559-582

# --- Excitation ---
@kernel
def excitation_pulse_with_timestamps(exp):
    """Single excitation attempt: Q-axis on, trap ramp down, pump, excite,
    SPCM timestamp collection. Returns (ts1_mu, ts2_mu, t0_mu)"""
    # Lines 528-641 (the core excitation cycle)

@kernel
def ramp_trap_amp(exp, start_amp, end_amp, n_steps, dt_us):
    """Adiabatic trap depth ramp on urukul1_ch1"""
    # Lines 185-206

# --- Post-detection ---
@kernel
def microwave_pi_pulse_z(exp, click_time_mu):
    """Z-basis: single MW pi pulse on 00 transition (5120 mu)"""
    # Lines 646-653 (Z parity)

@kernel
def microwave_two_photon_rotation(exp, click_time_mu, micro_freq, rf_freq):
    """X-basis: MW pi + two-photon rotation (MW IF + RF antenna, 106000 mu)"""
    # Lines 646-669 (X parity)

@kernel
def blowaway_f4(exp):
    """Resonant blowaway: off shims, cooling at 99 MHz + fiber Z, 500us"""
    # Lines 675-701

@kernel
def recool_and_check_survival(exp) -> TBool:
    """PGC + dual SPCM imaging for survival check"""
    # Lines 864-892

@kernel
def reset_to_mot(exp):
    """Reset all beams/shims to MOT configuration"""
    # Lines 951-971

# --- Feedback prep ---
@kernel
def prepare_for_feedback(exp):
    """Set beams for feedback measurement: turn off repump RF, set cooling to
    122 MHz, turn off trap, unblock excitation path"""
    # Lines 360-372

@kernel
def restore_after_feedback(exp):
    """Restore beams after feedback: block excitation, set cooling to MOT freq"""
    # Lines 387-396
```

**Source files:**
- `archive/Experiments/HS_code_20260401_Z_parity_v7_*.py` (970 lines - most complete)
- `archive/Experiments/HS_code_20260402_X_parity_v7_*.py` (X-basis additions)
- `archive/Calibration/Microwave_variables_scan_v7_*.py` (MW-specific sequences)

**2.2** Create `subroutines/aom_feedback.py` — **AUTO class only** (called from within experiments):
- `AOMPowerStabilizer` class that replaces `FeedbackRebalancer_5_revisedJan26`
- Loaded from `feedback_channels.json` config
- Device attributes assigned by caller (same pattern as current code: `stabilizer.core = self.core`, etc.)
- `@kernel rebalance(...)` method with same 25-parameter interface for ARTIQ-7 kernel compatibility (no list indexing with mixed types in kernels)
- **New**: publishes feedback results to broadcast datasets (`feedback_MOT_X`, `feedback_FORT`, etc.) so applets can monitor
- **New**: `@kernel monitor_only(...)` method that reads sampler without adjusting amplitudes
- **New**: returns updated amplitudes AND final measured voltages (for logging)
- Also includes `FeedbackRebalancer_exc` for excitation-only feedback (currently nested class)

**2.3** Create `manual_control/feedback_standalone.py` — **MANUAL class** (standalone dashboard experiment):
- Extracted from `FeedbackPIDExperiment_5channel_revisedJan26(EnvExperiment)` (lines 13-333 of current file)
- Full standalone `EnvExperiment` with its own `build()`, dashboard arguments, `run()` kernel
- Used independently to: characterize beam stability, set reference voltages, verify feedback performance
- Saves reference voltages to `ref_voltages_5.txt`
- Prints formatted report table
- Uses `BaseExperiment` for device setup instead of manual declarations

**2.4** Create `subroutines/k10cr1_functions.py`:
- Relocate from `archive/Devices_and_config/External Devices/Waveplates/Node3_K10CR1/k10cr1_functions.py`
- Fix import path (currently `from repository.tests.ndsp_tests.thorlabs.with_kasli.Node3_K10CR1.k10cr1_functions import *`)
- Same interface: `move_to_target_deg(experiment, name, target_deg, tolerance_deg=0.5)`
- Also copy `k10cr1_driver.py` if needed as a dependency

---

### Phase 3: Experiment Migration

Each experiment migration follows this pattern:
1. Remove all `setattr_device()` calls → handled by `BaseExperiment`
2. Remove all hardcoded parameters → read from datasets via `BaseExperiment`
3. Replace inline sequences → call `subroutines/sequences.py` functions
4. Replace CSV writing → `write_results()` HDF5
5. Add `set_dataset(..., broadcast=True)` calls for real-time monitoring
6. Keep dashboard `setattr_argument()` for experiment-specific scan parameters
7. Import `FeedbackRebalancer` from `subroutines/aom_feedback` (not from broken external path)

**3.1** `experiments/atom_load_monitor.py` — **Migrate first** (simplest, validates BaseExperiment)
- Source: `archive/Feedback_and_Manual_Control/Atom_load_applet_faster_release_double_spcm_251202.py`
- Note: this one imports from `aom_feedback_final` (3-channel), not 5-channel
- Continuous MOT loading loop, dual-SPCM readout
- Broadcasts: `SPCM1_counts`, `SPCM2_counts`, `loading_rate` per iteration
- Dashboard args: `num_runs`
- ~50 lines after refactoring (currently ~200+)

**3.2** `experiments/z_parity.py`
- Source: `archive/Experiments/HS_code_20260401_Z_parity_v7_*.py` (970 lines → ~120 lines)
- Dashboard args: HWP/QWP scan ranges, n_measurements, max_excitation_cycles, excitation timing, adiabatic ramp params
- Blue scope: `microwave_pi_pulse_z()` → blowaway → final imaging
- Broadcasts: parity data arrays, photon timestamps, measurement progress, loading stats
- HDF5 datasets: `SPCM1_timestamps`, `SPCM2_timestamps`, `counts_signal`, `counts_bg`, `HWP_angles`, `QWP_angles`, `trials_per_measurement`

**3.3** `experiments/x_parity.py`
- Source: `archive/Experiments/HS_code_20260402_X_parity_v7_*.py`
- Nearly identical to Z parity except Blue scope uses `microwave_two_photon_rotation()`
- Additional params: `micro_freq` = 249.383 MHz, `RF_freq` = 0.608 MHz (dashboard-settable)
- MW pi pulse first (5120 mu), then MW IF + RF simultaneous pulse (106000 mu)

**3.4** `experiments/microwave_scan.py`
- Source: `archive/Calibration/Microwave_variables_scan_v7_packing_20260317.py` (1015 lines)
- Most complex dashboard interface: 10 scan mode booleans, fixed frequencies for all 4 transitions, scan ranges, pi-pulse durations
- Q-axis parameters as dashboard arguments (was the source of the YAML bug — this ensures consistency)
- Modes: Frequency scan (00/01/11/2ph), Time scan (00/01/11/2ph), Ramsey scan (00/11/2ph)
- Each mode has distinct blue scope logic

**3.5** `experiments/trap_frequency.py`
- Source: `archive/Characterization/HS_code_20260331_Trap_freq_FrenchMethod.py`
- Parametric excitation scan to measure trap oscillation frequency

**3.6** `experiments/excitation_scan.py`
- Source: `archive/Calibration/HS_code_20260225_ext_long_trap_off.py`
- Excitation rate measurement with variable trap-off duration

**3.7** `experiments/general_variable_scan.py`
- New, modeled on reference repo's `GeneralVariableScan.py`
- Scan any ExperimentVariable over user-defined range
- Support 1D and 2D scans (nested loops)
- Override dataset values temporarily during scan

---

### Phase 4: Manual Control Utilities

**4.1** `manual_control/aom_control.py`
- Source: `archive/Feedback_and_Manual_Control/AOMurukulcontrol_artiq7.py`
- Uses `BaseExperiment` for device setup
- Dashboard arguments for all 12 DDS channels with human-readable names
- Set frequency and amplitude from dashboard, apply with single kernel call

**4.2** `manual_control/shim_coil_control.py`
- Source: `archive/Feedback_and_Manual_Control/ShimCoilSwitch_artiq7.py`
- Uses `BaseExperiment` for device setup
- Dashboard arguments for X/Y/Z1/Z2 voltages + TTL switch states
- Preset buttons for idle/PGC/off/Q-axis configurations

**4.3** `manual_control/feedback_standalone.py` (already described in Phase 2.3)

---

### Phase 5: Monitoring Applets

All applets are pyqtgraph scripts that subscribe to broadcast datasets via ARTIQ's applet infrastructure.

**5.1** `applets/plot_spcm_counts.py`
- Rolling line plot of SPCM1 and SPCM2 count rates over iterations
- Reference: `qn_artiq_routines-main/applets/plot_single_photons_by_iteration.py`

**5.2** `applets/plot_feedback_powers.py`
- Bar chart of 5 feedback channel powers normalized to setpoints
- Color-coded: green if within tolerance, yellow/red if drifting
- Reference: `qn_artiq_routines-main/applets/bar_plot_MOT_powers.py`

**5.3** `applets/plot_atom_loading.py`
- Loading success rate scatter plot over time
- Shows threshold line

**5.4** `applets/plot_parity.py`
- Parity oscillation data vs waveplate angle (updates live during scan)

**5.5** `applets/plot_histogram.py`
- SPCM count histogram with configurable threshold marker
- Auto-binning

---

### Phase 6: Fitting and Analysis

**6.1** `fitting/fit_base.py`
- scipy `curve_fit` wrapper with auto parameter estimation, bounds, error reporting
- Source: `qn_artiq_routines-main/fitting/FitBase.py`

**6.2** Fitting models (each subclasses FitBase):
- `lorentzian.py` — MW resonance peaks
- `sinusoid.py` — Ramsey fringes, parity oscillations
- `gaussian.py` — SPCM count histograms
- `exponential_decay.py` — Trap lifetime

**6.3** `analysis/h5_loader.py`
- `ExperimentData` class: load single HDF5 by path or RID
- `find_runs(date, experiment_name)`: batch file discovery in results/
- `load_csv()`: backward-compatible CSV loading for old data
- Source: `experiments_another_claude_attempt_incomplete/Analysis/h5_loader.py` + `qn_artiq_routines-main/Analysis/h5_data_utilities.py`

**6.4** `analysis/parity_analysis.py`
- Calculate entanglement fidelity from Z+X basis parity visibility
- Parity = P(same) - P(different) as function of waveplate angle
- Sinusoidal fit to extract visibility

---

## AOM Feedback Split (Key Change from Previous Plan)

The current `aom_feedback_final_5channel_revisedjan26.py` contains TWO distinct classes:

### 1. Manual/Standalone Class → `manual_control/feedback_standalone.py`
`FeedbackPIDExperiment_5channel_revisedJan26(EnvExperiment)` (lines 13-333)
- Full `EnvExperiment` with `build()`, `prepare()`, `run()` kernel
- Dashboard arguments: initial_amp, frequency, p_gain, tol, iterations, use_stored_ref (per channel)
- Initializes beams (D1_onoff_L at 250MHz, cooling at 122MHz), measures initial voltages
- Runs P-only feedback loop, prints formatted report, saves reference voltages to file
- **Purpose**: Run independently to characterize beam stability and calibrate reference setpoints

### 2. Auto/Inline Class → `subroutines/aom_feedback.py`
`FeedbackRebalancer_5_revisedJan26` (lines 335-432) + nested `FeedbackRebalancer_exc_revisedJan26` (lines 434-457)
- No `build()` — caller assigns device attributes manually
- `@kernel rebalance(25 scalar params) → (amp0, amp1, amp2, amp3, amp4)`: full 5-channel feedback
- `@kernel rebalance_exc(6 scalar params) → amp4`: excitation-only feedback
- **Purpose**: Called inline during experiments (e.g., in Yellow scope of parity measurement)
- Called with 25 explicit parameters because ARTIQ-7 kernels cannot index lists of mixed-type objects

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Base pattern | Composition (`self.base = BaseExperiment(self)`) | Proven in reference repo, avoids MRO issues with ARTIQ's EnvExperiment |
| Parameters | ExperimentVariables + dataset_db.pyon | Dashboard-editable, auto-persists, included in HDF5 for free |
| Naming | Clean Cs names (f_cooling_DP_MOT, not AOM_A1_freq) | Descriptive, matches actual hardware labels |
| Device config | device_aliases.json | Decouples human names from hardware channels |
| Feedback split | Manual standalone class + auto inline class | Different use cases: calibration vs in-experiment stabilization |
| Feedback interface | 25 scalar params (not lists/dicts) | ARTIQ-7 kernel limitation: no list indexing with mixed types |
| Data saving | HDF5 via write_results() @rpc | Complete metadata, debuggable, standard physics format |
| Results location | useful_codes/results/{date}/{hour}/ | Self-contained project directory |
| Calibration | Datasets (not YAML) | Single source of truth, dashboard-editable, avoids Q-axis mismatch |
| Original files | Archived in useful_codes/archive/ | Preserved for reference/comparison during validation |
| TTL logic | Document inverted logic (ON=blocked) | Critical for correctness, easy to get wrong |

## HDF5 Dataset Organization

Every experiment run saves:

```
{RID:09d}-{ExperimentName}.h5
├── metadata/
│   ├── rid                          # Run ID (int)
│   ├── start_time                   # ISO timestamp
│   ├── experiment_class             # e.g. "ZParityExperiment"
│   └── artiq_version                # ARTIQ version string
├── archive/                         # All ExperimentVariables snapshot
│   ├── nominal_trapdepth            # 0.24
│   ├── f_cooling_MOT                # 112e6
│   ├── shim_X_idle                  # 0.25
│   ├── ... (all ~150 parameters)
│   └── feedback_ref_voltages        # [ref0, ref1, ref2, ref3, ref4]
├── scan/
│   ├── scan_variable                # "HWP_angle" or "microwave_freq"
│   ├── scan_sequence                # [0, 30, 60, ..., 330]
│   └── scan_variable_2              # "QWP_angle" (if 2D scan)
├── data/
│   ├── SPCM1_timestamps            # float64 array (seconds relative to t0)
│   ├── SPCM2_timestamps            # float64 array
│   ├── counts_signal                # int32 array (per measurement)
│   ├── counts_background            # int32 array
│   ├── trials_per_measurement       # int32 array (loading attempts needed)
│   ├── HWP_angle                    # float64 array (angle at each measurement)
│   ├── QWP_angle                    # float64 array
│   └── photon_detected              # bool array
└── feedback/
    ├── MOT_X_power                  # float64 array (normalized, per feedback call)
    ├── MOT_Y_power                  # float64 array
    ├── MOT_Z_power                  # float64 array
    ├── FORT_power                   # float64 array
    ├── excitation_power             # float64 array
    └── final_amplitudes             # float64 array (DDS amplitudes after feedback)
```

---

## Verification Plan

1. **Phase 1 validation**: Run `ExperimentVariables` from dashboard → verify `dataset_db.pyon` created with all ~150 parameters. Run a trivial test experiment that does `BaseExperiment.initialize_hardware()` and reads sampler0 → verify all 12 DDS channels respond, Zotino writes, TTLs toggle.

2. **Phase 2 validation**: Run `feedback_standalone.py` from dashboard → verify 5 channels stabilize and report matches original `FeedbackPIDExperiment` output format. Run inline `AOMPowerStabilizer.rebalance()` in a test experiment → verify amplitudes converge.

3. **Phase 3 validation**: Run `atom_load_monitor.py` → verify SPCM counts match original. Then run `z_parity.py` with identical waveplate angles and parameters as a recent successful run → compare photon detection rates and timing distributions.

4. **Phase 5 validation**: Start parity experiment → verify applets show live SPCM counts, feedback bar chart, and parity oscillation updating in real time.

5. **Phase 6 validation**: Load HDF5 from a parity run → extract timestamps → fit sinusoid → verify visibility calculation matches manual analysis.

---

## Migration Order (risk-minimized)

1. Foundation (Phase 1) — no physics, just plumbing
2. Feedback split (Phase 2.2, 2.3) — test independently
3. K10CR1 relocation (Phase 2.4) — test waveplate movement
4. Sequences extraction (Phase 2.1) — test with atom loading
5. Atom load monitor (Phase 3.1) — simplest experiment, validates everything
6. Z parity (Phase 3.2) — most important experiment
7. X parity (Phase 3.3) — small delta from Z
8. MW scan (Phase 3.4) — complex dashboard interface
9. Remaining experiments (Phase 3.5-3.7)
10. Manual controls (Phase 4)
11. Applets (Phase 5)
12. Fitting & analysis (Phase 6)
