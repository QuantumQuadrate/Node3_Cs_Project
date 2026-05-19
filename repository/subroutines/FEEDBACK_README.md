# AOM Feedback System — Architecture and Usage

## Overview

The feedback system stabilizes AOM beam powers by measuring photodiode voltages on the Sampler ADC and adjusting DDS amplitudes via proportional (P-only) control. It runs inline within experiments — not as a separate process.

### Channels

| Index | Name | DDS | Sampler Ch | Status | Purpose |
|-------|------|-----|------------|--------|---------|
| 0 | MOT_X | urukul2_ch0 | 0 | Active | MOT cooling X fiber AOM |
| 1 | MOT_Y | urukul2_ch1 | 1 | Active | MOT cooling Y fiber AOM |
| 2 | MOT_Z | urukul2_ch2 | 2 | Active | MOT cooling Z fiber AOM |
| 3 | FORT | urukul1_ch1 | 3 | Active | Trap AOM @1064nm |
| 4 | excitation | urukul1_ch2 | 4 | **Disabled** (p_gain=0) | D1 on-off AOM R — i2v off, never stabilized |

---

## File Map

```
repository/
├── config/
│   ├── feedback_channels.json        # Tuning params (gain, tolerance, iterations, amp, freq, max_amp)
│   └── calibration_references.json   # Reference voltages per profile (FORT_loading, low_FORT, etc.)
├── subroutines/
│   ├── aom_feedback.py               # FeedbackRebalancer_5 class
│   │                                 #   rebalance()     — 5-channel @kernel
│   │                                 #   rebalance_3ch() — MOT-only @kernel
│   │                                 #   load_params_from_json(), load_profile(), save_diagnostics()
│   └── FEEDBACK_README.md            # This file
├── utilities/
│   ├── base_experiment.py            # _setup_rebalancer(): creates rebalancer, loads params + profile
│   └── calibration_store.py          # CalibrationStore class — reads/writes calibration_references.json
├── manual_control/
│   └── feedback_standalone.py        # Dashboard experiment for manual calibration
└── experiments/
    └── *.py                          # Call self.rebalancer.rebalance() from @kernel
```

---

## Data Flow

### At build time (once per experiment submission):
```
feedback_channels.json
    → BaseExperiment._setup_rebalancer()
        → creates FeedbackRebalancer_5 instance
        → assigns 9 device attributes (core, sampler, DDS channels, TTLs)
        → FeedbackRebalancer_5.load_params_from_json()
            → sets self.p0..p4, self.tol0..tol4, self.it0..it4,
              self.amp0..amp4, self.max_amp0..max_amp4, self.f0..f4
            → initializes diagnostic attributes (init_v0..v4, final_v0..v4, ref0..ref4)
            → creates CalibrationStore instance
```

### At prepare time (before each run):
```
calibration_references.json
    → CalibrationStore.get_refs(profile)
        → BaseExperiment.prepare(feedback_profile=...)
            → rebalancer.load_profile(profile)
                → sets self.ref0..ref4 from the chosen profile
```

### At kernel run time:
```
experiment @kernel run()
    → prepare_for_feedback(self)    # beams configured for measurement
    → self.rebalancer.rebalance()   # @kernel, reads self.*, applies DDS, returns amps
    → restore_after_feedback(self)  # beams restored for experiment
```

### After kernel (diagnostics):
```
rebalancer.save_diagnostics(final_amps)  # host-side, writes datasets to HDF5/Zarr
```

---

## Calibration Profiles

### calibration_references.json

Multiple profiles store reference voltages for different experimental phases:

| Profile | FORT ref | Use case | Used by |
|---------|----------|----------|---------|
| `FORT_loading` | 0.134V (high) | Atom loading at nominal_trapdepth | atom_load_monitor (active_profile default) |
| `low_FORT` | 0.036V (low) | Science phase, adiabatic ramp-down | z_parity, x_parity, microwave_scan |
| `blowaway` | uncalibrated | Reduced FORT during blowaway | (future) |

Experiments select their profile in `prepare()`:
```python
def prepare(self):
    self.base.prepare()                              # uses active_profile (FORT_loading)
    self.base.prepare(feedback_profile="low_FORT")   # explicit profile
```

Partial profiles only override specific channels — others fall back to the active profile.

**SAFETY NOTE:** If a profile has `ref_voltage: -1.0` (uncalibrated) and `p_gain > 0`, the feedback will drive the amplitude to ZERO. Always calibrate before enabling feedback on a channel.

### feedback_channels.json

Controls HOW the feedback loop behaves (independent of profile):

```json
{
    "channels": {
        "MOT_X": {
            "p_gain": 0.08,
            "tolerance": 0.01,
            "max_iterations": 20,
            "initial_amp": 0.13,
            "max_amp": 0.17,
            "frequency_hz": 80e6
        },
        "excitation": {
            "p_gain": 0.0,       // PERMANENTLY DISABLED — i2v is off
            "tolerance": 1.0,
            "max_iterations": 1
        }
    }
}
```

---

## Rebalancer Methods

### rebalance() — 5-channel @kernel
Stabilizes all 5 channels. FORT and excitation are included but excitation is effectively disabled (p_gain=0 in JSON). Captures initial and final voltages for diagnostics.

### rebalance_3ch() — MOT-only @kernel
Stabilizes MOT X/Y/Z only. Does NOT touch FORT or excitation, does NOT toggle TTL14/TTL6. Used by atom_load_monitor where FORT should not be adjusted during feedback.

### load_params_from_json() — host-side
Reads feedback_channels.json, sets all `self.p0..p4` etc. attributes. Called by BaseExperiment._setup_rebalancer() at build time.

### load_profile(profile=None) — host-side
Reads ref voltages from CalibrationStore for the given profile (or active_profile if None). Sets `self.ref0..ref4`. Called by BaseExperiment.prepare().

### save_diagnostics(final_amps) — host-side
Writes feedback diagnostics to experiment datasets. Must be called explicitly after kernel returns.

---

## How Experiments Use Feedback

```python
class MyExperiment(EnvExperiment):
    def build(self):
        self.base = BaseExperiment(self)
        self.base.build()
        # rebalancer created and configured automatically

    def prepare(self):
        self.base.prepare(feedback_profile="low_FORT")  # or omit for active_profile

    @kernel
    def run(self):
        self.base.initialize_hardware()
        ...
        prepare_for_feedback(self)
        amp0, amp1, amp2, amp3, amp4 = self.rebalancer.rebalance()
        trap_ramp_end_amp = amp3
        restore_after_feedback(self)
        ...
```

---

## FeedbackStandalone (Manual Dashboard Tool)

`manual_control/feedback_standalone.py` — per-channel control for calibration:

| Toggle | Default | Meaning |
|--------|---------|---------|
| `use_stored_ref_i` | True | True = use ref from JSON profile; False = measure fresh voltage |
| `save_ref_i` | False | True = write this channel's ref back to JSON after run |
| `use_json_params_i` | True | True = use gain/tol/iter from JSON; False = use dashboard override |
| `target_profile` | first in JSON | Which profile to read from and save to |

### Workflow: Calibrate FORT at loading depth
1. Set FORT amplitude to `nominal_trapdepth` from AOM control
2. Open FeedbackStandalone, set `target_profile = FORT_loading`
3. Set `use_stored_ref_3 = False` (measure fresh FORT voltage)
4. Set `save_ref_3 = True` (write it back)
5. All other channels: `use_stored_ref = True`, `save_ref = False`
6. Run → only FORT reference updates in the JSON

---

## Diagnostics Saved to Data Files

After each `rebalance()` call, `save_diagnostics()` writes to experiment datasets (archived in HDF5/Zarr):

| Dataset | Contents |
|---------|----------|
| `fb_profile_used` | Which calibration profile was active |
| `fb_initial_amplitudes` | [5] DDS amplitudes at start |
| `fb_final_amplitudes` | [5] DDS amplitudes after convergence |
| `fb_initial_voltages` | [5] Sampler voltages before feedback |
| `fb_final_voltages` | [5] Sampler voltages after convergence |
| `fb_ref_voltages` | [5] Target voltages from profile |
| `fb_params_gain` | [5] P-gains used |
| `fb_params_tolerance` | [5] Tolerances used |
| `fb_params_iterations` | [5] Max iterations used |

---

## SPCM Dual Readout — Parallel vs Sequential

Two versions exist in `sequences.py` for different count rate regimes:

| Function | Method | FIFO capacity | Use case |
|----------|--------|---------------|----------|
| `spcm_image_dual_parallel` | `with parallel:` both SPCMs gate simultaneously | Lower (overflows at high count rates) | Short gates: parity (4ms), MW scan |
| `spcm_image_dual_sequential` | Sequential gate_rising, count with 1ms delays | Higher (handles bright beams) | Long gates: atom_load_monitor (100ms) |
| `spcm_image_dual` | Alias → parallel | (backward compatible default) | Used by check_atom_loaded, background_measurement |

---

## ARTIQ Kernel Constraint

**CRITICAL: No kernel nesting via RPC.**

The Kasli FPGA can only run one kernel at a time. If a kernel calls a host-side method (RPC) that tries to launch a second kernel, the firmware crashes with `ConnectionResetError`.

This is why `rebalance()` is a `@kernel` method that reads from `self.*` attributes — NOT a host-side method that launches a kernel internally.

```
WORKS:     experiment @kernel → self.rebalancer.rebalance() [@kernel]
CRASHES:   experiment @kernel → self.rebalancer.rebalance() [host] → self._run_feedback() [@kernel]
```

---

## Future Optimization: Mid-Experiment Profile Switching

Currently, the profile is selected once in `prepare()` before the kernel starts. All `rebalance()` calls within a single kernel run use the same reference voltages.

### Strategy for mid-experiment switching:

Pre-load multiple profiles' ref voltages as separate attributes in `prepare()`:

```python
def prepare(self):
    refs_loading = self._cal_store.get_refs("FORT_loading")
    exp.rebalancer.ref0_loading = refs_loading[0]
    ...
    refs_low = self._cal_store.get_refs("low_FORT")
    exp.rebalancer.ref0_low = refs_low[0]
    ...
```

Then in the kernel, swap refs without any RPC:

```python
@kernel
def run(self):
    # Loading phase
    self.rebalancer.ref0 = self.rebalancer.ref0_loading
    ...
    self.rebalancer.rebalance()

    # Science phase
    self.rebalancer.ref0 = self.rebalancer.ref0_low
    ...
    self.rebalancer.rebalance()
```

This avoids kernel nesting — all values are pre-loaded, the kernel just copies floats. No JSON reading, no RPC, no host interaction during the kernel.
