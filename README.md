# SNAQ-Node-3 — Cs Neutral-Atom ARTIQ-7 Experiment

ARTIQ-7 control software for **SNAQ-Node-3**, a single-atom Cs neutral-atom node in the SNAQ quantum networking collaboration. Hardware: Kasli FPGA at `192.168.1.130` driving 3 Urukul boards (12 AD9910 DDS channels), a Sampler ADC, a Zotino DAC, TTL switches for MW / repump / excitation gating, and two SPCMs for single-photon detection.

## Layout

```
.
├── device_db.py                    Hardware channel map (Kasli + Urukul + Sampler + Zotino + TTLs)
├── Csnode_schematic.PNG            Hardware schematic for the optical / RF setup
├── manual.pdf                      ARTIQ system manual (reference copy)
│
├── archive/                        Original working code, kept for physics validation
│   ├── Calibration/                MW variable scan v7
│   ├── Characterization/           Trap frequency (French method)
│   ├── Devices_and_config/         K10CR1 rotator drivers and docs
│   ├── Experiments/                Z, X, Y parity (v7 + v2 optimized)
│   └── Feedback_and_Manual_Control/  AOM feedback, AOM control, shim coil switching
│
└── repository/                     Rectified modular framework
    ├── config/                     JSON: device aliases, feedback channels, calibration profiles
    ├── utilities/                  BaseExperiment, ExperimentVariables, write_results (HDF5 + Zarr)
    ├── subroutines/                Reusable @kernel physics sequences + AOM rebalancer
    ├── experiments/                z_parity, x_parity, y_parity, microwave_scan, atom_load_monitor
    ├── manual_control/             AOM, shim coil, feedback standalone tools
    ├── applets/                    pyqtgraph live-monitoring plots
    ├── fitting/                    Gaussian dip/peak, damped cosine (Rabi + Ramsey), parity cos⁴
    ├── analysis/                   HDF5/Zarr loaders + analysis notebooks for MW, Ramsey, parity
    ├── ndsp/k10cr1/                K10CR1 waveplate rotator NDSP server
    └── claude_work_ref/            Design docs (refactoring plan, Zarr integration plan)
```

## Quick start

1. **One-time** — submit `repository/utilities/experiment_variables.py` (`ExperimentVariables`) from the ARTIQ dashboard. This populates ~150 persistent datasets in `dataset_db.pyon` with default frequencies, amplitudes, timings, and shim values for the current operating point.

2. **Calibrate beam powers** — submit `repository/manual_control/feedback_standalone.py` (`FeedbackStandalone`) to set 5-channel AOM reference voltages (MOT X/Y/Z, FORT, excitation) into `repository/config/calibration_references.json`.

3. **Monitor atom loading** — submit `repository/experiments/atom_load_monitor.py` (`AtomLoadMonitor`) to see live SPCM count rates while you optimize the MOT.

4. **Run a calibration scan** — `repository/experiments/microwave_scan.py` (`MicrowaveScan`) supports 11 modes (frequency / time / Ramsey × 00 / 01 / 11 / 2ph). Toggle one boolean in the dashboard, set the scan range, run. Auto-calibration writes the fitted value back to the corresponding dataset.

5. **Run physics** — `z_parity.py`, `x_parity.py`, `y_parity.py` run the basis-selective parity oscillation scans over HWP/QWP waveplate angles, scheduling K10CR1 rotators via the NDSP server.

6. **Analyze** — open `repository/analysis/analyze_mw_scan_threshold.ipynb` (or the other notebooks in that folder) to load HDF5 results from `results/` and fit them.

## Conventions

- **Times** are stored in SI seconds in datasets, displayed in µs / ms on the dashboard. Machine-unit times (`*_mu`) are unitless integers (1 mu ≈ 1 ns on this core).
- **Frequencies** are stored in Hz, displayed in MHz.
- **TTL switches** for repump/MW/excitation use *active-low* logic: `.on()` = beam blocked, `.off()` = beam unblocked. The single exception is `ttl_UV` (UV lamp), which is non-inverted.
- **Datasets** are the single source of truth for physics parameters. Edit values in the dashboard; they persist to `dataset_db.pyon`. Two parameter sets currently exist (MW-scan-tuned timings vs Z-parity-v2-tuned shim/frequency values) — see `repository/utilities/experiment_variables.py` docstring comments for which defaults apply where.

## What is not committed

`.gitignore` excludes runtime state (`dataset_db.pyon`, `last_rid.pyon`), experiment output (`results/`, `analysis_results/`, loose `.h5`/`.csv`/`.zarr`), Python bytecode (`__pycache__/`), and IDE / environment folders (`.idea/`, `.vscode/`, `.conda-*`). Experiment data is regenerable from the code and config; if you need to share specific runs, attach them as release artifacts or send them out-of-band.

## Reference

Detailed design rationale lives under `repository/claude_work_ref/`:
- `refactoring_plan.md` — original phase-by-phase refactor design
- `zarr_integration_plan.md` — HDF5 + Zarr dual-write strategy

And under `repository/subroutines/`:
- `FEEDBACK_README.md` — AOM feedback architecture (5-channel inline rebalancer + standalone calibration tool)
