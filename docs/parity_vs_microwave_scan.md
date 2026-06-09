# Parity vs. microwave_scan — process schematics & comparison

Schematic diagrams of the per-run sequences of the **parity experiments**
(`repository/experiments/{x,y,z}_parity.py`) and the **microwave_scan experiment**
(`repository/experiments/microwave_scan.py`), and a comparison of how they differ.

Both are built on the same hardware primitives (load → image → state-prep → pulse →
blowaway → read-out), but they are fundamentally different *kinds* of experiment:
microwave_scan is a **driven, deterministic, per-shot** measurement; parity is a
**photon-heralded, conditional, retry-until-click** measurement.

---

## 1. Parity experiments (x / y / z) — atom–photon *heralded* parity

The qubit is a single Cs atom; the partner is a spontaneously-emitted 852 nm photon
whose polarization is entangled with the atomic Zeeman state. The "analysis phase" is
the **photon-polarization basis set by the HWP/QWP waveplate angles** that are scanned.
The single analysis pulse (`microwave_pi_pulse_z` for Z; `microwave_two_photon_x/y`
for X/Y) maps the chosen basis before fluorescence read-out. Aligned to the
`HS_code_*_*_Parity_Optimized_v2` archive structure.

```
initialize_hardware
│
├─ FOR qwp in QWP_scan ───────── physical QWP rotate (NDSP, ~15 s)        ┐ SCAN AXIS =
│  └─ FOR hwp in HWP_scan ────── physical HWP rotate (~5–15 s)            ┘ waveplate angle
│     └─ WHILE measurements < n_measurements:
│        │
│        │  ┌── YELLOW · load a CONFIRMED atom (retry) ──────────────┐
│        │  │  WHILE True:                                           │
│        │  │    feedback rebalance → restore                        │
│        │  │    background image                                    │
│        │  │    load_atom_deterministic                             │
│        │  │    image  (PGC · shot1 · PGC)                          │
│        │  │    if counts−bg > threshold:  BREAK   ◀ atom present   │
│        │  └────────────────────────────────────────────────────────┘
│        │
│        │  ┌── ORANGE · prep ONCE, then HERALD ─────────────────────┐
│        │  │  optical_pumping_F3 + Q-axis ON + trap ramp DOWN  (once)│
│        │  │  WHILE True:                                           │
│        │  │    FOR exc_cycle in range(max_excitation_cycles):      │
│        │  │      break_realtime          ◀ refresh RTIO slack/cycle│
│        │  │      optical pump (MOT + D1 + exc + repump)            │
│        │  │      ► EXCITATION pulse → SPCM gate → click1, click2   │
│        │  │      IF photon (exactly one SPCM clicks):              │
│        │  │         ► ANALYSIS pulse  (πz │ 2φ·x │ 2φ·y)  ◀ basis  │
│        │  │         ► blowaway(F=4)                                │
│        │  │         photon_detected=True ; BREAK                   │
│        │  │      ELSE: ttl_MW block   (no per-cycle PGC, v2 skips) │
│        │  │    IF photon_detected: BREAK                           │
│        │  │    ── recovery (no photon this round) ──               │
│        │  │    trap ramp UP                                        │
│        │  │    every Check_img_every: survival image → lost? BREAK │
│        │  │    else: PGC recool → retry                            │
│        │  └────────────────────────────────────────────────────────┘
│        │
│        │  ┌── BLUE · success (photon heralded) ──┐
│        │  │  trap ramp UP                         │
│        │  │  final_state_readout (shot 2)         │
│        │  │  save(hwp, qwp, timestamps, counts)   │
│        │  └───────────────────────────────────────┘
│        │  reset_to_mot ;  measurements += 1
│
└─ (offline) parity_cos⁴ fit of retention vs HWP angle
```

Reference: `z_parity.py` run() ≈ lines 105–300 (x/y identical modulo the analysis
pulse and RF phase). Loader `load_atom_deterministic`; per-cycle
`self.core.break_realtime()` (v2:559) is required for RTIO slack.

---

## 2. microwave_scan — *driven* MW spectroscopy / Rabi / Ramsey

Drives a microwave or two-photon transition on the atom and measures population
transfer (retention). Deterministic: every shot applies the pulse and reads out; the
swept parameter is a DDS frequency or pulse duration. Mode-dispatched (Freq_* =
spectroscopy, Time_* = Rabi, Ramsey_* = T2*).

```
initialize_hardware
│
├─ FOR loop in range(n_measurements):
│  └─ FOR scan_idx in range(len(scan_values)) ── DDS, fast    ◀ SCAN AXIS = MW freq OR pulse time
│     │   scan_val = scan_values[scan_idx]
│     │
│     │   feedback rebalance → restore
│     │
│     │   ┌── load (SINGLE attempt, NO retry) ──┐
│     │   │  background image                   │
│     │   │  load_atom_deterministic            │
│     │   │  image (shot 1) → loaded? (post-hoc)│
│     │   │  imaging_pgc (recool)               │
│     │   └─────────────────────────────────────┘
│     │
│     │   f3_state_prep
│     │   set_qaxis + trap ramp DOWN + optical_pumping_mw_scan
│     │
│     │   ┌── MODE DISPATCH · ONE MW/2-photon pulse ──┐
│     │   │  Freq_00/01/11/2ph/m10   (spectroscopy)   │  ◀ the pulse IS
│     │   │  Time_*                  (Rabi)           │     the measurement
│     │   │  Ramsey_00/11/2ph        (T2*)            │
│     │   └───────────────────────────────────────────┘
│     │
│     │   blowaway(F=4) + trap ramp UP + pgc_long
│     │   imaging_readout_dual (shot 2) → retained? (post-hoc)
│     │   save_point(scan_val, bg, shot1, shot2)
│     │   reset_mw_scan
│
└─ (in-run) _analyze_and_update → Rabi / Ramsey / Gaussian fit
```

Reference: `microwave_scan.py` run() ≈ lines 285–404; mode dispatch lines 334–373;
`_save_point` (threshold → loaded/retained) lines 406–435.

---

## 3. Key differences

| Aspect | **Parity (x/y/z)** | **microwave_scan** |
|---|---|---|
| **Physics goal** | Measure atom–**photon** entanglement (parity of a heralded Bell pair) | Drive an atomic transition; measure population transfer |
| **Scan axis** | **Waveplate angle** (HWP/QWP) — physical rotators via NDSP, seconds per step | **MW frequency or pulse duration** — DDS, microseconds |
| **Core operation** | **Excitation → spontaneous photon → SPCM click HERALDS** the entangled event | **One MW/2-photon pulse** applied unconditionally |
| **Conditional?** | Yes — everything downstream is **gated on a photon click** | No — every shot pulses and reads out |
| **Herald loop** | Inner `for exc_cycle` ×`max_excitation_cycles`, **retries until a photon**, with recovery/survival between rounds | **None** — single pulse per shot |
| **Loading** | `while True` **retry until a confirmed atom** (only proceeds with a verified atom) | **Single** load attempt; loaded/retained recorded as data |
| **The "pulse"** | **Excitation** pulse (entangles) **+ analysis** pulse (πz / 2φ·x / 2φ·y) that maps the chosen basis | The transition pulse itself (Rabi/Ramsey/spectroscopy), mode-dispatched |
| **State-prep cadence** | F3 + Q-axis + ramp **once** before the herald loop | F3 + Q-axis + ramp **once per shot** (no herald loop) |
| **Survival logic** | `Check_img_every` survival imaging; atom carried across many heralding rounds | None — one shot, done |
| **Analysis** | Offline `parity_cos⁴` fit vs angle | **In-run** fit (`_analyze_and_update`: Rabi/Ramsey/Gaussian) |
| **Read-out** | blowaway(F=4) → `final_state_readout` | blowaway(F=4) → `pgc_long` → `imaging_readout_dual` |
| **External dependency** | Requires the **K10CR1 NDSP server** (`ndsp/k10cr1/launcher_multi_rotor.py`) running, to move the waveplates | None beyond the core device |

**One-line summary:** microwave_scan is a **deterministic, per-shot driven
measurement** that sweeps a DDS parameter and fits a curve in-run; parity is a
**heralded, conditional, retry-until-photon** measurement that sweeps physical
waveplate angles and extracts a parity oscillation offline. They share the
load/image/prep/blowaway/read-out scaffolding, but parity wraps it in a
photon-heralding loop (excitation + SPCM gate + analysis-pulse-on-click + survival
recovery) that microwave_scan has no equivalent of.
