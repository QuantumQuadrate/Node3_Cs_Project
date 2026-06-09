# HARDWARE_SAFETY.md — SNAQ-Node-3 hardware safety limits

**Authoritative safety reference for all code that drives hardware on SNAQ-Node-3.**
Every agent (and human) MUST consult this file before writing or editing code that
commands a DDS amplitude/frequency, a Zotino/DAC voltage, a TTL output, or the FORT
trap laser. If a proposed value would violate a limit here, do NOT write it — stop
and flag it.

> **Status of the numbers below**
> - ✅ **AUTHORITATIVE** values are taken directly from this repo
>   (`repository/config/device_aliases.json`, `experiment_variables.py`) or from
>   the device datasheets (Urukul AD9910, Zotino AD5372). Trust and enforce them.
> - ⚠️ **CONFIRM** values are lab-/laser-/coil-specific safe operating limits that
>   are NOT yet recorded in the repo. They are placeholders. **An operator must fill
>   the real value before this limit can be enforced.** Until filled, treat the
>   subsystem conservatively and ask the human for the number — never guess one.

---

## 0. The cardinal rule

**Code must never drive an actuator past a limit in this file, and must always
leave hardware in a safe state when a sequence ends or aborts.** When in doubt,
command the *lower* power / *smaller* field and ask. Atom loss is recoverable;
a cooked AOM, an overheated coil, or a damaged laser is not.

---

## 1. DDS amplitudes (Urukul AD9910)

DDS amplitude is the dimensionless **ASF** (amplitude scale factor), range **0.0–1.0**.

- ✅ **Hard ceiling: 1.0.** Commanding `set_amplitude` / ASF > 1.0 is invalid and
  must never appear in code.
- ✅ **Per-channel servo caps** (from `config/device_aliases.json` `feedback.max_amp`).
  These are the real safe operating ceilings; the intensity servo will not exceed
  them, and open-loop code must not either:

  | Alias | Device | ✅ max_amp | Purpose |
  |---|---|---|---|
  | `dds_FORT` | urukul1_ch1 | **0.25** | 1064 nm trap AOM |
  | `dds_excitation` | urukul1_ch2 | **0.5** | D1 on/off AOM R, excitation |
  | `dds_MOT_X` | urukul2_ch0 | **0.17** | MOT cooling X fiber AOM |
  | `dds_MOT_Y` | urukul2_ch1 | **0.17** | MOT cooling Y fiber AOM |
  | `dds_MOT_Z` | urukul2_ch2 | **0.17** | MOT cooling Z fiber AOM |

- Channels **without** a servo cap in the config (`dds_mw_IF`, `dds_cooling_DP`,
  `dds_repump`, `dds_D1_pump`, `dds_RF`, `dds_D1_onoff_L`, `dds_pump_repumper`):
  the only hard limit on record is the 1.0 ASF ceiling. The **safe drive level for
  the downstream AOM / RF amplifier is ⚠️ CONFIRM** — do not raise these above their
  current `ExperimentVariables` values without an operator confirming the amplifier
  can take it.
  - `dds_RF` (RF antenna, two-photon): RF-amp input limit ⚠️ CONFIRM (___ ASF max).
  - `dds_mw_IF` (microwave IF): amp limit ⚠️ CONFIRM (___ ASF max).

**Rule:** any code setting a capped channel's amplitude must respect the table
above. If a value comes from a dataset, the dataset's dashboard `max` must not
exceed the cap. Flag any `NumberValue(..., max=...)` that allows exceeding it.

---

## 2. FORT trap laser (1064 nm) — `dds_FORT`

- ✅ Commanded via `dds_FORT` amplitude; nominal setpoint is the `nominal_trapdepth`
  dataset (dashboard default **0.24**, dashboard `max=1`).
- ✅ **Servo cap 0.25** (§1). **`nominal_trapdepth` must never be set above 0.25.**
  The dashboard currently *allows* up to 1.0 — that is a latent hazard; treat 0.25
  as the true ceiling and flag any attempt to exceed it.
- ⚠️ **CONFIRM: absolute max trap laser optical power / AOM RF drive** before which
  the AOM or 1064 nm path is at thermal risk (___). Until recorded, never raise FORT
  drive above the established 0.25 cap.
- **Do not slam FORT power.** Large abrupt changes risk heating the atom out of the
  trap (atom loss) and thermally shocking the AOM. Use the existing ramp pattern if
  one is in `sequences.py`; if changing trap depth, change it the way the archive
  reference does.
- **End-of-sequence:** FORT returns to `nominal_trapdepth` (the drop-in loader
  contract), not off, unless the sequence's explicit purpose is to release the atom.

---

## 3. Zotino DAC / shim coils (AD5372) — `shim_X / Y / Z1 / Z2`

- ✅ **Hardware range: ±10 V** (AD5372). The dashboard `NumberValue`s already enforce
  `min=-10, max=10` for every shim dataset — keep it that way; never widen those
  bounds.
- ⚠️ **CONFIRM: safe shim-coil voltage / current (thermal) limit.** The coils may
  overheat well below ±10 V if held continuously. Record the safe steady-state
  voltage per axis (X ___, Y ___, Z1 ___, Z2 ___) and the max hold time. Until
  recorded, do not command shims near ±10 V for extended periods.
- **Do not step shims discontinuously to large values** — ramp where the archive
  does. A sudden large dB/dt also disturbs the atom.
- **End-of-sequence / abort:** return shims to a safe idle (the "off"/B≈0 config),
  never leave a large field energized.
- Per the project rule, log the **physical shim voltage** with each shot, not a
  slot index.

---

## 4. UV lamp — `ttl_UV` (ttl4, default OFF)

- ✅ Default state **off**. Increases MOT density when on.
- ⚠️ **CONFIRM: max continuous UV on-time and/or duty cycle** before lamp wear or
  heating becomes a concern (___). Until recorded, keep UV on only for the loading
  window and ensure every code path turns it **off** afterward.
- **Never leave UV on across an abort.** Any timeout/error path must drive
  `ttl_UV.off()`.

---

## 5. TTL safe states (return-to-default contract)

Outputs and their ✅ safe/default states (from `config/device_aliases.json`).
ON = RF/beam **blocked** for the switch lines. Every sequence — including its
timeout and error paths — must leave these in their default state:

| Alias | TTL | ✅ default | Meaning of default |
|---|---|---|---|
| `ttl_UV` | ttl4 | off | UV lamp off |
| `ttl_repump_switch` | ttl5 | on | MOT repump RF blocked |
| `ttl_exc_atten1` | ttl6 | on | Excitation atten 1 blocked |
| `ttl_MW_switch` | ttl7 | on | Microwave blocked |
| `ttl_exc_atten2` | ttl12 | on | Excitation atten 2 blocked |
| `ttl_D1_exc_switch` | ttl14 | on | D1 pump + excitation blocked |

**Rule:** if a sequence opens any of these (drives a switch to the unblocked state),
it MUST close it again on every exit path. A review should reject code that can exit
with excitation/MW/repump left unblocked.

---

## 6. Beam-on / RF-on duration (AOM thermal)

- ⚠️ **CONFIRM: max continuous on-time for each high-power AOM / RF amplifier**
  (FORT, MOT, RF antenna, excitation) before thermal drift or damage (___).
- Until recorded: do not introduce indefinite or unbounded `delay`/loop windows that
  hold a high-power AOM driven continuously. Bounded, archive-matching durations only.
- This is the one place an RTIO/logic bug (e.g. a beam left on in a stuck loop) can
  become a *hardware* problem, not just bad data — review loop exit conditions.

---

## 7. Pre-edit checklist (every editing agent runs this)

Before writing/approving code that drives hardware, confirm:

1. **Amplitudes** — every `dds_*` amplitude written is ≤ its §1 cap (and ≤ 1.0).
   FORT ≤ 0.25.
2. **Dashboard bounds** — no `NumberValue(max=...)` exceeds a §1/§2/§3 limit; shim
   bounds remain ±10.
3. **DAC voltages** — every shim voltage is within the confirmed safe range (or, if
   unconfirmed, within ±10 and not held long).
4. **TTL contract** — every switch opened is closed on all exit paths (success,
   timeout, abort). Hardware ends in §5 default states.
5. **FORT** — ends at `nominal_trapdepth`; no abrupt slams; never > 0.25.
6. **UV** — off on every exit path.
7. **Durations** — no unbounded high-power beam/RF window.
8. **Units** — exact; no `NumberValue(unit=...)` + manual scale double-conversion
   (a units bug here can mean a 1000× over-drive).
9. If a value is governed by a ⚠️ CONFIRM limit that isn't filled yet, **stop and
   ask the operator** for the number instead of guessing.

---

## 8. Maintaining this file

- When a ⚠️ CONFIRM limit is established in the lab, replace the placeholder with the
  real value **and its source** (who measured it / datasheet / date), and change the
  marker to ✅.
- If hardware changes (new AOM, new servo cap in `device_aliases.json`, rewired
  channel), update the relevant section here in the same change.
- This file is the single source of truth for safe limits. If code and this file
  disagree, the *more conservative* of the two wins until a human resolves it.
