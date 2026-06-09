# Bell State Tomography — Parity Analysis

Analysis pipeline for **atom–photon Bell-state tomography** via parity-oscillation
measurements. It fits parity-vs-angle data in the Z, X, and Y bases to a physical
`cos`-model, extracts the four quadrature populations, and computes the
**entanglement fidelity** of the prepared Bell state.

This package is self-contained: drop in a new measurement CSV from the experiment,
run one function, and get the per-QWP fidelity estimate plus publication figures.

---

## 1. Install

Requires **Python 3.10+** (tested on 3.13).

```bash
# from inside this folder
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies are only `numpy`, `pandas`, `matplotlib`, `scipy` (+ `jupyter` for the notebook).

---

## 2. What's in here

| File | Purpose |
|------|---------|
| `parity_lib.py`         | Analysis engine: fitting, quadrature extraction, fidelity, plotting. |
| `registry.py`           | Batch bookkeeping — scans `Data/`, scores every QWP slice, writes `registry.csv`. |
| `Parity_ANALYSIS.ipynb` | Main interactive analysis / figure-making notebook. |
| `Data/`                 | Raw measurement CSVs (one per basis run). |
| `registry.csv`          | Per-(file, QWP) fidelity log. Safe to delete and regenerate. |

---

## 3. Input data format

Each measurement CSV (one row per experimental shot) must have these columns:

| Column     | Meaning |
|------------|---------|
| `SPCM1`    | Detector 1 reading. `-2` (or `<=0`) means "no click". |
| `SPCM2`    | Detector 2 reading. Same convention. |
| `trials`   | Shots accumulated for that point. |
| `counts`   | Atom fluorescence counts (used to tag atom-present vs. absent). |
| `counts_bg`| Background counts. |
| `HWP`      | Half-wave-plate / analysis angle θ in **degrees** (the swept axis). |
| `QWP`      | Quarter-wave-plate setting — a single file can hold several QWP values. |

**Filename convention** (the basis and timestamp are parsed from it):

```
<BASIS>_basis_parity_measurement[_Optimized]_<YYYYMMDD>_<HHMM>.csv
e.g.  Z_basis_parity_measurement_Optimized_20260502_1056.csv
```

`<BASIS>` is one of `X`, `Y`, `Z`. Put new files in the `Data/` folder.

---

## 4. Quick start — register a new measurement

After taking data, copy the CSV into `Data/` and run:

```bash
python quickstart.py
```

This auto-discovers any new files, scores every QWP slice, and updates
`registry.csv`. Open that CSV to see, per QWP: the fitted visibilities, the
optimal angle `theta_opt`, the estimated max fidelity `max_fid ± max_fid_err`,
and an `is_best_in_file` flag marking the winning QWP.

Equivalently, from Python:

```python
from registry import register_file, new_files_since_registry

for f in new_files_since_registry("Data"):   # only the not-yet-analyzed files
    register_file(f)                          # appends/updates registry.csv
```

---

## 5. Analyzing a single file in code

```python
from parity_lib import analyze_file, select_theta_opt, extract_quadratures_fit

# Pick a QWP value that exists in that file (see the QWP column of your CSV)
data = analyze_file("Data/Z_basis_parity_measurement_Optimized_20260417_1501.csv",
                    target_qwp=175.0)

theta_opt, idx, plus_dominant = select_theta_opt(data, use_fits=True)
vals, errs = extract_quadratures_fit(theta_opt, data["fits"])
print("theta_opt =", theta_opt, "quadratures =", vals)
```

`data["fits"]` holds the `cos`-fit (visibility, parameters, covariance, smooth
curve) for each of the four populations and the two parity combinations.

---

## 6. Computing fidelity

Two estimators are provided in `parity_lib.py`:

- **`fidelity_two_basis(...)`** — Blinov lower bound from two bases (Z + one of X/Y):
  `F ≥ ½(Z_b1+Z_b2+X_b1+X_b2 − X_s1−X_s2 − 2·sqrt(Z_s1·Z_s2))`.
- **`fidelity_three_basis(per_basis)`** — full three-basis fidelity:
  `F = ¼(1 + Σ_i (B_i1+B_i2 − S_i1−S_i2))` over Z, X, Y.
- **`correlator_per_basis(per_basis)`** — per-basis correlator, to see which basis
  is dragging the fidelity down.

Both return `(F, sigma_F)` with full error propagation. See
`Parity_ANALYSIS.ipynb` for an end-to-end worked example that builds the
`per_basis` dict and renders the figure.

---

## 7. Making the publication figure

`create_figure(per_basis_data, ...)` renders the multi-panel parity + bar figure
and writes both `.svg` and `.pdf`. The notebook walks through assembling
`per_basis_data` for the Z/X/Y rows; copy that cell as your template.

---

## Notes / gotchas

- `hist_thres` (default **15**) is the atom-present count threshold. Re-tune it if
  your fluorescence levels change — it directly affects the atom/no-atom split.
- The fit frequency is locked to `f = 4` (`fixed_freq=4.0`) for the cos model; pass
  a different value to `fit_cos4` if your sweep geometry differs.
- `registry.csv` is regenerable: delete it and re-run `quickstart.py` to rebuild
  from scratch.
- A slice needs **≥ 4 distinct HWP angles** to be fit; otherwise it's skipped.

Questions about the analysis assumptions are in the inline docstrings of
`parity_lib.py`.
