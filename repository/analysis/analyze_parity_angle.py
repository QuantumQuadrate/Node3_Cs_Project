"""Parity-angle experiment analysis: parity oscillation vs waveplate angle.

Analyzes a single `ParityAngleExperiment` run (the free-analysis-angle parity
experiment, `experiments/parity_angle.py`). For each QWP slice it builds the
parity correlations vs HWP angle, fits the cos^4 parity oscillation, and reports
the visibility.

Ported from analyze_parity.ipynb (the Z/X/Y notebook) and adapted for the
free-angle experiment:
  - reads the run's analysis angle from the `basis_phase_deg` tag (0=X, 90=Y),
    instead of the old `basis_axis` string;
  - globs `*ParityAngleExperiment.h5`;
  - the platform-safe cos^4 fit (least_squares + Cramer's-rule covariance, no
    BLAS/LAPACK — those crash on this Windows numpy/scipy build) is inlined from
    that notebook, so this module is self-contained (no parity_lib dependency).

Definitions (unchanged from analyze_parity):
    S0 = click on SPCM1, S1 = click on SPCM2
    L  = atom lost after readout, E = atom still present (counts >= threshold)
    parity_plus  = p(S0,L) + p(S1,E)
    parity_minus = p(S1,L) + p(S0,E)
The cos^4 fit visibility V of parity_plus / parity_minus is the parity-oscillation
contrast for that QWP setting.

Usage::

    python analyze_parity_angle.py                 # latest ParityAngleExperiment run
    python analyze_parity_angle.py 1042            # a specific run ID (RID)
    python analyze_parity_angle.py path/to/run.h5  # an explicit file
    python analyze_parity_angle.py <file> --thres 19 --save-plot fig.png

Options:
    --thres N     Atom-present threshold on counts_signal (default: value in file).
    --no-plot     Skip the per-QWP figures, just print the table.
    --save-plot P Save the summary plot to PATH (per-QWP figures are shown only).
    --qwp Q       Restrict analysis/plots to one QWP value (deg).
"""

import sys, os, glob, argparse, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.optimize import least_squares

RESULTS_DIR = Path(
    r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes\repository\results"
)


# =========================================================================
#  MODEL (inlined from parity_lib.cos4_physical)
# =========================================================================

def cos4_physical(x_deg, C, V, phase_deg, f):
    """Physical cos^4 parity model: C * (1 + V * cos(f*x - phase))."""
    return C * (1 + V * np.cos(np.deg2rad(f * x_deg - phase_deg)))


# =========================================================================
#  RUN DISCOVERY / METADATA
# =========================================================================

def _rid_of(path):
    try:
        return int(os.path.basename(path).split("-")[0])
    except Exception:
        return -1


def find_runs():
    """All ParityAngleExperiment .h5 files under RESULTS_DIR, sorted by RID."""
    files = glob.glob(str(RESULTS_DIR / "**" / "*ParityAngleExperiment.h5"),
                      recursive=True)
    return sorted(files, key=_rid_of)


def read_basis_phase(h5_path):
    """Return the run's analysis angle `basis_phase_deg` (float, degrees), or NaN.

    ParityAngleExperiment records it via set_dataset(..., broadcast=True), which
    lands both as a dataset and as an attr on the datasets group.
    """
    try:
        with h5py.File(h5_path, "r", locking=False) as hf:
            if "datasets/basis_phase_deg" in hf:
                v = np.asarray(hf["datasets/basis_phase_deg"][()]).reshape(-1)
                if v.size:
                    return float(v[0])
            if "datasets" in hf and "basis_phase_deg" in hf["datasets"].attrs:
                return float(hf["datasets"].attrs["basis_phase_deg"])
    except Exception:
        pass
    return float("nan")


def _nearest_basis_label(phase_deg):
    """Cosmetic label for a free angle: nearest of X (0/180) or Y (90/270)."""
    if not np.isfinite(phase_deg):
        return "?"
    p = phase_deg % 180.0
    return "Y-like" if 45.0 <= p < 135.0 else "X-like"


# =========================================================================
#  LOADER  (from analyze_parity.ipynb cell 4)
# =========================================================================

def load_h5_as_df(h5_path):
    """Load a ParityAngleExperiment HDF5 file into a DataFrame with columns
    SPCM1, SPCM2, trials, counts, counts_bg, HWP, QWP. Also returns the
    atom_threshold recorded in the file (default 12.0)."""
    with h5py.File(h5_path, "r", locking=False) as hf:
        def _ds(key):
            return hf[f"datasets/{key}"][()]
        spcm1     = _ds("SPCM1_timestamps").astype(float)
        spcm2     = _ds("SPCM2_timestamps").astype(float)
        trials    = _ds("trials_per_meas").astype(int)
        counts    = _ds("counts_signal").astype(int)
        counts_bg = _ds("counts_background").astype(int)
        hwp       = _ds("HWP_angles").astype(float)
        qwp       = _ds("QWP_angles").astype(float)
        if "archive/atom_threshold" in hf:
            atom_threshold = float(hf["archive/atom_threshold"][()])
        else:
            atom_threshold = 12.0
    df = pd.DataFrame({
        "SPCM1": spcm1, "SPCM2": spcm2, "trials": trials,
        "counts": counts, "counts_bg": counts_bg, "HWP": hwp, "QWP": qwp,
    })
    return df, atom_threshold


# =========================================================================
#  PLATFORM-SAFE cos^4 FIT  (from analyze_parity.ipynb cell 4)
# =========================================================================

def _inv3x3(M):
    """Manual 3x3 inverse via Cramer's rule (avoids the LAPACK/BLAS crash)."""
    a, b, c = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    d, e, f = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    g, h, k = float(M[2, 0]), float(M[2, 1]), float(M[2, 2])
    det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-15:
        return np.eye(3) * 1e6
    inv = np.empty((3, 3), dtype=float)
    inv[0, 0] = (e * k - f * h) / det; inv[0, 1] = -(b * k - c * h) / det; inv[0, 2] = (b * f - c * e) / det
    inv[1, 0] = -(d * k - f * g) / det; inv[1, 1] = (a * k - c * g) / det; inv[1, 2] = -(a * f - c * d) / det
    inv[2, 0] = (d * h - e * g) / det; inv[2, 1] = -(a * h - b * g) / det; inv[2, 2] = (a * e - b * d) / det
    return inv


def safe_fit_cos4(x, y, yerr, fixed_freq=4.0):
    """Platform-safe cos^4 fit (no tight-bound curve_fit, no BLAS/LAPACK).

    Returns (popt, info); popt = [C, V, phase_deg, fixed_freq];
    info = {'vis', 'vis_err', 'popt', 'pcov' (4x4), 'y_smooth'}.
    """
    C0 = max(float(np.mean(y)), 1e-9)
    amp_est = (float(np.max(y)) - float(np.min(y))) / 2.0
    V0 = float(np.clip(amp_est / C0, 0.05, 0.95))

    best_p0 = [C0, V0, 0.0]
    best_cost = np.inf
    for ph in range(0, 360, 45):
        y_mod = cos4_physical(x, C0, V0, float(ph), fixed_freq)
        cost = float(np.sum(((y_mod - y) / np.maximum(yerr, 1e-9)) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_p0 = [C0, V0, float(ph)]

    def residuals(p):
        C_r, V_r, phi = p
        return (cos4_physical(x, max(float(C_r), 1e-9),
                              float(np.clip(V_r, 0.0, 1.0)),
                              float(phi), fixed_freq) - y) / np.maximum(yerr, 1e-9)

    try:
        result = least_squares(residuals, best_p0, method='lm', max_nfev=5000,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        C_f = max(float(result.x[0]), 0.0)
        V_f = float(np.clip(result.x[1], 0.0, 1.0))
        phi_f = float(result.x[2])
        popt = np.array([C_f, V_f, phi_f, fixed_freq])

        J = np.asarray(result.jac, dtype=float)
        n_d, n_p = J.shape
        chi2 = float(np.sum(result.fun ** 2))
        dof = max(n_d - n_p, 1)
        JtJ = np.einsum('ki,kj->ij', J, J)      # no BLAS call
        pcov_3 = _inv3x3(JtJ) * (chi2 / dof)

        pcov = np.zeros((4, 4), dtype=float)
        pcov[:3, :3] = pcov_3
        V_err = float(np.sqrt(max(0.0, pcov[1, 1])))
        theta_sm = np.linspace(float(x.min()), float(x.max()), 500)
        y_smooth = cos4_physical(theta_sm, C_f, V_f, phi_f, fixed_freq)
        return popt, {'vis': V_f, 'vis_err': V_err, 'popt': popt,
                      'pcov': pcov, 'y_smooth': y_smooth}
    except Exception:
        C0_ = max(float(best_p0[0]), 0.0)
        V0_ = float(np.clip(best_p0[1], 0.0, 1.0))
        phi0 = float(best_p0[2])
        popt_fb = np.array([C0_, V0_, phi0, fixed_freq])
        theta_sm = np.linspace(float(x.min()), float(x.max()), 500)
        return popt_fb, {'vis': 0.0, 'vis_err': 0.0, 'popt': popt_fb,
                         'pcov': np.zeros((4, 4)),
                         'y_smooth': cos4_physical(theta_sm, C0_, V0_, phi0, fixed_freq)}


# =========================================================================
#  PER-QWP ANALYSIS  (from analyze_parity.ipynb cell 4)
# =========================================================================

def safe_analyze_qwp_slice(d, hist_thres=18):
    """Per-QWP parity analysis. Returns {'angles','theta_smooth','res','fits'}
    or None if fewer than 4 valid single-click HWP angles."""
    if len(d) == 0:
        return None

    frac_m2 = float(np.mean((d['SPCM1'] == -2) | (d['SPCM2'] == -2)))
    det0 = (d['SPCM1'] != -2) if frac_m2 > 0.01 else (d['SPCM1'] > 0)
    det1 = (d['SPCM2'] != -2) if frac_m2 > 0.01 else (d['SPCM2'] > 0)
    atom = (d['counts'] >= hist_thres)

    single_click = np.logical_xor(det0, det1)
    valid_mask = (single_click.values
                  & np.isfinite(d['HWP'].values)
                  & np.isfinite(d['counts'].values))

    d_valid = d[valid_mask]
    det0v = det0.values[valid_mask]
    det1v = det1.values[valid_mask]
    atomv = atom.values[valid_mask]
    hwp = d_valid['HWP'].values

    angles = np.sort(np.unique(hwp))
    if len(angles) < 4:
        return None

    keys = ['p_S0_L', 'p_S1_L', 'p_S0_E', 'p_S1_E',
            'e_S0_L', 'e_S1_L', 'e_S0_E', 'e_S1_E',
            'parity_plus', 'parity_minus', 'e_parity_plus', 'e_parity_minus']
    res = {k: [] for k in keys}

    for ang in angles:
        mask = (hwp == ang)
        N = int(np.sum(mask))
        if N == 0:
            continue
        d0a = det0v[mask]; d1a = det1v[mask]; aa = atomv[mask]
        k_S0_L = int(np.sum(d0a & (~aa)))
        k_S1_L = int(np.sum(d1a & (~aa)))
        k_S0_E = int(np.sum(d0a & aa))
        k_S1_E = int(np.sum(d1a & aa))

        def _pe(k, n):
            p = k / n
            e = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
            return p, max(e, 1e-4)

        p_S0_L, e_S0_L = _pe(k_S0_L, N)
        p_S1_L, e_S1_L = _pe(k_S1_L, N)
        p_S0_E, e_S0_E = _pe(k_S0_E, N)
        p_S1_E, e_S1_E = _pe(k_S1_E, N)

        res['p_S0_L'].append(p_S0_L); res['e_S0_L'].append(e_S0_L)
        res['p_S1_L'].append(p_S1_L); res['e_S1_L'].append(e_S1_L)
        res['p_S0_E'].append(p_S0_E); res['e_S0_E'].append(e_S0_E)
        res['p_S1_E'].append(p_S1_E); res['e_S1_E'].append(e_S1_E)
        res['parity_plus'].append(p_S0_L + p_S1_E)
        res['parity_minus'].append(p_S1_L + p_S0_E)
        res['e_parity_plus'].append(np.sqrt(e_S0_L ** 2 + e_S1_E ** 2))
        res['e_parity_minus'].append(np.sqrt(e_S1_L ** 2 + e_S0_E ** 2))

    for k in res:
        res[k] = np.array(res[k])

    fits = {}
    target_keys = ['p_S0_L', 'p_S1_L', 'p_S0_E', 'p_S1_E', 'parity_plus', 'parity_minus']
    err_keys = ['e_S0_L', 'e_S1_L', 'e_S0_E', 'e_S1_E', 'e_parity_plus', 'e_parity_minus']
    theta_smooth = np.linspace(float(min(angles)), float(max(angles)), 500)
    for k, ek in zip(target_keys, err_keys):
        _, info = safe_fit_cos4(angles, res[k], res[ek], fixed_freq=4.0)
        fits[k] = info

    return {'angles': angles, 'theta_smooth': theta_smooth, 'res': res, 'fits': fits}


def select_theta_opt(result, use_fits=True, parity_convention="plus"):
    """Optimal HWP angle for entanglement, from the smooth cos^4 fit curves.

    The HWP scan usually spans > 90 deg, and the parity oscillation has a 90 deg
    period (cos(4*HWP)), so there are two antinode families 45 deg apart with
    EQUAL contrast: one gives bars [0, 0.5, 0.5, 0] (parity_plus dominant), the
    other [0.5, 0, 0, 0.5] (parity_minus dominant). Selecting by |P+ - P-| lets
    noise pick either one, which flips the bar plot between runs.

    To keep the bar pattern consistent we pick by SIGNED contrast:
      parity_convention="plus"  -> antinode maximizing (P+ - P-)  -> [0, .5, .5, 0]
      parity_convention="minus" -> antinode maximizing (P- - P+)  -> [.5, 0, 0, .5]

    Returns (theta_opt_deg, idx, plus_dominant).
    """
    if use_fits:
        plus = np.asarray(result["fits"]["parity_plus"]["y_smooth"], dtype=float)
        minus = np.asarray(result["fits"]["parity_minus"]["y_smooth"], dtype=float)
        thetas = result["theta_smooth"]
    else:
        plus = np.asarray(result["res"]["parity_plus"], dtype=float)
        minus = np.asarray(result["res"]["parity_minus"], dtype=float)
        thetas = result["angles"]
    diff = plus - minus
    idx = int(np.argmax(-diff if parity_convention == "minus" else diff))
    return float(thetas[idx]), idx, bool(diff[idx] > 0)


def extract_quadratures_fit(result, theta_deg):
    """Evaluate the four category probabilities from their cos^4 fits at one HWP
    angle. Returns (vals, errs) in bar order [p_S1_L, p_S0_L, p_S1_E, p_S0_E].

    vals come from the smooth fits at theta_deg; errs are borrowed from the
    nearest measured HWP point (a simple, honest error bar for the bar chart).
    """
    fits = result["fits"]
    angles = result["angles"]
    res = result["res"]
    j = int(np.argmin(np.abs(angles - theta_deg)))

    order = ["p_S1_L", "p_S0_L", "p_S1_E", "p_S0_E"]
    err_order = ["e_S1_L", "e_S0_L", "e_S1_E", "e_S0_E"]
    vals = [float(cos4_physical(theta_deg, *fits[k]["popt"])) for k in order]
    errs = [float(res[ek][j]) for ek in err_order]
    return vals, errs


def fit_qwp_visibility(analyzed_by_qwp, n_grid=400):
    """Fit parity visibility vs QWP angle with a sinusoid and return the QWP that
    maximizes the FIT (a continuous value, not the discrete experimental peak).

    Returns a dict: {qwps, vis, verr, best_qwp, best_vis, model, qwp_smooth,
    vis_smooth, popt}. The peak is searched only within the measured QWP range
    (never extrapolated). With < 4 QWP points, falls back to the discrete max.
    """
    qwps = np.array(sorted(analyzed_by_qwp), dtype=float)
    vis = np.array([analyzed_by_qwp[q]["vis_plus"] for q in qwps], dtype=float)
    verr = np.array([max(analyzed_by_qwp[q]["vis_plus_err"], 1e-3) for q in qwps],
                    dtype=float)
    out = {"qwps": qwps, "vis": vis, "verr": verr,
           "qwp_smooth": None, "vis_smooth": None, "popt": None}

    if len(qwps) < 4:
        j = int(np.argmax(vis))
        out.update(best_qwp=float(qwps[j]), best_vis=float(vis[j]), model="discrete")
        return out

    C0 = float(np.mean(vis))
    A0 = max((float(vis.max()) - float(vis.min())) / 2.0, 0.05)

    # grid-search frequency (cycles over a 360 deg argument) + phase, then refine
    best = None
    for f in np.linspace(0.5, 4.0, 24):
        for phdeg in range(0, 360, 20):
            model = C0 + A0 * np.cos(np.deg2rad(f * qwps - phdeg))
            cost = float(np.sum(((model - vis) / verr) ** 2))
            if best is None or cost < best[0]:
                best = (cost, float(f), float(phdeg))
    f0, ph0 = best[1], best[2]

    def resid(p):
        C, A, f, ph = p
        return (C + A * np.cos(np.deg2rad(f * qwps - ph)) - vis) / verr

    try:
        r = least_squares(resid, [C0, A0, f0, ph0], method="lm", max_nfev=5000)
        C, A, f, ph = (float(v) for v in r.x)
    except Exception:
        C, A, f, ph = C0, A0, f0, ph0

    qs = np.linspace(float(qwps.min()), float(qwps.max()), n_grid)
    vs = C + A * np.cos(np.deg2rad(f * qs - ph))
    j = int(np.argmax(vs))
    out.update(best_qwp=float(qs[j]), best_vis=float(vs[j]),
               qwp_smooth=qs, vis_smooth=vs, model="sinusoid", popt=(C, A, f, ph))
    return out


# =========================================================================
#  TOP-LEVEL ANALYSIS
# =========================================================================

def analyze_run(path, thres=None, qwp_only=None, parity_convention="plus"):
    """Analyze one ParityAngleExperiment file. Returns (meta, analyzed_by_qwp, df).

    parity_convention ("plus"/"minus") fixes which antinode family HWP_opt lands
    on, so the bar plot doesn't flip between runs (see select_theta_opt)."""
    df, atom_threshold_file = load_h5_as_df(path)
    hist_thres = int(atom_threshold_file if thres is None else thres)

    phase_deg = read_basis_phase(path)
    unique_qwp = np.sort(df["QWP"].unique())
    if qwp_only is not None:
        unique_qwp = unique_qwp[np.isclose(unique_qwp, qwp_only)]

    analyzed_by_qwp = {}
    for qv in unique_qwp:
        slice_df = df[df["QWP"] == qv].copy()
        if len(slice_df["HWP"].unique()) < 4:
            continue
        result = safe_analyze_qwp_slice(slice_df, hist_thres=hist_thres)
        if result is None:
            continue
        fits = result["fits"]
        theta_opt, idx_opt, plus_dom = select_theta_opt(
            result, use_fits=True, parity_convention=parity_convention)
        analyzed_by_qwp[qv] = {
            "result": result,
            "vis_plus": fits["parity_plus"]["vis"],
            "vis_plus_err": fits["parity_plus"]["vis_err"],
            "vis_minus": fits["parity_minus"]["vis"],
            "vis_minus_err": fits["parity_minus"]["vis_err"],
            "hwp_opt": theta_opt,          # HWP angle for maximum entanglement
            "plus_dominant": plus_dom,     # True: parity_plus dominant at hwp_opt
            "n_angles": len(result["angles"]),
        }

    qwp_fit = fit_qwp_visibility(analyzed_by_qwp) if analyzed_by_qwp else None
    meta = {
        "file": Path(path).name,
        "path": str(path),
        "rid": _rid_of(path),
        "basis_phase_deg": phase_deg,
        "basis_label": _nearest_basis_label(phase_deg),
        "hist_thres": hist_thres,
        "atom_threshold_file": atom_threshold_file,
        "n_rows": len(df),
        "n_qwp": len(unique_qwp),
        "success_prob": (len(df) / df['trials'].sum()) if df['trials'].sum() else float("nan"),
        "qwp_fit": qwp_fit,
        "best_qwp_fit": (qwp_fit["best_qwp"] if qwp_fit else float("nan")),
    }
    return meta, analyzed_by_qwp, df


# =========================================================================
#  TABLE
# =========================================================================

def print_summary(meta, analyzed_by_qwp):
    ph = meta["basis_phase_deg"]
    print(f"File: {meta['file']}   RID={meta['rid']}   "
          f"basis_phase={ph:.2f} deg ({meta['basis_label']})")
    print(f"  rows: {meta['n_rows']}   QWP points: {meta['n_qwp']}   "
          f"threshold: {meta['hist_thres']}   "
          f"success: {meta['success_prob']*100:.2f}%")
    if not analyzed_by_qwp:
        print("  >> No QWP slice had >= 4 valid HWP angles; collect more HWP points.")
        return None

    print()
    hdr = (f"{'QWP (deg)':>10}  {'n_hwp':>5}  {'vis+':>8}  {'vis+_err':>8}  "
           f"{'vis-':>8}  {'vis-_err':>8}")
    print(hdr)
    print("-" * len(hdr))
    for qv in sorted(analyzed_by_qwp):
        r = analyzed_by_qwp[qv]
        print(f"{qv:10.1f}  {r['n_angles']:5d}  "
              f"{r['vis_plus']:8.4f}  {r['vis_plus_err']:8.4f}  "
              f"{r['vis_minus']:8.4f}  {r['vis_minus_err']:8.4f}")

    best_qwp = max(analyzed_by_qwp, key=lambda q: analyzed_by_qwp[q]["vis_plus"])
    print(f"\nBest QWP by vis+ : {best_qwp:.1f} deg  "
          f"(vis+ = {analyzed_by_qwp[best_qwp]['vis_plus']:.4f})")
    return best_qwp


# =========================================================================
#  PLOTS  (self-contained; no parity_lib)
# =========================================================================

def plot_qwp_slice(meta, qv, entry, out=None):
    """Parity_plus / parity_minus vs HWP for one QWP, with cos^4 fit curves."""
    import matplotlib.pyplot as plt

    result = entry["result"]
    angles = result["angles"]
    ts = result["theta_smooth"]
    res = result["res"]
    fits = result["fits"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(angles, res["parity_plus"], yerr=res["e_parity_plus"],
                fmt="o", color="C0", capsize=3,
                label=r"$\mathcal{P}_+ = p_{S0,L}+p_{S1,E}$")
    ax.plot(ts, fits["parity_plus"]["y_smooth"], "-", color="C0", lw=1.6,
            label=f"fit  V={entry['vis_plus']:.3f}±{entry['vis_plus_err']:.3f}")
    ax.errorbar(angles, res["parity_minus"], yerr=res["e_parity_minus"],
                fmt="s", color="C3", capsize=3,
                label=r"$\mathcal{P}_- = p_{S1,L}+p_{S0,E}$")
    ax.plot(ts, fits["parity_minus"]["y_smooth"], "--", color="C3", lw=1.6,
            label=f"fit  V={entry['vis_minus']:.3f}±{entry['vis_minus_err']:.3f}")

    if "hwp_opt" in entry:
        ax.axvline(entry["hwp_opt"], color="green", ls=":", lw=1.5,
                   label=f"HWP_opt = {entry['hwp_opt']:.1f}° (max entanglement)")

    ax.set_xlabel("HWP angle (deg)")
    ax.set_ylabel("Parity correlation")
    ax.set_title(f"Parity oscillation  |  RID {meta['rid']}  |  "
                 f"basis {meta['basis_phase_deg']:.1f}°  |  QWP = {qv:.1f}°")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"Plot saved: {out}")
    else:
        plt.show()


def plot_visibility_summary(meta, analyzed_by_qwp, out=None):
    """Visibility vs QWP bar chart across the run's QWP points."""
    import matplotlib.pyplot as plt

    if not analyzed_by_qwp:
        print("Nothing to plot: no valid QWP slices.")
        return
    qvals = sorted(analyzed_by_qwp)
    x = np.arange(len(qvals))
    vp = [analyzed_by_qwp[q]["vis_plus"] for q in qvals]
    vpe = np.clip([analyzed_by_qwp[q]["vis_plus_err"] for q in qvals], 0, None)
    vm = [analyzed_by_qwp[q]["vis_minus"] for q in qvals]
    vme = np.clip([analyzed_by_qwp[q]["vis_minus_err"] for q in qvals], 0, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, vp, width=0.35, yerr=vpe, capsize=5, color="C0",
           alpha=0.85, label=r"vis ($\mathcal{P}_+$)")
    ax.bar(x + 0.2, vm, width=0.35, yerr=vme, capsize=5, color="#888888",
           alpha=0.8, label=r"vis ($\mathcal{P}_-$)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q:.1f}" for q in qvals])
    ax.set_xlabel("QWP angle (deg)")
    ax.set_ylabel("Parity visibility V")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Parity visibility vs QWP  |  RID {meta['rid']}  |  "
                 f"basis {meta['basis_phase_deg']:.1f}°")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"Summary plot saved: {out}")
    else:
        plt.show()


# =========================================================================
#  MULTI-RUN  (basis-angle scan over a RID set YOU choose)
# =========================================================================

def path_for_rid(rid):
    """Resolve an integer RID to its ParityAngleExperiment .h5 path (or None)."""
    matches = [f for f in find_runs() if _rid_of(f) == int(rid)]
    return matches[-1] if matches else None


def analyze_run_set(rids, thres=None, qwp_only=None, parity_convention="plus"):
    """Analyze a SET of runs you assign, as a basis-angle scan.

    `rids` is a list you pick yourself, e.g. [1071, 1077, 1080, 1083, 1086]
    (ints/RID-strings, or explicit .h5 paths). Each run is analyzed on its own;
    per run we keep the QWP slice with the highest vis_plus. Any RID that is
    missing or has no valid slice is SKIPPED with a printed note -- so a dropped
    angle (e.g. your 15 deg) just doesn't appear, nothing crashes.

    Returns a list of per-run summary dicts sorted by basis_phase_deg.
    """
    summaries = []
    for r in rids:
        if isinstance(r, str) and os.path.exists(r):
            path = str(r)
        else:
            try:
                path = path_for_rid(int(r))
            except (TypeError, ValueError):
                print(f"[skip] {r!r}: not a valid RID or path")
                continue
            if path is None:
                print(f"[skip] RID {r}: no ParityAngleExperiment file found")
                continue
        meta, analyzed, df = analyze_run(path, thres=thres, qwp_only=qwp_only,
                                         parity_convention=parity_convention)
        if not analyzed:
            print(f"[skip] RID {meta['rid']}: no QWP slice with >= 4 HWP angles")
            continue
        # Best QWP from the SINUSOIDAL FIT (continuous), not the discrete max.
        qf = meta["qwp_fit"]
        best_qwp = qf["best_qwp"]
        # Use the measured QWP slice nearest the fitted peak for the HWP analysis
        # and bar plot (the fit peak usually falls between grid points).
        best_qwp_slice = min(analyzed, key=lambda q: abs(q - best_qwp))
        e = analyzed[best_qwp_slice]
        summaries.append({
            "rid": meta["rid"],
            "basis_phase_deg": meta["basis_phase_deg"],
            "best_qwp": best_qwp,               # fitted (continuous) best QWP
            "best_qwp_slice": best_qwp_slice,   # nearest measured QWP used for HWP/bars
            "max_vis": qf["best_vis"],          # visibility from the QWP fit peak
            "max_vis_err": e["vis_plus_err"],   # error borrowed from nearest slice
            "hwp_opt": e["hwp_opt"],            # HWP for maximum entanglement (from fit)
            "plus_dominant": e["plus_dominant"],
            "vis_minus": e["vis_minus"], "vis_minus_err": e["vis_minus_err"],
            "qwp_fit_model": qf["model"],
            "n_qwp": len(analyzed),
            "n_rows": meta["n_rows"],
            "_meta": meta,                       # kept for plot helpers
            "_entry": e,
            "_qwp_fit": qf,
        })
    summaries.sort(key=lambda s: (s["basis_phase_deg"], s["rid"]))
    return summaries


def print_run_set_table(summaries):
    """Per-RID: best QWP, its maximum visibility, and the HWP for max entanglement."""
    if not summaries:
        print("No runs analyzed in this set.")
        return
    hdr = (f"{'RID':>9}  {'phase_deg':>9}  {'QWP_fit':>8}  {'(slice)':>8}  {'max_vis':>8}  "
           f"{'vis_err':>8}  {'HWP_opt':>8}  {'parity':>7}  {'n_rows':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        parity = "P+" if s["plus_dominant"] else "P-"
        print(f"{s['rid']:>9}  {s['basis_phase_deg']:>9.2f}  {s['best_qwp']:>8.2f}  "
              f"{s['best_qwp_slice']:>8.1f}  {s['max_vis']:>8.4f}  {s['max_vis_err']:>8.4f}  "
              f"{s['hwp_opt']:>8.2f}  {parity:>7}  {s['n_rows']:>7}")
    print("\nQWP_fit = best QWP from the sinusoidal fit of visibility vs QWP "
          "(continuous); (slice) = nearest measured QWP used for the HWP fit / bars.")
    print("HWP_opt = HWP where |parity_plus - parity_minus| peaks (from the cos^4 fit); "
          "parity = which correlation dominates (fixed by parity_convention, so it "
          "should NOT flip between RIDs).")


def plot_run_set_slices(summaries):
    """For each RID, show the parity oscillation of its BEST-visibility QWP slice
    (the optimal HWP is marked inside plot_qwp_slice)."""
    for s in summaries:
        plot_qwp_slice(s["_meta"], s["best_qwp"], s["_entry"])


def plot_qwp_fit(meta, out=None):
    """Plot parity visibility vs QWP for one run, with the sinusoidal fit and the
    fitted best QWP marked."""
    import matplotlib.pyplot as plt

    qf = meta.get("qwp_fit")
    if not qf:
        print(f"RID {meta['rid']}: no QWP fit (no valid slices).")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(qf["qwps"], qf["vis"], yerr=np.clip(qf["verr"], 0, None),
                fmt="o", color="C4", capsize=3, label="visibility (per QWP)")
    if qf["qwp_smooth"] is not None:
        ax.plot(qf["qwp_smooth"], qf["vis_smooth"], "-", color="C4", lw=1.6,
                label=f"sinusoid fit")
    ax.axvline(qf["best_qwp"], color="green", ls=":", lw=1.6,
               label=f"best QWP = {qf['best_qwp']:.2f}° (V={qf['best_vis']:.3f})")
    ax.set_xlabel("QWP angle (deg)")
    ax.set_ylabel("Parity visibility V")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"QWP vs visibility  |  RID {meta['rid']}  |  "
                 f"basis {meta['basis_phase_deg']:.1f}°  |  fit: {qf['model']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"QWP-fit plot saved: {out}")
    else:
        plt.show()


def plot_run_set_qwpfits(summaries):
    """For each RID, the QWP-vs-visibility sinusoidal fit (best QWP marked)."""
    for s in summaries:
        plot_qwp_fit(s["_meta"])


def plot_bar_split(meta, qv, entry, out=None):
    """Bar chart of the four category probabilities at the optimal HWP -- the
    'splitting' between the two parity pairs that signals entanglement.

    parity_plus  = p_S0_L + p_S1_E   (blue + green bars)
    parity_minus = p_S1_L + p_S0_E   (red + orange bars)
    """
    import matplotlib.pyplot as plt

    theta = entry["hwp_opt"]
    vals, errs = extract_quadratures_fit(entry["result"], theta)
    labels = [r"$p_{S1,\mathrm{lost}}$", r"$p_{S0,\mathrm{lost}}$",
              r"$p_{S1,\mathrm{exist}}$", r"$p_{S0,\mathrm{exist}}$"]
    colors = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]

    # bar order is [p_S1_L, p_S0_L, p_S1_E, p_S0_E]; the dominant parity pair
    # (P+ = p_S0_L + p_S1_E vs P- = p_S1_L + p_S0_E) is what "splits" high vs low.

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(4)
    ax.bar(x, vals, yerr=np.clip(errs, 0, None), capsize=5,
           color=colors, alpha=0.88, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(1.0, max(vals) * 1.2))
    parity = "P+ dominant" if entry["plus_dominant"] else "P- dominant"
    ax.set_title(f"RID {meta['rid']}  |  basis {meta['basis_phase_deg']:.1f}°  |  "
                 f"QWP {qv:.1f}°  HWP_opt {theta:.1f}°  ({parity})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"Bar-split plot saved: {out}")
    else:
        plt.show()


def plot_run_set_bars(summaries):
    """For each RID, the bar-splitting plot at its optimal HWP."""
    for s in summaries:
        plot_bar_split(s["_meta"], s["best_qwp"], s["_entry"])


def plot_waveplate_vs_phase(summaries, out=None):
    """How the optimal waveplate settings change with the basis angle:
    best QWP and HWP_opt vs basis_phase_deg."""
    import matplotlib.pyplot as plt

    if not summaries:
        print("Nothing to plot: empty run set.")
        return
    ph = [s["basis_phase_deg"] for s in summaries]
    qwp = [s["best_qwp"] for s in summaries]
    hwp = [s["hwp_opt"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ph, qwp, "o-", color="C4", lw=1.8)
    ax1.set_xlabel("basis_phase_deg  (analysis angle)")
    ax1.set_ylabel("Best QWP angle (deg)")
    ax1.set_title("Best QWP vs basis angle")
    ax1.grid(alpha=0.3)

    ax2.plot(ph, hwp, "s-", color="C2", lw=1.8)
    ax2.set_xlabel("basis_phase_deg  (analysis angle)")
    ax2.set_ylabel("HWP_opt angle (deg)")
    ax2.set_title("HWP for max entanglement vs basis angle")
    ax2.grid(alpha=0.3)

    for ax, ys in ((ax1, qwp), (ax2, hwp)):
        for xp, yp, s in zip(ph, ys, summaries):
            ax.annotate(str(s["rid"]), (xp, yp), textcoords="offset points",
                        xytext=(0, 6), fontsize=7, ha="center")

    fig.suptitle("Optimal waveplate settings vs basis angle", fontsize=13)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"Waveplate-vs-phase plot saved: {out}")
    else:
        plt.show()


def plot_run_set(summaries, out=None):
    """Plot parity visibility vs basis angle for the chosen RID set."""
    import matplotlib.pyplot as plt
    if not summaries:
        print("Nothing to plot: empty run set.")
        return
    ph = [s["basis_phase_deg"] for s in summaries]
    vp = [s["max_vis"] for s in summaries]
    vpe = np.clip([s["max_vis_err"] for s in summaries], 0, None)
    vm = [s["vis_minus"] for s in summaries]
    vme = np.clip([s["vis_minus_err"] for s in summaries], 0, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ph, vp, yerr=vpe, fmt="o-", color="C0", capsize=3,
                label=r"max vis (best QWP)")
    ax.errorbar(ph, vm, yerr=vme, fmt="s--", color="C3", capsize=3,
                label=r"vis ($\mathcal{P}_-$)")
    for s in summaries:
        ax.annotate(str(s["rid"]), (s["basis_phase_deg"], s["max_vis"]),
                    textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set_xlabel("basis_phase_deg  (analysis angle)")
    ax.set_ylabel("Parity visibility V")
    ax.set_ylim(0, 1.1)
    ax.set_title("Parity visibility vs basis angle  |  chosen RID set")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"Run-set plot saved: {out}")
    else:
        plt.show()


# =========================================================================
#  CLI
# =========================================================================

def _resolve_path(arg):
    if arg is None:
        files = find_runs()
        if not files:
            sys.exit(f"No ParityAngleExperiment result files under {RESULTS_DIR}.")
        print(f"(Using latest ParityAngleExperiment run: {Path(files[-1]).name})")
        return files[-1]
    p = Path(arg)
    if p.exists():
        return str(p)
    try:
        rid = int(arg)
    except ValueError:
        sys.exit(f"Not a file or run ID: {arg}")
    files = [f for f in find_runs() if _rid_of(f) == rid]
    if not files:
        sys.exit(f"No ParityAngleExperiment run with RID {rid}.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None,
                    help="ParityAngleExperiment .h5 path, RID, or omit for latest")
    ap.add_argument("--thres", type=int, default=None,
                    help="atom-present threshold on counts_signal (default: file value)")
    ap.add_argument("--qwp", type=float, default=None,
                    help="restrict to one QWP value (deg)")
    ap.add_argument("--no-plot", action="store_true", help="table only")
    ap.add_argument("--save-plot", type=str, default=None, metavar="PATH",
                    help="save the visibility summary plot to PATH")
    args = ap.parse_args()

    path = _resolve_path(args.run)
    meta, analyzed, df = analyze_run(path, thres=args.thres, qwp_only=args.qwp)
    print_summary(meta, analyzed)

    if not args.no_plot and analyzed:
        for qv in sorted(analyzed):
            plot_qwp_slice(meta, qv, analyzed[qv])
        plot_visibility_summary(meta, analyzed, out=args.save_plot)


if __name__ == "__main__":
    main()
