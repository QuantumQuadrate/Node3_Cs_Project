"""Analyze ParametricHeating runs: atom survival vs trap-modulation frequency.

Reads a `*-ParametricHeating.h5` (from experiments/Parametric_heating.py, one row
per shot) and extracts the trap frequencies. Modulating the FORT depth at f_mod
parametrically heats the atom and loss is resonant at

        f_mod = 2 f_trap      ->      f_trap = f_dip / 2

so each dip in the survival curve marks a mode.

  A  survival vs f_mod, Wilson error bars, the multi-dip fit, 0 Hz baseline
  B  shot-1 / shot-2 count histograms vs the threshold (is the atom call clean?)
  C  loading and survival vs scan repeat (did anything drift during the run?)

=============================================================================
TWO THINGS THIS MODULE DELIBERATELY DOES NOT TAKE ON TRUST
=============================================================================
1. THE ATOM THRESHOLD. The experiment decided `loaded`/`survived` on the core
   device using whatever `atom_threshold` was live, and stored the booleans --
   but it also stored the raw counts, so the decision can be REMADE here. Pass
   `threshold=` to any entry point and loading/survival are recomputed from
   (shot - background); the default is the run's archived atom_threshold, and
   `threshold="stored"` uses the booleans the kernel wrote. summarize() reports
   how many shots the choice flips, so threshold sensitivity is visible rather
   than assumed. Check panel B: the threshold belongs in the valley between the
   no-atom and one-atom peaks.

2. HOW MANY DIPS THERE ARE. A single Gaussian is the wrong model for a real
   trap: an anisotropic FORT has two radial frequencies, and intensity
   modulation of an anisotropic trap drives 2f_x, 2f_y AND the sum f_x+f_y, so
   three dips is a physically expected pattern, not an artefact. fit_dips()
   therefore fits a SUM of Gaussian dips,

        S(f) = S0 - sum_i A_i exp(-(f - fc_i)^2 / 2 sigma_i^2)

   adding one dip at a time at the largest remaining residual and refining each
   (fc_i, sigma_i) by coordinate descent, with the linear parameters
   (S0, A_1..A_N) solved exactly at every step. `n_dips=None` keeps adding while
   each addition improves chi2 by more than min_dchi2 (~3 sigma); pass n_dips=3
   to force what your eyes see.

   The linear solve is Gaussian elimination with partial pivoting written out
   longhand -- no np.linalg, no scipy. That is deliberate: the BLAS in
   artiq7_claude segfaults, and a hand-rolled solve also cannot silently fail to
   converge the way curve_fit can.

GROUPING IS ON ph_f_mod_actual, the frequency the AD9910 really played -- never
ph_f_mod_set. The RAM step is an integer, so realised frequencies are quantised
and unevenly spaced.

CAVEATS
  * Depth modulation drives 2 f_trap. Weaker features can appear at f_trap and
    at 2 f_trap / n, so not every dip is a distinct mode.
  * A SATURATED dip (survival floored near 0) has a flat bottom, so its centre
    is poorly defined and the delta-chi2 error badly understates the truth. Each
    dip is flagged, and saturated ones are reported to 1 decimal with a warning.
  * mod_depth is a fraction of the DDS amplitude, not of the trap depth (the AOM
    is nonlinear). Dip POSITIONS are unaffected; a heating RATE needs a
    photodiode calibration.

Every run is written twice (the master's copy under results/, write_results'
copy under repository/results/); only the master's has `expid`, so that one is
preferred and duplicate RIDs dropped.

Plot in Python 3.11, NOT artiq7_claude -- matplotlib segfaults there. Run:
    python analyze_parametric_heating.py            # latest run
    python analyze_parametric_heating.py 2678       # one RID
    python analyze_parametric_heating.py 2678 10    # ... with threshold 10
    python analyze_parametric_heating.py 2678 10 3  # ... and force 3 dips
"""

import os
import sys
import glob
import json
from pathlib import Path

import numpy as np
import h5py

_BASE = Path(r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes")
RESULTS_DIRS = [_BASE / "results", _BASE / "repository" / "results"]

PATTERN = "*-ParametricHeating.h5"

_ROW_KEYS = ("ph_f_mod_set", "ph_f_mod_actual", "ph_ram_step", "ph_n_periods",
             "ph_t_mod_set", "ph_t_mod_actual", "ph_mod_depth", "ph_center_amp",
             "ph_bg1", "ph_bg2",
             "ph_shot1_spcm1", "ph_shot1_spcm2",
             "ph_shot2_spcm1", "ph_shot2_spcm2",
             "ph_loaded", "ph_survived", "ph_loop_index")


# --------------------------------------------------------------------------
#  IO
# --------------------------------------------------------------------------
def _rid_of(path):
    return int(os.path.basename(path).split("-")[0])


def _has_expid(path):
    try:
        with h5py.File(path, "r", locking=False) as hf:
            return "expid" in hf
    except Exception:
        return False


def _all_files():
    """One file per RID, preferring the copy that carries `expid`."""
    files = []
    for root in RESULTS_DIRS:
        files += glob.glob(str(root / "**" / PATTERN), recursive=True)
    by_rid = {}
    for f in set(files):
        by_rid.setdefault(_rid_of(f), []).append(f)
    out = []
    for rid in sorted(by_rid):
        c = by_rid[rid]
        out.append(c[0] if len(c) == 1
                   else next((x for x in c if _has_expid(x)), c[0]))
    return out


def find_run(rid=None):
    """Newest run file, or the one for `rid`."""
    files = _all_files()
    if not files:
        raise FileNotFoundError(
            "No " + PATTERN + " under "
            + " or ".join(str(r) for r in RESULTS_DIRS))
    if rid is None:
        return files[-1]
    for f in files:
        if _rid_of(f) == int(rid):
            return f
    raise FileNotFoundError("No ParametricHeating .h5 for RID {}".format(rid))


def find_runs():
    """Every run file on disk (one per RID), oldest first."""
    return _all_files()


def load_run(path, threshold=None):
    """Return (data, meta, rid).

    `threshold` (counts, on shot-minus-background):
        None       -> the run's archived atom_threshold (default)
        a number   -> recompute loading/survival with that cut
        "stored"   -> use the booleans the kernel wrote on the core device

    data gains `s1`, `s2` (background-subtracted shot counts), `loaded` and
    `survived` (bool). meta gains threshold_used / threshold_source and
    n_flipped (how many shots differ from the kernel's own decision).
    """
    data, meta = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in _ROW_KEYS:
            data[k] = (np.atleast_1d(ds[k][()]).astype(float)
                       if k in ds else np.array([]))
        meta["atom_threshold_archived"] = (
            float(hf["archive/atom_threshold"][()])
            if "archive/atom_threshold" in hf else float("nan"))
        meta["nominal_trapdepth"] = (float(hf["archive/nominal_trapdepth"][()])
                                     if "archive/nominal_trapdepth" in hf
                                     else float("nan"))
        meta["args"] = {}
        if "expid" in hf:
            raw = hf["expid"][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            try:
                meta["args"] = json.loads(raw).get("arguments", {})
            except Exception:
                pass

    n = min(a.size for a in data.values())
    for k in data:
        data[k] = data[k][:n]
    meta["n_shots"] = n

    # Background-subtracted shot counts -- the quantity the threshold acts on.
    bg = data["ph_bg1"] + data["ph_bg2"]
    data["s1"] = data["ph_shot1_spcm1"] + data["ph_shot1_spcm2"] - bg
    data["s2"] = data["ph_shot2_spcm1"] + data["ph_shot2_spcm2"] - bg

    stored_loaded = data["ph_loaded"] > 0.5
    stored_surv = data["ph_survived"] > 0.5

    if isinstance(threshold, str) and threshold == "stored":
        data["loaded"], data["survived"] = stored_loaded, stored_surv
        meta["threshold_used"] = meta["atom_threshold_archived"]
        meta["threshold_source"] = "stored (kernel's own decision)"
    else:
        thr = (meta["atom_threshold_archived"] if threshold is None
               else float(threshold))
        if thr != thr:                       # no archive value -> fall back
            data["loaded"], data["survived"] = stored_loaded, stored_surv
            meta["threshold_used"] = float("nan")
            meta["threshold_source"] = "stored (no archived threshold)"
        else:
            data["loaded"] = data["s1"] > thr
            data["survived"] = data["s2"] > thr
            meta["threshold_used"] = thr
            meta["threshold_source"] = ("archived" if threshold is None
                                        else "override")
    meta["n_flipped"] = int(np.count_nonzero(data["loaded"] != stored_loaded)
                            + np.count_nonzero(data["survived"] != stored_surv))

    for key, src in (("mod_depth", "ph_mod_depth"),
                     ("center_amp", "ph_center_amp"),
                     ("t_mod_set", "ph_t_mod_set")):
        meta[key] = float(data[src][0]) if n else float("nan")
    return data, meta, _rid_of(path)


# --------------------------------------------------------------------------
#  STATS  (sums + hand-rolled linear algebra -- no BLAS, no scipy)
# --------------------------------------------------------------------------
def wilson(k, n, z=1.0):
    """Wilson score interval (z=1 ~ 68%). Returns (p, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def survival_curve(data):
    """Group shots by the PLAYED frequency. Returns a dict of 1-D arrays.

    survival = survived / loaded; shots that never loaded are EXCLUDED rather
    than counted as loss, or the loading probability would dilute the dips.
    `err` is the mean Wilson half-width, floored so p = 0 or 1 still carries a
    usable fit weight.
    """
    fa = data["ph_f_mod_actual"]
    loaded, surv = data["loaded"], data["survived"]

    f, nl, ns, p, lo, hi = [], [], [], [], [], []
    for fv in np.unique(fa):
        m = fa == fv
        k_l = int(np.count_nonzero(m & loaded))
        k_s = int(np.count_nonzero(m & loaded & surv))
        pv, l, h = wilson(k_s, k_l)
        f.append(fv); nl.append(k_l); ns.append(k_s)
        p.append(pv); lo.append(l); hi.append(h)

    out = {k: np.asarray(v, dtype=float) for k, v in
           (("f", f), ("n_loaded", nl), ("n_survived", ns),
            ("p", p), ("lo", lo), ("hi", hi))}
    err = 0.5 * (out["hi"] - out["lo"])
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.where(out["n_loaded"] > 0, 0.5 / out["n_loaded"], 1.0)
    out["err"] = np.maximum(err, floor)
    return out


def _gauss_solve(m, v):
    """Solve m x = v by Gaussian elimination with partial pivoting.

    Longhand on purpose: np.linalg / scipy are unusable in artiq7_claude (broken
    BLAS segfaults), and these systems are tiny (<= 6x6). Returns None if
    singular.
    """
    n = len(v)
    a = [list(row) + [v[i]] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-30:
            return None
        a[col], a[piv] = a[piv], a[col]
        inv = 1.0 / a[col][col]
        for r in range(n):
            if r == col:
                continue
            fac = a[r][col] * inv
            if fac == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= fac * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def _fit_amplitudes(f, s, w, centers, sigmas, s0):
    """Exact weighted solve for the dip amplitudes at FIXED centres/widths/S0.

    Model S = S0 - sum A_i g_i, linear in the A_i. Returns ([A_i], chi2, model).

    The baseline S0 is an INPUT, not a fitted parameter. Letting it float is
    what wrecked the first version of this fit: with S0 free and wide Gaussians
    allowed, a flat plateau can be re-expressed as a large S0 minus two broad
    dips parked at the ends of the scan, which is exactly what happened
    (S0 = -0.42 with 85 kHz-wide "dips" at 21 and 141 kHz). Pinning S0 to the
    measured no-heating plateau removes that degeneracy entirely, and it is the
    physically meaningful quantity anyway -- the 0 Hz control measures it.
    """
    if not centers:
        model = np.full_like(f, s0)
        return [], float(np.sum(w * (s - model) ** 2)), model
    cols = [np.exp(-0.5 * ((f - c) / sg) ** 2)
            for c, sg in zip(centers, sigmas)]
    y = s0 - s                                  # the loss the dips must explain
    k = len(cols)
    mm = [[float(np.sum(w * cols[i] * cols[j])) for j in range(k)]
          for i in range(k)]
    vv = [float(np.sum(w * cols[i] * y)) for i in range(k)]
    x = _gauss_solve(mm, vv)
    if x is None:
        return None, float("inf"), None
    model = np.full_like(f, s0)
    for i in range(k):
        model = model - x[i] * cols[i]
    return x, float(np.sum(w * (s - model) ** 2)), model


def _estimate_baseline(s, w):
    """No-heating survival plateau: robust high level of the curve.

    Median of the upper half -- dips only ever pull points DOWN, so the upper
    half is dip-free by construction as long as less than half the scan is on
    resonance.
    """
    if s.size == 0:
        return 1.0
    upper = s[s >= np.median(s)]
    s0 = float(np.median(upper)) if upper.size else float(np.median(s))
    return min(1.0, max(0.05, s0))


def fit_dips(curve, n_dips=None, max_dips=4, min_dchi2=25.0, baseline=None,
             n_fc_coarse=40, n_fc_fine=15, n_sig=24, n_rounds=2):
    """Fit a sum of Gaussian dips to the survival curve.

    Dips are added greedily at the largest remaining weighted residual, then
    every (fc_i, sigma_i) is refined by coordinate descent with the amplitudes
    solved exactly at each node. With n_dips=None, addition stops once a new dip
    no longer improves chi2 by min_dchi2 (25 ~ 5 sigma; deliberately strict,
    because a greedy fitter will always find *something*). Pass n_dips=3 to
    force what you can see.

    `baseline` is the no-heating survival plateau: None estimates it from the
    curve (and refines it once from the off-resonant points), or pass a number.
    It is NOT fitted -- see _fit_amplitudes for why that matters.

    sigma is capped at span/8 so a dip cannot impersonate baseline curvature.

    Returns a dict with `dips` (list, deepest first) or None if too few points.
    """
    m = curve["f"] > 0
    f, s, e = curve["f"][m], curve["p"][m], curve["err"][m]
    good = np.isfinite(s) & np.isfinite(e) & (e > 0)
    f, s, e = f[good], s[good], e[good]
    if f.size < 8:
        return None
    w = 1.0 / (e * e)
    span = f.max() - f.min()
    sig_lo = max(span / 300.0, 1.0)
    sig_hi = span / 8.0
    sig_grid = np.exp(np.linspace(np.log(sig_lo), np.log(sig_hi), n_sig))

    s0 = _estimate_baseline(s, w) if baseline is None else float(baseline)
    baseline_fixed = baseline is not None

    def _scan(i, cand_c, cand_s, fcs, sgs):
        """Best (chi2, fc, sigma) for dip i over the given grids."""
        loc = (float("inf"), cand_c[i], cand_s[i])
        for fc in fcs:
            for sg in sgs:
                tc = list(cand_c); tc[i] = float(fc)
                ts = list(cand_s); ts[i] = float(sg)
                aa, cc, _ = _fit_amplitudes(f, s, w, tc, ts, s0)
                # Reject non-dips (A <= 0 is a bump) and dips deeper than the
                # curve can physically go (survival cannot be negative).
                if aa is None or min(aa) <= 0 or max(aa) > s0 + 0.15:
                    continue
                if cc < loc[0]:
                    loc = (cc, float(fc), float(sg))
        return loc

    def _refine(cand_c, cand_s):
        """Coordinate descent on every (fc, sigma). Returns (chi2, c, s).

        COARSE-TO-FINE: a full fine grid at every coordinate step costs
        n_fc x n_sig solves per dip per round (~7 s for 2 dips over 70 points,
        which makes the threshold-sensitivity sweep painful). Scanning coarse
        first and then refining only around that optimum gives the same answer
        for ~5x less work.
        """
        sig_coarse = sig_grid[::2]
        best = (float("inf"), list(cand_c), list(cand_s))
        for _ in range(n_rounds):
            for i in range(len(cand_c)):
                lo = max(f.min(), cand_c[i] - span / 4.0)
                hi = min(f.max(), cand_c[i] + span / 4.0)
                fcs = np.linspace(lo, hi, n_fc_coarse)
                loc = _scan(i, cand_c, cand_s, fcs, sig_coarse)
                # Fine pass: +-1.5 coarse steps in fc, +-2 nodes in sigma.
                dfc = (hi - lo) / max(1, n_fc_coarse - 1)
                fcs2 = np.linspace(max(lo, loc[1] - 1.5 * dfc),
                                   min(hi, loc[1] + 1.5 * dfc), n_fc_fine)
                k = int(np.argmin(np.abs(sig_grid - loc[2])))
                sgs2 = sig_grid[max(0, k - 2):k + 3]
                loc2 = _scan(i, cand_c, cand_s, fcs2, sgs2)
                if loc2[0] < loc[0]:
                    loc = loc2
                cand_c[i], cand_s[i] = loc[1], loc[2]
                if loc[0] < best[0]:
                    best = (loc[0], list(cand_c), list(cand_s))
        return best

    centers, sigmas = [], []
    amps, chi2, model = _fit_amplitudes(f, s, w, centers, sigmas, s0)
    target = max_dips if n_dips is None else int(n_dips)

    for _pass in range(2):            # pass 2 re-runs with a refined baseline
        centers, sigmas = [], []
        amps, chi2, model = _fit_amplitudes(f, s, w, centers, sigmas, s0)
        while len(centers) < target:
            resid = (model - s) * np.sqrt(w)
            for c, sg in zip(centers, sigmas):
                resid = np.where(np.abs(f - c) < sg, -np.inf, resid)
            if not np.any(np.isfinite(resid)) or np.nanmax(resid) <= 0:
                break
            c_new = float(f[int(np.nanargmax(resid))])
            cc, nc, ns_ = _refine(centers + [c_new],
                                  sigmas + [float(np.median(sig_grid))])
            if nc is None or len(nc) == len(centers):
                break
            if n_dips is None and (chi2 - cc) < min_dchi2:
                break                  # not worth another dip
            centers, sigmas = nc, ns_
            amps, chi2, model = _fit_amplitudes(f, s, w, centers, sigmas, s0)

        if baseline_fixed or not centers:
            break
        # Refine the baseline from the points the dips do not touch, then redo.
        dipsum = s0 - model
        quiet = dipsum < 0.05
        if np.count_nonzero(quiet) < 3:
            break
        s0_new = float(np.sum(w[quiet] * s[quiet]) / np.sum(w[quiet]))
        s0_new = min(1.0, max(0.05, s0_new))
        if abs(s0_new - s0) < 0.005:
            break
        s0 = s0_new

    if not centers:
        return None

    # Per-dip delta-chi2 = 1 error on the centre (others held fixed).
    dips = []
    for i, (c, sg, a) in enumerate(zip(centers, sigmas, amps)):
        lo = max(f.min(), c - 3 * sg)
        hi = min(f.max(), c + 3 * sg)
        grid = np.linspace(lo, hi, 160)
        prof = []
        for fc in grid:
            tc = list(centers); tc[i] = float(fc)
            _, cc, _ = _fit_amplitudes(f, s, w, tc, sigmas, s0)
            prof.append(cc)
        prof = np.asarray(prof)
        inside = np.where(prof <= chi2 + 1.0)[0]
        c_err = (0.5 * (grid[inside[-1]] - grid[inside[0]])
                 if inside.size > 1 else float(grid[1] - grid[0]))
        near = np.abs(f - c) < 1.1775 * sg          # within the FWHM
        n_sat = int(np.count_nonzero(s[near] < 0.10))
        dips.append({"fc": float(c), "fc_err": float(c_err),
                     "sigma": float(sg), "fwhm": float(2.3548 * sg),
                     "amplitude": float(a),
                     "f_trap": float(c / 2.0),
                     "f_trap_err": float(c_err / 2.0),
                     "n_saturated": n_sat, "saturated": bool(n_sat >= 2),
                     "significant": bool(a > 2.0 * float(np.mean(e)))})
    dips.sort(key=lambda d: -d["amplitude"])

    dof = max(1, f.size - (1 + 3 * len(dips)))
    return {"dips": dips, "baseline": float(s0), "chi2": float(chi2),
            "dof": int(dof), "chi2_red": float(chi2 / dof),
            "n_points": int(f.size), "n_dips": len(dips),
            "centers": [d["fc"] for d in dips]}


def sum_resonance_check(fit, tol=0.08):
    """With 3 dips, test the anisotropic-trap pattern 2f_x, f_x+f_y, 2f_y.

    For intensity modulation of an anisotropic trap the sum resonance sits at
    f_x + f_y, i.e. exactly midway between 2f_x and 2f_y. Returns a dict with
    the predicted middle dip and the fractional mismatch, or None.
    """
    if fit is None or fit["n_dips"] != 3:
        return None
    c = sorted(fit["centers"])
    mid_pred = 0.5 * (c[0] + c[2])
    rel = abs(c[1] - mid_pred) / mid_pred if mid_pred else float("nan")
    return {"f_low": c[0], "f_mid": c[1], "f_high": c[2],
            "mid_predicted": mid_pred, "rel_mismatch": rel,
            "consistent": bool(rel < tol),
            "f_x": c[0] / 2.0, "f_y": c[2] / 2.0}


def summarize(path, threshold=None, n_dips=None, verbose=True):
    """Survival curve + multi-dip fit for one run. Returns a dict."""
    data, meta, rid = load_run(path, threshold=threshold)
    if meta["n_shots"] == 0:
        if verbose:
            print("RID {}: no shots in {}".format(rid, path))
        return None

    curve = survival_curve(data)
    fit = fit_dips(curve, n_dips=n_dips)
    sumres = sum_resonance_check(fit)

    loaded, surv = data["loaded"], data["survived"]
    zero = curve["f"] == 0
    base_p = base_lo = base_hi = float("nan")
    if np.any(zero):
        i = int(np.where(zero)[0][0])
        base_p, base_lo, base_hi = curve["p"][i], curve["lo"][i], curve["hi"][i]

    res = {"rid": rid, "path": path, "curve": curve, "fit": fit,
           "sum_resonance": sumres,
           "n_shots": int(meta["n_shots"]),
           "n_loaded": int(np.count_nonzero(loaded)),
           "n_survived": int(np.count_nonzero(loaded & surv)),
           "load_fraction": float(np.mean(loaded)),
           "baseline_0hz": base_p, "baseline_lo": base_lo,
           "baseline_hi": base_hi,
           "mod_depth": meta["mod_depth"], "center_amp": meta["center_amp"],
           "t_mod": meta["t_mod_set"],
           "threshold": meta["threshold_used"],
           "threshold_source": meta["threshold_source"],
           "threshold_archived": meta["atom_threshold_archived"],
           "n_flipped": meta["n_flipped"],
           "n_freqs": int(curve["f"].size), "meta": meta}

    if verbose:
        print("--- ParametricHeating  RID {} ---".format(rid))
        print("file            : {}".format(path))
        print("threshold       : {:.0f} counts  ({}; archived {:.0f})".format(
            res["threshold"], res["threshold_source"],
            res["threshold_archived"]))
        if res["n_flipped"]:
            print("                  {} of {} shot decisions differ from the "
                  "kernel's own call".format(res["n_flipped"],
                                             2 * res["n_shots"]))
        print("shots           : {}  (loaded {} = {:.0f}%, survived {})".format(
            res["n_shots"], res["n_loaded"], 100 * res["load_fraction"],
            res["n_survived"]))
        print("frequencies     : {}  ({:.1f} .. {:.1f} kHz)".format(
            res["n_freqs"], curve["f"].min() / 1e3, curve["f"].max() / 1e3))
        print("modulation      : depth {:.3f} of DDS amplitude, t_mod "
              "{:.3f} ms".format(res["mod_depth"], 1e3 * res["t_mod"]))
        if np.isfinite(base_p):
            print("0 Hz control    : survival {:.3f} [{:.3f}, {:.3f}]".format(
                base_p, base_lo, base_hi))
        print()
        if fit is None:
            print("DIP FIT: too few frequency points to fit.")
            return res

        print("DIP FIT  S(f) = S0 - sum_i A_i exp(-(f-fc_i)^2/2 sigma_i^2)")
        print("   baseline S0 : {:.3f}   chi2/dof : {:.2f}  ({} points, "
              "{} dips)".format(fit["baseline"], fit["chi2_red"],
                                fit["n_points"], fit["n_dips"]))
        print()
        print("   {:>3}  {:>12}  {:>9}  {:>7}  {:>8}   {:>16}".format(
            "#", "centre [kHz]", "FWHM", "depth", "sat?", "f_trap = fc/2"))
        for i, d in enumerate(fit["dips"], 1):
            if d["saturated"]:
                ft = "~{:.1f} kHz  (sat)".format(d["f_trap"] / 1e3)
            else:
                ft = "{:.2f} +- {:.2f} kHz".format(d["f_trap"] / 1e3,
                                                   d["f_trap_err"] / 1e3)
            print("   {:>3}  {:>12}  {:>7.2f}  {:>7.3f}  {:>8}   {:>16}".format(
                i, "{:.2f}+-{:.2f}".format(d["fc"] / 1e3, d["fc_err"] / 1e3),
                d["fwhm"] / 1e3, d["amplitude"],
                "{}".format(d["n_saturated"]) if d["saturated"] else "-", ft))
        if any(d["saturated"] for d in fit["dips"]):
            print()
            print("   !! A saturated dip has a FLAT bottom: its centre is not "
                  "well defined and the delta-chi2 error understates the truth."
                  " Re-measure with weaker drive (lower mod_depth or shorter "
                  "t_mod) so the minimum lands near 0.3-0.5.")
        if sumres is not None:
            print()
            print("   THREE DIPS -- anisotropic-trap test:")
            print("      2f_x = {:.2f}, sum = {:.2f}, 2f_y = {:.2f} kHz".format(
                sumres["f_low"] / 1e3, sumres["f_mid"] / 1e3,
                sumres["f_high"] / 1e3))
            print("      f_x + f_y would put the middle dip at {:.2f} kHz "
                  "(measured {:.2f}, off by {:.1f}%)".format(
                      sumres["mid_predicted"] / 1e3, sumres["f_mid"] / 1e3,
                      100 * sumres["rel_mismatch"]))
            if sumres["consistent"]:
                print("      CONSISTENT with 2f_x / f_x+f_y / 2f_y for a trap "
                      "with TWO radial frequencies:")
                print("         f_x = {:.2f} kHz,  f_y = {:.2f} kHz".format(
                    sumres["f_x"] / 1e3, sumres["f_y"] / 1e3))
            else:
                print("      NOT the sum pattern -- the three dips are more "
                      "likely independent modes and/or subharmonics.")
    return res


# --------------------------------------------------------------------------
#  FIGURE
# --------------------------------------------------------------------------
def analyze(path, threshold=None, n_dips=None, out_png=None):
    """Three-panel figure. Uses the active backend; returns the Figure."""
    import matplotlib
    import matplotlib.pyplot as plt

    res = summarize(path, threshold=threshold, n_dips=n_dips)
    if res is None:
        return None
    data, meta, rid = load_run(path, threshold=threshold)
    curve, fit = res["curve"], res["fit"]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.8),
                                        constrained_layout=True)

    # --- A: the measurement + fit ---
    m = curve["f"] > 0
    # Clamp the bar lengths: the Wilson interval is centred on (p+z^2/2n)/d and
    # clipped to [0,1], so at p exactly 0 or 1 the subtraction lands on -0.0,
    # which matplotlib rejects outright. These are zero-length, not negative.
    err_lo = np.maximum(0.0, curve["p"][m] - curve["lo"][m])
    err_hi = np.maximum(0.0, curve["hi"][m] - curve["p"][m])
    axA.errorbar(curve["f"][m] / 1e3, curve["p"][m], yerr=[err_lo, err_hi],
                 fmt="o", ms=4, lw=1, capsize=2, color="C0", label="survival")
    if np.isfinite(res["baseline_0hz"]):
        axA.axhline(res["baseline_0hz"], color="C7", ls="--", lw=1.2,
                    label="0 Hz control = {:.3f}".format(res["baseline_0hz"]))
    title = "A. Survival vs modulation frequency"
    if fit is not None:
        ff = np.linspace(curve["f"][m].min(), curve["f"][m].max(), 1200)
        model = np.full_like(ff, fit["baseline"])
        for d in fit["dips"]:
            model = model - d["amplitude"] * np.exp(
                -0.5 * ((ff - d["fc"]) / d["sigma"]) ** 2)
        axA.plot(ff / 1e3, model, "-", color="C3", lw=1.6,
                 label="{} -dip fit".format(fit["n_dips"]))
        for i, d in enumerate(fit["dips"], 1):
            axA.axvline(d["fc"] / 1e3, color="C3", ls=":", lw=1.0)
            axA.annotate("{}".format(i), (d["fc"] / 1e3, 1.01),
                         ha="center", fontsize=9, color="C3")
        title += "  |  {} dips, f_trap = {}".format(
            fit["n_dips"],
            ", ".join("{:.1f}".format(d["f_trap"] / 1e3)
                      for d in sorted(fit["dips"], key=lambda x: x["fc"])))
        title += " kHz"
    axA.set_xlabel("modulation frequency played [kHz]")
    axA.set_ylabel("survival  (survived / loaded)")
    axA.set_ylim(0, 1.08)
    axA.set_title(title, fontsize=10)
    axA.legend(fontsize=9)
    axA.grid(alpha=0.3)

    # --- B: is the atom call clean? ---
    s1, s2 = data["s1"], data["s2"]
    lo = min(s1.min(), s2.min())
    hi = max(s1.max(), s2.max())
    step = max(1, int(np.ceil((hi - lo + 1) / 60)))
    bins = np.arange(lo, hi + step + 1, step) - 0.5
    axB.hist(s1, bins=bins, color="C0", alpha=0.6, label="shot 1 (after load)")
    axB.hist(s2, bins=bins, color="C2", alpha=0.6, label="shot 2 (after mod)")
    if np.isfinite(res["threshold"]):
        axB.axvline(res["threshold"], color="C3", ls="--", lw=1.4,
                    label="threshold = {:.0f} ({})".format(
                        res["threshold"], res["threshold_source"]))
    if (np.isfinite(res["threshold_archived"])
            and res["threshold_archived"] != res["threshold"]):
        axB.axvline(res["threshold_archived"], color="C7", ls=":", lw=1.2,
                    label="archived = {:.0f}".format(res["threshold_archived"]))
    axB.set_xlabel("SPCM1+2 counts - background")
    axB.set_ylabel("shots")
    axB.set_title("B. Atom detection  ({:.0f}% loaded)".format(
        100 * res["load_fraction"]))
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3)

    # --- C: drift across the run ---
    loop = data["ph_loop_index"]
    loaded, surv = data["loaded"], data["survived"]
    loops = np.unique(loop)
    lf, sf = [], []
    for lv in loops:
        mm = loop == lv
        lf.append(np.mean(loaded[mm]) if np.any(mm) else np.nan)
        nl = np.count_nonzero(mm & loaded)
        sf.append(np.count_nonzero(mm & loaded & surv) / nl if nl else np.nan)
    axC.plot(loops, lf, "-o", ms=4, color="C1", label="loading fraction")
    axC.plot(loops, sf, "-o", ms=4, color="C0", label="survival (all freqs)")
    axC.set_xlabel("scan repeat (ph_loop_index)")
    axC.set_ylabel("fraction")
    axC.set_ylim(0, 1.05)
    axC.set_title("C. Stability across the run")
    axC.legend(fontsize=9)
    axC.grid(alpha=0.3)

    fig.suptitle(
        "ParametricHeating RID {}  |  {} shots, {} freqs  |  depth {:.3f}, "
        "t_mod {:.2f} ms  |  threshold {:.0f}".format(
            rid, res["n_shots"], res["n_freqs"], res["mod_depth"],
            1e3 * res["t_mod"], res["threshold"]),
        fontsize=13)

    if out_png is None:
        out_png = str(Path(path).with_name(
            "parametric_heating_RID{}.png".format(rid)))
    fig.savefig(out_png, dpi=130)
    print("Saved figure: {}".format(out_png))
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in ("inline", "ipympl", "nbagg", "widget")):
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    argv = sys.argv[1:]
    rid = int(argv[0]) if len(argv) > 0 else None
    thr = float(argv[1]) if len(argv) > 1 else None
    nd = int(argv[2]) if len(argv) > 2 else None
    analyze(find_run(rid), threshold=thr, n_dips=nd)
