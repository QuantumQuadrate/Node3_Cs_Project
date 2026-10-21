"""Analyze OpticalPumpResoScan runs: the D1 optical-pumping resonance.

Reads a `*-OpticalPumpResoScan.h5` (from experiments/optical_pump_resonance_scan.py,
one row per shot) and finds the pumping resonance as a DIP in atom retention.

With the repumper off, a resonant D1 pump drives atoms out of F=3; whatever
decays into F=4 stays there, and the F=4 blowaway ejects it. So

    retention LOW  -> pump was resonant (atoms reached F=4 and were blown away)
    retention HIGH -> pump was off resonance (atoms stayed in F=3)

and the dip centre is the value to put in `f_pump_D1`.

  A  retention vs pump frequency, Wilson error bars, Gaussian-dip fit
  B  shot-1 / shot-2 count histograms vs the threshold (is the atom call clean?)
  C  loading and retention vs scan repeat (did anything drift?)

=============================================================================
THE DIAGNOSTIC THAT MATTERS MOST: CONTRAST
=============================================================================
This method is brutally sensitive -- ONE scattering event that lands in F=4 is
enough to lose the atom. So if the pump is strong or long, atoms are pumped out
even far off resonance, retention goes low EVERYWHERE, and the scan has no
contrast: there is no dip to find because the whole window is the dip.

summarize() therefore reports, before any fit:
  * baseline / edge retention -- retention in the outer fifth of the scan, i.e.
    the most off-resonant points available
  * contrast = edge - minimum
and warns when the edges are already depleted. The fix is a WEAKER or SHORTER
pump (toward the calibrated t_optical_pump = 350 us and amp_pump_D1 = 0.04),
until off-resonant retention climbs back toward the no-pump survival. Only then
is the dip centre meaningful.

Mind the scale, too: the scanned AOM frequency is not the optical detuning
(double-passed, an AOM step is 2x in the light), and the Cs D1 natural linewidth
is ~4.6 MHz. A +-2 MHz AOM scan covers only about one linewidth of optical
detuning, so a saturated line can easily be broader than the entire scan.
If the curve is flat, widen the range before concluding anything.

THRESHOLD IS NOT TAKEN ON TRUST. The experiment decided loaded/retained on the
core device with whatever `atom_threshold` was live, but it stored the raw
counts, so the call is remade here. Pass `threshold=` to any entry point;
default is the run's archived value, `"stored"` uses the kernel's booleans.
summarize() reports how many decisions the choice flips. Check panel B: the
threshold belongs in the valley between the no-atom and one-atom peaks.

THE FIT pins the baseline to the measured off-resonant plateau instead of
fitting it, and caps sigma at span/4. A free baseline is degenerate -- a flat
curve can be re-expressed as a huge baseline minus a broad dip parked at the
scan edge. The amplitude is then solved exactly (one linear parameter), and
(fc, sigma) come from a grid search. No scipy, no np.linalg: the BLAS in
artiq7_claude segfaults, and a closed-form solve also cannot fail to converge.

Note the experiment itself already runs this kind of fit on the ARTIQ worker
(`opr_fit_resonance_Hz`, via microwave_analysis). This module is independent and
is the one to trust for publication numbers -- it reports contrast first and
refuses to quote a centre when the scan has none.

Every run is written twice (the master's copy under results/ and
write_results' copy under repository/results/); only the master's carries
`expid`, so that one is preferred and duplicate RIDs dropped. A run still in
progress may exist ONLY as the repository/results copy.

Plot in Python 3.11, NOT artiq7_claude -- matplotlib segfaults there. Run:
    python analyze_op_resonance.py            # latest run
    python analyze_op_resonance.py 2711       # one RID
    python analyze_op_resonance.py 2711 10    # ... with threshold 10
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

PATTERN = "*-OpticalPumpResoScan.h5"

_ROW_KEYS = ("opr_f_op", "opr_bg1", "opr_bg2",
             "opr_shot1_spcm1", "opr_shot1_spcm2",
             "opr_shot2_spcm1", "opr_shot2_spcm2",
             "opr_loaded", "opr_retained", "opr_retention", "opr_loop_index")


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
    raise FileNotFoundError("No OpticalPumpResoScan .h5 for RID {}".format(rid))


def find_runs():
    """Every run file on disk (one per RID), oldest first."""
    return _all_files()


def load_run(path, threshold=None):
    """Return (data, meta, rid).

    `threshold` (counts on shot-minus-background):
        None     -> the run's archived atom_threshold (default)
        a number -> recompute loading/retention with that cut
        "stored" -> use the booleans the kernel wrote on the core device
    """
    data, meta = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in _ROW_KEYS:
            data[k] = (np.atleast_1d(ds[k][()]).astype(float)
                       if k in ds else np.array([]))
        # Run settings the experiment broadcast as scalars.
        for name, key in (("t_pump_s", "opr_t_pump_s"),
                          ("amp_d1", "opr_amp_d1"),
                          ("amp_repump", "opr_amp_repump"),
                          ("n_loops", "opr_n_loops"),
                          ("fit_resonance_Hz", "opr_fit_resonance_Hz")):
            meta[name] = float(ds[key][()]) if key in ds else float("nan")
        meta["atom_threshold_archived"] = (
            float(hf["archive/atom_threshold"][()])
            if "archive/atom_threshold" in hf else float("nan"))
        meta["f_pump_D1_live"] = (float(hf["archive/f_pump_D1"][()])
                                  if "archive/f_pump_D1" in hf
                                  else float("nan"))
        meta["t_optical_pump_live"] = (
            float(hf["archive/t_optical_pump"][()])
            if "archive/t_optical_pump" in hf else float("nan"))
        meta["amp_pump_D1_live"] = (float(hf["archive/amp_pump_D1"][()])
                                    if "archive/amp_pump_D1" in hf
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
        meta["has_expid"] = "expid" in hf

    n = min(a.size for a in data.values())
    for k in data:
        data[k] = data[k][:n]
    meta["n_shots"] = n

    bg = data["opr_bg1"] + data["opr_bg2"]
    data["s1"] = data["opr_shot1_spcm1"] + data["opr_shot1_spcm2"] - bg
    data["s2"] = data["opr_shot2_spcm1"] + data["opr_shot2_spcm2"] - bg

    stored_loaded = data["opr_loaded"] > 0.5
    stored_ret = data["opr_retained"] > 0.5
    if isinstance(threshold, str) and threshold == "stored":
        data["loaded"], data["retained"] = stored_loaded, stored_ret
        meta["threshold_used"] = meta["atom_threshold_archived"]
        meta["threshold_source"] = "stored (kernel's own decision)"
    else:
        thr = (meta["atom_threshold_archived"] if threshold is None
               else float(threshold))
        if thr != thr:
            data["loaded"], data["retained"] = stored_loaded, stored_ret
            meta["threshold_used"] = float("nan")
            meta["threshold_source"] = "stored (no archived threshold)"
        else:
            data["loaded"] = data["s1"] > thr
            data["retained"] = data["s2"] > thr
            meta["threshold_used"] = thr
            meta["threshold_source"] = ("archived" if threshold is None
                                        else "override")
    meta["n_flipped"] = int(
        np.count_nonzero(data["loaded"] != stored_loaded)
        + np.count_nonzero(data["retained"] != stored_ret))
    return data, meta, _rid_of(path)


# --------------------------------------------------------------------------
#  STATS  (sums + closed-form solves -- no BLAS, no scipy)
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


def retention_curve(data):
    """Group shots by pump frequency. Returns a dict of 1-D arrays.

    retention = retained / loaded; shots that never loaded are EXCLUDED rather
    than counted as loss, or the loading probability would dilute the dip.
    """
    fo = data["opr_f_op"]
    loaded, ret = data["loaded"], data["retained"]
    f, nl, ns, p, lo, hi = [], [], [], [], [], []
    for fv in np.unique(fo):
        m = fo == fv
        k_l = int(np.count_nonzero(m & loaded))
        k_s = int(np.count_nonzero(m & loaded & ret))
        pv, l, h = wilson(k_s, k_l)
        f.append(fv); nl.append(k_l); ns.append(k_s)
        p.append(pv); lo.append(l); hi.append(h)
    out = {k: np.asarray(v, dtype=float) for k, v in
           (("f", f), ("n_loaded", nl), ("n_retained", ns),
            ("p", p), ("lo", lo), ("hi", hi))}
    err = 0.5 * (out["hi"] - out["lo"])
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.where(out["n_loaded"] > 0, 0.5 / out["n_loaded"], 1.0)
    out["err"] = np.maximum(err, floor)
    return out


def contrast_report(curve, edge_frac=0.2):
    """Is there any dip to fit? Returns a dict.

    `edge` is the weighted mean retention over the outermost `edge_frac` of the
    scan on each side -- the most off-resonant points available. Without
    contrast between that and the minimum there is nothing to centre a fit on.
    """
    f, p, e = curve["f"], curve["p"], curve["err"]
    good = np.isfinite(p)
    f, p, e = f[good], p[good], e[good]
    if f.size < 3:
        return None
    span = f.max() - f.min()
    k = max(1, int(round(edge_frac * f.size / 2)))
    order = np.argsort(f)
    idx = np.concatenate([order[:k], order[-k:]])
    w = 1.0 / (e[idx] ** 2)
    edge = float(np.sum(w * p[idx]) / np.sum(w))
    i_min = int(np.argmin(p))
    return {"edge": edge, "edge_n": int(np.sum(curve["n_loaded"][good][idx])),
            "min": float(p[i_min]), "f_min": float(f[i_min]),
            "contrast": float(edge - p[i_min]),
            "span": float(span),
            "flat": bool((edge - p[i_min]) < 3.0 * float(np.mean(e))),
            "depleted": bool(edge < 0.5)}


def fit_dip(curve, baseline=None, n_fc=400, n_sig=60):
    """Fit retention = S0 - A exp(-(f-fc)^2 / 2 sigma^2).

    S0 is PINNED (measured off-resonant plateau, or pass a number) and A is
    solved in closed form at each (fc, sigma) grid node -- a free baseline is
    degenerate against a broad edge dip. sigma is capped at span/4. fc's error
    is the delta-chi2 = 1 crossing of the profile.

    Returns a dict, or None if there are too few points.
    """
    f, p, e = curve["f"], curve["p"], curve["err"]
    good = np.isfinite(p) & np.isfinite(e) & (e > 0)
    f, p, e = f[good], p[good], e[good]
    if f.size < 5:
        return None
    w = 1.0 / (e * e)
    span = f.max() - f.min()

    if baseline is None:
        upper = p[p >= np.median(p)]
        s0 = float(np.median(upper)) if upper.size else float(np.median(p))
    else:
        s0 = float(baseline)
    s0 = min(1.0, max(0.02, s0))

    fc_grid = np.linspace(f.min(), f.max(), n_fc)
    sig_grid = np.exp(np.linspace(np.log(max(span / 200.0, 1.0)),
                                  np.log(span / 4.0), n_sig))

    def _amp(fc, sg):
        """Closed-form weighted amplitude for S = s0 - A g."""
        g = np.exp(-0.5 * ((f - fc) / sg) ** 2)
        den = float(np.sum(w * g * g))
        if den <= 0:
            return None, float("inf")
        a = float(np.sum(w * g * (s0 - p))) / den
        model = s0 - a * g
        return a, float(np.sum(w * (p - model) ** 2))

    best = (float("inf"), None, None, None)
    chi2_fc = np.full(fc_grid.size, np.inf)
    for i, fc in enumerate(fc_grid):
        for sg in sig_grid:
            a, chi2 = _amp(fc, sg)
            if a is None or a <= 0:
                continue
            if chi2 < chi2_fc[i]:
                chi2_fc[i] = chi2
            if chi2 < best[0]:
                best = (chi2, float(fc), float(sg), a)
    chi2, fc, sg, a = best
    if fc is None:
        return None

    fc_err = float("nan")
    below = np.where(chi2_fc <= chi2 + 1.0)[0]
    if below.size > 1:
        fc_err = 0.5 * float(fc_grid[below[-1]] - fc_grid[below[0]])
    if not (fc_err > 0):
        fc_err = float(fc_grid[1] - fc_grid[0])

    dof = max(1, f.size - 3)
    near = np.abs(f - fc) < 1.1775 * sg
    return {"fc": fc, "fc_err": fc_err, "sigma": sg, "fwhm": 2.3548 * sg,
            "amplitude": a, "baseline": s0,
            "chi2": chi2, "dof": dof, "chi2_red": chi2 / dof,
            "n_points": int(f.size),
            "n_saturated": int(np.count_nonzero(p[near] < 0.05)),
            "significant": bool(a > 2.0 * float(np.mean(e)))}


def summarize(path, threshold=None, verbose=True):
    """Retention curve + contrast report + dip fit. Returns a dict."""
    data, meta, rid = load_run(path, threshold=threshold)
    if meta["n_shots"] == 0:
        if verbose:
            print("RID {}: no shots in {}".format(rid, path))
        return None

    curve = retention_curve(data)
    con = contrast_report(curve)
    fit = fit_dip(curve)
    loaded, ret = data["loaded"], data["retained"]

    res = {"rid": rid, "path": path, "curve": curve, "fit": fit,
           "contrast": con,
           "n_shots": int(meta["n_shots"]),
           "n_loaded": int(np.count_nonzero(loaded)),
           "n_retained": int(np.count_nonzero(loaded & ret)),
           "load_fraction": float(np.mean(loaded)),
           "threshold": meta["threshold_used"],
           "threshold_source": meta["threshold_source"],
           "threshold_archived": meta["atom_threshold_archived"],
           "n_flipped": meta["n_flipped"],
           "n_freqs": int(curve["f"].size), "meta": meta}

    if verbose:
        print("--- OpticalPumpResoScan  RID {} ---".format(rid))
        print("file            : {}".format(path))
        if not meta["has_expid"]:
            print("                  (no expid -- this is the write_results "
                  "copy, i.e. a run still in progress or crashed)")
        print("threshold       : {:.0f} counts  ({}; archived {:.0f})".format(
            res["threshold"], res["threshold_source"],
            res["threshold_archived"]))
        if res["n_flipped"]:
            print("                  {} of {} decisions differ from the "
                  "kernel's call".format(res["n_flipped"], 2 * res["n_shots"]))
        print("shots           : {}  (loaded {} = {:.0f}%, retained {})".format(
            res["n_shots"], res["n_loaded"], 100 * res["load_fraction"],
            res["n_retained"]))
        print("frequencies     : {}  ({:.4f} .. {:.4f} MHz)".format(
            res["n_freqs"], curve["f"].min() / 1e6, curve["f"].max() / 1e6))
        print("pump            : {:.1f} us at amp {:.3f}, repump amp {:.3f}"
              .format(1e6 * meta["t_pump_s"], meta["amp_d1"],
                      meta["amp_repump"]))
        print("   calibrated   : t_optical_pump {:.1f} us, amp_pump_D1 {:.3f},"
              " f_pump_D1 {:.4f} MHz".format(
                  1e6 * meta["t_optical_pump_live"], meta["amp_pump_D1_live"],
                  meta["f_pump_D1_live"] / 1e6))
        print()
        if con is None:
            print("CONTRAST: too few points.")
            return res

        print("CONTRAST (check this BEFORE the fit)")
        print("   off-resonant edge retention : {:.3f}".format(con["edge"]))
        print("   minimum retention           : {:.3f} at {:.4f} MHz".format(
            con["min"], con["f_min"] / 1e6))
        print("   contrast                    : {:.3f}".format(con["contrast"]))
        if con["depleted"]:
            print("   !! EDGES ARE ALREADY DEPLETED (edge retention < 0.5).")
            print("      The pump removes atoms even far off resonance, so the")
            print("      whole window is inside the resonance and there is no")
            print("      dip to centre. ONE F=4-landing scattering event loses")
            print("      the atom, so this method saturates easily.")
            print("      FIX: weaker/shorter pump -- move t_pump toward "
                  "{:.0f} us and amp toward {:.3f} -- until the edges recover"
                  .format(1e6 * meta["t_optical_pump_live"],
                          meta["amp_pump_D1_live"]))
            print("      and/or widen the scan: {:.1f} MHz of AOM is only ~"
                  "{:.1f} linewidths of optical detuning (Cs D1 ~4.6 MHz, and"
                  " a double-passed AOM doubles the step).".format(
                      con["span"] / 1e6, 2 * con["span"] / 1e6 / 4.6))
        elif con["flat"]:
            print("   !! FLAT: contrast is within ~3x the mean error bar, so "
                  "there is no resolved dip yet. Take more shots per point, or"
                  " widen/recentre the scan.")
        print()
        if fit is None:
            print("DIP FIT: too few frequency points.")
        elif con["depleted"] or not fit["significant"]:
            print("DIP FIT: SUPPRESSED -- the curve has no usable contrast, so"
                  " a centre would be fitted to noise.")
            print("         (grid minimum sits at {:.4f} MHz, reported for "
                  "reference only)".format(fit["fc"] / 1e6))
        else:
            print("DIP FIT  retention = S0 - A exp(-(f-fc)^2/2 sigma^2)")
            print("   baseline S0 : {:.3f}   depth A : {:.3f}".format(
                fit["baseline"], fit["amplitude"]))
            print("   FWHM        : {:.4f} MHz".format(fit["fwhm"] / 1e6))
            print("   chi2/dof    : {:.2f}  ({} points)".format(
                fit["chi2_red"], fit["n_points"]))
            if fit["n_saturated"] >= 2:
                print("   !! {} points inside the FWHM are floored below 0.05 "
                      "retention: the bottom is flat, so the centre is only "
                      "approximate.".format(fit["n_saturated"]))
            print()
            print("   ==> OP RESONANCE f_pump_D1 = {:.4f} +- {:.4f} MHz"
                  .format(fit["fc"] / 1e6, fit["fc_err"] / 1e6))
            print("       (currently set to {:.4f} MHz)".format(
                meta["f_pump_D1_live"] / 1e6))
    return res


# --------------------------------------------------------------------------
#  FIGURE
# --------------------------------------------------------------------------
def analyze(path, threshold=None, out_png=None):
    """Three-panel figure. Uses the active backend; returns the Figure."""
    import matplotlib
    import matplotlib.pyplot as plt

    res = summarize(path, threshold=threshold)
    if res is None:
        return None
    data, meta, rid = load_run(path, threshold=threshold)
    curve, fit, con = res["curve"], res["fit"], res["contrast"]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.8),
                                        constrained_layout=True)

    # --- A: retention vs pump frequency ---
    # Clamp bar lengths: the Wilson interval is clipped to [0,1], so at p = 0
    # or 1 exactly the subtraction lands on -0.0, which matplotlib rejects.
    err_lo = np.maximum(0.0, curve["p"] - curve["lo"])
    err_hi = np.maximum(0.0, curve["hi"] - curve["p"])
    axA.errorbar(curve["f"] / 1e6, curve["p"], yerr=[err_lo, err_hi],
                 fmt="o", ms=4, lw=1, capsize=2, color="C0", label="retention")
    if con is not None:
        axA.axhline(con["edge"], color="C7", ls="--", lw=1.2,
                    label="edge (off-resonant) = {:.3f}".format(con["edge"]))
    if np.isfinite(meta["f_pump_D1_live"]):
        axA.axvline(meta["f_pump_D1_live"] / 1e6, color="C1", ls="-.", lw=1.2,
                    label="f_pump_D1 now = {:.3f}".format(
                        meta["f_pump_D1_live"] / 1e6))
    usable = (fit is not None and con is not None
              and not con["depleted"] and fit["significant"])
    if usable:
        ff = np.linspace(curve["f"].min(), curve["f"].max(), 600)
        model = fit["baseline"] - fit["amplitude"] * np.exp(
            -0.5 * ((ff - fit["fc"]) / fit["sigma"]) ** 2)
        axA.plot(ff / 1e6, model, "-", color="C3", lw=1.6, label="dip fit")
        axA.axvline(fit["fc"] / 1e6, color="C3", ls=":", lw=1.2)
        axA.set_title("A. Retention vs pump freq  |  resonance = "
                      "{:.4f} +- {:.4f} MHz".format(fit["fc"] / 1e6,
                                                    fit["fc_err"] / 1e6),
                      fontsize=10)
    else:
        why = ("edges depleted - no contrast" if (con and con["depleted"])
               else "no resolved dip")
        axA.set_title("A. Retention vs pump freq  |  NO FIT ({})".format(why),
                      fontsize=10)
    axA.set_xlabel("D1 pump frequency (AOM) [MHz]")
    axA.set_ylabel("retention  (retained / loaded)")
    axA.set_ylim(0, 1.05)
    axA.legend(fontsize=8)
    axA.grid(alpha=0.3)

    # --- B: is the atom call clean? ---
    s1, s2 = data["s1"], data["s2"]
    lo = min(s1.min(), s2.min())
    hi = max(s1.max(), s2.max())
    step = max(1, int(np.ceil((hi - lo + 1) / 60)))
    bins = np.arange(lo, hi + step + 1, step) - 0.5
    axB.hist(s1, bins=bins, color="C0", alpha=0.6, label="shot 1 (after load)")
    axB.hist(s2, bins=bins, color="C2", alpha=0.6, label="shot 2 (after blowaway)")
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
    loop = data["opr_loop_index"]
    loaded, ret = data["loaded"], data["retained"]
    loops = np.unique(loop)
    lf, rf = [], []
    for lv in loops:
        mm = loop == lv
        lf.append(np.mean(loaded[mm]) if np.any(mm) else np.nan)
        nl = np.count_nonzero(mm & loaded)
        rf.append(np.count_nonzero(mm & loaded & ret) / nl if nl else np.nan)
    axC.plot(loops, lf, "-o", ms=4, color="C1", label="loading fraction")
    axC.plot(loops, rf, "-o", ms=4, color="C0", label="retention (all freqs)")
    axC.set_xlabel("scan repeat (opr_loop_index)")
    axC.set_ylabel("fraction")
    axC.set_ylim(0, 1.05)
    axC.set_title("C. Stability across the run")
    axC.legend(fontsize=9)
    axC.grid(alpha=0.3)

    fig.suptitle(
        "OpticalPumpResoScan RID {}  |  {} shots, {} freqs  |  pump {:.0f} us "
        "@ {:.3f}, repump {:.3f}  |  threshold {:.0f}".format(
            rid, res["n_shots"], res["n_freqs"], 1e6 * meta["t_pump_s"],
            meta["amp_d1"], meta["amp_repump"], res["threshold"]),
        fontsize=13)

    if out_png is None:
        out_png = str(Path(path).with_name(
            "op_resonance_RID{}.png".format(rid)))
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
    analyze(find_run(rid), threshold=thr)
