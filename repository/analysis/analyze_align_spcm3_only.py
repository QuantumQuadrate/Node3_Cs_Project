"""Analyze AtomAlignSPCM3Only runs: the SPCM3 count histogram, one detector.

Reads a `*-AtomAlignSPCM3Only.h5` (from experiments/atom_align_spcm3_only.py,
which records SPCM3 alone -- no SPCM1/2 reference) and answers the one question
a single detector can honestly answer: does the SPCM3 count distribution split
into TWO peaks (no atom / atom), and where is the valley between them?

  A  histogram of every SPCM3 window, with the run's own readout_threshold AND
     the data-driven threshold found by Otsu's method
  B  the same counts as a time series, so you can see individual atom events

WHY NOT JUST USE align_contrast. With no independent reference, the experiment's
align_signal / align_background split SPCM3 against readout_threshold -- so
every "signal" window is above the threshold by construction and every
"background" window below it. That contrast therefore cannot fall below roughly
the threshold spacing and it MOVES when you change the threshold even if the
optics never moved. It is a relative indicator at a FIXED threshold, nothing
more. This module instead judges the counts on their own, independently of any
threshold, and reports:

    F = var/mean   the Fano factor   <-- THE METRIC to maximize/watch
    suggested      where to put readout_threshold on the dashboard
    separation     the Otsu split's peak separation (biased up -- see below)

**Use F, not separation.** Photon counting from one steady source is Poisson, so
F ~ 1 for a single blob; two populations (atom / no atom) inflate it far above 1.
`separation` cannot do this job: Otsu divides ANY distribution somewhere, and
cutting a single Poisson blob near its mean leaves two sub-populations each
narrower than the parent, so they land a couple of sigma apart by construction
-- a pure Poisson(30) blob measures 2.6 sigma. F tells one blob from two peaks;
separation only describes a split once F says there is one.

Datasets: `spcm_all3` (uncapped, every window -- preferred) falling back to
`spcm_counts3` (the applet's rolling window, last max_points only).
`readout_threshold` comes from the run's expid, the gate from
archive/t_readout_monitor, so counts are also reported as rates.

TWO COPIES OF EVERY RUN: the ARTIQ master writes results/<date>/<hour>/ and
utilities/write_results.py writes repository/results/<date>/<hour>/. Only the
master's copy carries `expid` (the threshold), so that one is preferred and
duplicate RIDs are dropped.

BLAS-free (histograms + sums only). Plot in Python 3.11, NOT artiq7_claude --
matplotlib segfaults there (see analyze_align_spcm3.ipynb cell 1). Run:
    python analyze_align_spcm3_only.py           # latest run
    python analyze_align_spcm3_only.py 2452      # one RID
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

PATTERN = "*-AtomAlignSPCM3Only.h5"


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
        cands = by_rid[rid]
        out.append(cands[0] if len(cands) == 1
                   else next((c for c in cands if _has_expid(c)), cands[0]))
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
    raise FileNotFoundError("No AtomAlignSPCM3Only .h5 for RID {}".format(rid))


def find_runs():
    """Every run file on disk (one per RID), oldest first."""
    return _all_files()


def load_run(path):
    """Return (c3, meta, rid).

    c3   : float array, one entry per imaging window.
    meta : readout_threshold, threshold_from, gate [s], n_windows, source.
    """
    meta = {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf

        # Prefer the uncapped record; fall back to the applet's rolling window.
        if "spcm_all3" in ds and ds["spcm_all3"].size:
            c3 = np.atleast_1d(ds["spcm_all3"][()]).astype(float)
            meta["source"] = "spcm_all3 (full run)"
        elif "spcm_counts3" in ds:
            c3 = np.atleast_1d(ds["spcm_counts3"][()]).astype(float)
            meta["source"] = "spcm_counts3 (rolling window)"
        else:
            c3 = np.array([])
            meta["source"] = "NONE -- no SPCM3 dataset in this file"

        meta["readout_threshold"] = float("nan")
        meta["threshold_from"] = "absent -- no expid in this file"
        if "expid" in hf:
            raw = hf["expid"][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            try:
                meta["readout_threshold"] = float(
                    json.loads(raw)["arguments"]["readout_threshold"])
                meta["threshold_from"] = "expid"
            except Exception:
                meta["threshold_from"] = "absent -- expid unreadable"

        meta["gate"] = (float(hf["archive/t_readout_monitor"][()])
                        if "archive/t_readout_monitor" in hf else float("nan"))

        # The experiment's own live metrics, for cross-checking.
        for k in ("align_signal", "align_background", "align_contrast"):
            key = "datasets/" + k
            meta[k] = float(hf[key][()]) if key in hf else float("nan")

    meta["n_windows"] = int(c3.size)
    return c3, meta, _rid_of(path)


# --------------------------------------------------------------------------
#  STATS  (histograms + sums only -- no BLAS)
# --------------------------------------------------------------------------
def otsu_threshold(c3):
    """Threshold maximizing between-class variance (Otsu), from the DATA.

    Independent of whatever readout_threshold was dialled in, so it is a fair
    way to ask "do these counts split in two?". Returns the integer threshold t
    where windows with count > t are the upper class, or None if the counts span
    a single value (nothing to split).
    """
    if c3.size < 2:
        return None
    c = c3.astype(int)
    lo, hi = int(c.min()), int(c.max())
    if hi <= lo:
        return None

    hist = np.bincount(c - lo, minlength=hi - lo + 1).astype(float)
    vals = np.arange(lo, hi + 1, dtype=float)
    cum_n = np.cumsum(hist)                 # count of windows <= vals[i]
    cum_s = np.cumsum(hist * vals)          # their summed counts
    n_tot, s_tot = cum_n[-1], cum_s[-1]

    best_t, best_var = None, -1.0
    for i in range(vals.size - 1):          # t = vals[i]; upper class is > t
        n0 = cum_n[i]
        n1 = n_tot - n0
        if n0 <= 0 or n1 <= 0:
            continue
        m0 = cum_s[i] / n0
        m1 = (s_tot - cum_s[i]) / n1
        var = n0 * n1 * (m0 - m1) ** 2      # between-class variance
        if var > best_var:
            best_var, best_t = var, vals[i]
    return None if best_t is None else int(best_t)


def fano_factor(c3):
    """variance / mean of the counts -- the bimodality test that actually works.

    Photon counting from ONE steady source is Poisson, so var/mean ~ 1. Two
    populations (atom / no atom) add a between-class term, which inflates it far
    above 1: for means m0,m1 with fractions f,1-f the excess is
    f(1-f)(m1-m0)^2 / mean. So F ~ 1 means a single blob no matter where you cut
    it, and F >> 1 means real structure.

    This is needed because the Otsu split's `separation` is BIASED UPWARD: Otsu
    divides any distribution somewhere, and cutting a single Poisson blob near
    its mean yields two sub-populations that are each narrower than the parent,
    so their means sit a couple of sigma apart by construction (measured: a pure
    Poisson(30) blob scores 2.6 sigma). separation alone cannot tell one blob
    from two peaks; F can.
    """
    if c3.size < 2:
        return float("nan")
    m = float(c3.mean())
    if m <= 0:
        return float("nan")
    return float(c3.var(ddof=1)) / m


def suggest_threshold(split):
    """Where to actually PUT readout_threshold, given the two classes.

    Otsu's own cut is weighted by class population, so with a big background
    population it sits close to the lower peak (measured: peaks at 20 and 90,
    70%/30%, put Otsu at 37 -- valid but tight). For a detection threshold you
    want the point equally many sigma from each peak,
        thr = (m0*s1 + m1*s0) / (s0 + s1),
    which balances the two misclassification rates. Falls back to the midpoint
    if a class has no spread.
    """
    if split is None:
        return None
    m0, s0 = split["mean_below"], split["sd_below"]
    m1, s1 = split["mean_above"], split["sd_above"]
    if m0 != m0 or m1 != m1:
        return None
    if (s0 + s1) <= 0:
        return int(round(0.5 * (m0 + m1)))
    return int(round((m0 * s1 + m1 * s0) / (s0 + s1)))


def _split_stats(c3, thr):
    """Mean/sd above and at-or-below `thr`, plus contrast and separation."""
    above = c3[c3 > thr]
    below = c3[c3 <= thr]

    def _ms(a):
        return (float(a.mean()) if a.size else float("nan"),
                float(a.std(ddof=1)) if a.size > 1 else 0.0)

    hi, hi_sd = _ms(above)
    lo, lo_sd = _ms(below)
    contrast = hi - lo
    denom = (0.5 * (hi_sd ** 2 + lo_sd ** 2)) ** 0.5
    return {
        "threshold": float(thr),
        "n_above": int(above.size), "n_below": int(below.size),
        "mean_above": hi, "sd_above": hi_sd,
        "mean_below": lo, "sd_below": lo_sd,
        "contrast": contrast,
        "separation": contrast / denom if denom > 0 else float("nan"),
    }


def summarize(path, verbose=True):
    """Histogram statistics for one run. Returns a dict."""
    c3, meta, rid = load_run(path)
    if c3.size == 0:
        if verbose:
            print("RID {}: no windows in {}  ({})".format(
                rid, path, meta["source"]))
        return None

    res = {"rid": rid, "path": path, "n_windows": int(c3.size),
           "gate": meta["gate"], "source": meta["source"],
           "min": float(c3.min()), "max": float(c3.max()),
           "mean": float(c3.mean()), "fano": fano_factor(c3)}

    # Is the distribution wider than a single Poisson source? This, not the
    # Otsu separation, is what decides "two peaks or one blob".
    res["bimodal"] = bool(res["fano"] == res["fano"] and res["fano"] > 2.0)

    # Split at the run's own threshold (self-referential -- see module docstring)
    thr_run = meta["readout_threshold"]
    res["run_split"] = (_split_stats(c3, thr_run)
                        if thr_run == thr_run else None)

    # Split found from the data, then the threshold to actually use.
    t_otsu = otsu_threshold(c3)
    res["otsu_threshold"] = t_otsu
    res["otsu_split"] = _split_stats(c3, t_otsu) if t_otsu is not None else None
    res["suggested_threshold"] = suggest_threshold(res["otsu_split"])

    if verbose:
        g = meta["gate"]

        def rate(v):
            return v / g if g == g and g > 0 else float("nan")

        print("--- AtomAlignSPCM3Only  RID {} ---".format(rid))
        print("file            : {}".format(path))
        print("source          : {}".format(meta["source"]))
        print("windows         : {}".format(res["n_windows"]))
        print("gate            : {:.1f} ms".format(1e3 * g))
        print("SPCM3 counts    : min {:.0f}  mean {:.1f}  max {:.0f}   "
              "({:.2f} kcps mean)".format(
                  res["min"], res["mean"], res["max"], rate(res["mean"]) / 1e3))

        if c3.max() == 0:
            print()
            print("ALL ZERO -- SPCM3 recorded no photons at all. Not an "
                  "alignment problem: check power, the BNC on ttl2, shutters.")
            return res

        if res["run_split"] is not None:
            s = res["run_split"]
            print()
            print("at the run's readout_threshold = {:.0f} "
                  "(from {}):".format(s["threshold"], meta["threshold_from"]))
            print("   above        : {:8.2f} +- {:5.2f}  (n={})".format(
                s["mean_above"], s["sd_above"], s["n_above"]))
            print("   at/below     : {:8.2f} +- {:5.2f}  (n={})".format(
                s["mean_below"], s["sd_below"], s["n_below"]))
            print("   contrast     : {:8.2f} counts   (self-referential -- "
                  "relative use only)".format(s["contrast"]))
            if s["n_above"] == 0 or s["n_below"] == 0:
                print("   !! every window fell on ONE side of this threshold; "
                      "it is not splitting the data at all.")
        else:
            print()
            print("readout_threshold {} -- cannot evaluate the run's own "
                  "split.".format(meta["threshold_from"]))

        # Bimodality: decided by the Fano factor, NOT by the Otsu separation
        # (which is biased upward -- see fano_factor's docstring).
        print()
        print("TWO PEAKS, OR ONE BLOB?  (threshold-independent)")
        print("   var/mean (F) : {:8.2f}   <-- THE METRIC. F~1 = single "
              "Poisson blob, F>>1 = real structure".format(res["fano"]))
        if not res["bimodal"]:
            print("   VERDICT: ONE BLOB -- SPCM3 is NOT resolving the atom "
                  "(F is Poisson-like). Keep walking the detector; watch F.")
        else:
            print("   VERDICT: structure present -- the counts are far wider "
                  "than one Poisson source.")

        if res["otsu_split"] is not None:
            s = res["otsu_split"]
            print()
            print("SPLIT FOUND IN THE DATA (Otsu cut at {:.0f}):".format(
                s["threshold"]))
            print("   upper peak   : {:8.2f} +- {:5.2f}  (n={})".format(
                s["mean_above"], s["sd_above"], s["n_above"]))
            print("   lower peak   : {:8.2f} +- {:5.2f}  (n={})".format(
                s["mean_below"], s["sd_below"], s["n_below"]))
            print("   separation   : {:8.2f} sigma   (biased UP -- a pure "
                  "Poisson blob scores ~2.6; read it with F)".format(
                      s["separation"]))
            if res["suggested_threshold"] is not None:
                print("   SUGGESTED readout_threshold = {:d} counts   "
                      "(equal-sigma point between the peaks)".format(
                          res["suggested_threshold"]))
                if not res["bimodal"]:
                    print("   ...but F says one blob, so this threshold is "
                          "splitting noise. Fix the alignment first.")

        live = meta.get("align_contrast", float("nan"))
        if live == live:
            print()
            print("(live align_contrast at end of run: {:.2f})".format(live))
    return res


# --------------------------------------------------------------------------
#  FIGURE
# --------------------------------------------------------------------------
def _int_bins(c3, max_bins=60):
    """Integer-centred bin edges spanning the counts."""
    lo, hi = float(c3.min()), float(c3.max())
    step = max(1, int(np.ceil((hi - lo + 1) / max_bins)))
    return np.arange(lo, hi + step + 1, step) - 0.5


def analyze(path, out_png=None):
    """Two-panel figure: count histogram + time series. Returns the Figure."""
    import matplotlib
    import matplotlib.pyplot as plt

    res = summarize(path)
    if res is None:
        return None
    c3, meta, rid = load_run(path)
    thr_run = meta["readout_threshold"]
    t_otsu = res["suggested_threshold"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8),
                                   constrained_layout=True)

    # A: the histogram -- does it split into two peaks?
    if c3.max() > c3.min():
        axA.hist(c3, bins=_int_bins(c3), color="C0", alpha=0.85,
                 edgecolor="black", lw=0.3)
    else:
        axA.hist(c3, bins=3, color="C0", alpha=0.85)
    if thr_run == thr_run:
        axA.axvline(thr_run, color="C3", ls="--", lw=1.4,
                    label="run threshold = {:.0f}".format(thr_run))
    if t_otsu is not None:
        axA.axvline(t_otsu, color="C2", ls="-", lw=1.4,
                    label="suggested = {:d}".format(t_otsu))
    axA.set_xlabel("SPCM3 counts per window  ({:.0f} ms gate)".format(
        1e3 * meta["gate"]))
    axA.set_ylabel("windows")
    axA.set_title("A. SPCM3 histogram  |  F = var/mean = {:.2f}  ({})".format(
        res["fano"],
        "structure" if res["bimodal"] else "one blob, Poisson-like"))
    axA.legend(fontsize=9)
    axA.grid(alpha=0.3)

    # B: time series -- individual atom events, and whether they are distinct.
    axB.plot(np.arange(c3.size), c3, "-o", ms=3, lw=0.8, color="C0", alpha=0.8)
    if thr_run == thr_run:
        axB.axhline(thr_run, color="C3", ls="--", lw=1.2,
                    label="run threshold")
    if t_otsu is not None:
        axB.axhline(t_otsu, color="C2", ls="-", lw=1.2, label="suggested")
    axB.set_xlabel("window number")
    axB.set_ylabel("SPCM3 counts")
    axB.set_title("B. Time series  ({} windows)".format(res["n_windows"]))
    axB.legend(fontsize=9)
    axB.grid(alpha=0.3)

    fig.suptitle(
        "AtomAlignSPCM3Only RID {}  |  {} windows @ {:.0f} ms  |  mean {:.1f} "
        "counts  |  suggested threshold {}".format(
            rid, res["n_windows"], 1e3 * meta["gate"], res["mean"],
            t_otsu if t_otsu is not None else "n/a"),
        fontsize=13)

    if out_png is None:
        out_png = str(Path(path).with_name(
            "align_spcm3_only_RID{}.png".format(rid)))
    fig.savefig(out_png, dpi=130)
    print("Saved figure: {}".format(out_png))
    # Display ONLY on a notebook backend: plt.show() on a GUI backend such as
    # qt5agg starts a blocking event loop (reads as a kernel crash in Jupyter,
    # hangs a script). The PNG is already written, so closing loses nothing.
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in ("inline", "ipympl", "nbagg", "widget")):
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")          # headless: save a PNG, no window
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze(find_run(rid))
