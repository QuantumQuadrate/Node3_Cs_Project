"""Analyze AtomAlignSPCM3 runs: SPCM3 count histograms vs the SPCM1+2 reference.

Reads a `*-AtomAlignSPCM3.h5` and turns the raw per-window counts into the
histograms you actually align on:
  - the SPCM1+SPCM2 reference histogram (atom / no-atom separation + threshold),
  - the SPCM3 histogram SPLIT by atom-present vs atom-absent  <-- the alignment
    figure of merit: the gap between the two peaks IS `align_contrast`,
  - the SPCM3 vs (SPCM1+SPCM2) scatter with the Pearson correlation.
A single lumped SPCM3 histogram says almost nothing; the conditional split is
what tells you whether SPCM3 sees the SAME atom as SPCM1/2.

Dataset convention (from experiments/atom_align_spcm3.py, one entry per imaging
window, all three counts from the SAME window):
  spcm_counts1 / spcm_counts2 -- reference SPCMs (ttl0 / ttl1)
  spcm_counts3                -- the SPCM being aligned (ttl2)
  spcm_sum12                  -- spcm_counts1 + spcm_counts2 (atom reference)
`readout_threshold` (the atom-present cutoff on spcm_sum12) is read from the
run's expid; the gate width comes from archive/t_readout_monitor, so counts are
also reported as rates [counts/s] and stay comparable across gate settings.

WHICH WINDOWS ARE ANALYZED -- the experiment writes two flavours of the counts:
  spcm_all1/2/3     uncapped, archive-only: EVERY window of the run.  PREFERRED.
  spcm_counts1/2/3  the applet datasets: a rolling window trimmed to the last
                    `max_points` (default 300) shots.
load_run() uses spcm_all* when present and falls back to the rolling datasets
for runs recorded before spcm_all* existed; the summary prints which one it got,
and warns when the data is the truncated kind. So current runs are analyzed in
full, not just the tail you happened to see in the applet.

POOLING RUNS -- summarize()/analyze() also accept a LIST of paths (see
find_runs()), concatenating several runs into one histogram. Each run's
atom-present mask is applied with ITS OWN readout_threshold before pooling.

BLAS-free (histograms + running sums only; no dot/matmul/linalg); matplotlib is
imported lazily inside analyze() with the Agg backend. Run:
    python analyze_align_spcm3.py                # latest run: summary + figure
    python analyze_align_spcm3.py 2088           # RID 2088
    python analyze_align_spcm3.py 2085 2088      # those RIDs, pooled
    python analyze_align_spcm3.py all            # every run, pooled
"""

import os
import sys
import glob
import json
from pathlib import Path

import numpy as np
import h5py

# AtomAlignSPCM3 can die before its write_results() call, so the file that always
# exists is the one the ARTIQ master writes. Search both result roots.
_BASE = Path(r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes")
RESULTS_DIRS = [_BASE / "results", _BASE / "repository" / "results"]

PATTERN = "*-AtomAlignSPCM3.h5"


# --------------------------------------------------------------------------
#  IO
# --------------------------------------------------------------------------
def _rid_of(path):
    return int(os.path.basename(path).split("-")[0])


def _has_expid(path):
    """True if the file carries the submitted expid (-> readout_threshold)."""
    try:
        with h5py.File(path, "r", locking=False) as hf:
            return "expid" in hf
    except Exception:
        return False


def _dedupe(files):
    """One file per RID, preferring the copy that carries `expid`.

    Every run is saved TWICE: the ARTIQ master writes results/<date>/<hour>/, and
    utilities/write_results.py writes repository/results/<date>/<hour>/. The two
    hold the same datasets, but only the master's copy has `expid` -- which is
    where readout_threshold is read from; without it the threshold silently falls
    back to a default. Returning both would also double-count every window when
    pooling runs.
    """
    by_rid = {}
    for f in files:
        by_rid.setdefault(_rid_of(f), []).append(f)
    out = []
    for rid in sorted(by_rid):
        cands = by_rid[rid]
        if len(cands) == 1:
            out.append(cands[0])
        else:                       # only touch the disk when it is ambiguous
            out.append(next((c for c in cands if _has_expid(c)), cands[0]))
    return out


def find_run(rid=None):
    """Newest *-AtomAlignSPCM3.h5 (or the one for `rid`) across both roots."""
    files = []
    for root in RESULTS_DIRS:
        files += glob.glob(str(root / "**" / PATTERN), recursive=True)
    if not files:
        raise FileNotFoundError(
            "No " + PATTERN + " under "
            + " or ".join(str(r) for r in RESULTS_DIRS))
    files = _dedupe(set(files))
    if rid is None:
        return files[-1]
    for f in files:
        if _rid_of(f) == int(rid):
            return f
    raise FileNotFoundError("No AtomAlignSPCM3 .h5 for RID {}".format(rid))


def find_runs(rids=None):
    """List of run files: every AtomAlignSPCM3 run, or just the given RIDs."""
    files = []
    for root in RESULTS_DIRS:
        files += glob.glob(str(root / "**" / PATTERN), recursive=True)
    files = _dedupe(set(files))     # one file per RID -- never pool duplicates
    if rids is None:
        return files
    wanted = {int(r) for r in rids}
    return [f for f in files if _rid_of(f) in wanted]


def load_run(path):
    """Return (data, meta, rid).

    data: c1, c2, c3, sum12 as float arrays, all the same length (truncated to
          the shortest if a run died mid-push).
    meta: readout_threshold [counts], gate [s], n_windows, source, truncated.

    Prefers the UNCAPPED spcm_all1/2/3 datasets (every window of the run, saved
    archive-only) and falls back to the rolling spcm_counts1/2/3 -- which the
    experiment trims to the last max_points -- for runs recorded before
    spcm_all* existed. meta["truncated"] says which one you got.
    """
    data, meta = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf

        full = all(k in ds for k in ("spcm_all1", "spcm_all2", "spcm_all3"))
        if full:
            names = (("c1", "spcm_all1"), ("c2", "spcm_all2"), ("c3", "spcm_all3"))
        else:
            names = (("c1", "spcm_counts1"), ("c2", "spcm_counts2"),
                     ("c3", "spcm_counts3"), ("sum12", "spcm_sum12"))
        meta["source"] = "spcm_all* (full run)" if full \
            else "spcm_counts* (rolling window)"
        meta["truncated"] = not full

        for key, name in names:
            data[key] = (np.atleast_1d(ds[name][()]).astype(float)
                         if name in ds else np.array([]))
        # spcm_all* has no stored sum; it is just c1+c2 (same window).
        if "sum12" not in data:
            data["sum12"] = np.array([])

        # Threshold from the submitted arguments (expid is JSON-compatible for
        # this experiment's scalar-only arguments; fall back if it ever isn't).
        meta["readout_threshold"] = 120.0
        meta["threshold_from"] = "DEFAULT -- no expid in this file"
        if "expid" in hf:
            raw = hf["expid"][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            try:
                meta["readout_threshold"] = float(
                    json.loads(raw)["arguments"]["readout_threshold"])
                meta["threshold_from"] = "expid"
            except Exception:
                meta["threshold_from"] = "DEFAULT -- expid unreadable"

        # Gate width -> rates. archive/ holds the ExperimentVariables snapshot.
        meta["gate"] = (float(hf["archive/t_readout_monitor"][()])
                        if "archive/t_readout_monitor" in hf else float("nan"))

    # sum12 is redundant with c1+c2; recompute if it is missing or short.
    if data["sum12"].size != data["c1"].size and data["c1"].size:
        data["sum12"] = data["c1"] + data["c2"]

    n = min(a.size for a in data.values())
    for k in data:
        data[k] = data[k][:n]
    meta["n_windows"] = n
    # Atom-present mask, using THIS run's own threshold.
    data["present"] = data["sum12"] > meta["readout_threshold"]
    meta["label"] = str(_rid_of(path))
    return data, meta, _rid_of(path)


def _as_paths(path_or_paths):
    """Accept a single path or a list of paths."""
    if isinstance(path_or_paths, (list, tuple, set)):
        return list(path_or_paths)
    return [path_or_paths]


def load_runs(paths):
    """Pool several runs into one dataset for a combined histogram.

    Each file's atom-present mask is computed with ITS OWN readout_threshold
    BEFORE concatenation, so pooling runs taken at different thresholds still
    labels every window correctly. Returns (data, meta, label) with the same
    keys as load_run; meta["readout_threshold"] is the first run's value (only
    used to draw the reference line) and meta["thresholds_differ"] flags a mix.
    """
    paths = _as_paths(paths)
    parts, metas, rids = [], [], []
    for p in paths:
        d, m, rid = load_run(p)
        if d["c3"].size == 0:
            continue
        parts.append(d)
        metas.append(m)
        rids.append(rid)
    if not parts:
        raise ValueError("No usable windows in: " + ", ".join(map(str, paths)))

    data = {k: np.concatenate([d[k] for d in parts])
            for k in ("c1", "c2", "c3", "sum12", "present")}
    data["present"] = data["present"].astype(bool)

    thrs = [m["readout_threshold"] for m in metas]
    gates = [m["gate"] for m in metas]
    meta = {
        "readout_threshold": thrs[0],
        "thresholds_differ": len(set(thrs)) > 1,
        "gate": gates[0],
        "gates_differ": len(set(gates)) > 1,
        "n_windows": int(data["c3"].size),
        "n_runs": len(parts),
        "rids": rids,
        "source": " + ".join(sorted({m["source"] for m in metas})),
        "truncated": any(m["truncated"] for m in metas),
        "threshold_from": ("expid" if all(m["threshold_from"] == "expid"
                                          for m in metas)
                           else "DEFAULT for at least one pooled run"),
    }
    meta["label"] = "{}-{}_x{}".format(min(rids), max(rids), len(rids))
    return data, meta, meta["label"]


# --------------------------------------------------------------------------
#  STATS  (running sums only -- no BLAS)
# --------------------------------------------------------------------------
def pearson(x, y):
    """Pearson r from running sums (matches the experiment's live align_corr)."""
    n = x.size
    if n < 2:
        return float("nan")
    sx = float(x.sum())
    sy = float(y.sum())
    cov = float((x * y).sum()) - sx * sy / n
    vx = float((x * x).sum()) - sx * sx / n
    vy = float((y * y).sum()) - sy * sy / n
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return max(-1.0, min(1.0, cov / (vx * vy) ** 0.5))


def summarize(path, verbose=True):
    """Alignment metrics recomputed from the raw windows. Returns a dict.

    `path` is one .h5 path, or a LIST of paths to pool into one combined
    result (see load_runs / find_runs).
    """
    paths = _as_paths(path)
    if len(paths) == 1:
        data, meta, rid = load_run(paths[0])
    else:
        data, meta, rid = load_runs(paths)
    c3, s12 = data["c3"], data["sum12"]
    if c3.size == 0:
        if verbose:
            print("RID {}: no windows in {}".format(rid, paths[0]))
        return None

    thr = meta["readout_threshold"]
    # Precomputed per run (load_run / load_runs), so pooled runs each use their
    # OWN threshold rather than one global cutoff.
    present = data["present"]
    absent = ~present

    def _ms(a):
        return (float(a.mean()),
                float(a.std(ddof=1)) if a.size > 1 else 0.0)

    sig, sig_sd = _ms(c3[present]) if present.any() else (float("nan"), 0.0)
    bg, bg_sd = _ms(c3[absent]) if absent.any() else (float("nan"), 0.0)
    contrast = sig - bg
    # Separation in units of the combined spread: >2 means the two SPCM3 peaks
    # are actually resolved, not merely different in the mean.
    denom = (0.5 * (sig_sd ** 2 + bg_sd ** 2)) ** 0.5
    sep = contrast / denom if denom > 0 else float("nan")

    res = {
        "rid": rid, "path": paths[0] if len(paths) == 1 else paths,
        "n_runs": meta.get("n_runs", 1), "n_windows": int(c3.size),
        "n_atom": int(present.sum()), "n_noatom": int(absent.sum()),
        "threshold": thr, "gate": meta["gate"],
        "source": meta.get("source", "?"),
        "truncated": bool(meta.get("truncated", True)),
        "signal": sig, "signal_sd": sig_sd,
        "background": bg, "background_sd": bg_sd,
        "contrast": contrast, "separation": sep,
        "corr": pearson(s12, c3),
    }

    if verbose:
        g = meta["gate"]

        def rate(v):
            return v / g if g == g and g > 0 else float("nan")

        print("--- AtomAlignSPCM3  RID {} ---".format(rid))
        if len(paths) == 1:
            print("file            : {}".format(paths[0]))
        else:
            print("files           : {} runs, RIDs {}".format(
                len(paths), meta.get("rids")))
        print("source          : {}".format(res["source"]))
        print("windows         : {}  (atom {} / no-atom {})".format(
            res["n_windows"], res["n_atom"], res["n_noatom"]))
        print("gate            : {:.1f} ms   threshold (sum12) : "
              "{:.0f} counts = {:.1f} kcps".format(
                  1e3 * g, thr, rate(thr) / 1e3))
        if not str(meta.get("threshold_from", "")).startswith("expid"):
            print("WARNING: readout_threshold is a {} -- the atom/no-atom "
                  "split may be wrong.".format(meta.get("threshold_from")))
        print("SPCM3 atom      : {:8.2f} +- {:5.2f} counts ({:6.2f} kcps)".format(
            sig, sig_sd, rate(sig) / 1e3))
        print("SPCM3 no atom   : {:8.2f} +- {:5.2f} counts ({:6.2f} kcps)".format(
            bg, bg_sd, rate(bg) / 1e3))
        print("contrast        : {:8.2f} counts ({:6.2f} kcps)   "
              "<-- MAXIMIZE".format(contrast, rate(contrast) / 1e3))
        print("separation      : {:8.2f} sigma".format(sep))
        print("corr(SPCM3,S12) : {:8.3f}   <-- -> 1".format(res["corr"]))
        if res["truncated"]:
            print("NOTE: at least one run predates spcm_all*, so only its LAST "
                  "max_points windows survive; earlier ones were dropped by "
                  "the rolling buffer before saving.")
        if meta.get("gates_differ"):
            print("WARNING: pooled runs used DIFFERENT gate widths -- raw "
                  "counts are not comparable across them; compare rates "
                  "(counts/gate) or pool only same-gate runs.")
        if meta.get("thresholds_differ"):
            print("NOTE: pooled runs used different readout_thresholds; each "
                  "run's own threshold was applied before pooling, but the "
                  "reference line in panel A shows only the first.")
    return res


# --------------------------------------------------------------------------
#  FIGURE
# --------------------------------------------------------------------------
def _int_bins(*arrays, **kwargs):
    """Integer-centred bin edges spanning all arrays (counts are integers)."""
    max_bins = kwargs.get("max_bins", 60)
    lo = min(float(a.min()) for a in arrays if a.size)
    hi = max(float(a.max()) for a in arrays if a.size)
    span = hi - lo + 1
    step = max(1, int(np.ceil(span / max_bins)))
    return np.arange(lo, hi + step + 1, step) - 0.5


def analyze(path, out_png=None):
    """3-panel figure for one run, or for a LIST of runs pooled together.

    Uses whatever matplotlib backend is already active, so a notebook keeps its
    interactive one (%matplotlib widget); the CLI below selects Agg itself.
    Returns the Figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    res = summarize(path)
    if res is None:
        return None
    paths = _as_paths(path)
    if len(paths) == 1:
        data, meta, rid = load_run(paths[0])
    else:
        data, meta, rid = load_runs(paths)
    c3, s12 = data["c3"], data["sum12"]
    thr = res["threshold"]
    present = data["present"]
    absent = ~present

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.8),
                                        constrained_layout=True)

    # Panel A: the REFERENCE histogram -- is the atom/no-atom split clean, and
    # is readout_threshold sitting in the valley between the two peaks?
    axA.hist(s12, bins=_int_bins(s12), color="C0", alpha=0.85,
             edgecolor="black", lw=0.3)
    axA.axvline(thr, color="C3", ls="--", lw=1.4,
                label="readout_threshold = {:.0f}".format(thr))
    axA.set_xlabel("SPCM1 + SPCM2 counts per window")
    axA.set_ylabel("occurrences")
    axA.set_title("A. Reference (SPCM1+2): atom present?")
    axA.legend(fontsize=9)
    axA.grid(alpha=0.3)

    # Panel B: THE alignment histogram -- SPCM3 conditioned on the reference.
    bins3 = _int_bins(c3)
    if absent.any():
        axB.hist(c3[absent], bins=bins3, color="C7", alpha=0.65,
                 edgecolor="black", lw=0.3,
                 label="no atom (n={})".format(res["n_noatom"]))
        axB.axvline(res["background"], color="C7", ls="--", lw=1.2)
    if present.any():
        axB.hist(c3[present], bins=bins3, color="C2", alpha=0.65,
                 edgecolor="black", lw=0.3,
                 label="atom (n={})".format(res["n_atom"]))
        axB.axvline(res["signal"], color="C2", ls="--", lw=1.2)
    axB.set_xlabel("SPCM3 counts per window")
    axB.set_ylabel("occurrences")
    axB.set_title("B. SPCM3 split by atom  |  contrast = {:.1f} counts "
                  "({:.1f} sigma)".format(res["contrast"], res["separation"]))
    axB.legend(fontsize=9)
    axB.grid(alpha=0.3)

    # Panel C: same-window scatter -- a threshold-independent alignment metric.
    axC.plot(s12[absent], c3[absent], "o", ms=4, color="C7", alpha=0.6,
             label="no atom")
    axC.plot(s12[present], c3[present], "o", ms=4, color="C2", alpha=0.6,
             label="atom")
    axC.axvline(thr, color="C3", ls="--", lw=1.0)
    axC.set_xlabel("SPCM1 + SPCM2 counts per window")
    axC.set_ylabel("SPCM3 counts per window")
    axC.set_title("C. Same-window scatter  |  r = {:.3f}".format(res["corr"]))
    axC.legend(fontsize=9)
    axC.grid(alpha=0.3)

    runs = "RID {}".format(rid) if len(paths) == 1 \
        else "RIDs {} ({} runs pooled)".format(rid, meta.get("n_runs"))
    fig.suptitle(
        "AtomAlignSPCM3 {}  |  {} windows @ {:.0f} ms  |  SPCM3 atom "
        "{:.1f} vs bg {:.1f} counts  |  r = {:.3f}".format(
            runs, res["n_windows"], 1e3 * meta["gate"], res["signal"],
            res["background"], res["corr"]),
        fontsize=13)

    if out_png is None:
        out_png = str(Path(paths[0]).with_name(
            "align_spcm3_RID{}.png".format(rid)))
    fig.savefig(out_png, dpi=130)
    print("Saved figure: {}".format(out_png))
    # Display ONLY on a notebook backend (inline / ipympl / nbagg). Calling
    # plt.show() on a GUI backend such as qt5agg starts a BLOCKING event loop:
    # inside a Jupyter kernel that reads as "The Kernel crashed while executing
    # code in the current cell", and in a plain script it simply hangs. "not
    # Agg" was the wrong test -- it treats every GUI backend as safe. The PNG is
    # already written above, so closing the figure loses nothing.
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in ("inline", "ipympl", "nbagg", "widget")):
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")          # headless: script runs save a PNG, no window
    argv = sys.argv[1:]
    if argv and argv[0].lower() == "all":
        analyze(find_runs())                       # every run, pooled
    elif len(argv) > 1:
        analyze(find_runs(argv))                   # the listed RIDs, pooled
    elif argv:
        analyze(find_run(int(argv[0])))            # one RID
    else:
        analyze(find_run())                        # latest run
