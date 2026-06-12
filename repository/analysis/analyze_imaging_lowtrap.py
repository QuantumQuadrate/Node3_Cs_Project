"""Analyze ImagingLowTrapScan runs: the best single-atom IMAGING condition (cooling-DP
frequency, amplitude) at the LOW trap, by two metrics.

Reads a `*-ImagingLowTrapScan.h5` and produces 2-D heatmaps over (f, amp):
  - Metric 1 (HEATING): survival@release=0 (does the atom survive two 4 ms images) and
    survival@release_probe (temperature sensitivity). HIGHER = less heating.
  - Metric 2 (SEPARATION / DETECTION FIDELITY): histogram of shot1 (atom) vs bg (no atom)
    per (f, amp). A double-Gaussian valley sets the threshold; fidelity = 0.5*(P_detect +
    P_reject). HIGHER = cleaner single-atom discrimination.
The best (f, amp) maximizes fidelity subject to survival@0 above a floor (so a bright but
atom-ejecting point can't win).

NOTE on the env: this imports scipy.optimize.curve_fit (via threshold.py) and uses
np.percentile/median -- run with the WORKING-BLAS `python` on PATH (numpy 2.x / scipy /
h5py / mpl 3.x), NOT the artiq7_claude env (broken BLAS hangs curve_fit / crashes mpl). Run:

    python analyze_imaging_lowtrap.py            # latest run
    python analyze_imaging_lowtrap.py 660        # RID 660
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from threshold import find_threshold_double_gaussian   # double-Gaussian valley threshold
# matplotlib imported lazily inside analyze() so importing the helpers (notebook use)
# does not force the headless "Agg" backend.

RESULTS_DIR = Path(
    r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes\repository\results"
)
SURVIVAL_FLOOR = 0.5   # best (f,a) must have survival@0 >= this (else "bright but ejected")


# --------------------------------------------------------------------------
#  IO
# --------------------------------------------------------------------------
def find_run(rid=None):
    files = sorted(
        glob.glob(str(RESULTS_DIR / "**" / "*-ImagingLowTrapScan.h5"), recursive=True),
        key=lambda f: int(os.path.basename(f).split("-")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No *-ImagingLowTrapScan.h5 under {RESULTS_DIR}")
    if rid is None:
        return files[-1]
    for f in files:
        if int(os.path.basename(f).split("-")[0]) == int(rid):
            return f
    raise FileNotFoundError(f"No ImagingLowTrapScan .h5 for RID {rid}")


def load_run(path):
    """Return (data dict of img_* arrays, snapshot dict, rid)."""
    data, snap = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in ds.keys():
            if k.startswith("img_"):
                v = ds[k][()]
                data[k] = np.atleast_1d(v) if hasattr(v, "__len__") and not isinstance(v, (bytes, str)) else v
            elif k.startswith("snapshot_"):
                v = ds[k][()]
                snap[k[len("snapshot_"):]] = v.decode() if isinstance(v, bytes) else v
    rid = int(os.path.basename(path).split("-")[0])
    return data, snap, rid


def wilson(k, n, z=1.0):
    """Wilson score interval (z=1 ~ 68%). Returns (p, lo, hi)."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def _iqr(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.percentile(x, 75) - np.percentile(x, 25))


# --------------------------------------------------------------------------
#  ANALYSIS
# --------------------------------------------------------------------------
def build_cells(data):
    """Filter img_loaded>0; return {(f, amp): {'shot1','shot2','bg','rel' lists}}."""
    f = np.asarray(data["img_f"]).astype(float)
    a = np.asarray(data["img_amp"]).astype(float)
    rel = np.asarray(data["img_release_us"]).astype(float)
    loaded = np.asarray(data["img_loaded"]).astype(int)
    s1 = (np.asarray(data["img_shot1_spcm1"]).astype(float)
          + np.asarray(data["img_shot1_spcm2"]).astype(float))
    s2 = (np.asarray(data["img_shot2_spcm1"]).astype(float)
          + np.asarray(data["img_shot2_spcm2"]).astype(float))
    bg = (np.asarray(data["img_bg_spcm1"]).astype(float)
          + np.asarray(data["img_bg_spcm2"]).astype(float))
    m = loaded > 0
    f, a, rel, s1, s2, bg = f[m], a[m], rel[m], s1[m], s2[m], bg[m]

    cells = {}
    for i in range(len(f)):
        key = (round(float(f[i]), 0), round(float(a[i]), 4))
        c = cells.setdefault(key, {"shot1": [], "shot2": [], "bg": [], "rel": []})
        c["shot1"].append(s1[i]); c["shot2"].append(s2[i])
        c["bg"].append(bg[i]); c["rel"].append(round(float(rel[i]), 1))
    for c in cells.values():
        for k in c:
            c[k] = np.asarray(c[k], dtype=float)
    return cells


def cell_fidelity(shot1, bg, n_bins=30):
    """Double-Gaussian valley threshold between atom (shot1) and no-atom (bg);
    fidelity = 0.5*(P_detect + P_reject). Returns (fidelity, threshold, gap, sep)."""
    shot1 = np.asarray(shot1, dtype=float)
    bg = np.asarray(bg, dtype=float)
    if shot1.size == 0 or bg.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    thr, _ = find_threshold_double_gaussian(np.concatenate([shot1, bg]), n_bins=n_bins)
    p_detect = float(np.mean(shot1 > thr))     # atom called present
    p_reject = float(np.mean(bg <= thr))       # empty called absent
    fid = 0.5 * (p_detect + p_reject)
    gap = float(np.median(shot1) - np.median(bg))
    sep = gap / (0.5 * (_iqr(shot1) + _iqr(bg)) + 1e-9)
    return fid, thr, gap, sep


def cell_survival(c, rel_value, threshold):
    """Survival = fraction of loaded points at rel_value whose shot2 counts exceed the
    PER-CELL atom/no-atom threshold (the same double-Gaussian valley cell_fidelity uses),
    so each (f, a) is judged by its OWN imaging brightness, not a fixed global bar -- this
    removes the bias where a dim (f, a) under-counts present atoms. The threshold is on raw
    counts (the valley between this cell's raw bg and raw shot1), so no bg subtraction is
    needed. Returns (p, lo, hi) Wilson, or None (no points / threshold not finite)."""
    if not np.isfinite(threshold):
        return None
    mask = np.isclose(c["rel"], rel_value)
    s2 = c["shot2"][mask]
    if s2.size == 0:
        return None
    k = int(np.sum(s2 > threshold))
    return wilson(k, int(s2.size))


def build_grids(data, snap):
    """Return fs, amps, M_surv0, M_survP, M_fid, M_gap, rel_probe, best_fa, best_fid.

    Survival uses the PER-CELL double-Gaussian threshold (computed once per cell and shared
    with the fidelity metric), NOT the global loading atom_threshold -- so a dim (f, a) is
    judged by its own atom/no-atom separation, removing the brightness bias. (snap is kept
    in the signature for call-site/snapshot compatibility; the global threshold is no longer
    used for survival.)"""
    cells = build_cells(data)
    fs = sorted({k[0] for k in cells})
    amps = sorted({k[1] for k in cells})
    rels = sorted({r for c in cells.values() for r in np.unique(c["rel"]).tolist()})
    rel_probe = max(rels) if rels else 0.0

    shape = (len(amps), len(fs))
    M_surv0 = np.full(shape, np.nan)
    M_survP = np.full(shape, np.nan)
    M_fid = np.full(shape, np.nan)
    M_gap = np.full(shape, np.nan)
    best_fa, best_fid = None, -1.0
    for i, a in enumerate(amps):
        for j, f in enumerate(fs):
            c = cells.get((f, a))
            if c is None:
                continue
            # per-cell atom/no-atom threshold, SHARED by fidelity AND survival
            fid, thr, gap, sep = cell_fidelity(c["shot1"], c["bg"])
            M_fid[i, j] = fid
            M_gap[i, j] = gap
            s0 = cell_survival(c, 0.0, thr)
            sp = cell_survival(c, rel_probe, thr)
            if s0 is not None:
                M_surv0[i, j] = s0[0]
            if sp is not None:
                M_survP[i, j] = sp[0]
            # best = max fidelity among cells that actually keep the atom
            if (s0 is not None and s0[0] >= SURVIVAL_FLOOR
                    and np.isfinite(fid) and fid > best_fid):
                best_fid = fid
                best_fa = (f, a)
    return fs, amps, M_surv0, M_survP, M_fid, M_gap, rel_probe, best_fa, best_fid


def analyze(path, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, snap, rid = load_run(path)
    n_total = len(np.asarray(data.get("img_f", [])))
    n_loaded = int(np.sum(np.asarray(data.get("img_loaded", [0])).astype(int)))
    print(f"RID {rid} | {n_total} points | {n_loaded} loaded")
    if n_loaded == 0:
        print("No loaded atoms -- nothing to analyze (check loading / atom_threshold).")
        return

    fs, amps, M_surv0, M_survP, M_fid, M_gap, rel_probe, best_fa, best_fid = build_grids(data, snap)
    fs_mhz = [f / 1e6 for f in fs]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    _heatmap(axes[0, 0], fs_mhz, amps, M_surv0,
             "A. survival @ 0 us  (survives two 4 ms images)", best_fa, plt, 0.0, 1.0)
    _heatmap(axes[0, 1], fs_mhz, amps, M_fid,
             "B. detection fidelity  0.5*(P_detect+P_reject)", best_fa, plt, 0.5, 1.0)
    _heatmap(axes[1, 0], fs_mhz, amps, M_survP,
             f"C. survival @ {rel_probe:g} us  (temperature proxy)", best_fa, plt, 0.0, 1.0)
    _heatmap(axes[1, 1], fs_mhz, amps, M_gap,
             "D. histogram gap  median(shot1)-median(bg)", best_fa, plt, None, None)
    fig.suptitle(f"Imaging low-trap scan -- RID {rid}", fontsize=13)

    if best_fa is not None:
        c = build_cells(data)[best_fa]
        fid, thr, gap, sep = cell_fidelity(c["shot1"], c["bg"])
        print(f"BEST imaging (f, amp) = {best_fa[0]/1e6:.4f} MHz, {best_fa[1]:.3f}  "
              f"fidelity={fid:.3f}  threshold={thr:.1f}  gap={gap:.1f}  sep={sep:.2f}")
        print(f"  baseline (nominal trap): f_cooling_imaging="
              f"{float(snap.get('f_cooling_imaging', 0))/1e6:.4f} MHz, "
              f"amp_cooling_imaging={snap.get('amp_cooling_imaging', '?')}")
        _hist_png(path, rid, best_fa, c, thr, plt)
    else:
        print(f"No (f, amp) with survival@0 >= {SURVIVAL_FLOOR:.0%} -- imaging ejects the "
              f"atom everywhere at this gate/trap. Shorten the gate or use a less shallow trap.")

    if out_png is None:
        out_png = str(Path(path).with_name(f"imaging_lowtrap_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved figure: {out_png}")

    # the whole-scan view: every (f, amp) histogram in one grid
    histogram_grid(path)


def _heatmap(ax, fs_mhz, amps, M, title, best_fa, plt, vmin, vmax):
    im = ax.pcolormesh(fs_mhz, amps, M, shading="nearest", cmap="viridis",
                       vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if best_fa is not None:
        ax.plot(best_fa[0] / 1e6, best_fa[1], "*", color="red", ms=16,
                markeredgecolor="white", label="best")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("imaging-DP frequency (MHz)")
    ax.set_ylabel("imaging-DP amplitude")
    ax.set_title(title, fontsize=11)


def _hist_png(path, rid, best_fa, c, thr, plt):
    """shot1 (atom) vs bg (no atom) histogram at the best (f, amp), with the threshold."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    allc = np.concatenate([c["shot1"], c["bg"]])
    bins = np.linspace(float(np.min(allc)), float(np.max(allc)), 31)
    ax.hist(c["bg"], bins=bins, color="C0", alpha=0.6, label="bg (no atom)")
    ax.hist(c["shot1"], bins=bins, color="C3", alpha=0.6, label="shot1 (atom)")
    ax.axvline(thr, color="k", ls="--", lw=1.5, label=f"threshold {thr:.0f}")
    ax.set_xlabel("SPCM1+SPCM2 counts")
    ax.set_ylabel("occurrences")
    ax.set_title(f"Best cell f={best_fa[0]/1e6:.3f} MHz amp={best_fa[1]:.3f} -- RID {rid}")
    ax.legend(fontsize=9)
    out = str(Path(path).with_name(f"imaging_lowtrap_hist_RID{rid}.png"))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved histogram: {out}")


def histogram_grid(path, out_png=None, n_bins=30):
    """Small-multiples: one shot1(atom, red)-vs-bg(no atom, blue) histogram per (f, amp)
    cell in ONE figure, each with its per-cell double-Gaussian threshold (dashed) and
    F=fidelity / s=survival@0. Rows = imaging amplitude (high at top, matching the heatmaps),
    cols = imaging frequency. The whole-scan separation picture at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, snap, rid = load_run(path)
    cells = build_cells(data)
    fs = sorted({k[0] for k in cells})
    amps = sorted({k[1] for k in cells})
    nf, na = len(fs), len(amps)
    if nf == 0 or na == 0:
        print("histogram_grid: no cells to plot.")
        return

    # shared bins so every panel is directly comparable
    allc = np.concatenate([np.concatenate([c["shot1"], c["bg"]]) for c in cells.values()])
    bins = np.linspace(float(allc.min()), float(allc.max()), n_bins + 1)

    fig, axes = plt.subplots(na, nf, sharex=True, sharey=True, squeeze=False,
                             figsize=(max(1.5 * nf, 6.0), max(1.4 * na, 4.0)))
    for i in range(na):
        a = amps[na - 1 - i]                       # row 0 = highest amp (match the heatmaps)
        for j in range(nf):
            f = fs[j]
            ax = axes[i][j]
            c = cells.get((f, a))
            if c is None:
                ax.axis("off")
                continue
            fid, thr, gap, sep = cell_fidelity(c["shot1"], c["bg"], n_bins=n_bins)
            s0 = cell_survival(c, 0.0, thr)
            ax.hist(c["bg"], bins=bins, color="C0", alpha=0.6)
            ax.hist(c["shot1"], bins=bins, color="C3", alpha=0.6)
            if np.isfinite(thr):
                ax.axvline(thr, color="k", ls="--", lw=0.7)
            s0t = f"{s0[0]:.2f}" if s0 is not None else "-"
            ax.text(0.96, 0.94, f"F{fid:.2f}\ns{s0t}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
            ax.tick_params(labelsize=4, length=2)
            if j == 0:
                ax.set_ylabel(f"a={a:.3f}", fontsize=6)
            if i == 0:
                ax.set_title(f"{f/1e6:.2f}", fontsize=6)

    fig.suptitle(f"Per-(f, amp) histograms -- shot1 (atom, red) vs bg (no atom, blue), "
                 f"dashed = threshold -- RID {rid}\n"
                 f"cols = imaging freq (MHz), rows = imaging amp;  F = fidelity, s = survival@0",
                 fontsize=10)
    if out_png is None:
        out_png = str(Path(path).with_name(f"imaging_lowtrap_histgrid_RID{rid}.png"))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram grid: {out_png}")


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze(find_run(rid))
