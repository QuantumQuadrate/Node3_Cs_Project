"""Analyze PGCLowTrapScan runs: best cooling-DP (frequency, amplitude) for PGC at
the low trap.

Reads a `*-PGCLowTrapScan.h5` and produces two 2-D heatmaps over (f, amp):
  - Panel A: recapture at the actual release time = the atom-temperature proxy
    (HIGHER = colder). Its peak = the best low-trap PGC (f, amp).
  - Panel B: retention at release=0 = "did the PGC keep the atom at all" (a
    sanity map; a region that's low here lost the atom regardless of recapture).

BLAS-free (sums + np.full only); but PREFER the working-BLAS `python` on PATH;
matplotlib savefig crashes in the artiq7_claude env. Run:

    python analyze_pgc_lowtrap.py            # latest run
    python analyze_pgc_lowtrap.py 640        # RID 640
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import h5py
# matplotlib imported lazily inside analyze() so importing the data helpers
# (notebook use) does not force the headless "Agg" backend.

RESULTS_DIR = Path(
    r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes\repository\results"
)


# --------------------------------------------------------------------------
#  IO
# --------------------------------------------------------------------------
def find_run(rid=None):
    files = sorted(
        glob.glob(str(RESULTS_DIR / "**" / "*-PGCLowTrapScan.h5"), recursive=True),
        key=lambda f: int(os.path.basename(f).split("-")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No *-PGCLowTrapScan.h5 under {RESULTS_DIR}")
    if rid is None:
        return files[-1]
    for f in files:
        if int(os.path.basename(f).split("-")[0]) == int(rid):
            return f
    raise FileNotFoundError(f"No PGCLowTrapScan .h5 for RID {rid}")


def load_run(path):
    """Return (data dict of pgc_* arrays, snapshot dict, rid)."""
    data, snap = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in ds.keys():
            if k.startswith("pgc_"):
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


# --------------------------------------------------------------------------
#  ANALYSIS
# --------------------------------------------------------------------------
def recapture_table(data):
    """Filter loaded==1; return {(f, amp, release_us): [n_retained, n_loaded]}."""
    f = np.asarray(data["pgc_f"]).astype(float)
    a = np.asarray(data["pgc_amp"]).astype(float)
    rel = np.asarray(data["pgc_release_us"]).astype(float)
    loaded = np.asarray(data["pgc_atom_loaded"]).astype(int)
    retained = np.asarray(data["pgc_atom_retained"]).astype(int)
    m = loaded > 0
    f, a, rel, retained = f[m], a[m], rel[m], retained[m]
    table = {}
    for ff, aa, rr, ret in zip(f, a, rel, retained):
        key = (round(float(ff), 0), round(float(aa), 4), round(float(rr), 1))
        if key not in table:
            table[key] = [0, 0]
        table[key][0] += int(ret)
        table[key][1] += 1
    return table


def build_grids(data):
    """Return (fs, amps, M_recap, M_retention, rel_probe, best_fa, best_p).

    M_recap[i_amp, i_f]  = recapture at rel_probe (temperature proxy);
    M_retention[i_amp, i_f] = retention at release=0.  NaN where unmeasured.
    """
    table = recapture_table(data)
    fs = sorted({k[0] for k in table})
    amps = sorted({k[1] for k in table})
    rels = sorted({k[2] for k in table})
    rel_probe = max(rels) if rels else 0.0   # the non-zero release = temperature proxy

    M_rec = np.full((len(amps), len(fs)), np.nan)
    M_ret = np.full((len(amps), len(fs)), np.nan)
    best_fa, best_p = None, -1.0
    for i, a in enumerate(amps):
        for j, f in enumerate(fs):
            d0 = table.get((f, a, 0.0))
            dp = table.get((f, a, rel_probe))
            if d0 and d0[1] > 0:
                M_ret[i, j] = d0[0] / d0[1]
            if dp and dp[1] > 0:
                M_rec[i, j] = dp[0] / dp[1]
                if M_rec[i, j] > best_p:
                    best_p = M_rec[i, j]
                    best_fa = (f, a)
    return fs, amps, M_rec, M_ret, rel_probe, best_fa, best_p


def analyze(path, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, snap, rid = load_run(path)
    n_total = len(np.asarray(data.get("pgc_f", [])))
    n_loaded = int(np.sum(np.asarray(data.get("pgc_atom_loaded", [0])).astype(int)))
    print(f"RID {rid} | {n_total} points | {n_loaded} loaded")
    if n_loaded == 0:
        print("No loaded atoms -- nothing to analyze (check loading / atom_threshold).")
        return

    fs, amps, M_rec, M_ret, rel_probe, best_fa, best_p = build_grids(data)
    fs_mhz = [f / 1e6 for f in fs]

    fig, (axR, axT) = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)
    _heatmap(axR, fs_mhz, amps, M_rec, f"A. recapture @ {rel_probe:g} us (temperature proxy)",
             best_fa, plt)
    _heatmap(axT, fs_mhz, amps, M_ret, "B. retention @ 0 us (did PGC keep the atom)",
             None, plt)
    fig.suptitle(f"PGC low-trap scan -- RID {rid}", fontsize=13)

    if best_fa is not None:
        print(f"BEST low-trap PGC: f = {best_fa[0]/1e6:.4f} MHz, amp = {best_fa[1]:.3f}  "
              f"(recapture {best_p:.3f} at {rel_probe:g} us)")
        print(f"  baseline deep-PGC: f = {float(snap.get('f_cooling_deep_PGC', 0))/1e6:.4f} MHz, "
              f"amp = {snap.get('amp_cooling_deep_PGC', '?')}")

    if out_png is None:
        out_png = str(Path(path).with_name(f"pgc_lowtrap_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


def _heatmap(ax, fs_mhz, amps, M, title, best_fa, plt):
    im = ax.pcolormesh(fs_mhz, amps, M, shading="nearest", cmap="viridis",
                       vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if best_fa is not None:
        ax.plot(best_fa[0] / 1e6, best_fa[1], "*", color="red", ms=16,
                markeredgecolor="white", label="best")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("cooling-DP frequency (MHz)")
    ax.set_ylabel("cooling-DP amplitude")
    ax.set_title(title, fontsize=11)


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze(find_run(rid))
