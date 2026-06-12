"""Analyze PGCLowTrapScan runs taken in the FIXED-(f,amp), scan-(t_PGC x release)
mode: the low-trap PGC cooling curve T(t_PGC) and the survival-vs-t_PGC guard.

Reads a `*-PGCLowTrapScan.h5` and produces:
  - Panel A: recapture vs release time, one curve per t_PGC -> the temperature at
    each PGC duration (a curve that decays LATER = colder atom). Reports the
    half-recapture release time per t_PGC = the temperature proxy.
  - Panel B: half-recapture release (temperature proxy, higher = colder) vs t_PGC
    AND retention@0 vs t_PGC -> the cooling 'knee' (shortest t_PGC that reaches the
    cold floor) and the loss guard (retention@0 falling = atom LOST, not cold).

BLAS-free (sums + Wilson CIs only) so it runs in either env, but PREFER the
working-BLAS `python` on PATH (numpy 2.x / h5py / mpl 3.x); matplotlib savefig
crashes in the artiq7_claude env. Run:

    python analyze_pgc_time_release.py            # latest run
    python analyze_pgc_time_release.py 642        # RID 642
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
KNEE_FRAC = 0.90   # knee = smallest t_PGC whose temperature proxy >= this * its max


# --------------------------------------------------------------------------
#  IO (BLAS-free, lock-free read)
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
def tpgc_release_table(data):
    """Filter loaded==1; return {(t_pgc_us, release_us): [n_retained, n_loaded]}."""
    tp = np.asarray(data["pgc_t_pgc_us"]).astype(float)
    rel = np.asarray(data["pgc_release_us"]).astype(float)
    loaded = np.asarray(data["pgc_atom_loaded"]).astype(int)
    retained = np.asarray(data["pgc_atom_retained"]).astype(int)
    m = loaded > 0
    tp, rel, retained = tp[m], rel[m], retained[m]
    table = {}
    for t, r, ret in zip(tp, rel, retained):
        key = (round(float(t), 1), round(float(r), 1))
        if key not in table:
            table[key] = [0, 0]
        table[key][0] += int(ret)
        table[key][1] += 1
    return table


def _half_release(rx, ry):
    """First release time where recapture crosses 0.5 (linear interp). BLAS-free.
    Higher value = colder atom (recapture stays high to longer release)."""
    for i in range(1, len(ry)):
        if ry[i - 1] >= 0.5 >= ry[i]:
            x0, x1, y0, y1 = rx[i - 1], rx[i], ry[i - 1], ry[i]
            if y1 == y0:
                return x1
            return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return None


def _curve(table, t, releases):
    """Recapture-vs-release (rx, ry) for one t_PGC."""
    rx, ry = [], []
    for r in releases:
        kn = table.get((t, r))
        if kn and kn[1] > 0:
            rx.append(r); ry.append(kn[0] / kn[1])
    return rx, ry


def analyze(path, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, snap, rid = load_run(path)
    n_total = len(np.asarray(data.get("pgc_t_pgc_us", [])))
    n_loaded = int(np.sum(np.asarray(data.get("pgc_atom_loaded", [0])).astype(int)))
    print(f"RID {rid} | {n_total} points | {n_loaded} loaded")
    if n_loaded == 0:
        print("No loaded atoms -- nothing to analyze (check loading / atom_threshold).")
        return

    table = tpgc_release_table(data)
    tpgcs = sorted({t for (t, r) in table})
    releases = sorted({r for (t, r) in table})

    # report the (fixed) f/amp actually present
    fvals = sorted({round(float(x), 0) for x in np.asarray(data["pgc_f"]).astype(float)})
    avals = sorted({round(float(x), 4) for x in np.asarray(data["pgc_amp"]).astype(float)})
    print(f"  f (MHz): {[round(x/1e6, 4) for x in fvals]}   amp: {avals}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # --- Panel A: recapture vs release, one curve per t_PGC ---
    print("\nPanel A -- recapture vs release per t_PGC (temperature proxy):")
    halves = []
    for t in tpgcs:
        rx, ry = _curve(table, t, releases)
        if len(rx) >= 2:
            axA.plot(rx, ry, "o-", label=f"{t:.0f} us", lw=1.5, ms=4)
            half = _half_release(rx, ry)
            halves.append((t, half))
            ht = f"{half:.1f} us" if half is not None else "n/a"
            print(f"  t_PGC={t:7.0f} us:  half-recapture release = {ht}")
        else:
            halves.append((t, None))
    axA.axhline(0.5, ls="--", color="gray", lw=1)
    axA.set_xlabel("release time (us)")
    axA.set_ylabel("recapture survival")
    axA.set_ylim(0, 1.05)
    axA.set_title(f"A. Recapture vs release per t_PGC (RID {rid})")
    axA.legend(fontsize=8, ncol=2, title="t_PGC")

    # --- Panel B: temperature proxy + retention@0 vs t_PGC ---
    tt = [t for (t, h) in halves if h is not None]
    hh = [h for (t, h) in halves if h is not None]
    r0t, r0p, r0lo, r0hi = [], [], [], []
    for t in tpgcs:
        kn = table.get((t, 0.0))
        if kn and kn[1] > 0:
            p, lo, hi = wilson(kn[0], kn[1])
            r0t.append(t); r0p.append(p)
            r0lo.append(max(0.0, p - lo)); r0hi.append(max(0.0, hi - p))

    if tt:
        axB.plot(tt, hh, "s-", color="C0", lw=1.7, ms=6,
                 label="half-recapture release (colder = higher)")
    axB.set_xlabel("t_PGC (us)")
    axB.set_ylabel("half-recapture release (us)  [temperature proxy]", color="C0")
    axB.tick_params(axis="y", labelcolor="C0")
    axR = axB.twinx()
    if r0t:
        axR.errorbar(r0t, r0p, yerr=[r0lo, r0hi], fmt="o--", color="C3", capsize=3,
                     lw=1.2, label="retention@0 (survival)")
    axR.set_ylabel("retention@0 (survival)", color="C3")
    axR.set_ylim(0, 1.05)
    axR.tick_params(axis="y", labelcolor="C3")
    lns, lbs = axB.get_legend_handles_labels()
    l2, lb2 = axR.get_legend_handles_labels()
    axB.legend(lns + l2, lbs + lb2, fontsize=8, loc="lower right")
    axB.set_title("B. Cooling knee + survival vs t_PGC")

    print("\nPanel B -- retention@0 vs t_PGC (loss guard; a drop = atom LOST, not cold):")
    for i in range(len(r0t)):
        print(f"  t_PGC={r0t[i]:7.0f} us:  retention@0 = {r0p[i]:.3f}")

    # knee = smallest t_PGC whose temperature proxy >= KNEE_FRAC * its max
    if hh:
        hmax = max(hh)
        knee = None
        for t, h in zip(tt, hh):
            if h >= KNEE_FRAC * hmax:
                knee = t
                break
        print(f"\nMax half-recapture release = {hmax:.1f} us; "
              f"knee (>= {KNEE_FRAC:.0%} of max) at t_PGC = {knee} us "
              f"-- the shortest PGC time that still reaches ~the cold floor.")

    if out_png is None:
        out_png = str(Path(path).with_name(f"pgc_time_release_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\nSaved figure: {out_png}")


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze(find_run(rid))
