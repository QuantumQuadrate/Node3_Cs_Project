"""Analyze ExcitationSurvivalScan runs: atom-survival ceiling + temperature proxy.

Reads a `*-ExcitationSurvivalScan.h5` and produces:
  - Panel A: survival vs total excitation cycles at release=0 (Wilson 68% CIs) ->
    read off the `max_excitation_cycles` ceiling (largest cycle count whose
    survival stays above a target). This is the "how many cycles does the atom
    survive" answer.
  - Panel B: recapture survival vs release time, one curve per cycle count ->
    the atom-temperature proxy (a longer release before recapture drops = colder
    atom). Reports the half-recapture release time per cycle count.

BLAS-free (sums + Wilson CIs only; no np.dot / np.linalg) so it runs in either
env, but PREFER the working-BLAS `python` on PATH (numpy 2.x / h5py / mpl 3.x);
matplotlib savefig crashes in the artiq7_claude env. Run:

    python analyze_excitation_survival.py            # latest run
    python analyze_excitation_survival.py 612        # RID 612
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import h5py
# matplotlib is imported lazily inside analyze() so that importing the BLAS-free
# data helpers (notebook use) does not force the headless "Agg" backend.

RESULTS_DIR = Path(
    r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes\repository\results"
)
SURVIVAL_TARGET = 0.95   # the "ceiling" = largest cycle count with survival >= this


# --------------------------------------------------------------------------
#  IO (BLAS-free, lock-free read)
# --------------------------------------------------------------------------
def find_run(rid=None):
    files = sorted(
        glob.glob(str(RESULTS_DIR / "**" / "*-ExcitationSurvivalScan.h5"), recursive=True),
        key=lambda f: int(os.path.basename(f).split("-")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No *-ExcitationSurvivalScan.h5 under {RESULTS_DIR}")
    if rid is None:
        return files[-1]
    for f in files:
        if int(os.path.basename(f).split("-")[0]) == int(rid):
            return f
    raise FileNotFoundError(f"No ExcitationSurvivalScan .h5 for RID {rid}")


def load_run(path):
    """Return (data dict of esv_* arrays, snapshot dict, rid)."""
    data, snap = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in ds.keys():
            if k.startswith("esv_"):
                v = ds[k][()]
                data[k] = np.atleast_1d(v) if hasattr(v, "__len__") else v
            elif k.startswith("snapshot_"):
                v = ds[k][()]
                snap[k[len("snapshot_"):]] = v.decode() if isinstance(v, bytes) else v
    rid = int(os.path.basename(path).split("-")[0])
    return data, snap, rid


def wilson(k, n, z=1.0):
    """Wilson score interval (z=1 -> ~68%). Returns (p, lo, hi). BLAS-free."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


# --------------------------------------------------------------------------
#  ANALYSIS
# --------------------------------------------------------------------------
def survival_table(data):
    """Filter loaded==1; return per-(cycles_per_group, release_us) (n_retained, n_loaded).

    Keys on esv_cpg (the scanned cycles_per_group). Falls back to esv_total_cycles
    for older runs that predate the cycles_per_group scan."""
    key_arr = data["esv_cpg"] if "esv_cpg" in data else data["esv_total_cycles"]
    cyc = np.asarray(key_arr).astype(int)
    rel = np.asarray(data["esv_release_us"]).astype(float)
    loaded = np.asarray(data["esv_atom_loaded"]).astype(int)
    retained = np.asarray(data["esv_atom_retained"]).astype(int)
    m = loaded > 0
    cyc, rel, retained = cyc[m], rel[m], retained[m]

    table = {}   # (cycles_per_group, release) -> [n_retained, n_loaded]
    for c, r, ret in zip(cyc, rel, retained):
        key = (int(c), round(float(r), 3))
        if key not in table:
            table[key] = [0, 0]
        table[key][0] += int(ret)
        table[key][1] += 1
    return table


def analyze(path, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, snap, rid = load_run(path)
    n_total = len(np.asarray(data.get("esv_cpg", data.get("esv_total_cycles", []))))
    n_loaded = int(np.sum(np.asarray(data.get("esv_atom_loaded", [0])).astype(int)))
    n_groups = int(snap.get("n_groups", 1))
    recool = snap.get("recool_mode", "?")
    print(f"RID {rid} | {n_total} points | {n_loaded} loaded | "
          f"n_groups={n_groups} | recool={recool}")
    if n_loaded == 0:
        print("No loaded atoms -- nothing to analyze (check loading / atom_threshold).")
        return

    table = survival_table(data)
    cycle_counts = sorted({c for (c, r) in table})
    releases = sorted({r for (c, r) in table})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # --- Panel A: survival vs cycles at release=0 (the ceiling) ---
    rel0 = releases[0] if 0.0 not in releases else 0.0
    xs, ps, los, his = [], [], [], []
    print(f"\nPanel A -- survival vs cycles_per_group at release={rel0:g} us:")
    ceiling = None
    for c in cycle_counts:
        kn = table.get((c, rel0))
        if not kn or kn[1] == 0:
            continue
        p, lo, hi = wilson(kn[0], kn[1])
        xs.append(c); ps.append(p)
        # clamp to >= 0 (Wilson is centred on its own centre, not p, so at p=0/1
        # the error-bar distance can be a tiny negative from FP -> errorbar rejects it).
        los.append(max(0.0, p - lo)); his.append(max(0.0, hi - p))
        print(f"  cycles_per_group={c:5d}  survival={p:.3f}  [{lo:.3f},{hi:.3f}]  (N={kn[1]})")
        if p >= SURVIVAL_TARGET:
            ceiling = c
    if xs:
        axA.errorbar(xs, ps, yerr=[los, his], fmt="o-", capsize=4, color="C0")
        axA.axhline(SURVIVAL_TARGET, ls="--", color="gray", lw=1,
                    label=f"target {SURVIVAL_TARGET:.0%}")
        if ceiling is not None:
            axA.axvline(ceiling, ls=":", color="C3", lw=1.5,
                        label=f"ceiling ≈ {ceiling} cpg")
        axA.legend(fontsize=9)
    axA.set_xlabel("cycles_per_group (cycles between recools)")
    axA.set_ylabel(f"survival (release={rel0:g} us)")
    axA.set_ylim(0, 1.05)
    axA.set_title(f"A. Survival vs cycles_per_group  (RID {rid})")
    if ceiling is not None:
        print(f"  -> max cycles_per_group with survival >= {SURVIVAL_TARGET:.0%}: "
              f"{ceiling}  (n_groups={n_groups} -> {ceiling * n_groups} total cycles)")

    # --- Panel B: recapture survival vs release per cycle count (temperature) ---
    print(f"\nPanel B -- recapture vs release (temperature proxy):")
    for c in cycle_counts:
        rx, ry = [], []
        for r in releases:
            kn = table.get((c, r))
            if kn and kn[1] > 0:
                rx.append(r); ry.append(kn[0] / kn[1])
        if len(rx) >= 2:
            axB.plot(rx, ry, "o-", label=f"{c} cpg", lw=1.5, ms=4)
            # half-recapture release time (temperature proxy)
            half = _half_release(rx, ry)
            ht = f"{half:.1f} us" if half is not None else "n/a"
            print(f"  cycles_per_group={c:5d}:  half-recapture release = {ht}")
    axB.axhline(0.5, ls="--", color="gray", lw=1)
    axB.set_xlabel("release time (us)")
    axB.set_ylabel("recapture survival")
    axB.set_ylim(0, 1.05)
    axB.set_title("B. Recapture vs release (colder = decays later)")
    axB.legend(fontsize=8, ncol=2)

    if out_png is None:
        out_png = str(Path(path).with_name(f"excitation_survival_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\nSaved figure: {out_png}")


def _half_release(rx, ry):
    """First release time where recapture crosses 0.5 (linear interp). BLAS-free."""
    for i in range(1, len(ry)):
        if ry[i - 1] >= 0.5 >= ry[i]:
            x0, x1, y0, y1 = rx[i - 1], rx[i], ry[i - 1], ry[i]
            if y1 == y0:
                return x1
            return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return None


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze(find_run(rid))
