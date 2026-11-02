"""Rate-experiment analysis with CORRECTED herald accounting.

The naive herald probability (total clicks / total excitation attempts) is biased
LOW, because the attempt count of the first herald after a (re)load bundles in the
unproductive attempts around atom turnover -- the dying old atom + the gap before
the periodic survival check notices the loss + the reload. Those zero-herald
attempts don't belong to a live atom's Bernoulli process.

This module separates the two cleanly using `how_many_atoms` (loads since the
previous click):
  - how_many_atoms == 0  -> REUSED: the click is on the SAME atom as the previous
    one, so its attempts are pure live-atom attempts (a clean sample).
  - how_many_atoms >= 1  -> FRESH: a reload happened since the last click, so the
    window spans a turnover and its attempt count is inflated.

Honest per-attempt herald probability on a LIVE atom:
    p_live = (# reused clicks) / (sum of trials over reused clicks)
This matches the parity experiment's per-attempt efficiency; the naive aggregate
does not.

It also reports the per-herald-position breakdown (1st herald on an atom, 2nd,
...) and the turnover overhead (extra attempts charged to each fresh herald) --
the latter quantifies the real wasted time that adaptive loss detection would
recover (a later, separate task; nothing here changes the experiment).

Usage::

    python analyze_rate_corrected.py                 # latest RateExperiment run
    python analyze_rate_corrected.py 3513            # a specific RID
    python analyze_rate_corrected.py path/to/run.h5  # explicit file
"""

import sys, os, glob, argparse
from pathlib import Path

import numpy as np
import h5py

RESULTS_DIR = Path(
    r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes\repository\results"
)


# =========================================================================
#  DISCOVERY / LOADING
# =========================================================================

def _rid_of(path):
    try:
        return int(os.path.basename(path).split("-")[0])
    except Exception:
        return -1


def find_runs():
    files = glob.glob(str(RESULTS_DIR / "**" / "*-RateExperiment.h5"), recursive=True)
    return sorted(files, key=_rid_of)


def load_run(path):
    """Load a RateExperiment .h5 into a dict of arrays + duration."""
    with h5py.File(path, "r", locking=False) as hf:
        def ds(k):
            return np.asarray(hf[f"datasets/{k}"][()])
        d = {
            "trials_per_click": ds("trials_per_click").astype(float),
            "how_many_atoms": ds("how_many_atoms").astype(float),
            "click_times_s": (ds("click_times_s").astype(float)
                              if "datasets/click_times_s" in hf else None),
            "click_detector": (ds("click_detector").astype(int)
                               if "datasets/click_detector" in hf else None),
        }
        dur = np.nan
        if "metadata" in hf and "duration_seconds" in hf["metadata"].attrs:
            dur = float(hf["metadata"].attrs["duration_seconds"])
    # Fallback duration from the click-time span.
    if not np.isfinite(dur) and d["click_times_s"] is not None and len(d["click_times_s"]) > 1:
        t = d["click_times_s"]
        dur = float(np.nanmax(t) - np.nanmin(t))
    d["duration_s"] = dur
    d["rid"] = _rid_of(path)
    d["file"] = Path(path).name
    return d


# =========================================================================
#  CORRECTED HERALD ACCOUNTING
# =========================================================================

def _herald_position(hma):
    """Position of each click within its atom's herald run: 1 for the first
    herald after a (re)load, 2 for the next on the same atom, etc."""
    pos = np.empty(len(hma), dtype=int)
    c = 0
    for i in range(len(hma)):
        c = 1 if hma[i] >= 1 else c + 1
        pos[i] = c
    return pos


def herald_stats(d):
    """Return naive vs corrected (live-atom) herald probability and diagnostics."""
    tpc = d["trials_per_click"]
    hma = d["how_many_atoms"]
    n = len(tpc)
    dur = d.get("duration_s", np.nan)

    fresh = hma >= 1          # first herald after a (re)load (turnover-contaminated)
    reused = ~fresh           # subsequent herald on the same atom (clean, live)

    def _p(mask):
        k = int(mask.sum())
        a = float(tpc[mask].sum())
        return k, a, (k / a if a > 0 else np.nan), (a / k if k > 0 else np.nan)

    nf, af, pf, mtf = _p(fresh)
    nr, ar, pr, mtr = _p(reused)
    tot_att = float(tpc.sum())
    n_loads = float(hma.sum())

    pos = _herald_position(hma)
    by_pos = []
    for k in range(1, int(pos.max()) + 1 if n else 1):
        m = pos == k
        if m.sum() > 0:
            mt = float(tpc[m].mean())
            by_pos.append({"pos": k, "n": int(m.sum()), "mean_trials": mt,
                           "p": (1.0 / mt if mt > 0 else np.nan)})

    return {
        "file": d.get("file"), "rid": d.get("rid"),
        "n_clicks": n, "total_attempts": tot_att, "duration_s": dur,
        "clicks_per_s": (n / dur if dur and np.isfinite(dur) else np.nan),
        "attempts_per_s": (tot_att / dur if dur and np.isfinite(dur) else np.nan),
        "p_naive": (n / tot_att if tot_att > 0 else np.nan),   # biased low
        "p_live": pr,                                          # HONEST per-attempt p
        "p_fresh": pf,                                         # turnover-contaminated
        "n_fresh": nf, "n_reused": nr,
        "mean_trials_fresh": mtf, "mean_trials_live": mtr,
        "n_loads": int(n_loads),
        "clicks_per_load": (n / n_loads if n_loads > 0 else np.nan),
        # extra attempts charged to each turnover (fresh window minus a clean one):
        "turnover_extra_attempts": (mtf - mtr if np.isfinite(mtf) and np.isfinite(mtr) else np.nan),
        "by_position": by_pos,
        "pos": pos,
    }


# =========================================================================
#  REPORTING
# =========================================================================

def print_summary(s):
    print(f"File: {s['file']}   RID={s['rid']}   duration={s['duration_s']:.1f} s")
    print(f"  clicks={s['n_clicks']}   attempts={int(s['total_attempts'])}   "
          f"loads={s['n_loads']}   clicks/load={s['clicks_per_load']:.2f}")
    if np.isfinite(s["clicks_per_s"]):
        print(f"  clicks/s={s['clicks_per_s']:.3f}   attempts/s={s['attempts_per_s']:.0f}")
    print()
    print(f"  herald p  (NAIVE, clicks/attempts)       = {s['p_naive']*100:.3f} %   <- biased LOW")
    print(f"  herald p  (LIVE atom, reused heralds)    = {s['p_live']*100:.3f} %   <- honest per-attempt")
    print(f"  herald p  (FRESH, first-after-load)      = {s['p_fresh']*100:.3f} %   "
          f"(turnover-inflated: {s['mean_trials_fresh']:.0f} vs {s['mean_trials_live']:.0f} attempts)")
    print(f"  turnover overhead ~ {s['turnover_extra_attempts']:.0f} extra attempts per reload "
          f"({s['n_fresh']} reloads-with-herald of {s['n_clicks']} clicks)")
    print("\n  per-herald-position on the same atom (flat beyond #1 => recool is fine):")
    print(f"    {'herald #':>8}  {'N':>5}  {'mean attempts':>13}  {'p %':>7}")
    for r in s["by_position"][:8]:
        print(f"    {r['pos']:>8}  {r['n']:>5}  {r['mean_trials']:>13.1f}  {r['p']*100:>7.3f}")


# =========================================================================
#  PLOTS
# =========================================================================

def plot_herald(s, d, out=None):
    """Two panels: (left) mean attempts per herald-position (bar), (right)
    histogram of trials_per_click split FRESH vs REUSED."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    bp = s["by_position"]
    xs = [r["pos"] for r in bp]
    mt = [r["mean_trials"] for r in bp]
    ax1.bar(xs, mt, color="C0", alpha=0.85, edgecolor="black", lw=0.5)
    ax1.axhline(s["mean_trials_live"], color="C3", ls="--", lw=1.5,
                label=f"live-atom mean = {s['mean_trials_live']:.0f}")
    ax1.set_xlabel("herald # on the same atom")
    ax1.set_ylabel("mean excitation attempts")
    ax1.set_title("Attempts per herald vs position\n(#1 inflated by turnover)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, axis="y")

    tpc = d["trials_per_click"]; hma = d["how_many_atoms"]
    fresh = tpc[hma >= 1]; reused = tpc[hma == 0]
    hi = float(np.percentile(tpc, 99)) if len(tpc) else 1.0
    bins = np.linspace(0, max(hi, 1.0), 40)
    ax2.hist(reused, bins=bins, alpha=0.6, color="C2", label=f"reused (live), N={reused.size}")
    ax2.hist(fresh, bins=bins, alpha=0.6, color="C1", label=f"fresh (turnover), N={fresh.size}")
    ax2.set_xlabel("attempts between heralds (trials_per_click)")
    ax2.set_ylabel("count")
    ax2.set_title("Clean live-atom windows vs\nturnover-contaminated windows")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle(f"RateExperiment RID {s['rid']}  |  "
                 f"honest herald p = {s['p_live']*100:.3f}%  (naive {s['p_naive']*100:.3f}%)",
                 fontsize=12)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140); print(f"saved {out}")
    else:
        plt.show()


# =========================================================================
#  CLI
# =========================================================================

def _resolve_path(arg):
    if arg is None:
        files = find_runs()
        if not files:
            sys.exit(f"No RateExperiment result files under {RESULTS_DIR}.")
        print(f"(Using latest run: {Path(files[-1]).name})")
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
        sys.exit(f"No RateExperiment run with RID {rid}.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    path = _resolve_path(args.run)
    d = load_run(path)
    s = herald_stats(d)
    print_summary(s)
    if not args.no_plot:
        plot_herald(s, d)


if __name__ == "__main__":
    main()
