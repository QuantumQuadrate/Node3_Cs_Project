"""Analyze RateExperiment runs: photon-click rate + excitation-attempt rate.

Reads a `*-RateExperiment.h5` (one row PER heralded photon click) and reports the
throughput numbers used to quantify the PGC-low-trap / imaging-low-trap /
optical-pump-time speed-up:
  - click rate     = n_clicks / duration              [clicks / s]
  - attempt rate   = sum(trials_per_click) / duration [excitation attempts / s]
  - load rate      = sum(how_many_atoms) / duration   [atom-load attempts / s]
  - herald prob    = n_clicks / total_attempts        [clicks per attempt, Wilson CI]
  - per-detector SPCM1 / SPCM2 click split.

`compare(rid_before, rid_after)` prints the speed-up factors between a high-trap
BASELINE run and a low-trap/OP-optimized run.

Dataset convention (from Rate_measurement.py, one row per click):
  click_times_s    -- click time in s since run start (RTIO-based)
  click_detector   -- 1 = SPCM1, 2 = SPCM2
  trials_per_click -- excitation attempts since the previous click (sum = total attempts)
  how_many_atoms   -- atom-load attempts since the previous click (sum = total loads)
duration is taken from the HDF5 metadata (duration_seconds), falling back to the
last click time if absent.

BLAS-free (sums + Wilson only); matplotlib imported lazily (Agg). Run:
    python analyze_rate.py              # latest run summary + figure
    python analyze_rate.py 661          # RID 661
    python analyze_rate.py 660 661      # compare baseline 660 vs optimized 661
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import h5py
# matplotlib imported lazily inside analyze() so notebook use of the data helpers
# doesn't force the headless "Agg" backend.

RESULTS_DIR = Path(
    r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes\repository\results"
)


# --------------------------------------------------------------------------
#  IO
# --------------------------------------------------------------------------
def find_run(rid=None):
    files = sorted(
        glob.glob(str(RESULTS_DIR / "**" / "*-RateExperiment.h5"), recursive=True),
        key=lambda f: int(os.path.basename(f).split("-")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No *-RateExperiment.h5 under {RESULTS_DIR}")
    if rid is None:
        return files[-1]
    for f in files:
        if int(os.path.basename(f).split("-")[0]) == int(rid):
            return f
    raise FileNotFoundError(f"No RateExperiment .h5 for RID {rid}")


def load_run(path):
    """Return (data dict of click arrays, metadata dict, rid)."""
    data, meta = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in ("click_times_s", "click_detector", "trials_per_click", "how_many_atoms"):
            if k in ds:
                data[k] = np.atleast_1d(ds[k][()]).astype(float)
            else:
                data[k] = np.array([])
        if "metadata" in hf:
            for k, v in hf["metadata"].attrs.items():
                meta[k] = v.decode() if isinstance(v, bytes) else v
    rid = int(os.path.basename(path).split("-")[0])
    return data, meta, rid


def wilson(k, n, z=1.0):
    """Wilson score interval (z=1 ~ 68%). Returns (p, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------------------
#  SUMMARY
# --------------------------------------------------------------------------
def summarize(path, verbose=True):
    """Compute the rate metrics for one run. Returns a dict (or None if no clicks)."""
    data, meta, rid = load_run(path)
    ts = data["click_times_s"]
    det = data["click_detector"]
    trials = data["trials_per_click"]
    loads = data["how_many_atoms"]
    n_clicks = int(ts.size)
    if n_clicks == 0:
        if verbose:
            print(f"RID {rid}: no clicks recorded.")
        return None

    # duration: prefer the wall-clock metadata; fall back to the last click time.
    duration = float(meta.get("duration_seconds", 0.0))
    dur_src = "metadata"
    if duration <= 0.0:
        duration = float(np.max(ts))
        dur_src = "last-click (RTIO)"

    total_attempts = int(np.sum(trials))
    total_loads = int(np.sum(loads))
    n1 = int(np.sum(det == 1))
    n2 = int(np.sum(det == 2))

    clicks_per_s = n_clicks / duration if duration > 0 else float("nan")
    attempts_per_s = total_attempts / duration if duration > 0 else float("nan")
    loads_per_s = total_loads / duration if duration > 0 else float("nan")
    click_rate_err = (np.sqrt(n_clicks) / duration) if duration > 0 else float("nan")  # Poisson
    p, plo, phi = wilson(n_clicks, total_attempts)

    res = dict(
        rid=rid, duration=duration, dur_src=dur_src,
        n_clicks=n_clicks, n_spcm1=n1, n_spcm2=n2,
        total_attempts=total_attempts, total_loads=total_loads,
        clicks_per_s=clicks_per_s, click_rate_err=click_rate_err,
        attempts_per_s=attempts_per_s, loads_per_s=loads_per_s,
        herald_prob=p, herald_lo=plo, herald_hi=phi,
        atoms_per_click=(total_loads / n_clicks if n_clicks else float("nan")),
    )

    if verbose:
        print(f"RID {rid}  |  {n_clicks} clicks (SPCM1 {n1} / SPCM2 {n2})  |  "
              f"duration {duration:.1f} s ({dur_src})")
        print(f"  click rate    = {clicks_per_s:.4f} +/- {click_rate_err:.4f} clicks/s")
        print(f"  attempt rate  = {attempts_per_s:.1f} attempts/s  ({total_attempts} total)")
        print(f"  load rate     = {loads_per_s:.3f} loads/s  ({total_loads} total loads, "
              f"{res['atoms_per_click']:.2f} loads/click)")
        print(f"  herald prob   = {100.0 * p:.4f} %  "
              f"[{100.0 * plo:.4f}, {100.0 * phi:.4f}]  (clicks/attempt)")
        print(f"  mean attempts between clicks = {(total_attempts / n_clicks):.1f}")
    return res


def compare(rid_before, rid_after):
    """Print baseline (before) vs optimized (after) and the speed-up factors."""
    print("=" * 70)
    print(f"BASELINE  RID {rid_before}")
    a = summarize(find_run(rid_before))
    print("-" * 70)
    print(f"OPTIMIZED RID {rid_after}")
    b = summarize(find_run(rid_after))
    if a is None or b is None:
        print("Cannot compare (a run had no clicks).")
        return
    print("=" * 70)

    def _ratio(x, y):
        return (y / x) if (x and x == x and x != 0) else float("nan")

    print("SPEED-UP (optimized / baseline):")
    print(f"  click rate    : {a['clicks_per_s']:.4f} -> {b['clicks_per_s']:.4f} clicks/s   "
          f"x{_ratio(a['clicks_per_s'], b['clicks_per_s']):.2f}")
    print(f"  attempt rate  : {a['attempts_per_s']:.1f} -> {b['attempts_per_s']:.1f} attempts/s   "
          f"x{_ratio(a['attempts_per_s'], b['attempts_per_s']):.2f}")
    print(f"  herald prob   : {100*a['herald_prob']:.4f} % -> {100*b['herald_prob']:.4f} %   "
          f"x{_ratio(a['herald_prob'], b['herald_prob']):.2f}  (should be ~1: physics unchanged)")
    print(f"  load rate     : {a['loads_per_s']:.3f} -> {b['loads_per_s']:.3f} loads/s   "
          f"x{_ratio(a['loads_per_s'], b['loads_per_s']):.2f}")
    return a, b


# --------------------------------------------------------------------------
#  FIGURE
# --------------------------------------------------------------------------
def analyze(path, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = summarize(path)
    if res is None:
        return
    data, meta, rid = load_run(path)
    ts = np.sort(data["click_times_s"])
    trials = data["trials_per_click"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # Panel A: cumulative clicks vs time (slope = click rate; straight = steady)
    cum = np.arange(1, ts.size + 1)
    axA.step(ts, cum, where="post", color="C0", lw=1.5)
    axA.plot([0, res["duration"]], [0, res["clicks_per_s"] * res["duration"]],
             "--", color="C3", lw=1,
             label=f"mean {res['clicks_per_s']:.3f} clicks/s")
    axA.set_xlabel("time since run start (s)")
    axA.set_ylabel("cumulative clicks")
    axA.set_title(f"A. Cumulative clicks vs time  (RID {rid})")
    axA.legend(loc="upper left", fontsize=9)
    axA.grid(alpha=0.3)

    # Panel B: distribution of excitation attempts between clicks (~geometric)
    axB.hist(trials, bins=min(40, max(5, trials.size // 3)), color="C1",
             alpha=0.8, edgecolor="black", lw=0.4)
    axB.axvline(float(np.mean(trials)), color="C3", ls="--", lw=1.2,
                label=f"mean {np.mean(trials):.1f} attempts/click")
    axB.set_xlabel("excitation attempts between clicks (trials_per_click)")
    axB.set_ylabel("count")
    axB.set_title("B. Attempts per click  (1/herald-prob)")
    axB.legend(loc="upper right", fontsize=9)
    axB.grid(alpha=0.3)

    fig.suptitle(
        f"RateExperiment RID {rid}  |  {res['clicks_per_s']:.3f} clicks/s  |  "
        f"{res['attempts_per_s']:.0f} attempts/s  |  herald {100*res['herald_prob']:.3f}%",
        fontsize=13,
    )

    if out_png is None:
        out_png = str(Path(path).with_name(f"rate_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    if len(args) >= 2:
        compare(args[0], args[1])
    else:
        rid = args[0] if args else None
        analyze(find_run(rid))
