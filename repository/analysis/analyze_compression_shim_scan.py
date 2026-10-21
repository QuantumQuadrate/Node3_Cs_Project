"""Compression-shim 4D scan analysis: loading & release-recapture survival.

Reads a CompressionShimScan run and, per (X, Y, Z1, Z2) compression-shim point:
  - loading probability   = fraction of trials that trapped an atom (cap_loaded)
  - survival probability  = fraction of the LOADED trials that survived the
                            release-and-recapture (cap_survived / cap_loaded)

Higher R&R survival = colder / better-held atoms, so the best shim setting is the
one that MAXIMIZES survival (temperature probe), among points that actually load
often enough to trust the number.

Deliverables:
  - a 3D map (X, Y, Z1 axes; color = value) of loading probability AND of R&R
    survival probability, one figure per Z2 scan value;
  - the (X, Y, Z1, Z2) point with the MAXIMUM survival probability over the whole
    4D scan (optionally restricted to points with enough loads).

Usage::

    python analyze_compression_shim_scan.py                 # latest run
    python analyze_compression_shim_scan.py 1234            # a specific RID
    python analyze_compression_shim_scan.py <file> --min-load-prob 0.3 --min-loads 10
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
    files = glob.glob(str(RESULTS_DIR / "**" / "*CompressionShimScan.h5"), recursive=True)
    return sorted(files, key=_rid_of)


def load_run(path):
    """Load a CompressionShimScan .h5 into a dict of arrays."""
    with h5py.File(path, "r", locking=False) as hf:
        def ds(k):
            return np.asarray(hf[f"datasets/{k}"][()])
        d = {
            "X": ds("cap_X").astype(float),
            "Y": ds("cap_Y").astype(float),
            "Z1": ds("cap_Z1").astype(float),
            "Z2": ds("cap_Z2").astype(float),
            "loaded": ds("cap_loaded").astype(float),
            "survived": ds("cap_survived").astype(float),
        }
        d["load_time_s"] = (ds("cap_load_time_s").astype(float)
                            if "datasets/cap_load_time_s" in hf else np.zeros_like(d["loaded"]))
        d["t_release_us"] = (float(np.asarray(hf["datasets/cap_t_release_us"][()]).reshape(-1)[0])
                             if "datasets/cap_t_release_us" in hf else float("nan"))
    d["rid"] = _rid_of(path)
    d["file"] = Path(path).name
    return d


# =========================================================================
#  AGGREGATION
# =========================================================================

def aggregate(d, round_dec=3):
    """Group trials by (X, Y, Z1, Z2). Returns parallel arrays:
    X, Y, Z1, Z2, n_trials, n_loaded, n_survived, load_prob,
    survival_prob (NaN if never loaded)."""
    keys = np.stack([np.round(d["X"], round_dec), np.round(d["Y"], round_dec),
                     np.round(d["Z1"], round_dec), np.round(d["Z2"], round_dec)], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    n = len(uniq)
    n_trials = np.zeros(n)
    n_loaded = np.zeros(n)
    n_surv = np.zeros(n)
    sum_lt = np.zeros(n)   # sum of loading time over loaded trials
    for i in range(len(inv)):
        g = inv[i]
        n_trials[g] += 1
        if d["loaded"][i] > 0.5:
            n_loaded[g] += 1
            sum_lt[g] += d["load_time_s"][i]
            if d["survived"][i] > 0.5:
                n_surv[g] += 1

    load_prob = np.where(n_trials > 0, n_loaded / np.maximum(n_trials, 1), 0.0)
    survival_prob = np.where(n_loaded > 0, n_surv / np.maximum(n_loaded, 1), np.nan)
    load_time = np.where(n_loaded > 0, sum_lt / np.maximum(n_loaded, 1), np.nan)

    return {
        "X": uniq[:, 0], "Y": uniq[:, 1], "Z1": uniq[:, 2], "Z2": uniq[:, 3],
        "n_trials": n_trials, "n_loaded": n_loaded, "n_survived": n_surv,
        "load_prob": load_prob, "survival_prob": survival_prob,
        "load_time": load_time,   # mean deterministic-loading time (s), loaded trials
    }


def find_max_survival(agg, min_load_prob=0.0, min_loads=1):
    """Aggregated point with the HIGHEST R&R survival probability, among points
    with load probability >= min_load_prob and n_loaded >= min_loads."""
    ok = (agg["load_prob"] >= min_load_prob) & (agg["n_loaded"] >= min_loads) \
        & np.isfinite(agg["survival_prob"])
    if not np.any(ok):
        return None
    idx_ok = np.where(ok)[0]
    j = idx_ok[int(np.argmax(agg["survival_prob"][idx_ok]))]
    return {
        "X": float(agg["X"][j]), "Y": float(agg["Y"][j]),
        "Z1": float(agg["Z1"][j]), "Z2": float(agg["Z2"][j]),
        "survival_prob": float(agg["survival_prob"][j]),
        "load_prob": float(agg["load_prob"][j]),
        "n_loaded": int(agg["n_loaded"][j]),
        "n_survived": int(agg["n_survived"][j]),
        "n_trials": int(agg["n_trials"][j]),
    }


def _pack_point(agg, j):
    return {
        "X": float(agg["X"][j]), "Y": float(agg["Y"][j]),
        "Z1": float(agg["Z1"][j]), "Z2": float(agg["Z2"][j]),
        "load_prob": float(agg["load_prob"][j]),
        "survival_prob": float(agg["survival_prob"][j]),
        "load_time": float(agg["load_time"][j]),
        "n_loaded": int(agg["n_loaded"][j]),
        "n_survived": int(agg["n_survived"][j]),
        "n_trials": int(agg["n_trials"][j]),
    }


def recommend(agg, min_load_prob=0.90, min_loads=1):
    """Gate on loading rate, then give three picks, each with a 0-100 grade.

    Only grid points whose LOADING probability is >= min_load_prob (default 90%)
    are eligible. If none qualify, returns qualified=False (caller prints
    'every point is bad').

    Grading (linear, over the qualifying points' data range):
      grade_time     : fastest loading time -> 100, slowest -> 0
                       grade = 100 * (t_max - t) / (t_max - t_min)
      grade_survival : highest R&R survival -> 100, lowest -> 0  (high = good)
                       grade = 100 * (s - s_min) / (s_max - s_min)
      grade_both     : grade_time + grade_survival (each out of 100 -> 0..200)

    Picks:
      best_time     -> highest grade_time     (fastest)
      best_survival -> highest grade_survival (coldest / best R&R)
      best_both     -> highest grade_both
    Each returned point dict carries grade_time / grade_survival / grade_both.
    """
    ok = (agg["load_prob"] >= min_load_prob) & (agg["n_loaded"] >= min_loads) \
        & np.isfinite(agg["survival_prob"]) & np.isfinite(agg["load_time"])
    idx = np.where(ok)[0]
    out = {"min_load_prob": min_load_prob, "n_qualified": int(idx.size),
           "qualified": bool(idx.size > 0), "grading": None,
           "best_time": None, "best_survival": None, "best_both": None}
    if idx.size == 0:
        return out

    surv = agg["survival_prob"][idx]
    time = agg["load_time"][idx]
    t_min, t_max = float(time.min()), float(time.max())
    s_min, s_max = float(surv.min()), float(surv.max())

    def grade_time(t):    # fast (low t) -> 100
        return 100.0 if t_max <= t_min else 100.0 * (t_max - t) / (t_max - t_min)

    def grade_surv(s):    # high survival -> 100
        return 100.0 if s_max <= s_min else 100.0 * (s - s_min) / (s_max - s_min)

    def pack(j):
        p = _pack_point(agg, j)
        p["grade_time"] = grade_time(p["load_time"])
        p["grade_survival"] = grade_surv(p["survival_prob"])
        p["grade_both"] = p["grade_time"] + p["grade_survival"]
        return p

    gboth = np.array([grade_time(t) + grade_surv(s) for t, s in zip(time, surv)])
    out["best_time"] = pack(idx[int(np.argmin(time))])
    out["best_survival"] = pack(idx[int(np.argmax(surv))])
    out["best_both"] = pack(idx[int(np.argmax(gboth))])
    out["grading"] = {"t_min_ms": t_min * 1e3, "t_max_ms": t_max * 1e3,
                      "s_min": s_min, "s_max": s_max}
    return out


def pick_case(rec, case):
    """Select the optimum for a case: 1=fastest, 2=best R&R survival, 3=both.
    Returns the point dict (with its scan values + grades), or None if no point
    passed the loading gate."""
    if not rec.get("qualified"):
        return None
    return {1: rec["best_time"], 2: rec["best_survival"],
            3: rec["best_both"]}.get(int(case))


# =========================================================================
#  GAUSSIAN 1D CUTS (loading time & survival vs each shim axis)
# =========================================================================

def _fit_gaussian(x, y, peak=True):
    """Fit y = a*exp(-(x-mu)^2/2 sigma^2) + c (BLAS-free least_squares).
    peak=True -> expect a maximum (survival); peak=False -> a dip (loading time).
    Returns dict with mu/sigma/xs/ys, or None if too few points."""
    from scipy.optimize import least_squares
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 4 or np.ptp(x) == 0:
        return None
    c0 = float(y.min() if peak else y.max())
    a0 = float((y.max() - y.min()) if peak else (y.min() - y.max()))
    mu0 = float(x[np.argmax(y)] if peak else x[np.argmin(y)])
    sig0 = max(float(np.ptp(x)) / 4.0, 1e-6)

    def resid(p):
        a, mu, sig, c = p
        sig = max(abs(sig), 1e-9)
        return a * np.exp(-(x - mu) ** 2 / (2 * sig * sig)) + c - y

    try:
        r = least_squares(resid, [a0, mu0, sig0, c0], method="lm", max_nfev=5000)
        a, mu, sig, c = (float(v) for v in r.x)
        sig = abs(sig)
        xs = np.linspace(float(x.min()), float(x.max()), 200)
        ys = a * np.exp(-(xs - mu) ** 2 / (2 * sig * sig)) + c
        return {"a": a, "mu": float(mu), "sigma": sig, "c": c, "xs": xs, "ys": ys}
    except Exception:
        return None


def plot_axis_gaussians(agg, center, out=None):
    """For the chosen optimum `center` (a point dict from pick_case), plot 1D
    cuts along each shim axis: fix the other three shims at the optimum and vary
    one, showing loading rate (Gaussian peak), loading time (Gaussian dip) and
    R&R survival (Gaussian peak), each with a Gaussian fit. Red dashed line = the
    chosen optimum on that axis."""
    import matplotlib.pyplot as plt
    if center is None:
        print("No optimum to cut through (no point passed the loading gate).")
        return

    axes = [("X", "shim X"), ("Y", "shim Y"), ("Z1", "shim Z1"), ("Z2", "shim Z2")]
    fig, axS = plt.subplots(len(axes), 3, figsize=(15, 3.0 * len(axes)))
    for i, (key, xlabel) in enumerate(axes):
        others = [a for a in ("X", "Y", "Z1", "Z2") if a != key]
        mask = np.ones(len(agg["X"]), dtype=bool)
        for o in others:
            mask &= np.isclose(agg[o], center[o], atol=1e-6)
        xv = agg[key][mask]
        order = np.argsort(xv)
        xv = xv[order]
        lprob = agg["load_prob"][mask][order]
        tim = agg["load_time"][mask][order] * 1e3   # ms
        surv = agg["survival_prob"][mask][order]

        # --- loading rate / probability (peak) ---
        a0 = axS[i, 0]
        fp = np.isfinite(lprob)
        a0.plot(xv[fp], lprob[fp], "^", color="C2")
        g = _fit_gaussian(xv[fp], lprob[fp], peak=True)
        if g:
            a0.plot(g["xs"], g["ys"], "-", color="C2", lw=1.6,
                    label=f"fit mu={g['mu']:.3f}")
            a0.axvline(g["mu"], ls=":", color="gray")
            a0.legend(fontsize=8)
        a0.axvline(center[key], ls="--", color="red", lw=1)
        a0.set_xlabel(xlabel); a0.set_ylabel("loading rate")
        a0.set_ylim(0, 1.05)
        if i == 0:
            a0.set_title("Loading rate (Gaussian peak)")

        # --- loading time (dip) ---
        a1 = axS[i, 1]
        ft = np.isfinite(tim)
        a1.plot(xv[ft], tim[ft], "o", color="C0")
        g = _fit_gaussian(xv[ft], tim[ft], peak=False)
        if g:
            a1.plot(g["xs"], g["ys"], "-", color="C0", lw=1.6,
                    label=f"fit mu={g['mu']:.3f}")
            a1.axvline(g["mu"], ls=":", color="gray")
            a1.legend(fontsize=8)
        a1.axvline(center[key], ls="--", color="red", lw=1)
        a1.set_xlabel(xlabel); a1.set_ylabel("loading time (ms)")
        if i == 0:
            a1.set_title("Loading time (Gaussian dip)")

        # --- R&R survival (peak) ---
        a2 = axS[i, 2]
        fs = np.isfinite(surv)
        a2.plot(xv[fs], surv[fs], "s", color="C3")
        g = _fit_gaussian(xv[fs], surv[fs], peak=True)
        if g:
            a2.plot(g["xs"], g["ys"], "-", color="C3", lw=1.6,
                    label=f"fit mu={g['mu']:.3f}")
            a2.axvline(g["mu"], ls=":", color="gray")
            a2.legend(fontsize=8)
        a2.axvline(center[key], ls="--", color="red", lw=1)
        a2.set_xlabel(xlabel); a2.set_ylabel("R&R survival")
        if i == 0:
            a2.set_title("R&R survival (Gaussian peak)")

    fig.suptitle(f"1D cuts through optimum "
                 f"(X={center['X']:.3f}, Y={center['Y']:.3f}, "
                 f"Z1={center['Z1']:.3f}, Z2={center['Z2']:.3f})   "
                 f"red dashed = chosen value", fontsize=12)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140); print(f"saved {out}")
    else:
        plt.show()


# =========================================================================
#  REPORTING
# =========================================================================

def _show_point(tag, b):
    if b is None:
        return
    print(f"\n  [{tag}]")
    print(f"    shim_X_compression={b['X']:.3f}  shim_Y_compression={b['Y']:.3f}  "
          f"shim_Z1_compression={b['Z1']:.3f}  shim_Z2_compression={b['Z2']:.3f}")
    print(f"    loading={b['load_prob']*100:.0f}%   survival={b['survival_prob']*100:.1f}%   "
          f"load time={b['load_time']*1e3:.0f} ms   ({b['n_loaded']}/{b['n_trials']} loaded)")
    if "grade_time" in b:
        print(f"    grades:  time={b['grade_time']:.0f}/100   "
              f"survival={b['grade_survival']:.0f}/100   both={b['grade_both']:.0f}/200")


def print_summary(d, agg, min_load_prob=0.90, min_loads=1):
    print(f"File: {d['file']}   RID={d['rid']}   t_release = {d['t_release_us']:.1f} us")
    print(f"  total trials: {len(d['loaded'])}   grid points: {len(agg['X'])}")
    print(f"  overall loading prob: {np.mean(d['loaded'])*100:.1f}%")
    n_load = float(np.sum(d['loaded']))
    if n_load > 0:
        print(f"  overall R&R survival (of loaded): {np.sum(d['survived'])/n_load*100:.1f}%")
        lt = d['load_time_s'][d['loaded'] > 0.5]
        if lt.size:
            print(f"  deterministic loading time (loaded trials): "
                  f"mean {lt.mean()*1e3:.0f} ms, min {lt.min()*1e3:.0f}, max {lt.max()*1e3:.0f}")

    rec = recommend(agg, min_load_prob=min_load_prob, min_loads=min_loads)
    thr = min_load_prob * 100
    print(f"\n=== RECOMMENDATION (require loading rate >= {thr:.0f}%) ===")
    if not rec["qualified"]:
        print(f"  EVERY POINT IS BAD: no grid point reaches the >= {thr:.0f}% "
              f"loading requirement.")
        print(f"  (Collect more data, widen the scan, or relax the {thr:.0f}% "
              f"threshold via --min-load-prob.)")
        return rec

    print(f"  {rec['n_qualified']} point(s) meet the >= {thr:.0f}% loading requirement.")
    _show_point("BEST by LOADING TIME only (fastest)", rec["best_time"])
    _show_point("BEST by SURVIVAL only (coldest)", rec["best_survival"])
    _show_point("BEST by BOTH (survival high + time low)", rec["best_both"])
    return rec


# =========================================================================
#  3D MAPS (one per Z2)
# =========================================================================

def plot_maps_per_z2(agg, value="survival_prob", out_dir=None, clim=None,
                     elev=30, azim=-60):
    """3D scatter (axes X, Y, Z1; color = `value`) -- one figure per Z2 value.

    value: 'load_prob', 'survival_prob' (R&R survival), or 'load_time'.
    clim:  (lo, hi) to force the color range; default None auto-scales survival
           and load_time to the data min..max (loading prob stays 0..1).
    elev, azim: 3D viewing angle in degrees (elevation, azimuth). Change these to
           rotate the maps to the angle you want (matplotlib default 30, -60).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)

    is_time = (value == "load_time")
    has_nan = (value in ("survival_prob", "load_time"))
    # Global color range across ALL grid points (so every Z2 map is on the SAME
    # scale and they stay comparable). For survival (and loading time) we scale
    # to the actual data min..max so small differences are visible instead of
    # being washed out by a fixed 0..1 range. `clim=(lo, hi)` overrides.
    finite_all = agg[value][np.isfinite(agg[value])]
    if clim is not None:
        g_lo, g_hi = float(clim[0]), float(clim[1])
    elif value in ("survival_prob", "load_time") and finite_all.size:
        g_lo, g_hi = float(finite_all.min()), float(finite_all.max())
        if g_lo == g_hi:                 # avoid a degenerate (flat) colorbar
            g_lo -= 0.01; g_hi += 0.01
    else:
        g_lo, g_hi = 0.0, 1.0            # loading probability keeps 0..1

    z2s = np.unique(agg["Z2"])
    for z2 in z2s:
        m = agg["Z2"] == z2
        x, y, z1 = agg["X"][m], agg["Y"][m], agg["Z1"][m]
        c = agg[value][m].copy()
        valid = np.isfinite(c)
        if is_time:
            c = c * 1e3   # ms
            label, cmap, title = "mean loading time (ms)", "viridis_r", "Deterministic loading time"
            vmin, vmax = g_lo * 1e3, g_hi * 1e3
        elif value == "survival_prob":
            label, cmap, title = "R&R survival probability", "viridis", "R&R survival"
            vmin, vmax = g_lo, g_hi
        else:
            label, cmap, title = "loading probability", "viridis", "Loading probability"
            vmin, vmax = g_lo, g_hi

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)   # controllable viewing angle
        sc = ax.scatter(x[valid], y[valid], z1[valid], c=c[valid],
                        cmap=cmap, s=60, depthshade=False,
                        edgecolors="k", linewidths=0.3, vmin=vmin, vmax=vmax)
        if has_nan and np.any(~valid):
            ax.scatter(x[~valid], y[~valid], z1[~valid], c="lightgray",
                       marker="x", s=30, label="never loaded")
            ax.legend(loc="upper left", fontsize=8)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label(label)
        ax.set_xlabel("shim X"); ax.set_ylabel("shim Y"); ax.set_zlabel("shim Z1")
        ax.set_title(f"{title}   |   Z2 = {z2:.3f}")
        fig.tight_layout()
        if out_dir:
            p = os.path.join(out_dir, f"map_{value}_Z2_{z2:.3f}.png")
            fig.savefig(p, dpi=140); print(f"saved {p}"); plt.close(fig)
        else:
            plt.show()


def plot_all_maps(agg, out_dir=None, elev=30, azim=-60):
    """Loading probability, R&R survival, AND loading-time 3D maps, per Z2 value."""
    plot_maps_per_z2(agg, value="load_prob", out_dir=out_dir, elev=elev, azim=azim)
    plot_maps_per_z2(agg, value="survival_prob", out_dir=out_dir, elev=elev, azim=azim)
    plot_maps_per_z2(agg, value="load_time", out_dir=out_dir, elev=elev, azim=azim)


# =========================================================================
#  CLI
# =========================================================================

def _resolve_path(arg):
    if arg is None:
        files = find_runs()
        if not files:
            sys.exit(f"No CompressionShimScan result files under {RESULTS_DIR}.")
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
        sys.exit(f"No CompressionShimScan run with RID {rid}.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--min-load-prob", type=float, default=0.90,
                    help="required loading rate (default 0.90); below it, points are ineligible")
    ap.add_argument("--min-loads", type=int, default=1,
                    help="ignore points with fewer loaded trials than this")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    path = _resolve_path(args.run)
    d = load_run(path)
    agg = aggregate(d)
    print_summary(d, agg, min_load_prob=args.min_load_prob, min_loads=args.min_loads)
    if not args.no_plot:
        plot_all_maps(agg)


if __name__ == "__main__":
    main()
