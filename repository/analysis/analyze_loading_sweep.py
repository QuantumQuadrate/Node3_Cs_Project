"""Quantitative sweep analysis for AtomLoadDeterministicTest.

Reads one HDF5 result file per sweep point and computes, per file:

  - Apparent loading rate:  fraction with det_load_success == 1
  - Confirmed loading rate: fraction with det_load_success == 1 AND
                            (confirm_c1+confirm_c2) > bg_threshold
  - Missed-load rate:       fraction with det_load_success == 0 AND
                            (confirm_c1+confirm_c2) > bg_threshold
                            (loader said "no" but an atom was actually there)
  - False-positive rate:    fraction of reported successes whose confirmation
                            shot is consistent with background
  - bg threshold:           median(bg) + 3 * max(MAD(bg), sqrt(median(bg)))
                            per file, so it adapts to whatever the
                            day's alignment gives you
  - Confirmation/background medians and the gap between them (counts)
  - Median load time (successful shots only)
  - Median MOT-only counts and equivalent per-SPCM rate (from
    det_first_unloaded_counts)
  - Mean number of feedback rounds per shot

Each rate gets a 95% Wilson confidence interval — that's what makes the
columns honest near 0% and 100% where the normal-approx sqrt(p(1-p)/n)
breaks down.

Usage::

    python analyze_loading_sweep.py file1.h5 file2.h5 file3.h5 ...
    python analyze_loading_sweep.py *.h5 --csv summary.csv --save-plot fig.png

Options:
    --no-plot       Skip the matplotlib plot, just print the table.
    --save-plot P   Save the plot to PATH (PNG/PDF) instead of showing.
    --csv P         Also write the table as CSV to PATH.
    --sigma S       Confirmation threshold = bg_med + S*sigma  (default 3).
    --x-key K       Dataset key to use as the sweep x-axis. Default is
                    det_t_MOT2PGC_used (in seconds, displayed as ms).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import numpy as np

from analysis.data_loader import load


# =========================================================================
#  STATISTICAL HELPERS
# =========================================================================

def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n Bernoulli trials.

    Better behaved than sqrt(p(1-p)/n) when p is near 0 or 1, which is
    exactly the regime we care about here.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def mad_robust_sigma(x):
    """Robust standard-deviation analog: 1.4826 * MAD.

    Equals sigma for Gaussian-distributed data, but is insensitive to
    outliers. Returns 0 for empty input.
    """
    if len(x) == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


# =========================================================================
#  PER-FILE ANALYSIS
# =========================================================================

def analyze_one(path, x_key="det_t_MOT2PGC_used", sigma=3.0):
    """Compute summary metrics for one HDF5 file.

    Returns a dict with one row of the summary table.
    """
    bundle = load(path)
    d = bundle["data"]

    # Scalar parameters echoed by the test experiment.
    x_value = float(d[x_key]) if x_key in d else float("nan")
    threshold_rate_cps = float(d.get("det_threshold_rate_used", float("nan")))
    gate_time_s = float(d.get("det_gate_time_used", float("nan")))

    # Per-shot arrays.
    success = np.asarray(d["det_load_success"], dtype=int)
    c1 = np.asarray(d["det_confirm_count1"], dtype=float)
    c2 = np.asarray(d["det_confirm_count2"], dtype=float)
    b1 = np.asarray(d["det_confirm_bg1"], dtype=float)
    b2 = np.asarray(d["det_confirm_bg2"], dtype=float)
    first_unloaded = np.asarray(d["det_first_unloaded_counts"], dtype=float)
    t_load = np.asarray(d["det_load_time_s"], dtype=float)
    n_fb = np.asarray(d["det_n_feedback_rounds"], dtype=float)

    n = int(len(success))
    confirm = c1 + c2
    bg = b1 + b2

    # Per-file confirmation threshold. Use MAD-based sigma but floor by
    # Poisson shot noise (sqrt(N)) so we never set an absurdly tight cut
    # when the background histogram happens to be flat in this run.
    bg_med = float(np.median(bg)) if len(bg) > 0 else 0.0
    bg_sigma = mad_robust_sigma(bg)
    shot_noise = float(np.sqrt(max(bg_med, 1.0)))
    atom_threshold = bg_med + sigma * max(bg_sigma, shot_noise)

    is_real_atom = confirm > atom_threshold
    sucs = success.astype(bool)

    # Headline rates.
    apparent_k = int(sucs.sum())
    apparent_p = apparent_k / n if n else 0.0
    apparent_ci = wilson_ci(apparent_k, n)

    confirmed_k = int(np.logical_and(sucs, is_real_atom).sum())
    confirmed_p = confirmed_k / n if n else 0.0
    confirmed_ci = wilson_ci(confirmed_k, n)

    missed_k = int(np.logical_and(~sucs, is_real_atom).sum())
    missed_p = missed_k / n if n else 0.0
    missed_ci = wilson_ci(missed_k, n)

    fp_k = int(np.logical_and(sucs, ~is_real_atom).sum())
    fp_p = (fp_k / apparent_k) if apparent_k else 0.0
    fp_ci = wilson_ci(fp_k, apparent_k) if apparent_k else (0.0, 0.0)

    # Load time, conditional on success.
    t_success = t_load[sucs] if sucs.any() else np.array([])
    t_load_med_ms = float(np.median(t_success) * 1e3) if len(t_success) else float("nan")

    # MOT-only baseline rate (per SPCM, kcps). first_unloaded is SPCM1+SPCM2
    # sum from the first polling try; dividing by 2*gate_time gives an
    # average per-SPCM rate.
    mot_only_med = float(np.median(first_unloaded)) if len(first_unloaded) else 0.0
    if gate_time_s > 0 and np.isfinite(gate_time_s):
        mot_only_rate_kcps = 0.5 * mot_only_med / gate_time_s / 1e3
    else:
        mot_only_rate_kcps = float("nan")

    return {
        "file": Path(path).name,
        "x": x_value,
        "n_shots": n,
        "apparent_p": apparent_p, "apparent_ci": apparent_ci,
        "confirmed_p": confirmed_p, "confirmed_ci": confirmed_ci,
        "missed_p": missed_p, "missed_ci": missed_ci,
        "fp_p": fp_p, "fp_ci": fp_ci,
        "bg_med": bg_med, "bg_sigma": bg_sigma,
        "atom_threshold": atom_threshold,
        "confirm_med": float(np.median(confirm)) if len(confirm) else 0.0,
        "gap_counts": (float(np.median(confirm)) if len(confirm) else 0.0) - bg_med,
        "t_load_med_ms": t_load_med_ms,
        "mot_only_med": mot_only_med,
        "mot_only_rate_kcps": mot_only_rate_kcps,
        "mean_fb_rounds": float(np.mean(n_fb)) if len(n_fb) else 0.0,
        "threshold_rate_cps": threshold_rate_cps,
        "gate_time_s": gate_time_s,
    }


# =========================================================================
#  TABLE & PLOT
# =========================================================================

def _fmt_rate_ci(p, ci):
    lo, hi = ci
    return f"{p*100:5.1f} [{lo*100:4.1f}-{hi*100:4.1f}]"


def print_table(rows, x_label="t_MOT2PGC (ms)", x_scale=1e3):
    rows = sorted(rows, key=lambda r: r["x"])

    cols = [
        (x_label, 13, lambda r: f"{r['x'] * x_scale:13.1f}"),
        ("N", 5, lambda r: f"{r['n_shots']:5d}"),
        ("apparent %", 20, lambda r: f"{_fmt_rate_ci(r['apparent_p'], r['apparent_ci']):>20s}"),
        ("confirmed %", 20, lambda r: f"{_fmt_rate_ci(r['confirmed_p'], r['confirmed_ci']):>20s}"),
        ("missed %", 20, lambda r: f"{_fmt_rate_ci(r['missed_p'], r['missed_ci']):>20s}"),
        ("FP/apparent %", 20, lambda r: f"{_fmt_rate_ci(r['fp_p'], r['fp_ci']):>20s}"),
        ("thresh", 7, lambda r: f"{r['atom_threshold']:7.1f}"),
        ("gap", 7, lambda r: f"{r['gap_counts']:7.1f}"),
        ("bg", 7, lambda r: f"{r['bg_med']:7.1f}"),
        ("conf", 7, lambda r: f"{r['confirm_med']:7.1f}"),
        ("t_load ms", 10, lambda r: f"{r['t_load_med_ms']:10.1f}"),
        ("MOT kcps", 9, lambda r: f"{r['mot_only_rate_kcps']:9.2f}"),
        ("fb_rnds", 8, lambda r: f"{r['mean_fb_rounds']:8.2f}"),
    ]

    header = " ".join(f"{name:>{w}s}" for name, w, _ in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" ".join(fn(r) for _, _, fn in cols))

    print()
    print("Legend:")
    print("  apparent   = loader returned success=True")
    print("  confirmed  = loader said yes AND confirmation > threshold")
    print("  missed     = loader said no  AND confirmation > threshold")
    print("  FP/app.    = of loader-reported successes, fraction that failed confirmation")
    print("  thresh     = per-file confirmation threshold = bg_med + sigma*max(MAD, sqrt(bg_med))")
    print("  gap        = median(confirm) - median(bg), in counts per imaging gate")
    print("  MOT kcps   = per-SPCM count rate, median first_unloaded / (2*gate_time)")


def write_csv(rows, path, x_label):
    import csv
    rows = sorted(rows, key=lambda r: r["x"])
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "file", x_label, "n_shots",
            "apparent_p", "apparent_ci_lo", "apparent_ci_hi",
            "confirmed_p", "confirmed_ci_lo", "confirmed_ci_hi",
            "missed_p", "missed_ci_lo", "missed_ci_hi",
            "fp_p", "fp_ci_lo", "fp_ci_hi",
            "bg_med", "bg_sigma", "atom_threshold",
            "confirm_med", "gap_counts",
            "t_load_med_ms", "mot_only_med", "mot_only_rate_kcps",
            "mean_fb_rounds", "threshold_rate_cps", "gate_time_s",
        ])
        for r in rows:
            w.writerow([
                r["file"], r["x"], r["n_shots"],
                r["apparent_p"], r["apparent_ci"][0], r["apparent_ci"][1],
                r["confirmed_p"], r["confirmed_ci"][0], r["confirmed_ci"][1],
                r["missed_p"], r["missed_ci"][0], r["missed_ci"][1],
                r["fp_p"], r["fp_ci"][0], r["fp_ci"][1],
                r["bg_med"], r["bg_sigma"], r["atom_threshold"],
                r["confirm_med"], r["gap_counts"],
                r["t_load_med_ms"], r["mot_only_med"], r["mot_only_rate_kcps"],
                r["mean_fb_rounds"], r["threshold_rate_cps"], r["gate_time_s"],
            ])
    print(f"CSV saved: {path}")


def plot_sweep(rows, out=None, x_label="t_MOT2PGC (ms)", x_scale=1e3):
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["x"])
    x = np.array([r["x"] * x_scale for r in rows])

    def _err_pair(rows, key):
        # Clamp to >= 0: the Wilson interval can put p a hair outside [lo, hi]
        # by floating-point noise, and matplotlib rejects negative yerr.
        p = np.array([r[key + "_p"] * 100 for r in rows])
        lo = np.clip(p - np.array([r[key + "_ci"][0] * 100 for r in rows]), 0, None)
        hi = np.clip(np.array([r[key + "_ci"][1] * 100 for r in rows]) - p, 0, None)
        return p, [lo, hi]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]})

    # Top: success rates with CIs
    ap, ap_e = _err_pair(rows, "apparent")
    co, co_e = _err_pair(rows, "confirmed")
    mi, mi_e = _err_pair(rows, "missed")

    ax_top.errorbar(x, ap, yerr=ap_e, fmt="o--", color="gray",
                    label="apparent (loader)", capsize=3)
    ax_top.errorbar(x, co, yerr=co_e, fmt="s-", color="C0",
                    label="confirmed (bg+3sigma)", capsize=3)
    ax_top.errorbar(x, mi, yerr=mi_e, fmt="^:", color="C3",
                    label="missed (loader failed but atom present)", capsize=3)
    ax_top.set_ylabel("Rate (%)")
    ax_top.set_ylim(-2, 105)
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="best", fontsize=9)
    ax_top.set_title("Deterministic loading: success vs sweep parameter")

    # Bottom: gap (confirm - bg) -- the SNR-style diagnostic
    gap = np.array([r["gap_counts"] for r in rows])
    thr = np.array([r["atom_threshold"] - r["bg_med"] for r in rows])
    confirm_med = np.array([r["confirm_med"] for r in rows])
    bg_med = np.array([r["bg_med"] for r in rows])

    ax_bot.plot(x, confirm_med, "s-", color="C0", label="median confirm")
    ax_bot.plot(x, bg_med, "o-", color="0.4", label="median bg")
    ax_bot.fill_between(x, bg_med, bg_med + thr, color="0.85",
                        label="bg + 3sigma (threshold band)")
    ax_bot.set_xlabel(x_label)
    ax_bot.set_ylabel("Counts per gate")
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"Plot saved: {out}")
    else:
        plt.show()


# =========================================================================
#  CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path,
                   help="HDF5 result files (one per sweep point)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save-plot", type=str, default=None, metavar="PATH",
                   help="Save plot to PATH instead of showing it")
    p.add_argument("--csv", type=str, default=None, metavar="PATH",
                   help="Also write the summary table as CSV")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Confirmation threshold sigma above bg (default 3)")
    p.add_argument("--x-key", type=str, default="det_t_MOT2PGC_used",
                   help="Dataset key for the sweep x-axis "
                        "(default det_t_MOT2PGC_used)")
    p.add_argument("--x-label", type=str, default=None,
                   help="Override x-axis label (auto-detected from --x-key)")
    p.add_argument("--x-scale", type=float, default=None,
                   help="Multiplier applied to x values for display "
                        "(default 1e3 for t_* keys, 1 otherwise)")
    args = p.parse_args()

    # Sensible defaults for the time-style sweep keys.
    if args.x_label is None:
        if args.x_key == "det_t_MOT2PGC_used":
            args.x_label = "t_MOT2PGC (ms)"
        elif args.x_key.startswith("det_t_"):
            args.x_label = args.x_key[4:] + " (ms)"
        else:
            args.x_label = args.x_key
    if args.x_scale is None:
        args.x_scale = 1e3 if args.x_key.startswith("det_t_") else 1.0

    rows = []
    for path in args.files:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        try:
            row = analyze_one(path, x_key=args.x_key, sigma=args.sigma)
            rows.append(row)
        except Exception as e:
            print(f"WARNING: failed on {path}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    if not rows:
        sys.exit("No usable files.")

    print_table(rows, x_label=args.x_label, x_scale=args.x_scale)

    if args.csv:
        write_csv(rows, args.csv, x_label=args.x_label)

    if not args.no_plot:
        plot_sweep(rows, out=args.save_plot,
                   x_label=args.x_label, x_scale=args.x_scale)


if __name__ == "__main__":
    main()
