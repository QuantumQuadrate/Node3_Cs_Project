"""Excitation-amplitude scan analysis: photon collection probability vs amplitude.

Reads ONE ExcitationScan HDF5 result file (a single run already contains the full
amplitude scan, with one row per measurement) and computes, per excitation
amplitude, the photon collection probability

    P = (photons detected) / (excitation attempts)

aggregated over the measurements where the atom was actually present during the
excitation (see --gate), so the detections come from a real trapped atom.

  - x-axis: excitation amplitude  (dataset `exc_scan_amp`)
  - y-axis: photon collection probability P  (per excitation attempt)

SPCM1, SPCM2, and combined (SPCM1+SPCM2) probabilities are reported, each with a
95% Wilson confidence interval — honest near 0% and 100%, where the normal-approx
binomial error sqrt(p(1-p)/n) breaks down. The peak-probability amplitude is
highlighted and cross-checked against the experiment's own `exc_best_amp`.

With --timing, also plots the photon arrival-time profile (from the per-photon
`SPCM*_timestamps` tagged by `SPCM*_ts_amp`).

Usage::

    python analyze_excitation_scan.py                  # latest ExcitationScan run
    python analyze_excitation_scan.py 512              # a specific run ID (RID)
    python analyze_excitation_scan.py path/to/run.h5   # an explicit file
    python analyze_excitation_scan.py <file> --save-plot fig.png --csv out.csv
    python analyze_excitation_scan.py <file> --timing  # also: arrival-time profile

Options:
    --gate G        Which measurements to include: 'both' (default; atom loaded
                    AND retained), 'retained', 'loaded', or 'none' (all).
    --no-plot       Skip the probability plot, just print the table.
    --save-plot P   Save the probability plot to PATH instead of showing it.
    --csv P         Also write the per-amplitude table as CSV.
    --timing        Also show the photon arrival-time histogram.
    --no-spcm       Plot only the combined probability (hide SPCM1/SPCM2 curves).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import numpy as np

from analysis.data_loader import load, find_runs


# =========================================================================
#  STATISTICAL HELPER
# =========================================================================

def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n Bernoulli trials.

    Better behaved than sqrt(p(1-p)/n) when p is near 0 or 1 — exactly the
    regime of a low photon-collection probability. BLAS-free.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


# =========================================================================
#  AGGREGATION
# =========================================================================

def _gate_mask(d, gate):
    """Per-measurement boolean mask selecting which measurements to include."""
    loaded = np.asarray(d["exc_atom_loaded"], dtype=int).astype(bool)
    retained = np.asarray(d["exc_atom_retained"], dtype=int).astype(bool)
    if gate == "none":
        return np.ones(len(loaded), dtype=bool)
    if gate == "loaded":
        return loaded
    if gate == "retained":
        return retained
    return loaded & retained   # 'both' (default): present before AND after


def collection_probability(path, gate="both"):
    """Aggregate photon collection probability per excitation amplitude.

    Returns (meta, rows). `rows` is a list of per-amplitude dicts (sorted by
    amplitude) with detection counts, attempts, and Wilson CIs for SPCM1,
    SPCM2, and combined.
    """
    bundle = load(path)
    d = bundle["data"]

    amp = np.asarray(d["exc_scan_amp"], dtype=float)
    det1 = np.asarray(d["exc_det_spcm1"], dtype=float)
    det2 = np.asarray(d["exc_det_spcm2"], dtype=float)
    attempts = np.asarray(d["exc_attempts"], dtype=float)

    mask = _gate_mask(d, gate)
    # amplitudes come from min + i*step; round to a stable grouping key
    amp_key = np.round(amp, 6)
    uniq = np.unique(amp_key[mask]) if mask.any() else np.array([])

    rows = []
    for a in uniq:
        m = mask & (amp_key == a)
        N = float(np.sum(attempts[m]))        # total excitation attempts at this amp
        k1 = float(np.sum(det1[m]))           # SPCM1 detections
        k2 = float(np.sum(det2[m]))           # SPCM2 detections
        kc = k1 + k2                          # combined detections
        rows.append({
            "amp": float(a),
            "n_meas": int(np.sum(m)),
            "attempts": N,
            "k1": k1, "p1": (k1 / N if N > 0 else 0.0), "ci1": wilson_ci(k1, N),
            "k2": k2, "p2": (k2 / N if N > 0 else 0.0), "ci2": wilson_ci(k2, N),
            "kc": kc, "pc": (kc / N if N > 0 else 0.0), "cic": wilson_ci(kc, N),
        })

    def _scalar(key):
        if key not in d:
            return float("nan")
        v = d[key]
        try:
            return float(np.asarray(v).reshape(-1)[0]) if np.asarray(v).size else float(v)
        except (TypeError, ValueError):
            return float("nan")

    loaded_all = np.asarray(d["exc_atom_loaded"], dtype=int) if "exc_atom_loaded" in d else np.zeros(len(amp), dtype=int)
    retained_all = np.asarray(d["exc_atom_retained"], dtype=int) if "exc_atom_retained" in d else np.zeros(len(amp), dtype=int)
    meta = {
        "file": Path(path).name,
        "path": str(path),
        "gate": gate,
        "n_total": int(len(amp)),
        "n_gated": int(np.sum(mask)),
        "n_loaded": int(np.sum(loaded_all > 0)),
        "n_retained": int(np.sum(retained_all > 0)),
        "n_det1": int(np.sum(det1)),
        "n_det2": int(np.sum(det2)),
        "exc_best_amp": _scalar("exc_best_amp"),
        "exc_best_efficiency": _scalar("exc_best_efficiency"),
    }
    return meta, rows


# =========================================================================
#  TABLE / CSV
# =========================================================================

def print_table(meta, rows):
    rows = sorted(rows, key=lambda r: r["amp"])
    print(f"File: {meta['file']}   gate={meta['gate']}   "
          f"measurements kept: {meta['n_gated']}/{meta['n_total']}")
    print(f"  atoms loaded: {meta['n_loaded']}/{meta['n_total']}   "
          f"retained: {meta['n_retained']}/{meta['n_total']}   "
          f"photons detected: SPCM1={meta['n_det1']}, SPCM2={meta['n_det2']}")
    if not rows:
        print(f"  >> Nothing passed gate='{meta['gate']}'. With loaded={meta['n_loaded']} "
              f"and retained={meta['n_retained']}, the run had ~no surviving atoms. "
              f"Set GATE='none' to plot raw per-amplitude detection rates regardless of "
              f"survival (this is data, not an analysis bug).")
        return
    print()
    hdr = (f"{'amp':>8}  {'n_meas':>6}  {'attempts':>9}  "
           f"{'P1 % [95% CI]':>20}  {'P2 % [95% CI]':>20}  {'P_comb % [95% CI]':>22}")
    print(hdr)
    print("-" * len(hdr))

    def _fmt(p, ci):
        return f"{p*100:6.3f} [{ci[0]*100:5.3f}-{ci[1]*100:5.3f}]"

    for r in rows:
        print(f"{r['amp']:8.3f}  {r['n_meas']:6d}  {int(r['attempts']):9d}  "
              f"{_fmt(r['p1'], r['ci1']):>20}  {_fmt(r['p2'], r['ci2']):>20}  "
              f"{_fmt(r['pc'], r['cic']):>22}")

    best = max(rows, key=lambda r: r["pc"])
    print()
    print(f"Peak combined collection probability: {best['pc']*100:.3f}%  "
          f"at amplitude {best['amp']:.3f}  "
          f"({int(best['kc'])}/{int(best['attempts'])} detections/attempts)")
    if np.isfinite(meta["exc_best_amp"]):
        print(f"(experiment _summarize reported best amp = {meta['exc_best_amp']:.3f}, "
              f"efficiency = {meta['exc_best_efficiency']*100:.3f}%)")


def write_csv(meta, rows, path):
    import csv
    rows = sorted(rows, key=lambda r: r["amp"])
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "amp", "n_meas", "attempts",
            "k_spcm1", "p_spcm1", "p_spcm1_ci_lo", "p_spcm1_ci_hi",
            "k_spcm2", "p_spcm2", "p_spcm2_ci_lo", "p_spcm2_ci_hi",
            "k_comb", "p_comb", "p_comb_ci_lo", "p_comb_ci_hi",
        ])
        for r in rows:
            w.writerow([
                r["amp"], r["n_meas"], int(r["attempts"]),
                int(r["k1"]), r["p1"], r["ci1"][0], r["ci1"][1],
                int(r["k2"]), r["p2"], r["ci2"][0], r["ci2"][1],
                int(r["kc"]), r["pc"], r["cic"][0], r["cic"][1],
            ])
    print(f"CSV saved: {path}")


# =========================================================================
#  PLOTS
# =========================================================================

def plot_scan(meta, rows, out=None, show_spcm=True):
    """The requested plot: x = excitation amplitude, y = photon collection probability."""
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["amp"])
    if not rows:
        print(f"Nothing to plot: 0/{meta['n_total']} measurements passed gate='{meta['gate']}'. "
              f"Atoms loaded {meta['n_loaded']}/{meta['n_total']}, retained "
              f"{meta['n_retained']}/{meta['n_total']}; photons detected "
              f"SPCM1={meta['n_det1']}, SPCM2={meta['n_det2']}.\n"
              f">> Set GATE='none' (cell 2) to plot raw detection rates regardless of atom "
              f"survival. An empty plot here means the run had ~no surviving atoms, not an "
              f"analysis bug.")
        return
    x = np.array([r["amp"] for r in rows])

    def _pe(pkey, cikey):
        p = np.array([r[pkey] * 100 for r in rows])
        lo = np.clip(p - np.array([r[cikey][0] * 100 for r in rows]), 0, None)
        hi = np.clip(np.array([r[cikey][1] * 100 for r in rows]) - p, 0, None)
        return p, [lo, hi]

    fig, ax = plt.subplots(figsize=(8, 5))

    pc, pc_e = _pe("pc", "cic")
    ax.errorbar(x, pc, yerr=pc_e, fmt="o-", color="C0", capsize=3, lw=1.8,
                label="combined (SPCM1+SPCM2)")
    if show_spcm:
        p1, p1_e = _pe("p1", "ci1")
        p2, p2_e = _pe("p2", "ci2")
        ax.errorbar(x, p1, yerr=p1_e, fmt="s--", color="C1", capsize=2,
                    alpha=0.7, label="SPCM1")
        ax.errorbar(x, p2, yerr=p2_e, fmt="^--", color="C2", capsize=2,
                    alpha=0.7, label="SPCM2")

    i_best = int(np.argmax(pc))
    ax.axvline(x[i_best], color="red", ls=":", lw=1.5,
               label=f"peak @ {x[i_best]:.3f}  ({pc[i_best]:.2f}%)")

    ax.set_xlabel("Excitation amplitude")
    ax.set_ylabel("Photon collection probability (%)")
    ax.set_title(f"Excitation amplitude scan  |  {meta['file']}  |  gate={meta['gate']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=140)
        print(f"Plot saved: {out}")
    else:
        plt.show()


def plot_timing(path, amp=None, out=None, bins=60):
    """Photon arrival-time histogram (relative to excitation start), optionally
    restricted to one excitation amplitude via `amp`."""
    import matplotlib.pyplot as plt

    bundle = load(path)
    d = bundle["data"]
    ts1 = np.asarray(d.get("SPCM1_timestamps", []), dtype=float)
    a1 = np.asarray(d.get("SPCM1_ts_amp", []), dtype=float)
    ts2 = np.asarray(d.get("SPCM2_timestamps", []), dtype=float)
    a2 = np.asarray(d.get("SPCM2_ts_amp", []), dtype=float)

    if amp is not None:
        key = round(float(amp), 6)
        if ts1.size:
            ts1 = ts1[np.round(a1, 6) == key]
        if ts2.size:
            ts2 = ts2[np.round(a2, 6) == key]

    if ts1.size == 0 and ts2.size == 0:
        print("No photon timestamps to plot"
              + (f" at amp={amp:.3f}" if amp is not None else "") + ".")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if ts1.size:
        ax.hist(ts1 * 1e9, bins=bins, alpha=0.6, color="C1",
                label=f"SPCM1 (N={ts1.size})")
    if ts2.size:
        ax.hist(ts2 * 1e9, bins=bins, alpha=0.6, color="C2",
                label=f"SPCM2 (N={ts2.size})")
    ax.set_xlabel("Photon arrival time after excitation start (ns)")
    ax.set_ylabel("Counts")
    ttl = "Photon arrival-time profile"
    if amp is not None:
        ttl += f"   (amp = {amp:.3f})"
    ax.set_title(f"{ttl}  |  {Path(path).name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=140)
        print(f"Timing plot saved: {out}")
    else:
        plt.show()


# =========================================================================
#  CLI
# =========================================================================

def _resolve_path(arg):
    """Resolve the CLI argument to an .h5 path: explicit file, RID, or latest."""
    if arg is None:
        files = find_runs(experiment="ExcitationScan")
        if not files:
            sys.exit("No ExcitationScan result files found under results/.")
        print(f"(Using latest ExcitationScan run: {Path(files[-1]).name})")
        return files[-1]

    p = Path(arg)
    if p.exists():
        return str(p)

    try:
        rid = int(arg)
    except ValueError:
        sys.exit(f"Not a file or run ID: {arg}")

    files = find_runs(experiment="ExcitationScan")
    matches = [f for f in files if Path(f).name.startswith(f"{rid:09d}-")]
    if not matches:
        sys.exit(f"No ExcitationScan run with RID {rid} under results/.")
    return matches[-1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None,
                    help="ExcitationScan .h5 path, RID, or omit for the latest run")
    ap.add_argument("--gate", choices=["both", "retained", "loaded", "none"],
                    default="both",
                    help="Measurements to include (default: both = loaded AND retained)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the probability plot, just print the table")
    ap.add_argument("--save-plot", type=str, default=None, metavar="PATH",
                    help="Save the probability plot to PATH instead of showing it")
    ap.add_argument("--csv", type=str, default=None, metavar="PATH",
                    help="Also write the per-amplitude table as CSV")
    ap.add_argument("--timing", action="store_true",
                    help="Also show the photon arrival-time histogram")
    ap.add_argument("--timing-amp", type=float, default=None, metavar="AMP",
                    help="Restrict the arrival-time histogram to one amplitude")
    ap.add_argument("--no-spcm", action="store_true",
                    help="Plot only the combined probability (hide SPCM1/SPCM2)")
    args = ap.parse_args()

    path = _resolve_path(args.run)
    meta, rows = collection_probability(path, gate=args.gate)

    print_table(meta, rows)

    if args.csv:
        write_csv(meta, rows, args.csv)

    if not args.no_plot:
        plot_scan(meta, rows, out=args.save_plot, show_spcm=not args.no_spcm)

    if args.timing:
        plot_timing(path, amp=args.timing_amp)


if __name__ == "__main__":
    main()
