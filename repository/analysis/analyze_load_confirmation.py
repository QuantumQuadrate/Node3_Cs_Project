"""Loader-vs-ground-truth confirmation analysis for AtomLoadDeterministicTest.

The loading test now takes a CONFIRMATION image after every load and records
whether an atom is REALLY in the trap (`det_confirmed`), independent of the
loader's own threshold verdict (`det_load_success`). This module scores how
honest that verdict is -- in particular the PHANTOM rate: declared loads with no
real atom. Phantoms are what hurt the real experiment (parity/rate herald "atom
present" on empty traps -> wasted cycles), and they grow as the MOT gets brighter
at longer formation times and its background light alone trips the loader.

Per MOT value it reports the loader as a binary classifier against truth:
  success %    loader said 'loaded'                      (det_load_success)
  confirmed %  atom really there                         (det_confirmed)
  PHANTOM %    loaded but NOT confirmed = P(no atom | loader said yes)  <- the cost
  fidelity %   loaded AND confirmed / loaded              (= 1 - phantom)
  miss %       confirmed but loader said NO               (false negative)

Plots:
  1. success % vs confirmed % vs MOT -- the widening gap IS the phantom rate.
  2. phantom (false-positive) rate vs MOT, with a tolerance line.
  3. loader winning-count histogram, REAL vs PHANTOM, with the current loader
     threshold -- shows directly WHERE to move single_atom_threshold_for_loading.
  4. confirmation-image histogram, REAL vs PHANTOM (validates the ground truth).

It also RECOMMENDS a loader threshold: the lowest count cut that rejects >=95%
of phantoms, and how many real atoms that keeps.

Requires a run from the confirmation version of the experiment (the one with
det_confirmed). Older runs without it are reported as "no confirmation data".

Usage::

    python analyze_load_confirmation.py            # latest run
    python analyze_load_confirmation.py 3610        # a specific RID
    python analyze_load_confirmation.py --max-phantom 0.02   # stricter cut
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
    files = glob.glob(
        str(RESULTS_DIR / "**" / "*-AtomLoadDeterministicTest.h5"), recursive=True)
    return sorted(files, key=_rid_of)


def load_run(path):
    """Load the confirmation-relevant per-shot arrays from one scan .h5."""
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"]

        def arr(k):
            return np.asarray(ds[k][()]) if k in ds else None

        def scal(k, default=np.nan):
            return float(ds[k][()]) if k in ds else default

        d = {
            "mot_s":     arr("det_mot_formation_s"),
            "success":   arr("det_load_success"),
            "confirmed": arr("det_confirmed"),
            "poll_s":    arr("det_load_time_s"),
            "s1_load":   arr("det_spcm1_at_load"),
            "s2_load":   arr("det_spcm2_at_load"),
            "confirm1":  arr("det_confirm_count1"),
            "confirm2":  arr("det_confirm_count2"),
            "bg1":       arr("det_confirm_bg1"),
            "bg2":       arr("det_confirm_bg2"),
            "gate_s":    scal("det_gate_time_used"),
            "threshold_cps": scal("det_threshold_rate_used"),
        }
    d["rid"] = _rid_of(path)
    d["file"] = Path(path).name
    d["has_confirmation"] = (d["confirmed"] is not None
                             and d["mot_s"] is not None)
    if d["has_confirmation"]:
        d["success"] = d["success"].astype(float)
        d["confirmed"] = d["confirmed"].astype(float)
        d["mot_s"] = d["mot_s"].astype(float)
        d["poll_s"] = (d["poll_s"].astype(float)
                       if d["poll_s"] is not None else np.zeros_like(d["mot_s"]))
        d["load_net"] = (d["s1_load"] + d["s2_load"]).astype(float)   # loader winning sum
        d["confirm_net"] = ((d["confirm1"] + d["confirm2"])
                            - (d["bg1"] + d["bg2"])).astype(float)      # image signal
    return d


# =========================================================================
#  STATS
# =========================================================================

def _sum_threshold(d):
    """Loader count-level threshold: rate = 0.5*(c0+c1)/gate > thr_cps
    <=> (c0+c1) > 2*thr_cps*gate."""
    g, t = d["gate_s"], d["threshold_cps"]
    return (2.0 * t * g) if (np.isfinite(g) and np.isfinite(t)) else np.nan


def per_mot(d):
    """One classifier-scorecard row per distinct MOT value, including load time
    and the throughput-honest 'time per confirmed atom'."""
    mot, succ, conf, poll = d["mot_s"], d["success"], d["confirmed"], d["poll_s"]
    rows = []
    for val in sorted(set(np.round(mot, 6))):
        m = np.isclose(mot, val)
        s = succ[m] > 0.5
        c = conf[m] > 0.5
        nloads = int(s.sum())
        conf_rate = float(c.mean())
        # total load time per shot = fixed MOT + measured poll (both in ms).
        poll_med_ms = float(np.median(poll[m])) * 1e3
        poll_mean_ms = float(np.mean(poll[m])) * 1e3
        mot_ms = val * 1e3
        total_med_ms = mot_ms + poll_med_ms
        total_mean_ms = mot_ms + poll_mean_ms
        # Time to get ONE REAL atom = mean load time / confirmed fraction. This
        # penalizes slow loading (big total) AND phantoms (small confirmed_rate),
        # so its minimum over MOT is the honest optimum.
        t_per_conf_ms = (total_mean_ms / conf_rate) if conf_rate > 0 else np.inf
        rows.append({
            "mot_ms": mot_ms, "n": int(m.sum()),
            "success_rate": float(s.mean()),
            "confirmed_rate": conf_rate,
            "phantom_rate": float((s & ~c).sum() / nloads) if nloads else np.nan,
            "fidelity": float((s & c).sum() / nloads) if nloads else np.nan,
            "miss_rate": float((c & ~s).mean()),
            "poll_med_ms": poll_med_ms,
            "total_med_ms": total_med_ms,
            "total_mean_ms": total_mean_ms,
            "t_per_conf_ms": t_per_conf_ms,
        })
    return rows


def recommend_threshold(d, max_phantom=0.05):
    """Lowest loader count-cut that keeps <= max_phantom of phantom triggers;
    report how much of the real-atom signal it retains. Uses the loader's
    winning-gate counts (load_net) split by truth."""
    s = d["success"] > 0.5
    c = d["confirmed"] > 0.5
    real = d["load_net"][s & c]
    phantom = d["load_net"][s & ~c]
    cur = _sum_threshold(d)
    if phantom.size == 0:
        return {"ok": False, "reason": "no phantoms -- threshold already clean",
                "cur_sum": cur, "cur_cps": d["threshold_cps"],
                "n_real": real.size, "n_phantom": 0}
    lo = int(np.floor(min(real.min() if real.size else 0, phantom.min())))
    hi = int(np.ceil(max(real.max() if real.size else 1, phantom.max())))
    chosen = None
    for t in range(lo, hi + 1):
        ph_keep = float(np.mean(phantom >= t))
        if ph_keep <= max_phantom:
            re_keep = float(np.mean(real >= t)) if real.size else 1.0
            chosen = (t, re_keep, ph_keep)
            break
    if chosen is None:
        t = hi + 1
        chosen = (t, 0.0, 0.0)
    t, re_keep, ph_keep = chosen
    g = d["gate_s"]
    return {"ok": True, "cur_sum": cur, "cur_cps": d["threshold_cps"],
            "rec_sum": t, "rec_cps": (t / (2.0 * g) if np.isfinite(g) else np.nan),
            "real_kept": re_keep, "phantom_kept": ph_keep,
            "n_real": real.size, "n_phantom": phantom.size,
            "real_med": float(np.median(real)) if real.size else np.nan,
            "phantom_med": float(np.median(phantom))}


# =========================================================================
#  REPORTING
# =========================================================================

def print_report(d, rows, rec):
    print(f"File: {d['file']}   RID={d['rid']}   shots={len(d['success'])}")
    sum_thr = _sum_threshold(d)
    print(f"  loader threshold: {d['threshold_cps']:.0f} counts/s "
          f"(= {sum_thr:.0f} summed counts/gate);  atom confirm vs image.")
    print()
    print(f"  {'MOT ms':>7}  {'N':>4}  {'succ %':>7}  {'CONF %':>7}  "
          f"{'PHANTOM %':>9}  {'total ms':>8}  {'t/atom ms':>9}")
    for r in rows:
        tpa = r["t_per_conf_ms"]
        tpa_s = f"{tpa:>9.0f}" if np.isfinite(tpa) else f"{'inf':>9}"
        print(f"  {r['mot_ms']:>7.0f}  {r['n']:>4}  {r['success_rate']*100:>7.1f}  "
              f"{r['confirmed_rate']*100:>7.1f}  {r['phantom_rate']*100:>9.1f}  "
              f"{r['total_med_ms']:>8.0f}  {tpa_s}")

    # ---- OPTIMAL MOT (headline) ----
    # Throughput-honest: minimize time per CONFIRMED atom.
    best = min(rows, key=lambda r: r["t_per_conf_ms"])
    print()
    print(f"  >>> OPTIMAL MOT formation: {best['mot_ms']:.0f} ms <<<")
    print(f"      time per CONFIRMED atom = {best['t_per_conf_ms']:.0f} ms  (minimum)")
    print(f"      at {best['mot_ms']:.0f} ms: total load {best['total_med_ms']:.0f} ms, "
          f"confirmed {best['confirmed_rate']*100:.0f}%, phantom {best['phantom_rate']*100:.0f}%")
    print(f"      metric = mean load time / confirmed fraction -- penalizes both slow")
    print(f"      loading AND phantom (empty-trap) loads, so its min is the real optimum.")

    # Cross-check: fastest raw total among points that stay clean (phantom<=10%).
    clean = [r for r in rows if np.isfinite(r["phantom_rate"]) and r["phantom_rate"] <= 0.10]
    if clean:
        fastclean = min(clean, key=lambda r: r["total_med_ms"])
        hi = max(clean, key=lambda r: r["mot_ms"])
        print(f"  Cross-check: fastest total among phantom<=10% is {fastclean['mot_ms']:.0f} ms "
              f"({fastclean['total_med_ms']:.0f} ms/load); clean up to {hi['mot_ms']:.0f} ms.")
        dirty = [r for r in rows if r["mot_ms"] > hi["mot_ms"]]
        if dirty:
            print(f"  Above {hi['mot_ms']:.0f} ms the loader false-triggers on the bright MOT "
                  f"(phantom up to {max(r['phantom_rate'] for r in dirty)*100:.0f}%).")
    else:
        print("  WARNING: phantom rate >10% at EVERY MOT -- the loader threshold is too low; "
              "raise it (see recommendation below) and/or shorten the MOT.")

    print()
    if not rec["ok"]:
        print(f"  Threshold: {rec['reason']} "
              f"(current {rec['cur_cps']:.0f} counts/s = {rec['cur_sum']:.0f} counts).")
    else:
        print(f"  Loader counts: real atoms median {rec['real_med']:.0f}, "
              f"phantoms median {rec['phantom_med']:.0f} "
              f"(current cut {rec['cur_sum']:.0f}).")
        print(f"  RECOMMEND threshold -> {rec['rec_cps']:.0f} counts/s "
              f"(= {rec['rec_sum']:.0f} summed counts): rejects "
              f"{(1-rec['phantom_kept'])*100:.0f}% of phantoms, keeps "
              f"{rec['real_kept']*100:.0f}% of real atoms.")
        if rec["real_kept"] < 0.95:
            print(f"  NOTE: that cut also drops {(1-rec['real_kept'])*100:.0f}% of REAL "
                  f"atoms -- real and phantom counts overlap, so detection is "
                  f"intrinsically marginal here (shorten the MOT).")
    return best


# =========================================================================
#  PLOTS
# =========================================================================

def plot_report(d, rows, rec, max_phantom=0.05, out=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    x = np.array([r["mot_ms"] for r in rows])
    succ = np.array([r["success_rate"] * 100 for r in rows])
    conf = np.array([r["confirmed_rate"] * 100 for r in rows])
    phantom = np.array([r["phantom_rate"] * 100 for r in rows])
    tpa = np.array([r["t_per_conf_ms"] for r in rows])
    best_i = int(np.argmin(tpa))
    best_mot = x[best_i]

    # (1) success vs confirmed; gap = phantoms. Optimal MOT marked.
    a = ax[0, 0]
    a.plot(x, succ, "o-", color="C0", label="loader success")
    a.plot(x, conf, "s-", color="C2", label="CONFIRMED (real atom)")
    a.fill_between(x, conf, succ, color="C3", alpha=0.2, label="phantom gap")
    a.axvline(best_mot, color="C1", ls="--", lw=1.5, label=f"optimal {best_mot:.0f} ms")
    a.set_xlabel("MOT formation time [ms]"); a.set_ylabel("%")
    a.set_ylim(0, 103); a.set_title("Loader verdict vs ground truth")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # (2) phantom rate + time-per-confirmed-atom vs MOT. The t/atom curve's
    # minimum IS the optimal MOT (star); its U-shape shows the trade-off.
    a = ax[0, 1]
    a.plot(x, phantom, "o-", color="C3", label="phantom rate")
    a.axhline(max_phantom * 100, color="k", ls=":", lw=1,
              label=f"{max_phantom*100:.0f}% tol")
    a.set_xlabel("MOT formation time [ms]")
    a.set_ylabel("phantom rate [%]", color="C3")
    a.tick_params(axis="y", labelcolor="C3")
    a.axvline(best_mot, color="C1", ls="--", lw=1.5)
    a2 = a.twinx()
    a2.plot(x, tpa, "D-", color="C4", label="time / confirmed atom")
    a2.plot(best_mot, tpa[best_i], "*", color="C1", ms=18, zorder=6)
    a2.set_ylabel("time per confirmed atom [ms]", color="C4")
    a2.tick_params(axis="y", labelcolor="C4")
    a.set_title(f"Phantoms + time/atom  (optimal {best_mot:.0f} ms)")
    a.legend(fontsize=8, loc="upper left")
    a2.legend(fontsize=8, loc="upper right")

    # (3) loader winning-count histogram, real vs phantom.
    a = ax[1, 0]
    s = d["success"] > 0.5
    c = d["confirmed"] > 0.5
    real = d["load_net"][s & c]
    phan = d["load_net"][s & ~c]
    hi = 1.0
    for v in (real, phan):
        if v.size:
            hi = max(hi, float(np.percentile(v, 99)))
    bins = np.linspace(0, hi * 1.1, 40)
    if real.size:
        a.hist(real, bins=bins, color="C2", alpha=0.6, label=f"real, N={real.size}")
    if phan.size:
        a.hist(phan, bins=bins, color="C3", alpha=0.6, label=f"phantom, N={phan.size}")
    cur = _sum_threshold(d)
    if np.isfinite(cur):
        a.axvline(cur, color="k", ls="--", lw=1.5, label=f"current cut = {cur:.0f}")
    if rec.get("ok"):
        a.axvline(rec["rec_sum"], color="C1", ls="-", lw=1.5,
                  label=f"recommend = {rec['rec_sum']:.0f}")
    a.set_xlabel("loader winning-gate counts (SPCM1+2)")
    a.set_ylabel("count"); a.set_title("Where the loader threshold should sit")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # (4) confirmation-image histogram, real vs phantom (ground-truth sanity).
    a = ax[1, 1]
    cr = d["confirm_net"][s & c]
    cp = d["confirm_net"][s & ~c]
    allc = d["confirm_net"][s]
    hi = float(np.percentile(allc, 99)) if allc.size else 1.0
    lo = float(np.percentile(allc, 1)) if allc.size else 0.0
    bins = np.linspace(min(lo, 0), hi * 1.1, 40)
    if cr.size:
        a.hist(cr, bins=bins, color="C2", alpha=0.6, label=f"confirmed, N={cr.size}")
    if cp.size:
        a.hist(cp, bins=bins, color="C3", alpha=0.6, label=f"not confirmed, N={cp.size}")
    a.set_xlabel("confirmation image net counts (signal - bg)")
    a.set_ylabel("count"); a.set_title("Ground-truth separation (confirm image)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.suptitle(f"Load confirmation -- RID {d['rid']}", fontsize=13)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140); print(f"saved {out}")
    else:
        plt.show()


# =========================================================================
#  DRIVER / CLI
# =========================================================================

def analyze(path, max_phantom=0.05, plot=True):
    d = load_run(path)
    if not d["has_confirmation"]:
        print(f"{d['file']} (RID {d['rid']}) has NO confirmation data "
              f"(det_confirmed missing). Re-run the confirmation version of "
              f"AtomLoadDeterministicTest.")
        return None
    rows = per_mot(d)
    rec = recommend_threshold(d, max_phantom=max_phantom)
    print_report(d, rows, rec)
    if plot:
        plot_report(d, rows, rec, max_phantom=max_phantom)
    return {"d": d, "rows": rows, "rec": rec}


def _resolve(arg):
    p = Path(arg)
    if p.exists():
        return str(p)
    try:
        rid = int(arg)
    except ValueError:
        sys.exit(f"Not a file or RID: {arg}")
    files = [f for f in find_runs() if _rid_of(f) == rid]
    if not files:
        sys.exit(f"No AtomLoadDeterministicTest run with RID {rid}.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None, help="RID or file path")
    ap.add_argument("--max-phantom", type=float, default=0.05,
                    help="phantom tolerance for the threshold recommendation (default 0.05)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.run:
        path = _resolve(args.run)
    else:
        files = find_runs()
        if not files:
            sys.exit(f"No AtomLoadDeterministicTest runs under {RESULTS_DIR}.")
        path = files[-1]
        print(f"(Using latest run: {Path(path).name})")
    analyze(path, max_phantom=args.max_phantom, plot=not args.no_plot)


if __name__ == "__main__":
    main()
