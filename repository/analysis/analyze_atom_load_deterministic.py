"""Analysis for AtomLoadDeterministicTest -- the atom-loading time budget.

What this answers
-----------------
Each load splits into two waits:
  * MOT formation   -- a FIXED delay (the loader's t_mot_formation, default
    100 ms). This is NOT measured per shot; it is a known constant.
  * poll (+ compression) -- the variable part: how long the loader watches the
    SPCMs until an atom appears. This IS measured per shot and stored as
    `det_load_time_s` (poll only -- t_before_atom_mu is taken AFTER the MOT
    formation, so the 100 ms is excluded).

So the total wall time per load ~ mot_formation + det_load_time_s (+ small
per-shot overhead: the background shot and the confirmation image, which this
experiment adds but the real experiments mostly don't). The headline question
is which of the two waits dominates -- that decides whether shortening the MOT
is worth it.

Single-run mode prints the budget + success + n_tries, and plots:
  (1) the MOT-vs-poll time split, (2) poll-time histogram, (3) n_tries
  histogram, (4) threshold calibration (MOT-only vs atom-loaded count
  histograms with the current threshold line).

Scan mode (several runs at different MOT-formation times) plots success rate,
poll time and TOTAL load time vs MOT-formation time, and reports the shortest
MOT that still loads reliably -- the point of the t_MOT2PGC_override_ms sweep.

MOT-formation value per run:
  * runs taken AFTER the loader was made argument-driven store the true value
    in `det_mot_formation_s_used` -- used directly.
  * OLDER runs stored `det_t_MOT2PGC_used`, which was NEVER the MOT time the
    deterministic loader used (it hardcoded 100 ms). For those we fall back to
    100 ms (the real value) and flag it, NOT the stale dataset number.
  Override with --mot-ms if you know a specific old run was different.

Usage::

    python analyze_atom_load_deterministic.py                 # latest run
    python analyze_atom_load_deterministic.py 3600            # one RID
    python analyze_atom_load_deterministic.py 3573 3574 3575  # scan across RIDs
    python analyze_atom_load_deterministic.py --scan 5        # last 5 runs as a scan
    python analyze_atom_load_deterministic.py --mot-ms 100    # force MOT value

A single run from the SCAN experiment (per-shot MOT tags, det_mot_formation_s)
is auto-detected: pointing at that one RID plots the whole MOT scan from it.
Old single-MOT-per-run files still analyze as one point, and several of them
can still be combined with the multi-RID / --scan forms above.
"""

import sys, os, glob, argparse
from pathlib import Path

import numpy as np
import h5py

RESULTS_DIR = Path(
    r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes\repository\results"
)

# MOT value assumed for runs predating the argument-driven loader (it hardcoded
# 100 ms regardless of the t_MOT2PGC dataset).
LEGACY_MOT_S = 0.100


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


def load_run(path, mot_ms=None):
    """Load one AtomLoadDeterministicTest .h5 into a dict of arrays + scalars."""
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"]

        def arr(k):
            return np.asarray(ds[k][()]).astype(float) if k in ds else None

        def scal(k, default=np.nan):
            return float(ds[k][()]) if k in ds else default

        d = {
            "success":        arr("det_load_success"),
            "n_tries":        arr("det_n_tries"),
            "n_fb":           arr("det_n_feedback_rounds"),
            "s1_load":        arr("det_spcm1_at_load"),
            "s2_load":        arr("det_spcm2_at_load"),
            "poll_s":         arr("det_load_time_s"),
            "first_unloaded": arr("det_first_unloaded_counts"),
            "confirmed":      arr("det_confirmed"),
            "confirm1":       arr("det_confirm_count1"),
            "confirm2":       arr("det_confirm_count2"),
            "bg1":            arr("det_confirm_bg1"),
            "bg2":            arr("det_confirm_bg2"),
            "gate_s":         scal("det_gate_time_used"),
            "threshold_cps":  scal("det_threshold_rate_used"),
        }
        dur = np.nan
        if "metadata" in hf and "duration_seconds" in hf["metadata"].attrs:
            dur = float(hf["metadata"].attrs["duration_seconds"])

    # MOT-formation time for this run.
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"]
        # NEW single-file-scan runs tag EVERY shot with its MOT time; this array
        # is what makes one-file multi-point analysis possible.
        d["mot_per_shot"] = (np.asarray(ds["det_mot_formation_s"][()]).astype(float)
                             if "det_mot_formation_s" in ds else None)
        if mot_ms is not None:
            d["mot_s"] = float(mot_ms) * 1e-3
            d["mot_src"] = "override"
        elif d["mot_per_shot"] is not None and len(d["mot_per_shot"]):
            # scalar fallback for single-point summaries: mean of the per-shot values
            d["mot_s"] = float(np.mean(d["mot_per_shot"]))
            d["mot_src"] = "per-shot"
        elif "det_mot_formation_s_used" in ds:
            d["mot_s"] = float(ds["det_mot_formation_s_used"][()])
            d["mot_src"] = "logged"
        else:
            d["mot_s"] = LEGACY_MOT_S
            d["mot_src"] = "legacy-100ms (pre-edit run; det_t_MOT2PGC_used ignored)"

    d["duration_s"] = dur
    d["rid"] = _rid_of(path)
    d["file"] = Path(path).name
    return d


# =========================================================================
#  STATS
# =========================================================================

def _med_mean_max(a):
    a = np.asarray(a, float)
    return float(np.median(a)), float(a.mean()), float(a.max())


def summarize(d):
    succ = d["success"]
    n = len(succ)
    poll = d["poll_s"]
    nt = d["n_tries"]
    mot = d["mot_s"]

    poll_med, poll_mean, poll_max = _med_mean_max(poll)
    nt_med, nt_mean, nt_max = _med_mean_max(nt)

    # Total per-load wall time = fixed MOT + measured poll.
    total = mot + poll
    tot_med, tot_mean, tot_max = _med_mean_max(total)

    # Count-level threshold (the loader uses rate = 0.5*(c0+c1)/gate, so the
    # equivalent SUM-count threshold is 2 * rate_threshold * gate).
    gate = d["gate_s"]
    sum_thr = (2.0 * d["threshold_cps"] * gate
               if np.isfinite(d["threshold_cps"]) and np.isfinite(gate) else np.nan)

    # MOT-only sample: first-try counts on shots that did NOT win on try 1
    # (so the first try was genuinely atom-free). atom-loaded sample: the
    # winning-gate sum on successful shots.
    atom_sum = (d["s1_load"] + d["s2_load"])
    atom_loaded = atom_sum[(succ > 0.5) & (atom_sum > 0)]
    mot_only = d["first_unloaded"][nt > 1]

    # Ground-truth confirmed-load fraction (confirmation image), if present.
    conf = d.get("confirmed")
    confirmed_rate = (float(np.asarray(conf, float).mean())
                      if conf is not None and len(conf) else np.nan)

    return {
        "file": d["file"], "rid": d["rid"], "n": n,
        "mot_s": mot, "mot_src": d["mot_src"],
        "gate_s": gate, "threshold_cps": d["threshold_cps"], "sum_thr": sum_thr,
        "success_rate": float(succ.mean()),
        "confirmed_rate": confirmed_rate,
        "n_timeouts": int((succ < 0.5).sum()),
        "poll_med": poll_med, "poll_mean": poll_mean, "poll_max": poll_max,
        "ntries_med": nt_med, "ntries_mean": nt_mean, "ntries_max": nt_max,
        "total_med": tot_med, "total_mean": tot_mean, "total_max": tot_max,
        "mot_frac_med": (mot / tot_med if tot_med > 0 else np.nan),
        "fb_mean": float(d["n_fb"].mean()), "fb_frac": float((d["n_fb"] > 0).mean()),
        "duration_s": d["duration_s"],
        "loads_per_s": (n / d["duration_s"]
                        if d["duration_s"] and np.isfinite(d["duration_s"]) else np.nan),
        "atom_loaded": atom_loaded, "mot_only": mot_only,
    }


def distinct_mots(d):
    """Sorted distinct MOT values (ms, rounded) tagged on the shots of a
    single-file scan, or [] if the file is not a single-file scan."""
    mot = d.get("mot_per_shot")
    if mot is None or not len(mot):
        return []
    return sorted(set(np.round(np.asarray(mot, float) * 1e3).astype(int)))


def summarize_file_scan(d):
    """Split a single-file scan (per-shot det_mot_formation_s) into one summary
    row per distinct MOT value -- same row shape as summarize(), so print_scan /
    plot_scan work on it unchanged."""
    mot = d.get("mot_per_shot")
    if mot is None:
        raise ValueError("not a single-file scan (no det_mot_formation_s dataset)")
    mot = np.asarray(mot, float)
    arr_keys = ("success", "n_tries", "n_fb", "s1_load", "s2_load", "poll_s",
                "first_unloaded", "confirmed", "confirm1", "confirm2", "bg1", "bg2")
    rows = []
    for val in sorted(set(np.round(mot, 6))):
        mask = np.isclose(mot, val)
        sub = dict(d)
        for k in arr_keys:
            if d.get(k) is not None:
                sub[k] = np.asarray(d[k])[mask]
        sub["mot_s"] = float(val)
        sub["mot_src"] = "file-scan"
        rows.append(summarize(sub))
    return sorted(rows, key=lambda r: r["mot_s"])


# =========================================================================
#  REPORTING
# =========================================================================

def print_summary(s):
    print(f"File: {s['file']}   RID={s['rid']}   shots={s['n']}")
    print(f"  MOT formation      : {s['mot_s']*1e3:.0f} ms   [{s['mot_src']}]")
    print(f"  gate / threshold   : {s['gate_s']*1e6:.0f} us   /   {s['threshold_cps']:.3e} counts/s"
          f"   (= {s['sum_thr']:.0f} summed counts/gate)")
    print()
    print(f"  success rate       : {s['success_rate']*100:.1f} %"
          f"   ({s['n_timeouts']} timeouts of {s['n']})   [loader's own verdict]")
    if np.isfinite(s.get("confirmed_rate", np.nan)):
        print(f"  CONFIRMED atom     : {s['confirmed_rate']*100:.1f} %"
              f"   [ground-truth image; gap vs success = false triggers]")
    print(f"  n_tries  med/mean  : {s['ntries_med']:.0f} / {s['ntries_mean']:.1f}   (max {s['ntries_max']:.0f})")
    print(f"  poll time med/mean : {s['poll_med']*1e3:.0f} / {s['poll_mean']*1e3:.0f} ms   (max {s['poll_max']*1e3:.0f} ms)")
    print(f"  feedback fallback  : {s['fb_frac']*100:.0f} % of shots  (mean {s['fb_mean']:.2f} rounds)")
    print()
    print(f"  TIME BUDGET per load (median):")
    print(f"     MOT formation   = {s['mot_s']*1e3:6.0f} ms   ({s['mot_frac_med']*100:.0f} %)")
    print(f"     poll (+compress)= {s['poll_med']*1e3:6.0f} ms   ({(1-s['mot_frac_med'])*100:.0f} %)")
    print(f"     -------------------------------")
    print(f"     total per load  = {s['total_med']*1e3:6.0f} ms   (mean {s['total_mean']*1e3:.0f} ms)")
    if np.isfinite(s["loads_per_s"]):
        print(f"  wall rate          : {s['loads_per_s']:.2f} loads/s"
              f"   (incl. this test's bg + confirm-image overhead)")
    # One-line verdict.
    if np.isfinite(s["mot_frac_med"]):
        if s["mot_frac_med"] > 0.6:
            print(f"\n  => MOT formation dominates ({s['mot_frac_med']*100:.0f}%). "
                  f"Scan t_MOT2PGC_override_ms DOWN to find the shortest MOT that still loads.")
        elif s["mot_frac_med"] < 0.4:
            print(f"\n  => Poll wait dominates ({(1-s['mot_frac_med'])*100:.0f}%). "
                  f"The lever is the loading RATE (beam power / alignment / MOT size), not the MOT delay.")
        else:
            print(f"\n  => MOT and poll are comparable; both matter.")


def print_scan(rows):
    has_conf = any(np.isfinite(r.get("confirmed_rate", np.nan)) for r in rows)
    print("\n=== MOT-formation scan ===")
    hdr = f"  {'RID':>6}  {'MOT ms':>7}  {'succ %':>7}"
    if has_conf:
        hdr += f"  {'CONF %':>7}"
    hdr += f"  {'poll ms':>8}  {'total ms':>9}  {'n_tries':>8}"
    print(hdr)
    for r in rows:
        line = f"  {r['rid']:>6}  {r['mot_s']*1e3:>7.0f}  {r['success_rate']*100:>7.1f}"
        if has_conf:
            cr = r.get("confirmed_rate", np.nan)
            line += (f"  {cr*100:>7.1f}" if np.isfinite(cr) else f"  {'--':>7}")
        line += f"  {r['poll_med']*1e3:>8.0f}  {r['total_med']*1e3:>9.0f}  {r['ntries_med']:>8.0f}"
        print(line)

    # Optimize over points that actually produce REAL atoms. With a confirmation
    # image, gate on confirmed-load rate (immune to the loader false-triggering on
    # a bright MOT); otherwise fall back to the loader's own success.
    if has_conf:
        ok = [r for r in rows if r.get("confirmed_rate", 0.0) >= 0.90]
        gate = "confirmed-atom >=90%"
    else:
        ok = [r for r in rows if r["success_rate"] >= 0.99]
        gate = "success >=99%"
    if not ok:
        print(f"\n  No scanned point reached {gate} -- "
              "loading/detection fails across this whole range.")
        return

    # Among qualifying points, the optimizer is MIN TOTAL TIME (MOT + poll).
    fast = min(ok, key=lambda r: r["total_med"])
    floor = min(ok, key=lambda r: r["mot_s"])
    cur = max(rows, key=lambda r: r["mot_s"])
    print(f"\n  (optimizing over points with {gate})")
    print(f"  Fastest TOTAL per load : MOT {fast['mot_s']*1e3:.0f} ms "
          f"(RID {fast['rid']}), {fast['total_med']*1e3:.0f} ms/load "
          f"= {fast['mot_s']*1e3:.0f} MOT + {fast['poll_med']*1e3:.0f} poll.")
    save = cur["total_med"] - fast["total_med"]
    if save > 0:
        print(f"                           ~{save*1e3:.0f} ms/load faster than the "
              f"longest MOT scanned ({cur['mot_s']*1e3:.0f} ms, {cur['total_med']*1e3:.0f} ms).")
    if floor["mot_s"] < fast["mot_s"]:
        print(f"  Qualifying floor       : down to {floor['mot_s']*1e3:.0f} ms MOT, "
              f"but that costs {floor['total_med']*1e3:.0f} ms/load -- not worth it.")
    if has_conf:
        bad = [r for r in rows if np.isfinite(r.get("confirmed_rate", np.nan))
               and r["confirmed_rate"] < 0.90]
        if bad:
            mots = ", ".join(f"{r['mot_s']*1e3:.0f}" for r in bad)
            print(f"  WARNING: {len(bad)} MOT point(s) [{mots} ms] show <90% CONFIRMED atoms "
                  f"despite high loader success -- the loader is false-triggering on the "
                  f"bright MOT there. Excluded from the optimum, and BAD for the real "
                  f"experiment (phantom heralds). Keep MOT short enough to stay confirmed.")


# =========================================================================
#  PLOTS
# =========================================================================

def plot_single(s, out=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (1) time budget split.
    a = ax[0, 0]
    a.bar(["per load"], [s["mot_s"] * 1e3], color="C3", label="MOT formation")
    a.bar(["per load"], [s["poll_med"] * 1e3], bottom=[s["mot_s"] * 1e3],
          color="C0", label="poll (+compress), median")
    a.set_ylabel("time [ms]")
    a.set_title(f"Load time budget  (MOT = {s['mot_frac_med']*100:.0f}% of {s['total_med']*1e3:.0f} ms)")
    a.legend()
    a.grid(alpha=0.3, axis="y")

    # (2) poll-time histogram.
    a = ax[0, 1]
    poll_ms = s.get("_poll_ms")
    poll_ms = np.asarray(poll_ms) if poll_ms is not None else None
    if poll_ms is not None and poll_ms.size:
        a.hist(poll_ms, bins=40, color="C0", alpha=0.8, edgecolor="k", lw=0.3)
        a.axvline(np.median(poll_ms), color="C3", ls="--",
                  label=f"median {np.median(poll_ms):.0f} ms")
        a.legend()
    a.set_xlabel("poll time [ms]  (excludes MOT formation)")
    a.set_ylabel("shots")
    a.set_title("Polling wait distribution")
    a.grid(alpha=0.3)

    # (3) n_tries histogram.
    a = ax[1, 0]
    nt = s.get("_n_tries")
    nt = np.asarray(nt) if nt is not None else None
    if nt is not None and nt.size:
        mx = int(nt.max())
        a.hist(nt, bins=np.arange(0.5, mx + 1.5), color="C2", alpha=0.8,
               edgecolor="k", lw=0.3)
        a.axvline(np.median(nt), color="C3", ls="--",
                  label=f"median {np.median(nt):.0f}")
        a.legend()
    a.set_xlabel("polling gates until atom (n_tries)")
    a.set_ylabel("shots")
    a.set_title("Tries to catch an atom")
    a.grid(alpha=0.3)

    # (4) threshold calibration.
    a = ax[1, 1]
    mot_only = s["mot_only"]
    atom = s["atom_loaded"]
    hi = 1.0
    for v in (mot_only, atom):
        if v is not None and len(v):
            hi = max(hi, float(np.percentile(v, 99)))
    bins = np.linspace(0, hi * 1.1, 40)
    if mot_only is not None and len(mot_only):
        a.hist(mot_only, bins=bins, color="C1", alpha=0.6,
               label=f"MOT only (1st try), N={len(mot_only)}")
    if atom is not None and len(atom):
        a.hist(atom, bins=bins, color="C2", alpha=0.6,
               label=f"atom loaded, N={len(atom)}")
    if np.isfinite(s["sum_thr"]):
        a.axvline(s["sum_thr"], color="k", ls="--",
                  label=f"threshold = {s['sum_thr']:.0f} cts")
    a.set_xlabel("SPCM1+SPCM2 counts in a gate")
    a.set_ylabel("count")
    a.set_title("Threshold calibration\n(valley between the two -> good threshold)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.suptitle(f"AtomLoadDeterministicTest  RID {s['rid']}   "
                 f"success {s['success_rate']*100:.0f}%   "
                 f"{s['total_med']*1e3:.0f} ms/load", fontsize=12)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140); print(f"saved {out}")
    else:
        plt.show()


def _group_by_mot(rows):
    """Collapse runs that share a MOT time (nearest ms) into one point each.

    Returns a sorted list of dicts carrying, per MOT value, the MEAN across its
    runs plus the min/max (for error bars) and the raw per-run values (for a
    faint scatter overlay). Averaging repeats smooths run-to-run loading drift,
    and the error bar shows how big that drift was."""
    groups = {}
    for r in rows:
        groups.setdefault(round(r["mot_s"] * 1e3), []).append(r)

    def stat(g, field, scale):
        v = np.array([gr[field] * scale for gr in g], dtype=float)
        return {"mean": float(v.mean()), "min": float(v.min()),
                "max": float(v.max()), "raw": v}

    out = []
    for mot_ms in sorted(groups):
        g = groups[mot_ms]
        out.append({
            "mot_ms": mot_ms, "n": len(g),
            "succ": stat(g, "success_rate", 100.0),
            "poll": stat(g, "poll_med", 1e3),
            "total": stat(g, "total_med", 1e3),
        })
    return out


def plot_scan(rows, out=None, show_points=True):
    """MOT-formation scan, one averaged point per MOT time. Error bars = the
    min..max spread of repeats (the run-to-run loading drift); faint dots are
    the individual runs (set show_points=False to hide)."""
    import matplotlib.pyplot as plt
    G = _group_by_mot(rows)
    x = np.array([g["mot_ms"] for g in G], dtype=float)

    def mean_err(key):
        m = np.array([g[key]["mean"] for g in G])
        lo = np.array([g[key]["mean"] - g[key]["min"] for g in G])
        hi = np.array([g[key]["max"] - g[key]["mean"] for g in G])
        return m, np.vstack([lo, hi])

    def scatter(ax, key, color):
        if not show_points:
            return
        for g in G:
            if g["n"] > 1:
                ax.plot([g["mot_ms"]] * g["n"], g[key]["raw"],
                        ".", color=color, alpha=0.3, ms=7, zorder=1)

    succ_m, succ_e = mean_err("succ")
    poll_m, poll_e = mean_err("poll")
    total_m, total_e = mean_err("total")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    scatter(a1, "succ", "C2")
    a1.errorbar(x, succ_m, yerr=succ_e, fmt="o-", color="C2", capsize=3, zorder=3)
    a1.axhline(99, color="C3", ls="--", lw=1, label="99% target")
    a1.set_xlabel("MOT formation time [ms]")
    a1.set_ylabel("success rate [%]")
    a1.set_title("Loading reliability vs MOT time")
    a1.set_ylim(min(90, succ_m.min() - 2), 101)
    a1.legend(); a1.grid(alpha=0.3)

    scatter(a2, "total", "C0")
    scatter(a2, "poll", "C4")
    a2.errorbar(x, total_m, yerr=total_e, fmt="o-", color="C0",
                capsize=3, zorder=3, label="total per load (avg)")
    a2.errorbar(x, poll_m, yerr=poll_e, fmt="s--", color="C4",
                capsize=3, zorder=3, label="poll only (avg)")
    a2.plot(x, x, ":", color="C3", lw=1, label="MOT only (y=x)")
    # Mark the total-time minimum.
    imin = int(np.argmin(total_m))
    a2.annotate(f"min {total_m[imin]:.0f} ms\n@ {x[imin]:.0f} ms MOT",
                xy=(x[imin], total_m[imin]),
                xytext=(x[imin], total_m[imin] + 0.18 * (total_m.max() - total_m.min() + 1)),
                ha="center", fontsize=9, color="C0",
                arrowprops=dict(arrowstyle="->", color="C0"))
    a2.set_xlabel("MOT formation time [ms]")
    a2.set_ylabel("time per load [ms]")
    a2.set_title("Load time vs MOT time")
    a2.legend(); a2.grid(alpha=0.3)

    npts = sum(g["n"] for g in G)
    fig.suptitle(f"AtomLoadDeterministicTest -- MOT-formation scan "
                 f"({len(G)} MOT values, {npts} runs, averaged)", fontsize=12)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140); print(f"saved {out}")
    else:
        plt.show()


# =========================================================================
#  CLI
# =========================================================================

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


def analyze_scan_file(path, plot=True):
    """Analyze ONE single-file scan (per-shot MOT tags) as a MOT scan."""
    d = load_run(path)
    rows = summarize_file_scan(d)
    print(f"File: {d['file']}   RID={d['rid']}   "
          f"{len(rows)} MOT points, {len(d['success'])} shots total "
          f"({len(d['success']) // max(len(rows),1)} per point)")
    print_scan(rows)
    if plot:
        plot_scan(rows)
    return rows


def analyze_single(path, mot_ms=None, plot=True):
    d = load_run(path, mot_ms=mot_ms)
    # If this one file is itself a multi-point scan, analyze it as a scan.
    if mot_ms is None and len(distinct_mots(d)) > 1:
        return analyze_scan_file(path, plot=plot)
    s = summarize(d)
    # stash raw arrays the plot wants (kept out of summarize's return contract).
    s["_poll_ms"] = d["poll_s"] * 1e3
    s["_n_tries"] = d["n_tries"]
    print_summary(s)
    if plot:
        plot_single(s)
    return s


def analyze_scan(paths, mot_ms=None, plot=True):
    rows = [summarize(load_run(p, mot_ms=mot_ms)) for p in paths]
    rows = sorted(rows, key=lambda r: r["mot_s"])
    print_scan(rows)
    if plot:
        plot_scan(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="RIDs or file paths (2+ => scan)")
    ap.add_argument("--scan", type=int, metavar="N",
                    help="treat the last N runs as a MOT-formation scan")
    ap.add_argument("--mot-ms", type=float, default=None,
                    help="force the MOT-formation value (ms) for all runs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    plot = not args.no_plot

    if args.scan:
        files = find_runs()
        if len(files) < 1:
            sys.exit(f"No AtomLoadDeterministicTest runs under {RESULTS_DIR}.")
        analyze_scan(files[-args.scan:], mot_ms=args.mot_ms, plot=plot)
    elif len(args.runs) >= 2:
        analyze_scan([_resolve(a) for a in args.runs], mot_ms=args.mot_ms, plot=plot)
    else:
        if args.runs:
            path = _resolve(args.runs[0])
        else:
            files = find_runs()
            if not files:
                sys.exit(f"No AtomLoadDeterministicTest runs under {RESULTS_DIR}.")
            path = files[-1]
            print(f"(Using latest run: {Path(path).name})")
        analyze_single(path, mot_ms=args.mot_ms, plot=plot)


if __name__ == "__main__":
    main()
