"""Analyze OpticalPumpScan runs: mF=0 pumping fidelity vs. optical-pump TIME, and
the recommended (shortest-sufficient) t_optical_pump.

Reads a `*-OpticalPumpScan.h5`. For each pump time we have two interleaved
configs: pi (cfg=1) and no-pi reference (cfg=0). Among LOADED atoms,
  lost = 1 - retention   (atom blown away by F=4 blowaway = it ended in F=4).
The clean mF=0 transfer signal subtracts the F=4-leftover background:
  signal(t) = lost_pi(t) - lost_noPi(t).
signal(t) rises and SATURATES; the knee is the shortest pump time that still
prepares mF=0. We fit  signal(t) = S_inf - (S_inf - S_0) * exp(-t / tau)  and
report tau and t_rise98 = tau*ln(50) (time to close 98% of the S_0->S_inf rise)
as the recommended t_optical_pump. A fit-free empirical knee is printed too.

BLAS-free: only elementwise exp + sums + a hand-rolled 2x2 least-squares (the
artiq7_claude BLAS segfaults on np.linalg.inv / hangs curve_fit). But PREFER the
working-BLAS `python` on PATH; matplotlib savefig crashes in artiq7_claude. Run:

    python analyze_optical_pump_scan.py            # latest run
    python analyze_optical_pump_scan.py 712        # RID 712
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
        glob.glob(str(RESULTS_DIR / "**" / "*-OpticalPumpScan.h5"), recursive=True),
        key=lambda f: int(os.path.basename(f).split("-")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No *-OpticalPumpScan.h5 under {RESULTS_DIR}")
    if rid is None:
        return files[-1]
    for f in files:
        if int(os.path.basename(f).split("-")[0]) == int(rid):
            return f
    raise FileNotFoundError(f"No OpticalPumpScan .h5 for RID {rid}")


def load_run(path):
    """Return (data dict of op_* arrays, snapshot dict, rid)."""
    data, snap = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in ds.keys():
            if k.startswith("op_"):
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
#  TABLES
# --------------------------------------------------------------------------
def retention_table(data):
    """Filter loaded==1; return {(t_us, cfg): [n_retained, n_loaded]}."""
    t = np.asarray(data["op_t_pump_us"]).astype(float)
    cfg = np.asarray(data["op_apply_pi"]).astype(int)
    loaded = np.asarray(data["op_atom_loaded"]).astype(int)
    retained = np.asarray(data["op_atom_retained"]).astype(int)
    m = loaded > 0
    t, cfg, retained = t[m], cfg[m], retained[m]
    table = {}
    for tt, cc, ret in zip(t, cfg, retained):
        key = (round(float(tt), 2), int(cc))
        if key not in table:
            table[key] = [0, 0]
        table[key][0] += int(ret)
        table[key][1] += 1
    return table


def build_curves(data):
    """Return a dict of per-time arrays (all aligned to `ts`):
        ts, lost_pi, lpi_lo, lpi_hi, n_pi,
            lost_no, lno_lo, lno_hi, n_no,
            signal, n_eff
    `lost = 1 - retention`; CIs are the Wilson CIs of `lost` (= flipped Wilson of
    retention). signal = lost_pi - lost_no (NaN where either config is missing)."""
    table = retention_table(data)
    ts = sorted({k[0] for k in table})

    out = {k: [] for k in ("ts", "lost_pi", "lpi_lo", "lpi_hi", "n_pi",
                           "lost_no", "lno_lo", "lno_hi", "n_no",
                           "signal", "n_eff")}
    for t in ts:
        dpi = table.get((t, 1))
        dno = table.get((t, 0))
        # lost = 1 - retention -> p_lost = 1 - p_ret; CI flips around 1.
        # lost = 1 - retention; the flipped Wilson bounds are clamped around the
        # point value so a tiny float inversion at retention==0 or 1 (where the
        # Wilson c+/-h rounds just past the [0,1] edge) cannot make a yerr go
        # negative (matplotlib errorbar rejects negative yerr).
        if dpi and dpi[1] > 0:
            pr, lo, hi = wilson(dpi[0], dpi[1])
            lpi = 1.0 - pr
            lpi_lo, lpi_hi, n_pi = min(1.0 - hi, lpi), max(1.0 - lo, lpi), dpi[1]
        else:
            lpi, lpi_lo, lpi_hi, n_pi = np.nan, np.nan, np.nan, 0
        if dno and dno[1] > 0:
            pr, lo, hi = wilson(dno[0], dno[1])
            lno = 1.0 - pr
            lno_lo, lno_hi, n_no = min(1.0 - hi, lno), max(1.0 - lo, lno), dno[1]
        else:
            lno, lno_lo, lno_hi, n_no = np.nan, np.nan, np.nan, 0
        sig = (lpi - lno) if (n_pi > 0 and n_no > 0) else np.nan
        out["ts"].append(t)
        out["lost_pi"].append(lpi); out["lpi_lo"].append(lpi_lo); out["lpi_hi"].append(lpi_hi); out["n_pi"].append(n_pi)
        out["lost_no"].append(lno); out["lno_lo"].append(lno_lo); out["lno_hi"].append(lno_hi); out["n_no"].append(n_no)
        out["signal"].append(sig); out["n_eff"].append(min(n_pi, n_no))
    for k in out:
        out[k] = np.asarray(out[k], dtype=float)
    return out


# --------------------------------------------------------------------------
#  SATURATION FIT  (BLAS-free: tau grid-search + hand-rolled 2x2 weighted LS)
# --------------------------------------------------------------------------
def fit_saturation(ts, sig, weights=None, n_tau=600):
    """Fit signal(t) = S_inf - (S_inf - S_0) * exp(-t / tau).

    For each tau on a log grid, (S_inf, S_0) are LINEAR in the basis
    u = 1 - exp(-t/tau), v = exp(-t/tau)  (signal = S_inf*u + S_0*v); solved by a
    closed-form 2x2 weighted least-squares (NO np.linalg -- BLAS is broken).
    Returns dict(S_inf, S0, tau, t_rise98, ssr, ok) or {"ok": False}."""
    m = np.isfinite(ts) & np.isfinite(sig)
    ts, sig = np.asarray(ts)[m], np.asarray(sig)[m]
    w = np.ones_like(sig) if weights is None else np.asarray(weights, dtype=float)[m]
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    if ts.size < 3:
        return {"ok": False}

    pos = ts[ts > 0]
    t_lo = (pos.min() * 0.1) if pos.size else 1.0
    t_hi = max(ts.max() * 5.0, t_lo * 10.0)
    taus = np.logspace(np.log10(max(t_lo, 1e-3)), np.log10(t_hi), n_tau)

    best = {"ok": False, "ssr": np.inf}
    sw = w.sum()
    for tau in taus:
        v = np.exp(-ts / tau)
        u = 1.0 - v
        uu = float((w * u * u).sum()); uv = float((w * u * v).sum()); vv = float((w * v * v).sum())
        uy = float((w * u * sig).sum()); vy = float((w * v * sig).sum())
        det = uu * vv - uv * uv
        if abs(det) < 1e-12:
            continue
        S_inf = (uy * vv - vy * uv) / det
        S0 = (uu * vy - uv * uy) / det
        resid = sig - (S_inf * u + S0 * v)
        ssr = float((w * resid * resid).sum()) / sw
        if ssr < best["ssr"]:
            best = {"ok": True, "ssr": ssr, "tau": float(tau),
                    "S_inf": float(S_inf), "S0": float(S0)}
    if best["ok"]:
        best["t_rise98"] = best["tau"] * np.log(50.0)   # close 98% of the S0->S_inf rise
    return best


def empirical_knee(curves):
    """Fit-free: smallest pump time whose signal upper-CI reaches the max signal
    (i.e. first time the signal is statistically indistinguishable from its peak).
    Uses Wilson CIs of `lost` propagated to the signal (independent configs)."""
    ts, sig = curves["ts"], curves["signal"]
    # signal CI half-width ~ quadrature of the two lost CIs.
    half = np.sqrt(np.nan_to_num((curves["lpi_hi"] - curves["lpi_lo"]) / 2.0) ** 2
                   + np.nan_to_num((curves["lno_hi"] - curves["lno_lo"]) / 2.0) ** 2)
    m = np.isfinite(sig)
    if not m.any():
        return None
    s_max = np.nanmax(sig)
    for i in range(len(ts)):
        if m[i] and (sig[i] + half[i] >= s_max):
            return float(ts[i])
    return float(ts[np.nanargmax(sig)])


# --------------------------------------------------------------------------
#  DRIVER
# --------------------------------------------------------------------------
def analyze(path, out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, snap, rid = load_run(path)
    n_total = len(np.asarray(data.get("op_t_pump_us", [])))
    n_loaded = int(np.sum(np.asarray(data.get("op_atom_loaded", [0])).astype(int)))
    print(f"RID {rid} | {n_total} points | {n_loaded} loaded")
    if n_loaded == 0:
        print("No loaded atoms -- nothing to analyze (check loading / atom_threshold).")
        return

    curves = build_curves(data)
    ts = curves["ts"]
    print("  t_pump(us)   lost_pi   lost_noPi   signal   n_pi  n_noPi")
    for i in range(len(ts)):
        print(f"  {ts[i]:9.2f}   {curves['lost_pi'][i]:7.3f}   {curves['lost_no'][i]:9.3f}   "
              f"{curves['signal'][i]:6.3f}   {int(curves['n_pi'][i]):4d}  {int(curves['n_no'][i]):4d}")

    fit = fit_saturation(ts, curves["signal"], weights=curves["n_eff"])
    knee = empirical_knee(curves)
    t_max = float(np.nanmax(ts)) if len(ts) else 0.0

    if fit.get("ok"):
        rec = min(fit["t_rise98"], t_max) if t_max > 0 else fit["t_rise98"]
        clamp = " (clamped to scanned range)" if fit["t_rise98"] > t_max > 0 else ""
        print(f"\nFIT signal(t) = S_inf - (S_inf - S0) exp(-t/tau):")
        print(f"  S_inf (mF=0 fidelity ceiling) = {fit['S_inf']:.3f}   S0 = {fit['S0']:.3f}   "
              f"tau = {fit['tau']:.1f} us")
        print(f"  RECOMMENDED t_optical_pump ~ t_rise98 = tau*ln(50) = "
              f"{fit['t_rise98']:.1f} us  -> use {rec:.1f} us{clamp}")
    else:
        print("\nFIT: not enough valid points for a saturating fit.")
    if knee is not None:
        print(f"  empirical knee (signal within CI of its peak): t_pump = {knee:g} us")
    print("  (recommend-only: set t_optical_pump on the dashboard yourself; this script does NOT write it.)")

    # ---- plots ----
    fig, (axL, axS) = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)

    # Panel A: lost_pi and lost_noPi vs t
    _errbar(axL, ts, curves["lost_pi"], curves["lpi_lo"], curves["lpi_hi"],
            "o-", "C0", "lost_pi (pi applied)", plt)
    _errbar(axL, ts, curves["lost_no"], curves["lno_lo"], curves["lno_hi"],
            "s-", "C1", "lost_noPi (reference)", plt)
    axL.set_xlabel("optical-pump time (us)")
    axL.set_ylabel("lost fraction  (1 - retention)")
    axL.set_title("A. lost fraction: pi vs no-pi reference")
    axL.set_ylim(-0.02, 1.02)
    axL.legend(loc="best", fontsize=9)
    axL.grid(alpha=0.3)

    # Panel B: signal + fit + recommended time
    sig_half = np.sqrt(np.nan_to_num((curves["lpi_hi"] - curves["lpi_lo"]) / 2.0) ** 2
                       + np.nan_to_num((curves["lno_hi"] - curves["lno_lo"]) / 2.0) ** 2)
    axS.errorbar(ts, curves["signal"], yerr=sig_half, fmt="o", color="C3",
                 capsize=3, label="signal = lost_pi - lost_noPi")
    if fit.get("ok"):
        tt = np.linspace(0.0, t_max if t_max > 0 else 1.0, 300)
        yy = fit["S_inf"] - (fit["S_inf"] - fit["S0"]) * np.exp(-tt / fit["tau"])
        axS.plot(tt, yy, "-", color="C3", alpha=0.7,
                 label=f"fit (tau={fit['tau']:.0f} us)")
        axS.axvline(min(fit["t_rise98"], t_max) if t_max > 0 else fit["t_rise98"],
                    color="k", ls="--", alpha=0.7,
                    label=f"t_rise98 = {fit['t_rise98']:.0f} us")
    if knee is not None:
        axS.axvline(knee, color="g", ls=":", alpha=0.7, label=f"empirical knee = {knee:g} us")
    axS.set_xlabel("optical-pump time (us)")
    axS.set_ylabel("clean mF=0 signal")
    axS.set_title("B. mF=0 transfer signal (background-subtracted)")
    axS.legend(loc="best", fontsize=9)
    axS.grid(alpha=0.3)

    fig.suptitle(f"Optical-pump time scan -- RID {rid}  "
                 f"(live t_optical_pump = {float(snap.get('t_optical_pump', 0)) * 1e6:.1f} us)",
                 fontsize=13)

    if out_png is None:
        out_png = str(Path(path).with_name(f"optical_pump_scan_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


def _errbar(ax, x, y, lo, hi, fmt, color, label, plt):
    yerr_lo = np.clip(np.nan_to_num(y - lo), 0.0, None)   # guard float-edge negatives
    yerr_hi = np.clip(np.nan_to_num(hi - y), 0.0, None)
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt=fmt, color=color,
                capsize=3, label=label)


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze(find_run(rid))
