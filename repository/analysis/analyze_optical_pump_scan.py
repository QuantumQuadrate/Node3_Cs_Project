"""Analyze OpticalPumpScan runs: mF=0 pumping fidelity vs. one or two selectable
knobs ({t_optical_pump, f_pump_repumper, amp_pump_repumper_active}).

Reads a `*-OpticalPumpScan.h5`. At each grid point we have two interleaved configs:
pi (cfg=1) and no-pi reference (cfg=0). Among LOADED atoms,
  lost = 1 - retention   (atom blown away by F=4 blowaway = it ended in F=4).
The clean mF=0 transfer signal subtracts the F=4-leftover background:
  signal = lost_pi - lost_noPi   (lost_noPi = the repumper F=4 leftover).
analyze() auto-detects the shape:
  1-D (one axis) -> lost-fraction + signal curve; a saturating fit
      signal(t) = S_inf - (S_inf - S_0)*exp(-t/tau), tau & t_rise98 = tau*ln(50),
      ONLY when the axis is the pump time;
  2-D -> heatmaps of signal (maximize) and lost_noPi (minimize), best = argmax signal.

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
    """Return (data dict of op_* arrays, snapshot dict, rid).

    NOTE: write_results stores STRING-valued datasets (e.g. snapshot_axis_x /
    op_axis_x_name) as HDF5 GROUP ATTRIBUTES, not as datasets -- so we read both
    ds.keys() (numeric datasets) AND ds.attrs (string values)."""
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
        for k, v in ds.attrs.items():        # string-valued op_*/snapshot_* live here
            val = v.decode() if isinstance(v, bytes) else v
            if k.startswith("snapshot_"):
                snap[k[len("snapshot_"):]] = val
            elif k.startswith("op_"):
                data[k] = val
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


def get_xy(data, snap):
    """Return (x, y, x_name, y_name) for the run.

    New 2-D runs carry op_axis_x_val / op_axis_y_val + snapshot axis_x / axis_y.
    Old 1-D pump-time runs carry op_t_pump_us -> x = pump time, y = constant 0.
    """
    if "op_axis_x_val" in data:
        x = np.atleast_1d(data["op_axis_x_val"]).astype(float)
        y = (np.atleast_1d(data["op_axis_y_val"]).astype(float)
             if "op_axis_y_val" in data else np.zeros_like(x))
        x_name = str(snap.get("axis_x", "x"))
        y_name = str(snap.get("axis_y", "none"))
    else:
        x = np.atleast_1d(data["op_t_pump_us"]).astype(float)
        y = np.zeros_like(x)
        x_name, y_name = "t_optical_pump", "none"
    return x, y, x_name, y_name


def axis_label(name):
    """Human axis label (with unit) for a scan-knob name."""
    return {"t_optical_pump": "optical-pump time (us)",
            "f_pump_repumper": "pump-repumper freq (MHz)",
            "amp_pump_repumper_active": "pump-repumper amp",
            "f_pump_D1": "D1-pump freq (MHz)",
            "amp_pump_D1": "D1-pump amp"}.get(name, name)


# --------------------------------------------------------------------------
#  TABLES
# --------------------------------------------------------------------------
def retention_table(data, snap):
    """Filter loaded==1; return {(x, y, cfg): [n_retained, n_loaded]}."""
    x, y, _, _ = get_xy(data, snap)
    cfg = np.asarray(data["op_apply_pi"]).astype(int)
    loaded = np.asarray(data["op_atom_loaded"]).astype(int)
    retained = np.asarray(data["op_atom_retained"]).astype(int)
    m = loaded > 0
    x, y, cfg, retained = x[m], y[m], cfg[m], retained[m]
    table = {}
    for xx, yy, cc, ret in zip(x, y, cfg, retained):
        key = (round(float(xx), 4), round(float(yy), 4), int(cc))
        if key not in table:
            table[key] = [0, 0]
        table[key][0] += int(ret)
        table[key][1] += 1
    return table


def build_curves(data, snap):
    """1-D curves vs the X axis (marginalised over Y), keyed on `ts` (= X values):
        ts, lost_pi, lpi_lo, lpi_hi, n_pi, lost_no, lno_lo, lno_hi, n_no, signal, n_eff
    `lost = 1 - retention`; CIs are the (clamped) flipped Wilson CIs of retention.
    signal = lost_pi - lost_no (NaN where either config is missing)."""
    table2d = retention_table(data, snap)
    table = {}                              # collapse (x, y, cfg) -> (x, cfg)
    for (xk, yk, cfgk), v in table2d.items():
        k = (xk, cfgk)
        if k not in table:
            table[k] = [0, 0]
        table[k][0] += v[0]
        table[k][1] += v[1]
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


def build_grids_2d(data, snap):
    """2-D grids over (X, Y). Returns (xs, ys, M_signal, M_lost_no, M_lost_pi,
    best_xy, best_sig). M[iy, ix]; NaN where a cell lacks the needed config(s).
    best = argmax signal over cells with BOTH the pi and no-pi configs."""
    table = retention_table(data, snap)
    xs = sorted({k[0] for k in table})
    ys = sorted({k[1] for k in table})
    M_sig = np.full((len(ys), len(xs)), np.nan)
    M_lno = np.full((len(ys), len(xs)), np.nan)
    M_lpi = np.full((len(ys), len(xs)), np.nan)
    best_xy, best_sig = None, -np.inf
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            dpi = table.get((xv, yv, 1))
            dno = table.get((xv, yv, 0))
            if dpi and dpi[1] > 0:
                M_lpi[iy, ix] = 1.0 - dpi[0] / dpi[1]
            if dno and dno[1] > 0:
                M_lno[iy, ix] = 1.0 - dno[0] / dno[1]
            if dpi and dno and dpi[1] > 0 and dno[1] > 0:
                sig = M_lpi[iy, ix] - M_lno[iy, ix]
                M_sig[iy, ix] = sig
                if sig > best_sig:
                    best_sig = sig
                    best_xy = (xv, yv)
    return xs, ys, M_sig, M_lno, M_lpi, best_xy, best_sig


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
    """Dispatch on the scan shape: 1 unique Y -> 1-D curve (saturation fit only when
    the X axis is the pump time); >1 unique Y -> 2-D heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, snap, rid = load_run(path)
    x, y, x_name, y_name = get_xy(data, snap)
    n_total = int(x.size)
    n_loaded = int(np.sum(np.asarray(data.get("op_atom_loaded", [0])).astype(int)))
    print(f"RID {rid} | {n_total} points | {n_loaded} loaded | X={x_name}  Y={y_name}")
    if n_loaded == 0:
        print("No loaded atoms -- nothing to analyze (check loading / atom_threshold).")
        return

    n_y = len({round(float(v), 4) for v in y})
    if n_y > 1:
        _analyze_2d(path, data, snap, rid, x_name, y_name, plt, out_png)
    else:
        _analyze_1d(path, data, snap, rid, x_name, plt, out_png)


def _analyze_1d(path, data, snap, rid, x_name, plt, out_png):
    curves = build_curves(data, snap)
    ts = curves["ts"]
    is_time = (x_name == "t_optical_pump")
    xlabel = axis_label(x_name)

    print(f"  {x_name:>22}   lost_pi   lost_noPi   signal   n_pi  n_noPi")
    for i in range(len(ts)):
        print(f"  {ts[i]:22.4g}   {curves['lost_pi'][i]:7.3f}   {curves['lost_no'][i]:9.3f}   "
              f"{curves['signal'][i]:6.3f}   {int(curves['n_pi'][i]):4d}  {int(curves['n_no'][i]):4d}")

    fit = fit_saturation(ts, curves["signal"], weights=curves["n_eff"]) if is_time else {"ok": False}
    knee = empirical_knee(curves)
    t_max = float(np.nanmax(ts)) if len(ts) else 0.0

    if is_time and fit.get("ok"):
        rec = min(fit["t_rise98"], t_max) if t_max > 0 else fit["t_rise98"]
        clamp = " (clamped to scanned range)" if fit["t_rise98"] > t_max > 0 else ""
        print(f"\nFIT signal(t) = S_inf - (S_inf - S0) exp(-t/tau):")
        print(f"  S_inf = {fit['S_inf']:.3f}   S0 = {fit['S0']:.3f}   tau = {fit['tau']:.1f} us")
        print(f"  RECOMMENDED t_optical_pump ~ t_rise98 = tau*ln(50) = "
              f"{fit['t_rise98']:.1f} us  -> use {rec:.1f} us{clamp}")
    sig = curves["signal"]
    if np.isfinite(sig).any():
        ib = int(np.nanargmax(sig))
        print(f"  max signal = {sig[ib]:.3f} at {x_name} = {ts[ib]:g}")
    print("  (recommend-only: set the knob on the dashboard yourself; not written.)")

    # ---- plots ----
    fig, (axL, axS) = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)
    _errbar(axL, ts, curves["lost_pi"], curves["lpi_lo"], curves["lpi_hi"],
            "o-", "C0", "lost_pi (pi applied)", plt)
    _errbar(axL, ts, curves["lost_no"], curves["lno_lo"], curves["lno_hi"],
            "s-", "C1", "lost_noPi (reference)", plt)
    axL.set_xlabel(xlabel)
    axL.set_ylabel("lost fraction  (1 - retention)")
    axL.set_title("A. lost fraction: pi vs no-pi reference")
    axL.set_ylim(-0.02, 1.02)
    axL.legend(loc="best", fontsize=9)
    axL.grid(alpha=0.3)

    sig_half = np.sqrt(np.nan_to_num((curves["lpi_hi"] - curves["lpi_lo"]) / 2.0) ** 2
                       + np.nan_to_num((curves["lno_hi"] - curves["lno_lo"]) / 2.0) ** 2)
    axS.errorbar(ts, curves["signal"], yerr=sig_half, fmt="o", color="C3",
                 capsize=3, label="signal = lost_pi - lost_noPi")
    if is_time and fit.get("ok"):
        tt = np.linspace(0.0, t_max if t_max > 0 else 1.0, 300)
        yy = fit["S_inf"] - (fit["S_inf"] - fit["S0"]) * np.exp(-tt / fit["tau"])
        axS.plot(tt, yy, "-", color="C3", alpha=0.7, label=f"fit (tau={fit['tau']:.0f} us)")
        axS.axvline(min(fit["t_rise98"], t_max) if t_max > 0 else fit["t_rise98"],
                    color="k", ls="--", alpha=0.7, label=f"t_rise98 = {fit['t_rise98']:.0f} us")
        if knee is not None:
            axS.axvline(knee, color="g", ls=":", alpha=0.7, label=f"empirical knee = {knee:g} us")
    else:
        axS.plot(ts, curves["signal"], "-", color="C3", alpha=0.4)
    axS.set_xlabel(xlabel)
    axS.set_ylabel("clean mF=0 signal")
    axS.set_title("B. mF=0 transfer signal (background-subtracted)")
    axS.legend(loc="best", fontsize=9)
    axS.grid(alpha=0.3)

    fig.suptitle(f"OpticalPumpScan 1-D ({x_name}) -- RID {rid}", fontsize=13)
    if out_png is None:
        out_png = str(Path(path).with_name(f"optical_pump_scan_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


def _analyze_2d(path, data, snap, rid, x_name, y_name, plt, out_png):
    xs, ys, M_sig, M_lno, M_lpi, best_xy, best_sig = build_grids_2d(data, snap)
    xs, ys = np.asarray(xs), np.asarray(ys)
    print(f"  2-D grid: {len(xs)} X ({x_name}) x {len(ys)} Y ({y_name})")
    if best_xy is not None:
        ix = int(np.argmin(np.abs(xs - best_xy[0])))
        iy = int(np.argmin(np.abs(ys - best_xy[1])))
        print(f"  BEST signal = {best_sig:.3f} at {x_name}={best_xy[0]:g}, {y_name}={best_xy[1]:g}")
        print(f"    there: lost_noPi (F=4 leftover) = {M_lno[iy, ix]:.3f}, lost_pi = {M_lpi[iy, ix]:.3f}")
    else:
        print("  No cell has both pi and no-pi configs yet.")
    print("  (recommend-only: set f_pump_repumper / amp_pump_repumper_active on the dashboard.)")

    fig, (axS, axN) = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)
    _heatmap(axS, xs, ys, M_sig, "A. clean mF=0 signal (lost_pi - lost_noPi) -- maximize",
             best_xy, plt, x_name, y_name)
    _heatmap(axN, xs, ys, M_lno, "B. lost_noPi  (F=4 leftover -- minimize)",
             best_xy, plt, x_name, y_name)
    fig.suptitle(f"OpticalPumpScan 2-D ({x_name} x {y_name}) -- RID {rid}", fontsize=13)
    if out_png is None:
        out_png = str(Path(path).with_name(f"optical_pump_scan_RID{rid}.png"))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


def _heatmap(ax, xs, ys, M, title, best_xy, plt, x_name, y_name):
    im = ax.pcolormesh(xs, ys, M, shading="nearest", cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if best_xy is not None:
        ax.plot(best_xy[0], best_xy[1], "*", color="red", ms=16,
                markeredgecolor="white", label="best signal")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel(axis_label(x_name))
    ax.set_ylabel(axis_label(y_name))
    ax.set_title(title, fontsize=11)


def _errbar(ax, x, y, lo, hi, fmt, color, label, plt):
    yerr_lo = np.clip(np.nan_to_num(y - lo), 0.0, None)   # guard float-edge negatives
    yerr_hi = np.clip(np.nan_to_num(hi - y), 0.0, None)
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt=fmt, color=color,
                capsize=3, label=label)


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze(find_run(rid))
