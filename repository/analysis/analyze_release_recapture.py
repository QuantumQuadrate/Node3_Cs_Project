"""Release-and-recapture analysis: single-atom temperature in uK.

Reads `*-ReleaseRecapture.h5` (experiments/release_recapture.py).

Two results, deliberately separated:

1. MODEL-FREE: the recapture curve P(t) = recaptured / loaded per release time
   (Wilson intervals), and the half-recapture time t_1/2 where P(t)/P(0) crosses
   0.5. Needs no trap parameters. Good for comparing runs (longer t_1/2 =
   colder), useless as an absolute number.

2. TEMPERATURE: maximum-likelihood fit of the curve to a Monte Carlo of a
   thermal atom in the 3D Gaussian FORT (Tuchendler et al., PRA 78, 033425
   (2008)). Per temperature T:
     - draw positions from the harmonic approximation of the trap and
       Maxwell-Boltzmann velocities at T;
     - fly ballistically (with gravity) for t;
     - recaptured if 1/2 m v^2 + U(r, z) < 0 in the FULL Gaussian potential
         U = -U0 / (1 + (z/zR)^2) * exp(-2 (x^2 + y^2) / w(z)^2),
         w(z) = w0 sqrt(1 + (z/zR)^2),  zR = pi w0^2 / lambda.
   The fitted model is A * P_MC(t; T): A absorbs everything that loses an atom
   independently of the release (imaging, background collisions, detection
   infidelity) and is pinned mainly by the t = 0 point the experiment always
   takes. T and A are fitted by binomial maximum likelihood; T carries a
   profile-likelihood 1-sigma interval.

   THE FIT NEEDS U0 AND w0 AT THE RELEASE DEPTH. The run records whether it
   released from the loading trap (nominal_trapdepth) or the low trap (amp3).
   Pass U0 for THAT depth -- either directly (U0_uK) or via the radial trap
   frequency measured at that depth (f_radial_kHz, U0 = m w_r^2 w0^2 / 4, the
   same formula as fitting/trap_oscillation.py). The waist is always needed.
   T depends only WEAKLY on the U0 you assume: on RIDs 3748/3751/3754, going
   from U0 = 312 uK to 1018 uK moved T by x1.43 (T ~ U0^0.3), so a 20 % error
   in U0 is a ~6 % error in T. The fit quality does not discriminate between
   U0 values (chi2/dof ~ 1 for all), so U0 must come from an independent
   measurement. The model-free t_1/2 is unaffected.

   Validity: initial conditions use the harmonic approximation, fine while
   kT << U0. The fit warns above kT/U0 = 0.1.

numpy only, elementwise: no np.dot / matmul / linalg (the ARTIQ env's BLAS
segfaults -- memory reference_broken_blas_env), no scipy.
"""

import os
import sys
import glob
import json
import math
from pathlib import Path

import numpy as np
import h5py

_BASE = Path(r"C:\Networking Experiment\Artiq7_TEST_NEW\useful_codes")
RESULTS_DIRS = [_BASE / "results", _BASE / "repository" / "results"]
PATTERN = "*-ReleaseRecapture.h5"

# Physical constants (SI)
M_CS = 132.905451931 * 1.66053906660e-27     # kg
K_B = 1.380649e-23                           # J/K
G = 9.80665                                  # m/s^2
LAMBDA_FORT_NM = 1064.0                      # FORT wavelength (aom_control: "FORT 1064nm")


# --------------------------------------------------------------------------
#  IO
# --------------------------------------------------------------------------
def _rid_of(path):
    return int(os.path.basename(str(path))[:9])


def find_runs():
    """All ReleaseRecapture files, one per RID, preferring the master copy
    (it carries expid)."""
    best = {}
    for d in RESULTS_DIRS:
        for f in glob.glob(str(d / "**" / PATTERN), recursive=True):
            rid = _rid_of(f)
            try:
                with h5py.File(f, "r", locking=False) as hf:
                    has = "expid" in hf
            except OSError:
                continue
            if rid not in best or (has and not best[rid][1]):
                best[rid] = (f, has)
    return [best[r][0] for r in sorted(best)]


def find_run(rid=None):
    runs = find_runs()
    if not runs:
        raise FileNotFoundError("no %s under %s" % (PATTERN, RESULTS_DIRS))
    if rid is None:
        return runs[-1]
    for f in runs:
        if _rid_of(f) == int(rid):
            return f
    raise FileNotFoundError("RID %s not found" % rid)


_KEYS = ("rr_t_release_us", "rr_loaded", "rr_survived",
         "rr_bg_spcm1", "rr_bg_spcm2", "rr_shot1_spcm1", "rr_shot1_spcm2",
         "rr_shot2_spcm1", "rr_shot2_spcm2", "rr_release_amp")


def load_run(path):
    """Return (data, meta, rid). Arrays are truncated to a common length
    (a run in progress can be a row out of step between datasets)."""
    data, meta = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"]
        for k in _KEYS:
            data[k] = np.atleast_1d(ds[k][()]).astype(float) if k in ds else np.array([])
        for k in ("rr_release_at_low_trap", "rr_atom_threshold",
                  "rr_nominal_trapdepth", "rr_trap_ramp_start_amp",
                  "rr_n_measurements"):
            meta[k] = float(ds[k][()]) if k in ds else None
        meta["release_times_us"] = (np.atleast_1d(ds["rr_release_times_us"][()]).astype(float)
                                    if "rr_release_times_us" in ds else None)
        meta["args"] = {}
        meta["has_expid"] = "expid" in hf
        if meta["has_expid"]:
            raw = hf["expid"][()]
            raw = raw.decode() if isinstance(raw, bytes) else raw
            try:
                meta["args"] = json.loads(raw).get("arguments", {})
            except Exception:
                pass
    n = min(data[k].size for k in _KEYS)
    for k in _KEYS:
        data[k] = data[k][:n]
    meta["n_trials"] = n
    meta["release_at_low_trap"] = bool(meta["rr_release_at_low_trap"])
    return data, meta, _rid_of(path)


# --------------------------------------------------------------------------
#  MODEL-FREE CURVE
# --------------------------------------------------------------------------
def wilson(k, n, z=1.0):
    """Wilson score interval (default 1 sigma). Returns (p, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def recapture_curve(data, meta, threshold=None):
    """Recapture probability per release time, over LOADED trials.

    threshold=None uses the run's own loaded/survived flags; a number
    re-thresholds from the raw counts (shot - bg > threshold) for both images.
    """
    t = data["rr_t_release_us"]
    if threshold is None:
        loaded = data["rr_loaded"] > 0.5
        surv = data["rr_survived"] > 0.5
    else:
        bg = data["rr_bg_spcm1"] + data["rr_bg_spcm2"]
        s1 = data["rr_shot1_spcm1"] + data["rr_shot1_spcm2"]
        s2 = data["rr_shot2_spcm1"] + data["rr_shot2_spcm2"]
        loaded = (s1 - bg) > threshold
        surv = loaded & ((s2 - bg) > threshold)
    rows = []
    for tu in np.unique(t):
        m = (t == tu) & loaded
        n = int(m.sum())
        k = int((m & surv).sum())
        p, lo, hi = wilson(k, n)
        rows.append(dict(t_us=float(tu), n=n, k=k, p=p, lo=lo, hi=hi))
    return rows


def half_recapture_time(rows):
    """Model-free temperature proxy: first release time where P(t)/P(0) falls
    through 0.5 (linear interpolation). None if it never does in range."""
    rows = [r for r in rows if r["n"] > 0]
    if not rows or rows[0]["t_us"] != 0.0 or not rows[0]["p"] > 0:
        return None
    p0 = rows[0]["p"]
    for a, b in zip(rows[:-1], rows[1:]):
        ya, yb = a["p"] / p0, b["p"] / p0
        if ya >= 0.5 > yb:
            return a["t_us"] + (0.5 - ya) * (b["t_us"] - a["t_us"]) / (yb - ya)
    return None


# --------------------------------------------------------------------------
#  TRAP + MONTE CARLO
# --------------------------------------------------------------------------
def trap_params(w0_um, U0_uK=None, f_radial_kHz=None, wavelength_nm=LAMBDA_FORT_NM):
    """Gaussian-trap parameters. Give U0_uK directly, or f_radial_kHz (the
    radial trap frequency at the release depth) and it is converted with
    U0 = m w_r^2 w0^2 / 4."""
    if w0_um is None or w0_um <= 0:
        raise ValueError("the beam waist w0_um is required")
    w0 = w0_um * 1e-6
    lam = wavelength_nm * 1e-9
    if U0_uK is None:
        if f_radial_kHz is None:
            raise ValueError("give U0_uK or f_radial_kHz")
        wr = 2 * math.pi * f_radial_kHz * 1e3
        U0 = M_CS * wr * wr * w0 * w0 / 4.0
    else:
        U0 = U0_uK * 1e-6 * K_B
    zR = math.pi * w0 * w0 / lam
    return dict(U0=U0, U0_uK=U0 / K_B * 1e6, w0=w0, zR=zR, lam=lam,
                omega_r=math.sqrt(4 * U0 / (M_CS * w0 * w0)),
                omega_z=math.sqrt(2 * U0 / (M_CS * zR * zR)))


def _normals(n_atoms, seed):
    """Common random numbers: the same standard normals for every T, so the
    MC curve is smooth in T and the likelihood can be minimised on a grid."""
    return np.random.RandomState(seed).standard_normal((6, n_atoms))


def mc_recapture(t_us, T_uK, trap, Z, gravity="transverse"):
    """Fraction of a thermal ensemble at T recaptured after each release t.

    gravity: 'transverse' (beam horizontal, g along y), 'axial' (g along the
    beam), or 'none'. Over <= 100 us gravity moves the atom ~0.05 um, so the
    choice barely matters; it is kept for completeness.
    """
    T = T_uK * 1e-6
    kT = K_B * T
    sx = math.sqrt(kT / (M_CS * trap["omega_r"] ** 2))
    sz = math.sqrt(kT / (M_CS * trap["omega_z"] ** 2))
    sv = math.sqrt(kT / M_CS)
    x0, y0, z0 = Z[0] * sx, Z[1] * sx, Z[2] * sz
    vx, vy, vz = Z[3] * sv, Z[4] * sv, Z[5] * sv
    U0, w0, zR = trap["U0"], trap["w0"], trap["zR"]
    out = np.empty(len(t_us))
    for i, tu in enumerate(t_us):
        t = tu * 1e-6
        gy = G if gravity == "transverse" else 0.0
        gz = G if gravity == "axial" else 0.0
        x = x0 + vx * t
        y = y0 + vy * t - 0.5 * gy * t * t
        z = z0 + vz * t - 0.5 * gz * t * t
        vy_t = vy - gy * t
        vz_t = vz - gz * t
        q = 1.0 + (z / zR) ** 2
        U = -U0 / q * np.exp(-2.0 * (x * x + y * y) / (w0 * w0 * q))
        E = 0.5 * M_CS * (vx * vx + vy_t * vy_t + vz_t * vz_t) + U
        out[i] = float(np.mean(E < 0.0))
    return out


# --------------------------------------------------------------------------
#  FIT
# --------------------------------------------------------------------------
def _nll_binomial(k, n, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum())


def _best_A(k, n, f):
    """Amplitude A in (0, 1] minimising the binomial nll of A*f (golden section)."""
    a, b = 1e-3, 1.0
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = _nll_binomial(k, n, c * f), _nll_binomial(k, n, d * f)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _nll_binomial(k, n, c * f)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _nll_binomial(k, n, d * f)
    A = 0.5 * (a + b)
    return A, _nll_binomial(k, n, A * f)


def fit_temperature(rows, trap, T_min_uK=0.5, T_max_uK=2000.0, n_grid=120,
                    n_atoms=20000, seed=1, gravity="transverse"):
    """Binomial MLE of (T, A) for the model A * P_MC(t; T).

    Coarse log grid in T, then a fine grid around the minimum; T carries a
    profile-likelihood 1-sigma interval (delta nll = 0.5). Returns a dict, or
    None if there are fewer than 3 populated release times.
    """
    rows = [r for r in rows if r["n"] > 0]
    if len(rows) < 3:
        return None
    t = np.array([r["t_us"] for r in rows])
    k = np.array([r["k"] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    Z = _normals(n_atoms, seed)

    def profile(Ts):
        out = []
        for T in Ts:
            f = mc_recapture(t, T, trap, Z, gravity)
            A, v = _best_A(k, n, f)
            out.append((T, v, A))
        return out

    coarse = profile(np.exp(np.linspace(math.log(T_min_uK), math.log(T_max_uK), n_grid)))
    i = int(np.argmin([c[1] for c in coarse]))
    lo = coarse[max(i - 3, 0)][0]
    hi = coarse[min(i + 3, len(coarse) - 1)][0]
    fine = profile(np.exp(np.linspace(math.log(lo), math.log(hi), 80)))
    allp = sorted(coarse + fine)
    Ts = np.array([a[0] for a in allp])
    nl = np.array([a[1] for a in allp])
    As = np.array([a[2] for a in allp])
    j = int(np.argmin(nl))
    T_best, nll_best, A_best = float(Ts[j]), float(nl[j]), float(As[j])
    ok = np.where(nl - nll_best <= 0.5)[0]
    res = dict(T_uK=T_best, T_lo_uK=float(Ts[ok.min()]), T_hi_uK=float(Ts[ok.max()]),
               A=A_best, nll=nll_best, T_grid=Ts, nll_grid=nl,
               t_us=t, model=A_best * mc_recapture(t, T_best, trap, Z, gravity),
               edge=(j == 0 or j == len(Ts) - 1),
               kT_over_U0=K_B * T_best * 1e-6 / trap["U0"])
    # goodness of fit: Pearson chi2 against the fitted model
    p = np.clip(res["model"], 1e-9, 1 - 1e-9)
    res["chi2"] = float((((k - n * p) ** 2) / (n * p * (1 - p))).sum())
    res["dof"] = len(rows) - 2
    res["_Z"], res["_k"], res["_n"] = Z, k, n
    return res


# --------------------------------------------------------------------------
#  REPORT + FIGURE
# --------------------------------------------------------------------------
def analyze(path=None, w0_um=None, U0_uK=None, f_radial_kHz=None,
            threshold=None, gravity="transverse", n_atoms=20000, out_png=None,
            show=True):
    """Model-free curve always; temperature fit when the trap is specified.
    Returns dict(rows, t_half_us, fit, trap, meta, rid)."""
    path = path or find_run()
    data, meta, rid = load_run(path)
    rows = recapture_curve(data, meta, threshold)
    t_half = half_recapture_time(rows)

    depth = "LOW trap (amp3, after the adiabatic ramp)" if meta["release_at_low_trap"] \
        else "LOADING trap (nominal_trapdepth = %s)" % meta["rr_nominal_trapdepth"]
    amps = data["rr_release_amp"][data["rr_loaded"] > 0.5]
    print("--- ReleaseRecapture  RID %d ---" % rid)
    print("file          : %s%s" % (path, "" if meta["has_expid"] else
                                    "   (no expid: run in progress)"))
    print("released from : %s" % depth)
    if amps.size:
        print("release amp   : %.4f  (min %.4f / max %.4f over loaded trials)"
              % (amps.mean(), amps.min(), amps.max()))
    print("trials        : %d, loaded %d (%.0f%%)%s"
          % (meta["n_trials"], int((data["rr_loaded"] > 0.5).sum()),
             100 * np.mean(data["rr_loaded"] > 0.5) if meta["n_trials"] else 0,
             "" if threshold is None else "   [re-thresholded at %g]" % threshold))
    print()
    print("  t_release    recaptured / loaded    P (1 sigma Wilson)")
    for r in rows:
        print("  %7.2f us   %5d / %-5d            %.3f  [%.3f, %.3f]"
              % (r["t_us"], r["k"], r["n"], r["p"], r["lo"], r["hi"]))
    print()
    print("half-recapture time t_1/2 (model-free, longer = colder): %s"
          % ("%.1f us" % t_half if t_half is not None else "not reached in range"))

    fit, trap = None, None
    if w0_um is not None and (U0_uK is not None or f_radial_kHz is not None):
        trap = trap_params(w0_um, U0_uK, f_radial_kHz)
        print()
        print("trap at release: U0 = %.0f uK (%.3f mK), w0 = %.3f um, zR = %.2f um,"
              " f_r = %.1f kHz, f_z = %.2f kHz"
              % (trap["U0_uK"], trap["U0_uK"] / 1e3, w0_um, trap["zR"] * 1e6,
                 trap["omega_r"] / 2 / math.pi / 1e3, trap["omega_z"] / 2 / math.pi / 1e3))
        fit = fit_temperature(rows, trap, n_atoms=n_atoms, gravity=gravity)
        if fit is None:
            print("TEMPERATURE   : not enough populated release times to fit")
        else:
            print("TEMPERATURE   : T = %.1f uK  [%.1f, %.1f] (1 sigma)   A = %.3f"
                  % (fit["T_uK"], fit["T_lo_uK"], fit["T_hi_uK"], fit["A"]))
            print("                kT/U0 = %.3f   chi2/dof = %.2f / %d"
                  % (fit["kT_over_U0"], fit["chi2"], fit["dof"]))
            if fit["edge"]:
                print("   !! best T sits on the edge of the search grid -- widen it")
            if fit["kT_over_U0"] > 0.1:
                print("   !! kT/U0 > 0.1: the harmonic initial-condition sampling is "
                      "no longer accurate; treat T as indicative")
    else:
        print()
        print("TEMPERATURE   : not fitted -- pass w0_um and (U0_uK or f_radial_kHz)"
              " for the release depth")

    _figure(rid, rows, fit, trap, t_half, depth, out_png or
            str(Path(path).with_name("release_recapture_RID%d.png" % rid)), show)
    return dict(rows=rows, t_half_us=t_half, fit=fit, trap=trap, meta=meta, rid=rid)


def _figure(rid, rows, fit, trap, t_half, depth, out_png, show):
    import matplotlib
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["n"] > 0]
    ncol = 2 if fit is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 4.6), constrained_layout=True)
    ax = axes[0] if ncol == 2 else axes
    t = np.array([r["t_us"] for r in rows])
    p = np.array([r["p"] for r in rows])
    elo = np.maximum(0.0, p - np.array([r["lo"] for r in rows]))
    ehi = np.maximum(0.0, np.array([r["hi"] for r in rows]) - p)
    ax.errorbar(t, p, yerr=[elo, ehi], fmt="o", color="C0", ms=5, capsize=2,
                label="data (recaptured / loaded)")
    if fit is not None:
        tt = np.linspace(0.0, t.max(), 120)
        Z = fit["_Z"]
        for T, ls, lab in ((fit["T_uK"], "-", "MC fit  T = %.1f uK" % fit["T_uK"]),
                           (fit["T_lo_uK"], ":", "1 sigma band"),
                           (fit["T_hi_uK"], ":", None)):
            ax.plot(tt, fit["A"] * mc_recapture(tt, T, trap, Z), ls, color="C3",
                    lw=1.8 if ls == "-" else 1.0, label=lab)
    if t_half is not None:
        ax.axvline(t_half, color="C7", ls="--", lw=1, label="t_1/2 = %.1f us" % t_half)
    ax.set_xlabel("release time [us]")
    ax.set_ylabel("recapture probability")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("RID %d  |  released from the %s" % (rid, depth.split(" (")[0]), fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if fit is not None:
        ax2 = axes[1]
        ax2.plot(fit["T_grid"], fit["nll_grid"] - fit["nll"], "-", color="C0")
        ax2.axhline(0.5, color="C3", ls="--", lw=1, label="delta nll = 0.5 (1 sigma)")
        ax2.axvline(fit["T_uK"], color="C7", lw=1)
        ax2.set_xscale("log")
        ax2.set_ylim(-0.2, 10)
        ax2.set_xlabel("temperature [uK]")
        ax2.set_ylabel("nll - min")
        ax2.set_title("profile likelihood  |  T = %.1f [%.1f, %.1f] uK"
                      % (fit["T_uK"], fit["T_lo_uK"], fit["T_hi_uK"]), fontsize=10)
        ax2.grid(alpha=0.3, which="both")
        ax2.legend(fontsize=8)
    fig.savefig(out_png, dpi=130)
    print("Saved figure: %s" % out_png)
    backend = matplotlib.get_backend().lower()
    if show and any(s in backend for s in ("inline", "ipympl", "nbagg", "widget")):
        plt.show()
    else:
        plt.close(fig)
    return fig


def _int_bins(*arrays, max_bins=70):
    """Bin edges on half-integers (counts are integers), widened in whole-count
    steps if the range would need more than max_bins bins."""
    v = np.concatenate([np.asarray(a, dtype=float) for a in arrays if len(a)])
    lo, hi = math.floor(v.min()), math.ceil(v.max())
    step = max(1, int(math.ceil((hi - lo + 1) / max_bins)))
    return np.arange(lo - 0.5, hi + step + 0.5, step)


def signal_histograms(path=None, threshold=None, out_png=None, show=True):
    """Atom-signal histograms: net counts (SPCM1+SPCM2 minus that trial's
    background) for both images, with the threshold that decided loaded /
    survived.

      A  image 1, ALL trials: the no-atom and one-atom peaks. The threshold
         belongs in the valley between them; this decides `loaded`.
      B  image 2, LOADED trials only, split into t = 0 (no release: survival
         through the images alone) and t > 0. A recaptured atom lands in the
         same bright peak as image 1; a lost one drops into the no-atom peak.
         Image 2 is skipped for unloaded trials, which store shot2 = 0, so
         including them would add a false spike near -bg.

    threshold=None draws the run's own atom_threshold. Returns a dict with the
    net counts and a few separation numbers.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    path = path or find_run()
    data, meta, rid = load_run(path)
    thr = float(threshold) if threshold is not None else meta["rr_atom_threshold"]
    bg = data["rr_bg_spcm1"] + data["rr_bg_spcm2"]
    net1 = data["rr_shot1_spcm1"] + data["rr_shot1_spcm2"] - bg
    net2 = data["rr_shot2_spcm1"] + data["rr_shot2_spcm2"] - bg
    loaded = net1 > thr if threshold is not None else data["rr_loaded"] > 0.5
    t = data["rr_t_release_us"]
    n2_t0 = net2[loaded & (t == 0.0)]
    n2_rel = net2[loaded & (t > 0.0)]

    atom = net1[net1 > thr]
    empty = net1[net1 <= thr]
    out = dict(rid=rid, threshold=thr, net1=net1, net2_t0=n2_t0, net2_release=n2_rel,
               bg_mean=float(bg.mean()) if bg.size else float("nan"),
               atom_mean=float(atom.mean()) if atom.size else float("nan"),
               empty_mean=float(empty.mean()) if empty.size else float("nan"),
               frac_above=float(np.mean(net1 > thr)) if net1.size else float("nan"))
    print("--- atom signal  RID %d  (threshold %g%s) ---"
          % (rid, thr, "" if threshold is None else ", override"))
    print("background (both SPCMs)  : mean %.1f counts" % out["bg_mean"])
    print("image 1, all trials      : %d trials, %.0f%% above threshold" %
          (net1.size, 100 * out["frac_above"]))
    print("    no-atom peak  (<= thr): mean %.1f   |   atom peak (> thr): mean %.1f"
          % (out["empty_mean"], out["atom_mean"]))
    if n2_t0.size:
        print("image 2, t = 0 (no release): %d trials, %.0f%% above threshold"
              % (n2_t0.size, 100 * np.mean(n2_t0 > thr)))
    if n2_rel.size:
        print("image 2, t > 0             : %d trials, %.0f%% above threshold"
              % (n2_rel.size, 100 * np.mean(n2_rel > thr)))

    # Reference palette (dataviz skill), light mode: categorical slots 1-3.
    C1, C2, C3, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    if net1.size:
        b = _int_bins(net1, n2_t0, n2_rel)
        axA.hist(net1, bins=b, color=C1, alpha=0.85, label="image 1, all trials (%d)" % net1.size)
        axB.hist(n2_t0, bins=b, histtype="step", lw=1.8, color=C2,
                 label="image 2, t = 0 (%d)" % n2_t0.size)
        axB.hist(n2_rel, bins=b, histtype="step", lw=1.8, color=C3,
                 label="image 2, t > 0 (%d)" % n2_rel.size)
    for ax in (axA, axB):
        ax.axvline(thr, color=INK, ls="--", lw=1.2, label="threshold %g" % thr)
        ax.set_xlabel("net counts  (SPCM1 + SPCM2 - background)")
        ax.set_ylabel("trials")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axA.set_title("A. Image 1 -- decides LOADED  (%.0f%% above threshold)"
                  % (100 * out["frac_above"]), fontsize=10)
    axB.set_title("B. Image 2, loaded trials -- decides RECAPTURED", fontsize=10)
    fig.suptitle("ReleaseRecapture RID %d -- atom signal" % rid, fontsize=12)

    out_png = out_png or str(Path(path).with_name("release_recapture_signal_RID%d.png" % rid))
    fig.savefig(out_png, dpi=130)
    print("Saved figure: %s" % out_png)
    backend = matplotlib.get_backend().lower()
    if show and any(s in backend for s in ("inline", "ipympl", "nbagg", "widget")):
        plt.show()
    else:
        plt.close(fig)
    out["fig"] = fig
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    argv = sys.argv[1:]
    rid = int(argv[0]) if argv else None
    kw = {}
    for a in argv[1:]:
        k, v = a.split("=")
        kw[k] = float(v)
    analyze(find_run(rid), **kw)
