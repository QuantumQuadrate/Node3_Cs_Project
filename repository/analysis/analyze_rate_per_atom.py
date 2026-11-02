"""Per-atom heralding diagnostics for RateExperiment: why do atoms not herald?

Reads the per-CONFIRMED-atom datasets added to experiments/Rate_measurement.py
(`atom_confirm_net`, `atom_attempts`, `atom_heralds`, `atom_first_herald`,
`atom_end_reason`) and tests the competing explanations for the low heralding
probability. One row per atom that PASSED the high-trap confirm image, so
loader false-positives (confirm-fails) are already excluded -- that was a
separate effect and is not what this module is about.

=============================================================================
READ THIS FIRST: dud fraction is NOT evidence of heterogeneity
=============================================================================
A homogeneous population with per-attempt herald probability p and per-attempt
loss hazard h already produces a large fraction of atoms that never herald,
purely from shot noise: an atom lost after A attempts fails to herald with
probability (1-p)^A. With p ~ 0.8% and a mean lifetime of a few hundred
attempts, 35-50% duds is the EXPECTED outcome, not a signal. Two earlier
rounds of this investigation were misled by exactly that, so this module
deliberately does not lead with the dud count.

The test that actually separates "one population" from "a bad sub-population"
is OVERDISPERSION: compare the spread of the per-atom herald rates h_i/a_i
against the binomial spread expected at the pooled rate. With N atoms,

    D = (1/(N-1)) * sum_i (h_i - a_i*p)^2 / (a_i*p*(1-p)),      p = sum h / sum a

is ~1 for a single population and >1 when atoms differ intrinsically. Because
a_i*p is small here (a few clicks per atom), the chi-square approximation for D
is unreliable, so significance comes from a PARAMETRIC BOOTSTRAP: resample
h_i ~ Binomial(a_i, p) many times and compare the observed D against that null.
That is exact under the null regardless of how small a_i*p is.

=============================================================================
WHAT EACH DIAGNOSTIC DISTINGUISHES
=============================================================================
hazard_by_block()   Loss hazard vs attempt index, at the survival-check grid.
                    A CONSTANT hazard means atoms die at a uniform rate. An
                    ELEVATED FIRST BLOCK means extra early loss -- i.e. the atom
                    does not survive the transition from the loading trap into
                    the excitation sequence (MOT dissipation, the one-time trap
                    ramp-down, first optical pumping). That is the "lost right
                    after trapping" hypothesis, and this is the test for it.

                    NOTE the geometry: the confirm image happens AFTER
                    t_MOT_dissipation and the imaging PGC, so an atom lost while
                    the MOT disperses fails CONFIRM and never appears in this
                    file at all. What the first block can show is loss between
                    the confirm image and the first survival check -- which is
                    dominated by the n==0 trap ramp-down (trap_ramp_steps *
                    trap_ramp_dt_us, typically 10 ms) plus the first 48 attempts.

first_herald_profile()  P(first herald in block k | survived to block k).
                    FLAT = memoryless, the herald rate does not depend on how
                    long the atom has been held. RISING = a warm-up (the atom
                    needs conditioning before it heralds at full rate).

overdispersion()    One population vs a bad sub-population (above).

confirm_net_split() Confirm-image counts for heralders vs duds. A ~2x shoulder
                    among duds points at TWO-ATOM loads (bright at confirm,
                    collisionally ejected under excitation).

compare_runs()      Dud fraction and herald rate vs any expid argument -- built
                    for the trap-depth series (trap_ramp_start_amp 0.21 / 0.18 /
                    0.15 across RIDs 3632/3635/3638/3643).

=============================================================================
THE ARCHIVE LIES ABOUT trap_ramp_start_amp -- read expid
=============================================================================
`trap_ramp_start_amp`, `trap_ramp_steps`, `trap_ramp_dt_us`,
`max_excitation_cycles` and `Check_img_every` are FRONT-PANEL arguments of
Rate_measurement, and they SHADOW same-named ExperimentVariables datasets. The
HDF5 `archive/` group stores the dataset (stale) while `expid` stores what
actually ran. Measured: archive says trap_ramp_start_amp = 0.21 for RIDs
3632/3635/3638/3643, while expid says 0.21 / 0.21 / 0.18 / 0.15. Anything
reading the archive would compare four runs it believes are identical. This
module always reads expid and reports both when they disagree.

Only the master copy under results/ carries expid; write_results' copy under
repository/results/ does not, so a run still in progress may be readable only
from the latter and without its arguments.

BLAS-free (sums, bincount, a bootstrap loop); no scipy. Plot in Python 3.11,
NOT artiq7_claude -- matplotlib segfaults there. Run:
    python analyze_rate_per_atom.py            # latest run
    python analyze_rate_per_atom.py 3643       # one RID
    python analyze_rate_per_atom.py compare    # the trap-depth series
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

PATTERN = "*-RateExperiment.h5"

_ATOM_KEYS = ("atom_confirm_net", "atom_attempts", "atom_heralds",
              "atom_first_herald", "atom_end_reason")

# One row per photon CLICK (not per atom), so these are loaded and truncated
# independently of _ATOM_KEYS. click_delay_ns is the photon's arrival time
# inside the collection gate, measured from gate open.
_CLICK_KEYS = ("click_delay_ns", "click_detector", "click_times_s")

# Front-panel arguments that shadow same-named ExperimentVariables datasets.
_SHADOWED = ("trap_ramp_start_amp", "trap_ramp_steps", "trap_ramp_dt_us",
             "max_excitation_cycles", "Check_img_every", "t_spcm_collection_mu")


# --------------------------------------------------------------------------
#  IO
# --------------------------------------------------------------------------
def _rid_of(path):
    return int(os.path.basename(path).split("-")[0])


def _has_expid(path):
    try:
        with h5py.File(path, "r", locking=False) as hf:
            return "expid" in hf
    except Exception:
        return False


def _all_files(require_atoms=True):
    """One file per RID, preferring the copy with expid. Optionally require
    that the per-atom datasets are present and non-empty."""
    files = []
    for root in RESULTS_DIRS:
        files += glob.glob(str(root / "**" / PATTERN), recursive=True)
    by_rid = {}
    for f in set(files):
        by_rid.setdefault(_rid_of(f), []).append(f)
    out = []
    for rid in sorted(by_rid):
        c = by_rid[rid]
        pick = c[0] if len(c) == 1 else next((x for x in c if _has_expid(x)), c[0])
        if require_atoms:
            try:
                with h5py.File(pick, "r", locking=False) as hf:
                    ds = hf["datasets"] if "datasets" in hf else hf
                    if "atom_heralds" not in ds or ds["atom_heralds"].size == 0:
                        continue
            except Exception:
                continue
        out.append(pick)
    return out


def find_run(rid=None):
    """Newest run carrying per-atom data, or the one for `rid`."""
    files = _all_files()
    if not files:
        raise FileNotFoundError(
            "No " + PATTERN + " with per-atom datasets under "
            + " or ".join(str(r) for r in RESULTS_DIRS))
    if rid is None:
        return files[-1]
    for f in files:
        if _rid_of(f) == int(rid):
            return f
    raise FileNotFoundError("No instrumented RateExperiment for RID {}".format(rid))


def find_runs():
    """Every run carrying per-atom data, oldest first."""
    return _all_files()


def load_run(path):
    """Return (data, meta, rid).

    data: per-atom arrays (confirm_net, attempts, heralds, first, end_reason)
          plus derived `dud` and `censored` masks.
    meta: run settings, taken from expid where available (see module docstring
          on why the archive is not trusted for the shadowed names).
    """
    data, meta = {}, {}
    with h5py.File(path, "r", locking=False) as hf:
        ds = hf["datasets"] if "datasets" in hf else hf
        for k in _ATOM_KEYS:
            data[k] = (np.atleast_1d(ds[k][()]).astype(float)
                       if k in ds else np.array([]))
        meta["n_clicks_total"] = (int(ds["n_clicks"][()])
                                  if "n_clicks" in ds else None)

        # Per-click arrays. A run still in progress can have them differing by
        # a row or two (the incremental save can land between appends), so they
        # are truncated to their own common length, not to the per-atom one.
        for k in _CLICK_KEYS:
            data[k] = (np.atleast_1d(ds[k][()]).astype(float)
                       if k in ds else np.array([]))
        nc = min(data[k].size for k in _CLICK_KEYS)
        for k in _CLICK_KEYS:
            data[k] = data[k][:nc]
        meta["n_coincidences"] = (int(ds["n_coincidences"][()])
                                  if "n_coincidences" in ds else None)

        meta["args"] = {}
        meta["has_expid"] = "expid" in hf
        if meta["has_expid"]:
            raw = hf["expid"][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            try:
                meta["args"] = json.loads(raw).get("arguments", {})
            except Exception:
                pass
        arch = hf.get("archive")
        meta["archive"] = {}
        if arch is not None:
            for k in list(_SHADOWED) + ["nominal_trapdepth", "atom_threshold",
                                        "atom_threshold_lowtrap",
                                        "t_MOT_dissipation",
                                        "single_atom_threshold_for_loading",
                                        "exc_beam_duration_mu",
                                        "exc_spcm_gate_offset_mu"]:
                if k in arch:
                    try:
                        meta["archive"][k] = float(arch[k][()])
                    except Exception:
                        pass

    n = min(a.size for a in (data[k] for k in _ATOM_KEYS)) if data else 0
    for k in _ATOM_KEYS:
        data[k] = data[k][:n]
    meta["n_atoms"] = n

    data["dud"] = data["atom_heralds"] == 0
    data["censored"] = data["atom_end_reason"] > 0.5

    # Effective values: expid wins, archive is the fallback, and disagreements
    # are recorded so the caller can warn.
    meta["shadow_conflicts"] = {}
    for k in _SHADOWED:
        a_val = meta["archive"].get(k)
        e_val = meta["args"].get(k)
        meta[k] = float(e_val) if e_val is not None else a_val
        if (e_val is not None and a_val is not None
                and abs(float(e_val) - a_val) > 1e-12):
            meta["shadow_conflicts"][k] = (a_val, float(e_val))
    return data, meta, _rid_of(path)


# --------------------------------------------------------------------------
#  STATISTICS  (sums + bootstrap -- no BLAS, no scipy)
# --------------------------------------------------------------------------
def fit_decay(x, top, t0=None, n_tau=240, n_w=61):
    """Two-component MLE of the photon arrival profile on the window [0, top].

    Model: an exponential with onset t0 and decay constant tau, plus a flat
    background of weight w --

        f(t) = (1 - w) * exp(-(t - t0) / tau) / Z  +  w / top    (t >= t0)
               w / top                                           (t <  t0)
        Z    = tau * (1 - exp(-(top - t0) / tau))

    A SINGLE exponential is not enough: the real profile carries a weak slow
    tail (~10 % of clicks), and a one-component fit absorbs it and biases tau
    high -- 48 ns instead of 36 ns on RID 3646.

    tau carries a profile-likelihood 1-sigma interval (delta nll = 0.5),
    resolution-limited by the log-spaced tau grid (~2 % at the default n_tau).
    Returns a dict (t0, tau, tau_lo, tau_hi, w, n, L, flat) or None if there is
    too little data; `flat` means no decay is resolved -- tau comparable to the
    whole window, or nearly all the weight in the background.

    Written with einsum-free numpy only (no np.dot, no scipy): the ARTIQ env's
    BLAS segfaults -- see memory reference_broken_blas_env.

    KEEP IN SYNC with fit_decay() in applets/plot_rate_histograms.py and
    _fit_decay_host() in experiments/Rate_measurement.py. The applet subsamples
    above 1500 clicks and uses a coarser grid for live speed, so its tau can
    differ by a few percent; the model is identical.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0.0) & (x <= top)]
    if x.size < 10 or not np.isfinite(top) or top <= 0.0:
        return None
    if t0 is None:
        # 2nd percentile, not the minimum: robust to a single stray early click.
        t0 = float(np.percentile(x, 2.0)) if x.size >= 50 else float(x.min())
    L = top - t0
    if L <= 0.0:
        return None

    taus = np.exp(np.linspace(np.log(5.0), np.log(max(10.0, 0.6 * L)), n_tau))
    ws = np.linspace(0.0, 0.40, n_w)
    above = x >= t0
    dx = np.where(above, x - t0, 0.0)
    # Loop over tau, vectorise over w: memory stays O(N * n_w).
    nll = np.empty((n_tau, n_w))
    for i, tau in enumerate(taus):
        z = tau * (1.0 - np.exp(-L / tau))
        sig = np.where(above, np.exp(-dx / tau) / z, 0.0)
        dens = (1.0 - ws[None, :]) * sig[:, None] + ws[None, :] / top
        nll[i] = -np.log(np.maximum(dens, 1e-300)).sum(axis=0)

    i, j = np.unravel_index(np.argmin(nll), nll.shape)
    tau, w, best = float(taus[i]), float(ws[j]), float(nll[i, j])
    prof = nll.min(axis=1)
    ok = np.where(prof - best <= 0.5)[0]
    return dict(t0=float(t0), tau=tau,
                tau_lo=float(taus[ok.min()]), tau_hi=float(taus[ok.max()]),
                w=w, n=int(x.size), L=float(L),
                flat=bool(tau > 0.5 * L or w > 0.6))


def overdispersion(data, n_boot=2000, seed=0):
    """One population, or a bad sub-population? Returns a dict.

    D = (1/(N-1)) sum (h_i - a_i p)^2 / (a_i p (1-p)) at the pooled p.
    Significance from a parametric bootstrap (h_i ~ Binomial(a_i, p)), because
    a_i*p is only a few counts here and the chi-square approximation for D is
    not trustworthy in that regime.
    """
    a = data["atom_attempts"]
    h = data["atom_heralds"]
    keep = a > 0
    a, h = a[keep], h[keep]
    nn = a.size
    if nn < 10:
        return None
    p = float(h.sum() / a.sum())
    if not (0.0 < p < 1.0):
        return None

    def _disp(hh):
        num = (hh - a * p) ** 2
        den = a * p * (1.0 - p)
        return float(np.sum(num / den) / (nn - 1))

    d_obs = _disp(h)
    rng = np.random.RandomState(seed)
    null = np.empty(n_boot)
    for i in range(n_boot):
        null[i] = _disp(rng.binomial(a.astype(int), p).astype(float))
    p_val = float((np.sum(null >= d_obs) + 1.0) / (n_boot + 1.0))
    return {"D": d_obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_95": float(np.percentile(null, 95)),
            "p_value": p_val, "n_atoms": int(nn), "p_pooled": p,
            "overdispersed": bool(p_val < 0.05)}


def hazard_by_block(data, block):
    """Loss hazard per survival-check block. Tests 'lost right after trapping'.

    The survival check runs every `block` attempts, so an atom's recorded
    attempts-at-loss is a multiple of `block`. For block k:
        at_risk = atoms that reached block k
        deaths  = atoms whose (complete) life ended at block k
        hazard  = deaths / at_risk
    Censored atoms (run ended) are removed from at_risk after their last block
    rather than counted as deaths. A flat hazard = uniform loss; an elevated
    FIRST block = extra loss in the confirm -> ramp-down -> first-attempts
    transition.
    """
    a = data["atom_attempts"]
    cens = data["censored"]
    if a.size == 0 or not block or block <= 0:
        return None
    kmax = int(np.ceil(a.max() / block))
    rows = []
    for k in range(1, kmax + 1):
        edge = k * block
        at_risk = int(np.count_nonzero(a >= edge))
        deaths = int(np.count_nonzero((a == edge) & (~cens)))
        if at_risk <= 0:
            continue
        hz = deaths / at_risk
        se = float(np.sqrt(max(hz * (1 - hz), 0.0) / at_risk))
        rows.append({"block": k, "attempts": edge, "at_risk": at_risk,
                     "deaths": deaths, "hazard": hz, "se": se})
    if not rows:
        return None
    # Reference hazard from blocks 2+ (i.e. excluding the transition block).
    later = [r for r in rows[1:] if r["at_risk"] > 0]
    d_l = sum(r["deaths"] for r in later)
    n_l = sum(r["at_risk"] for r in later)
    h_ref = (d_l / n_l) if n_l else float("nan")
    first = rows[0]
    excess = first["hazard"] - h_ref
    z = (excess / first["se"]) if first["se"] > 0 else float("nan")
    return {"rows": rows, "h_first": first["hazard"], "h_later": h_ref,
            "excess": excess, "z_first": z,
            "early_loss": bool(np.isfinite(z) and z > 3.0)}


def first_herald_profile(data, block):
    """P(first herald in block k | survived to k). Flat = memoryless.

    Uses atom_first_herald, so it only needs per-atom totals. A rising profile
    would indicate a warm-up; a flat one says the herald rate does not depend
    on how long the atom has been held.
    """
    a = data["atom_attempts"]
    fh = data["atom_first_herald"]
    if a.size == 0 or not block or block <= 0:
        return None
    kmax = int(np.ceil(a.max() / block))
    rows = []
    for k in range(1, kmax + 1):
        lo, hi = (k - 1) * block, k * block
        # At risk: reached this block AND had not heralded before it.
        at_risk = np.count_nonzero((a >= hi) & ((fh < 0) | (fh > lo)))
        hits = np.count_nonzero((fh > lo) & (fh <= hi))
        if at_risk <= 0:
            continue
        pk = hits / at_risk
        rows.append({"block": k, "attempts": hi, "at_risk": int(at_risk),
                     "hits": int(hits), "p_first": pk,
                     "se": float(np.sqrt(max(pk * (1 - pk), 0.0) / at_risk))})
    return rows or None


def flatness(rows, key="p_first", min_at_risk=50):
    """Chi-square test that a per-block series is CONSTANT.

    Replaces comparing the first block with the last, which is what a noisy
    series will defeat: on RID 3643 the first-herald profile scatters
    0.157, 0.185, 0.160, 0.177, 0.119, 0.218 around a flat ~0.17, and a
    first-vs-last comparison called that a rising warm-up. A weighted chi-square
    against the weighted mean uses every block and its own error bar.

    Blocks thinner than `min_at_risk` are dropped -- their error bars are
    unreliable and they would dominate a chi-square.
    """
    use = [r for r in rows if r["at_risk"] >= min_at_risk and r["se"] > 0]
    if len(use) < 3:
        return None
    v = np.array([r[key] for r in use], dtype=float)
    s = np.array([r["se"] for r in use], dtype=float)
    w = 1.0 / s ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    chi2 = float(np.sum(((v - mean) / s) ** 2))
    dof = len(use) - 1
    # Normal approximation to the chi-square tail; adequate for dof >= 3 and we
    # only need "is this consistent with flat", not a precise p.
    z = (chi2 - dof) / np.sqrt(2.0 * dof)
    return {"mean": mean, "chi2": chi2, "dof": dof, "chi2_red": chi2 / dof,
            "z": float(z), "n_blocks": len(use),
            "flat": bool(z < 2.0)}


def two_rate_consistency(res):
    """Do the measured herald rate and loss hazard ALONE predict the duds?

    If there is one population, heralding and loss are competing per-attempt
    processes, so the fraction of atoms that never herald is simply

        dud_predicted = h / (h + p)

    with p the per-attempt herald rate and h the per-attempt loss hazard
    (converted from the per-block hazard). Agreement with the observed dud
    fraction means the whole system is described by those two numbers and there
    is nothing left for a bad sub-population to explain.
    """
    hz, od = res.get("hazard"), res.get("overdispersion")
    block, p = res.get("block"), res.get("p_pooled")
    if hz is None or not block or not p or p != p:
        return None
    # Per-block hazard -> per-attempt hazard.
    h_block = hz["h_later"]
    if not (0.0 < h_block < 1.0):
        return None
    h_att = 1.0 - (1.0 - h_block) ** (1.0 / float(block))
    pred = h_att / (h_att + p)
    obs = res["dud_frac"]
    n = max(res["n_atoms"], 1)
    se = float(np.sqrt(max(obs * (1 - obs), 0.0) / n))
    return {"h_per_attempt": h_att, "p_per_attempt": p,
            "dud_predicted": pred, "dud_observed": obs, "dud_se": se,
            "z": (obs - pred) / se if se > 0 else float("nan"),
            "agrees": bool(se > 0 and abs(obs - pred) < 3 * se)}


def confirm_net_split(data):
    """Confirm-image counts for heralders vs duds -- the two-atom-load test."""
    cn = data["atom_confirm_net"]
    dud = data["dud"]
    if cn.size == 0 or dud.all() or (~dud).all():
        return None

    def _s(x):
        return {"n": int(x.size), "median": float(np.median(x)),
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0}

    return {"duds": _s(cn[dud]), "heralders": _s(cn[~dud]),
            "ratio_median": (float(np.median(cn[dud]) / np.median(cn[~dud]))
                             if np.median(cn[~dud]) else float("nan"))}


def infer_block(data, min_complete=20):
    """Recover `Check_img_every` (the survival-check block) from the data.

    Needed because `Check_img_every` is a FRONT-PANEL argument: it is never
    read as a dataset, so it never reaches `archive/`, and a run still in
    progress has only the `repository/results` copy, which has no `expid`
    either. Without it `hazard_by_block`, `first_herald_profile`, `flatness`
    and `two_rate_consistency` all return None -- which is what used to make
    the notebook's headline cell raise TypeError on a live run.

    Loss is only ever DETECTED at a survival check, so every completed
    lifetime (`atom_end_reason == 0`) is an exact multiple of the block, and
    their gcd recovers it. Verified against the two runs that do carry an
    expid: RID 3643 -> 48, RID 3646 -> 12, both exact.

    Returns None if there are too few completed lifetimes to trust the gcd (a
    handful of atoms can share a spurious common factor).
    """
    a = data.get("atom_attempts")
    er = data.get("atom_end_reason")
    if a is None or er is None or a.size == 0:
        return None
    comp = a[er < 0.5].astype(np.int64)
    comp = comp[comp > 0]
    if comp.size < min_complete:
        return None
    g = int(np.gcd.reduce(comp))
    return g if g > 0 else None


def photon_profile(data, meta):
    """Is each herald ONE spontaneously emitted photon? Plus the stale-read audit.

    Three independent things come out of `click_delay_ns` (the photon's arrival
    time inside the collection gate, measured from gate open):

    1. THE SHAPE -> the photon number. An exponential at the Cs natural
       lifetime (D2 30.5 ns, D1 34.9 ns) means the atom emits one photon and is
       then pumped dark -- even though `exc_beam_duration_mu` drives it for
       ~10 lifetimes. A profile flat across the gate would instead mean
       multi-photon scattering or background. This needs NO detection
       efficiency: it is a shape, not a rate.

    2. THE BACKGROUND, model-independently. No signal can live below the
       emission onset, so clicks in a guard window under it measure the
       uniform dark/stray rate directly. Compare that with the fitted flat
       weight `w`: if w is much larger, the second component is NOT uniform
       but late-weighted, i.e. weak re-excitation during the rest of the drive
       rather than dark counts.

    3. STALE READS. `timestamp_mu` returns only the FIRST edge in a gate, so a
       second edge (an SPCM afterpulse 20-50 ns later) survives in the input
       FIFO and is handed to a LATER attempt carrying its old timestamp, which
       shows up as a hugely NEGATIVE delay. Each one is checked against the
       previous click on the SAME detector: if that sits within one gate, the
       mechanism is confirmed rather than assumed. Those heralds are attributed
       to the wrong attempt here; in parity the same value feeds
       `at_mu(click + 2000)` and schedules an MW edge in the PAST
       (RTIOUnderflow, see memory reference_stale_spcm_timestamp).

    Clicks PAST the nominal gate close are expected, not an error: the input
    gate's effective window is offset ~65 ns later in gateware than the
    timestamp reference, so the delay axis is shifted, not the shape.
    """
    d_all = data.get("click_delay_ns", np.array([]))
    if d_all.size == 0:
        return None
    det = data.get("click_detector", np.array([]))
    tclk = data.get("click_times_s", np.array([]))
    gate = meta.get("t_spcm_collection_mu")
    gate = float(gate) if gate else 400.0

    stale_mask = d_all < 0.0
    x = d_all[~stale_mask]
    out = {"n_clicks": int(d_all.size), "gate_ns": gate,
           "n_stale": int(np.count_nonzero(stale_mask)),
           "stale_frac": float(np.mean(stale_mask)),
           "most_negative": (float(d_all[stale_mask].min())
                             if np.any(stale_mask) else float("nan"))}

    # --- verify the stale mechanism against the click record ---------------
    pairs = []
    if det.size == d_all.size and tclk.size == d_all.size:
        for i in np.where(stale_mask)[0]:
            prev = np.where(det[:i] == det[i])[0]
            gap = ((tclk[i] - tclk[prev[-1]]) * 1e9 if prev.size
                   else float("nan"))
            pairs.append({"index": int(i), "detector": int(det[i]),
                          "delay_ns": float(d_all[i]), "gap_ns": float(gap),
                          "same_gate": bool(np.isfinite(gap)
                                            and abs(gap) < gate)})
    out["stale_pairs"] = pairs
    out["stale_confirmed"] = bool(pairs) and all(p["same_gate"] for p in pairs)

    if x.size < 10:
        out["fit"] = None
        return out

    top = max(gate, float(x.max()))
    out["top_ns"] = top
    out["n_past_gate"] = int(np.count_nonzero(x > gate))
    fit = fit_decay(x, top)
    out["fit"] = fit
    out["in_gate"] = int(x.size)

    if fit is not None and not fit["flat"]:
        # Uniform-background estimate from a guard window strictly BELOW the
        # onset. The onset itself is the 2nd percentile, so ~2 % of clicks sit
        # under it by construction -- using it directly would be circular.
        pre_hi = max(0.0, fit["t0"] - 20.0)
        n_pre = int(np.count_nonzero(x < pre_hi))
        if pre_hi > 0.0:
            dens = n_pre / pre_hi                       # clicks per ns
            out["bg_frac"] = float(dens * top / x.size)
            # Poisson 95 % upper limit on n_pre (3.0 when n_pre == 0)
            ul = 3.0 if n_pre == 0 else n_pre + 1.96 * math.sqrt(n_pre)
            out["bg_frac_ul"] = float((ul / pre_hi) * top / x.size)
            out["n_pre"] = n_pre
            out["pre_hi"] = pre_hi
            # A fitted flat weight far above the uniform estimate means the
            # second component is late-weighted, not dark counts.
            out["tail_is_late"] = bool(fit["w"] > 2.0 * out["bg_frac_ul"])
        out["tau_is_natural"] = bool(20.0 <= fit["tau"] <= 55.0)
    return out


def coincidence_bound(res, prof):
    """Bound the emitted photon number from two-detector coincidences.

    No detection efficiency needed. With per-detector probability
    q = clicks / (2 * attempts), nbar emitted photons give ~(nbar*q)^2
    coincidences per attempt, so a small coincidence count caps nbar.

    This is also what kills the alternative reading of the stale reads -- that
    the second edge in the gate is a second PHOTON rather than an afterpulse.
    A doubles rate of `stale/clicks` would need nbar = 1 + 2*(stale/clicks)/q,
    and that nbar predicts a coincidence count the data flatly excludes
    (RID 3646: nbar ~ 11 -> ~36 expected, 0 observed).
    """
    n_co = res["meta"].get("n_coincidences")
    nclk, natt = res["clicks"], res["attempts"]
    if n_co is None or not natt or not nclk:
        return None
    q = nclk / (2.0 * natt)
    out = {"n_coinc": int(n_co), "q_per_detector": float(q),
           "attempts": int(natt)}
    if n_co == 0:
        out["nbar_max"] = float(math.sqrt(3.0 / natt) / q)   # 95 % UL
        out["nbar_est"] = None
    else:
        out["nbar_est"] = float(math.sqrt(n_co / natt) / q)
        out["nbar_max"] = None

    # The "second edge = second photon" hypothesis, tested against this bound.
    n_stale = (prof or {}).get("n_stale", 0)
    if n_stale and nclk:
        r = n_stale / nclk
        nbar_needed = 1.0 + 2.0 * r / q
        exp_co = (nbar_needed * q) ** 2 * natt
        out["doubles_frac"] = float(r)
        out["nbar_if_photons"] = float(nbar_needed)
        out["coinc_expected_if_photons"] = float(exp_co)
        out["photon_doubles_excluded"] = bool(n_co == 0 and exp_co > 5.0)
        out["exclusion_p"] = float(math.exp(-exp_co)) if n_co == 0 else float("nan")
    return out


def summarize(path, verbose=True, n_boot=2000):
    """All per-atom diagnostics for one run. Returns a dict."""
    data, meta, rid = load_run(path)
    if meta["n_atoms"] == 0:
        if verbose:
            print("RID {}: no per-atom rows in {}".format(rid, path))
        return None

    a, h = data["atom_attempts"], data["atom_heralds"]
    # A run still in progress has no expid, and Check_img_every never reaches
    # the archive, so fall back to reading the block off the completed
    # lifetimes (see infer_block). Without this every block-based test returns
    # None on a live run.
    block = meta.get("Check_img_every")
    meta["block_inferred"] = False
    if not block:
        block = infer_block(data)
        meta["block_inferred"] = block is not None
    meta["block_effective"] = block
    res = {"rid": rid, "path": path, "data": data, "meta": meta,
           "n_atoms": int(meta["n_atoms"]),
           "n_censored": int(np.count_nonzero(data["censored"])),
           "clicks": int(h.sum()), "attempts": int(a.sum()),
           "dud_frac": float(np.mean(data["dud"])),
           "p_pooled": float(h.sum() / a.sum()) if a.sum() else float("nan"),
           "median_attempts_dud": float(np.median(a[data["dud"]]))
                                  if np.any(data["dud"]) else float("nan"),
           "median_attempts_herald": float(np.median(a[~data["dud"]]))
                                     if np.any(~data["dud"]) else float("nan"),
           "overdispersion": overdispersion(data, n_boot=n_boot),
           "hazard": hazard_by_block(data, block),
           "first_profile": first_herald_profile(data, block),
           "confirm": confirm_net_split(data),
           "trap_start": meta.get("trap_ramp_start_amp"),
           "block": block}
    res["flatness"] = (flatness(res["first_profile"])
                       if res["first_profile"] else None)
    res["two_rate"] = two_rate_consistency(res)
    res["photon"] = photon_profile(data, meta)
    res["coinc"] = coincidence_bound(res, res["photon"])

    if verbose:
        m = meta
        print("--- RateExperiment per-atom  RID {} ---".format(rid))
        print("file            : {}".format(path))
        if not m["has_expid"]:
            print("                  (no expid: write_results copy -- run in "
                  "progress; front-panel arguments unknown)")
        print("confirmed atoms : {}  ({} censored by run end)".format(
            res["n_atoms"], res["n_censored"]))
        print("clicks/attempts : {} / {}   pooled herald rate {:.3f}%".format(
            res["clicks"], res["attempts"], 100 * res["p_pooled"]))
        print("trap ramp start : {}   survival check every {} attempts{}".format
              (res["trap_start"], block,
               "  (INFERRED from the completed lifetimes -- no expid)"
               if m.get("block_inferred") else ""))
        if m["shadow_conflicts"]:
            print("   !! archive disagrees with expid (expid is what RAN):")
            for k, (av, ev) in m["shadow_conflicts"].items():
                print("      {:<24} archive {} -> actually {}".format(k, av, ev))
        print()
        print("dud fraction    : {:.0f}%   <- NOT evidence by itself; shot "
              "noise alone gives 35-50%".format(100 * res["dud_frac"]))
        print("median attempts : duds {:.0f}  vs heralders {:.0f}   <- also "
              "expected (living longer buys more chances)".format(
                  res["median_attempts_dud"], res["median_attempts_herald"]))
        print()

        od = res["overdispersion"]
        print("1) ONE POPULATION, OR A BAD SUB-POPULATION?  (the real test)")
        if od is None:
            print("   not enough atoms")
        else:
            print("   dispersion D   : {:.3f}   null {:.3f} +- {:.3f} "
                  "(95th pct {:.3f})".format(od["D"], od["null_mean"],
                                             od["null_sd"], od["null_95"]))
            print("   bootstrap p    : {:.4f}  ({} atoms, {} resamples)".format(
                od["p_value"], od["n_atoms"], n_boot))
            if od["overdispersed"]:
                print("   => OVERDISPERSED: atoms differ intrinsically. A "
                      "sub-population heralds worse than the rest.")
            else:
                print("   => CONSISTENT WITH ONE POPULATION. The duds are shot "
                      "noise on a single herald rate -- there is no bad "
                      "sub-population to find, and the limiter is the rate "
                      "itself (and the loss hazard below).")

        hz = res["hazard"]
        print()
        print("2) IS THE ATOM LOST RIGHT AFTER TRAPPING?  (hazard by block)")
        if hz is None:
            print("   not enough data")
        else:
            print("   {:>6} {:>9} {:>8} {:>8} {:>9}".format(
                "block", "attempts", "at risk", "deaths", "hazard"))
            for r in hz["rows"][:8]:
                print("   {:>6} {:>9} {:>8} {:>8} {:>8.3f}".format(
                    r["block"], r["attempts"], r["at_risk"], r["deaths"],
                    r["hazard"]))
            if len(hz["rows"]) > 8:
                print("   ... {} more blocks".format(len(hz["rows"]) - 8))
            print("   first block {:.3f}  vs later blocks {:.3f}   "
                  "excess {:+.3f}  ({:+.1f} sigma)".format(
                      hz["h_first"], hz["h_later"], hz["excess"], hz["z_first"]))
            if hz["early_loss"]:
                print("   => EXCESS EARLY LOSS. Atoms are dying in the "
                      "confirm -> ramp-down -> first-attempts transition, not "
                      "at a uniform rate. Supports losing the atom right after "
                      "trapping.")
            else:
                print("   => NO excess in the first block: loss looks uniform "
                      "across the atom's life. Does NOT support a special "
                      "loss right after trapping.")
            print("   (an atom lost while the MOT disperses would fail CONFIRM "
                   "and never appear here -- see the module docstring)")

        fp = res["first_profile"]
        print()
        print("3) WARM-UP?  P(first herald | survived to block)")
        if not fp:
            print("   not enough data")
        else:
            for r in fp[:6]:
                print("   block {:>2} (<= {:>4} attempts): {:>4}/{:<5} = "
                      "{:.3f} +- {:.3f}".format(r["block"], r["attempts"],
                                                r["hits"], r["at_risk"],
                                                r["p_first"], r["se"]))
            fl = res["flatness"]
            if fl is None:
                print("   too few well-populated blocks to test flatness")
            else:
                print("   constancy test over {} blocks: weighted mean {:.3f}, "
                      "chi2/dof = {:.2f}  ({:+.1f} sigma)".format(
                          fl["n_blocks"], fl["mean"], fl["chi2_red"], fl["z"]))
                if fl["flat"]:
                    print("   => FLAT: memoryless. The herald rate does NOT "
                          "depend on how long the atom has been held, so there "
                          "is no warm-up.")
                else:
                    print("   => NOT flat: the per-block herald rate varies "
                          "beyond its error bars. Inspect panel C before "
                          "calling it a warm-up (a dip can do this too).")

        tr = res["two_rate"]
        print()
        print("4) DO THE TWO RATES ALONE EXPLAIN THE DUDS?")
        if tr is None:
            print("   not enough data")
        else:
            print("   herald p   = {:.4f}%/attempt".format(
                100 * tr["p_per_attempt"]))
            print("   loss  h    = {:.4f}%/attempt  (from the later-block "
                  "hazard)".format(100 * tr["h_per_attempt"]))
            print("   predicted dud fraction h/(h+p) = {:.1f}%   observed "
                  "{:.1f}% +- {:.1f}%".format(100 * tr["dud_predicted"],
                                              100 * tr["dud_observed"],
                                              100 * tr["dud_se"]))
            if tr["agrees"]:
                print("   => THE DUDS ARE FULLY ACCOUNTED FOR by the herald "
                      "rate competing with the loss rate. Nothing is left over "
                      "for a bad sub-population, a warm-up or two-atom loads.")
                print("      The system is two numbers: raise p or lower h -- "
                      "there is no third mechanism to find.")
            else:
                print("   => MISMATCH ({:+.1f} sigma): the two rates do not "
                      "account for the duds, so something else is removing "
                      "atoms or suppressing heralds.".format(tr["z"]))

        cf = res["confirm"]
        print()
        print("5) TWO-ATOM LOADS?  confirm counts, duds vs heralders")
        if cf is None:
            print("   not enough data")
        else:
            print("   duds      : median {:.1f}  mean {:.1f} +- {:.1f} "
                  "(n={})".format(cf["duds"]["median"], cf["duds"]["mean"],
                                  cf["duds"]["sd"], cf["duds"]["n"]))
            print("   heralders : median {:.1f}  mean {:.1f} +- {:.1f} "
                  "(n={})".format(cf["heralders"]["median"],
                                  cf["heralders"]["mean"],
                                  cf["heralders"]["sd"], cf["heralders"]["n"]))
            print("   median ratio dud/herald = {:.2f}   => {}".format(
                cf["ratio_median"],
                "~2x: TWO-ATOM loads likely" if cf["ratio_median"] > 1.5
                else "no 2x shoulder: duds are not brighter"))

        ph, co = res["photon"], res["coinc"]
        print()
        print("6) IS EACH HERALD ONE PHOTON?  (arrival time inside the gate)")
        if ph is None:
            print("   no click_delay_ns in this run (added 2026-10-22)")
        else:
            fit = ph["fit"]
            drive = meta["archive"].get("exc_beam_duration_mu")
            if fit is None:
                print("   too few in-gate clicks to fit ({})".format(
                    ph.get("in_gate", 0)))
            elif fit["flat"]:
                print("   FLAT across the {:.0f} ns gate -> multi-photon "
                      "scattering or background, NOT a clean single "
                      "photon".format(ph["gate_ns"]))
            else:
                print("   onset {:.0f} ns, tau = {:.1f} ns "
                      "[{:.1f}-{:.1f}], flat component {:.0f}%  (N={})".format(
                          fit["t0"], fit["tau"], fit["tau_lo"], fit["tau_hi"],
                          100 * fit["w"], fit["n"]))
                print("   Cs natural lifetime: D2 30.5 ns, D1 34.9 ns")
                if ph.get("tau_is_natural"):
                    print("   => tau IS the natural lifetime: the atom emits "
                          "ONE photon and is then pumped dark.")
                    if drive:
                        print("      It is driven for {:.0f} ns (~{:.0f} "
                              "lifetimes) yet emits for ~{:.0f} ns, so the "
                              "remaining drive does nothing.".format(
                                  drive, drive / fit["tau"], 3 * fit["tau"]))
                else:
                    print("   => tau is NOT the natural lifetime: suspect "
                          "multi-photon scattering or background.")
                if "bg_frac" in ph:
                    print("   uniform background, from the {:.0f} ns guard "
                          "window below the onset ({} clicks): {:.1f}% of "
                          "heralds (95% UL {:.1f}%)".format(
                              ph["pre_hi"], ph["n_pre"], 100 * ph["bg_frac"],
                              100 * ph["bg_frac_ul"]))
                    if ph.get("tail_is_late"):
                        print("   => the fitted flat component ({:.0f}%) is far "
                              "above that, so it is LATE-weighted: a weak "
                              "re-excitation tail, not dark counts.".format(
                                  100 * fit["w"]))
                if ph.get("n_past_gate"):
                    print("   {} clicks past the nominal gate close -- "
                          "expected: the input gate's effective window is "
                          "offset later in gateware, so the delay axis is "
                          "shifted, not the shape.".format(ph["n_past_gate"]))

            if ph["n_stale"]:
                print()
                print("   STALE READS: {} of {} clicks ({:.2f}%) carried a "
                      "timestamp from an EARLIER gate (most negative "
                      "{:.0f} ns)".format(ph["n_stale"], ph["n_clicks"],
                                          100 * ph["stale_frac"],
                                          ph["most_negative"]))
                for p in ph["stale_pairs"]:
                    print("      det {}: {:+.0f} ns delay, {:+.0f} ns after the "
                          "previous click on the same detector{}".format(
                              p["detector"], p["delay_ns"], p["gap_ns"],
                              "  <- SAME GATE" if p["same_gate"] else ""))
                if ph["stale_confirmed"]:
                    print("   => MECHANISM CONFIRMED: every stale read follows "
                          "a same-detector click within one gate. timestamp_mu "
                          "returns only the FIRST edge, so a second edge waits "
                          "in the input FIFO and a later attempt reads it.")
                print("   impact: those heralds are attributed to the wrong "
                      "attempt, inflating p by {:.2f}%. In PARITY the same "
                      "value feeds at_mu(click + 2000) -> MW edge scheduled in "
                      "the PAST -> RTIOUnderflow, ~1 herald in {:.0f}.".format(
                          100 * ph["stale_frac"],
                          1.0 / max(ph["stale_frac"], 1e-9)))

        print()
        print("7) HOW MANY PHOTONS PER ATTEMPT?  (two-detector coincidences)")
        if co is None:
            print("   no n_coincidences in this run (added 2026-10-22)")
        else:
            if co["nbar_max"] is not None:
                print("   0 coincidences in {} attempts -> nbar < {:.1f} "
                      "emitted photons per attempt (95% CL)".format(
                          co["attempts"], co["nbar_max"]))
            else:
                print("   {} coincidences in {} attempts -> nbar ~ {:.1f} "
                      "photons per attempt".format(
                          co["n_coinc"], co["attempts"], co["nbar_est"]))
            if "nbar_if_photons" in co:
                print("   could the in-gate doubles be second PHOTONS rather "
                      "than afterpulses?")
                print("      that needs nbar = {:.1f}, which predicts {:.0f} "
                      "coincidences; {} observed{}".format(
                          co["nbar_if_photons"],
                          co["coinc_expected_if_photons"], co["n_coinc"],
                          "  => EXCLUDED (p = {:.0e})".format(co["exclusion_p"])
                          if co["photon_doubles_excluded"] else ""))
            if res["p_pooled"] == res["p_pooled"]:
                print("   => with ~1 photon per attempt the herald probability "
                      "IS the collection+detection efficiency:")
                print("      eta ~ {:.3f}% per emitted photon. The heralding "
                      "rate is COLLECTION-LIMITED -- raising it means more "
                      "light (NA, fibre coupling, cavity), or a lower h."
                      .format(100 * res["p_pooled"]))
    return res


def compare_runs(rids=None, arg="trap_ramp_start_amp", verbose=True,
                 n_boot=500):
    """Dud fraction / herald rate / early-loss vs a front-panel argument.

    Built for the trap-depth series: the argument comes from expid, which is
    the only place the real value lives.
    """
    files = find_runs() if rids is None else [find_run(r) for r in rids]
    rows = []
    for p in files:
        r = summarize(p, verbose=False, n_boot=n_boot)
        if r is None:
            continue
        od, hz = r["overdispersion"], r["hazard"]
        rows.append({"rid": r["rid"], "arg": r["meta"].get(arg),
                     "n_atoms": r["n_atoms"], "clicks": r["clicks"],
                     "dud_frac": r["dud_frac"], "p_pooled": r["p_pooled"],
                     "D": od["D"] if od else float("nan"),
                     "p_val": od["p_value"] if od else float("nan"),
                     "h_first": hz["h_first"] if hz else float("nan"),
                     "h_later": hz["h_later"] if hz else float("nan")})
    if verbose and rows:
        print("{:>6} {:>12} {:>7} {:>7} {:>8} {:>9} {:>7} {:>8} {:>8}".format(
            "RID", arg[:12], "atoms", "clicks", "dud%", "herald%", "D",
            "h_1st", "h_later"))
        for r in rows:
            print("{:>6} {:>12} {:>7} {:>7} {:>7.0f}% {:>8.3f}% {:>7.2f} "
                  "{:>8.3f} {:>8.3f}".format(
                      r["rid"], str(r["arg"]), r["n_atoms"], r["clicks"],
                      100 * r["dud_frac"], 100 * r["p_pooled"], r["D"],
                      r["h_first"], r["h_later"]))
        print()
        print("Compare rows at DIFFERENT {} -- if dud% and herald% track the "
              "argument, it matters; if they are flat, it does not.".format(arg))
    return rows


def compare_runs_legacy(arg="trap_ramp_start_amp", min_rid=0, verbose=True):
    """Compare runs on the LEGACY per-click datasets, which every run has.

    The per-atom instrumentation only exists from 2026-10-22 onward, so a
    parameter scan that started before it (e.g. the trap-depth series
    trap_ramp_start_amp 0.21 / 0.18 / 0.15) cannot be compared with
    compare_runs(). `trials_per_click` and `how_many_atoms` are present in all
    runs and give the herald rate directly:

        herald rate = n_clicks / sum(trials_per_click)

    The binomial error on that is sqrt(n_clicks)/attempts, which is what decides
    whether a difference between two runs is real. Note `clicks/load` is NOT a
    clean efficiency: how_many_atoms counts load cycles including confirm-fails,
    and it resets at every click.
    """
    rows = []
    for root in RESULTS_DIRS:
        for p in glob.glob(str(root / "**" / PATTERN), recursive=True):
            rid = _rid_of(p)
            if rid < min_rid or not _has_expid(p):
                continue
            try:
                with h5py.File(p, "r", locking=False) as hf:
                    ds = hf["datasets"] if "datasets" in hf else hf
                    if "trials_per_click" not in ds:
                        continue
                    tpc = np.atleast_1d(ds["trials_per_click"][()]).astype(float)
                    hma = (np.atleast_1d(ds["how_many_atoms"][()]).astype(float)
                           if "how_many_atoms" in ds else np.array([]))
                    raw = hf["expid"][()]
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    val = json.loads(raw).get("arguments", {}).get(arg)
            except Exception:
                continue
            nclk, att = tpc.size, float(tpc.sum())
            if att <= 0 or nclk == 0:
                continue
            p_hat = nclk / att
            rows.append({"rid": rid, "arg": val, "clicks": int(nclk),
                         "attempts": int(att), "p": p_hat,
                         "p_se": float(np.sqrt(nclk) / att),
                         "loads": int(hma.sum()) if hma.size else 0})
    rows.sort(key=lambda r: r["rid"])
    # Deduplicate RIDs (master + write_results copies).
    seen, uniq = set(), []
    for r in rows:
        if r["rid"] in seen:
            continue
        seen.add(r["rid"])
        uniq.append(r)
    rows = uniq

    if verbose and rows:
        print("{:>6} {:>8} {:>8} {:>10} {:>18} {:>8}".format(
            "RID", arg[:8], "clicks", "attempts", "herald rate", "loads"))
        for r in rows:
            print("{:>6} {:>8} {:>8} {:>10} {:>10.4f}% +-{:.4f} {:>8}".format(
                r["rid"], str(r["arg"]), r["clicks"], r["attempts"],
                100 * r["p"], 100 * r["p_se"], r["loads"]))
        # Pool by argument value and test the extremes against each other.
        groups = {}
        for r in rows:
            groups.setdefault(r["arg"], []).append(r)
        print()
        print("pooled by {}:".format(arg))
        pooled = []
        for v in sorted(groups, key=lambda x: (x is None, x)):
            g = groups[v]
            c = sum(r["clicks"] for r in g)
            a = sum(r["attempts"] for r in g)
            ph, se = c / a, np.sqrt(c) / a
            pooled.append((v, ph, se, c))
            print("   {:<8} {:>6} clicks / {:>8} attempts = {:.4f}% +- "
                  "{:.4f}%  ({} runs)".format(str(v), c, a, 100 * ph,
                                              100 * se, len(g)))
        if len(pooled) >= 2:
            (v1, p1, s1, _), (v2, p2, s2, _) = pooled[0], pooled[-1]
            dz = (p1 - p2) / np.sqrt(s1 ** 2 + s2 ** 2)
            print()
            print("   {} vs {}: difference {:+.4f}% ({:+.1f} sigma) => {}"
                  .format(v1, v2, 100 * (p1 - p2), dz,
                          "NO detectable effect" if abs(dz) < 2
                          else "REAL difference"))
    return rows


# --------------------------------------------------------------------------
#  FIGURE
# --------------------------------------------------------------------------
def analyze(path, out_png=None, n_boot=2000):
    """Six-panel figure. Uses the active backend; returns the Figure.

    A-D are the per-atom tests (one population / early loss / warm-up /
    two-atom loads); E-F are the photon-level tests added 2026-10-22, where F
    is E on a log y axis -- a straight line there IS the single-exponential
    result, which is the cleanest form of the evidence.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    res = summarize(path, n_boot=n_boot)
    if res is None:
        return None
    data, meta, rid = res["data"], res["meta"], res["rid"]
    a, h = data["atom_attempts"], data["atom_heralds"]
    dud = data["dud"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    axA, axB, axC, axD, axE, axF = axes.ravel()

    # --- A: per-atom herald rate vs the binomial null (overdispersion) ---
    od = res["overdispersion"]
    keep = a > 0
    rate = h[keep] / a[keep]
    bins = np.linspace(0, max(rate.max(), 1e-9) * 1.05, 25)
    axA.hist(rate, bins=bins, color="C0", alpha=0.75, edgecolor="black", lw=0.3,
             label="observed")
    if od is not None:
        rng = np.random.RandomState(1)
        sim = rng.binomial(a[keep].astype(int), od["p_pooled"]) / a[keep]
        axA.hist(sim, bins=bins, histtype="step", color="C3", lw=1.8,
                 label="binomial null (one population)")
        axA.set_title("A. Per-atom herald rate  |  D = {:.2f}, p = {:.3f} "
                      "({})".format(od["D"], od["p_value"],
                                    "overdispersed" if od["overdispersed"]
                                    else "one population"), fontsize=10)
    axA.axvline(res["p_pooled"], color="C7", ls="--", lw=1.2,
                label="pooled {:.3f}%".format(100 * res["p_pooled"]))
    axA.set_xlabel("heralds / attempts for one atom")
    axA.set_ylabel("atoms")
    axA.legend(fontsize=8)
    axA.grid(alpha=0.3)

    # --- B: hazard by block (the early-loss test) ---
    hz = res["hazard"]
    if hz is not None:
        xs = [r["attempts"] for r in hz["rows"]]
        ys = [r["hazard"] for r in hz["rows"]]
        es = [r["se"] for r in hz["rows"]]
        axB.errorbar(xs, ys, yerr=es, fmt="-o", ms=4, lw=1, capsize=2,
                     color="C0", label="loss hazard")
        axB.axhline(hz["h_later"], color="C3", ls="--", lw=1.4,
                    label="later-blocks mean {:.3f}".format(hz["h_later"]))
        axB.set_title("B. Loss hazard per {}-attempt block  |  first block "
                      "{:+.1f}$\\sigma$  ({})".format(
                          res["block"], hz["z_first"],
                          "EXCESS early loss" if hz["early_loss"]
                          else "no early excess"), fontsize=10)
    axB.set_xlabel("attempts since confirm")
    axB.set_ylabel("P(lost in this block | alive)")
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3)

    # --- C: first-herald profile (warm-up test) ---
    fp = res["first_profile"]
    if fp:
        xs = [r["attempts"] for r in fp]
        ys = [r["p_first"] for r in fp]
        es = [r["se"] for r in fp]
        axC.errorbar(xs, ys, yerr=es, fmt="-o", ms=4, lw=1, capsize=2,
                     color="C2")
        axC.set_title("C. P(first herald | survived to block) -- flat means "
                      "no warm-up", fontsize=10)
    axC.set_xlabel("attempts since confirm")
    axC.set_ylabel("P(first herald in block)")
    axC.grid(alpha=0.3)

    # --- D: confirm counts, duds vs heralders (two-atom test) ---
    cn = data["atom_confirm_net"]
    lo, hi = float(cn.min()), float(cn.max())
    step = max(1.0, (hi - lo + 1) / 40)
    b = np.arange(lo, hi + step, step)
    axD.hist(cn[~dud], bins=b, color="C2", alpha=0.6, label="heralders")
    axD.hist(cn[dud], bins=b, color="C3", alpha=0.6, label="duds")
    thr = meta["archive"].get("atom_threshold")
    if thr is not None:
        axD.axvline(thr, color="C7", ls="--", lw=1.2,
                    label="confirm threshold {:.0f}".format(thr))
    cf = res["confirm"]
    axD.set_title("D. Confirm-image counts  |  median ratio dud/herald = "
                  "{:.2f}".format(cf["ratio_median"] if cf else float("nan")),
                  fontsize=10)
    axD.set_xlabel("confirm net counts")
    axD.set_ylabel("atoms")
    axD.legend(fontsize=8)
    axD.grid(alpha=0.3)

    # --- E / F: photon arrival time inside the gate (the single-photon test) ---
    ph = res["photon"]
    for ax, logy in ((axE, False), (axF, True)):
        if ph is None or ph["fit"] is None:
            ax.set_title("{}. Photon arrival in gate -- no click_delay_ns"
                         .format("E" if not logy else "F"), fontsize=10)
            ax.set_xlabel("photon delay after gate open [ns]")
            continue
        fit = ph["fit"]
        d_all = data["click_delay_ns"]
        x = d_all[d_all >= 0.0]
        top, binw = ph["top_ns"], 10.0
        edges = np.arange(0.0, top + binw, binw)
        ax.hist(x, bins=edges, color="C1", alpha=0.75, edgecolor="black",
                lw=0.3, label="clicks ({})".format(x.size))
        ts = np.linspace(0.0, top, 500)
        z = fit["tau"] * (1.0 - np.exp(-fit["L"] / fit["tau"]))
        pdf = (np.where(ts >= fit["t0"],
                        (1.0 - fit["w"]) * np.exp(-(ts - fit["t0"]) / fit["tau"]) / z,
                        0.0) + fit["w"] / top)
        ax.plot(ts, x.size * binw * pdf, color="C0", lw=2,
                label=r"fit: $\tau$ = {:.1f} ns, bg {:.0f}%".format(
                    fit["tau"], 100 * fit["w"]))
        ax.axvline(ph["gate_ns"], color="C3", ls="--", lw=1.4,
                   label="nominal gate close")
        ax.axvline(fit["t0"], color="C7", ls=":", lw=1.4,
                   label="onset {:.0f} ns".format(fit["t0"]))
        ax.set_xlabel("photon delay after gate open [ns]")
        ax.set_ylabel("clicks / {:.0f} ns".format(binw))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            ax.set_title("F. Same, log y -- a STRAIGHT line is a single "
                         "exponential  |  Cs natural 30.5/34.9 ns", fontsize=10)
        else:
            stale = ("  |  {} stale rejected".format(ph["n_stale"])
                     if ph["n_stale"] else "")
            ax.set_title("E. Photon arrival in gate  |  {}{}".format(
                "tau IS the natural lifetime => ~1 photon/attempt"
                if ph.get("tau_is_natural") else "tau is NOT the natural lifetime",
                stale), fontsize=10)

    co = res["coinc"]
    tail = ""
    if co is not None and co["nbar_max"] is not None:
        tail = "  |  nbar < {:.1f}, eta ~ {:.2f}%".format(
            co["nbar_max"], 100 * res["p_pooled"])
    fig.suptitle("RateExperiment RID {}  |  {} atoms, {} clicks, pooled "
                 "{:.3f}%  |  trap_ramp_start_amp = {}  |  dud {:.0f}%{}".format(
                     rid, res["n_atoms"], res["clicks"], 100 * res["p_pooled"],
                     res["trap_start"], 100 * res["dud_frac"], tail),
                 fontsize=13)

    if out_png is None:
        out_png = str(Path(path).with_name(
            "rate_per_atom_RID{}.png".format(rid)))
    fig.savefig(out_png, dpi=130)
    print("Saved figure: {}".format(out_png))
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in ("inline", "ipympl", "nbagg", "widget")):
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    argv = sys.argv[1:]
    if argv and argv[0].lower() == "compare":
        compare_runs()
    else:
        rid = int(argv[0]) if argv else None
        analyze(find_run(rid))
