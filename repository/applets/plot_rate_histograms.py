"""Live 3-panel histogram monitor for Rate_measurement.py.

Pops up automatically when RateExperiment runs (spawned via CCB). Panels:
  1. HIGH-trap atom readout  -- net confirm-image counts in the loading loop,
     with the atom_threshold line. Should be bimodal: empty-trap (near 0) vs
     loaded atom (bright); the threshold should sit in the valley between them.
  2. LOW-trap survival readout -- raw survival-image counts in the recovery loop,
     with the atom_threshold_lowtrap line. THIS is the diagnostic: if the atom
     and empty-trap peaks are NOT separated, or the line sits inside the atom
     peak / below the empty-trap counts, the survival check is passing empty
     traps (false negatives) and wasting excitation attempts on a dead trap.
  3. ATOM DECAY PROFILE -- where each heralded single photon arrived INSIDE the
     collection gate (dataset click_delay_ns, measured from gate open).

Panel 3 is the single-photon test, and it needs no detection efficiency: it is
a SHAPE, not a rate.
  exponential, tau = 30-35 ns   (Cs natural lifetime: D2 30.5, D1 34.9)
      -> the atom emits ONE photon and goes dark, so each herald is a single
         spontaneous emission -- even though exc_beam_duration_mu drives it for
         ~10 lifetimes (300 ns), the atom is pumped dark after ~1 scattering.
  flat across the gate
      -> multi-photon scattering, or background / dark counts, NOT a clean
         single photon.
The fit (cyan overlay) is the photon arrival distribution the physics
predicts: the excitation pulse, modelled as a GAUSSIAN (centre mu, rms width
sigma), CONVOLVED with the exponential spontaneous decay (tau) -- an
exponentially modified Gaussian -- plus a flat background fraction w, all
truncated to the gate and fitted by maximum likelihood on the 1 ns delay grid.
A bare exponential with a hard onset forces the finite pulse width into the
decay and biases tau high (33-38 ns on RIDs 3646-3652); the convolution
separates the two: RIDs 3649 / 3652 give sigma = 4.2 / 4.7 ns (FWHM ~10 ns,
consistent with the ~12 ns excitation switch pulse) and tau = 30.1 / 29.5 ns,
i.e. the Cs D2 lifetime (30.5 ns). With sigma -> 0 it reduces to the old
exponential model.
The red dashed line is the NOMINAL gate close; clicks past it are normal --
the input gate effective window is offset ~65 ns later in gateware than the
timestamp reference, so the delay axis is shifted, not the shape.

NEGATIVE delays are rejected and counted in the panel title. They are not early
photons but STALE timestamps: `timestamp_mu` returns only the FIRST edge in a
gate, so a second edge (an SPCM afterpulse ~20-50 ns later) survives in the
input FIFO and is handed to a LATER attempt carrying its old, milliseconds-old
timestamp. Worth watching: in parity that same value feeds
`at_mu(click + 2000)` and schedules an MW edge in the PAST (RTIOUnderflow).

Usage (also the CCB command the experiment issues):
  ${python} plot_rate_histograms.py rate_readout_hightrap
      --lowtrap rate_readout_lowtrap --delays click_delay_ns
      --gate rate_gate_ns
      --thr_hi rate_atom_threshold --thr_lo rate_atom_threshold_lowtrap
"""

import math

import numpy as np
import PyQt5  # ensure Qt5
from PyQt5.QtCore import Qt
import pyqtgraph

from artiq.applets.simple import SimpleApplet


def _get(data, name):
    """Value of a dataset by name, or None if absent/empty-named."""
    if not name:
        return None
    entry = data.get(name)
    return entry[1] if entry else None


_SQRT2 = math.sqrt(2.0)


def _log_erfc(x):
    """log(erfc(x)), vectorised and underflow-free.

    Numerical Recipes `erfcc` (fractional error < 1.2e-7 everywhere), kept in
    log form for x >= 0 so the far Gaussian tail does not underflow to 0 and
    then multiply an overflowing exponential. Checked against math.erfc to
    1.1e-7 absolute in log. numpy only: no scipy, and no np.dot (the BLAS in
    the ARTIQ env segfaults -- memory reference_broken_blas_env).
    """
    x = np.asarray(x, dtype=float)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = (-1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
            + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
            + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    out = np.log(t) - z * z + poly                    # log erfc(|x|)
    neg = x < 0.0
    out[neg] = np.log(2.0 - np.exp(out[neg]))         # erfc(-z) = 2 - erfc(z)
    return out


def _emg_logpdf(k, mu, sigma, tau):
    """log of the exponentially modified Gaussian density at k (ns):
    Gaussian pulse (mu, sigma) convolved with (1/tau) exp(-t/tau), t >= 0.

        f(k) = 1/(2 tau) exp(sigma^2/(2 tau^2) - (k-mu)/tau)
                         * erfc((sigma/tau - (k-mu)/sigma) / sqrt 2)
    """
    d = k - mu
    z = (sigma / tau - d / sigma) / _SQRT2
    return (-math.log(2.0 * tau) + 0.5 * (sigma / tau) ** 2 - d / tau
            + _log_erfc(z))


def emg_pmf(top, mu, sigma, tau, w):
    """Model probability per 1 ns bin on k = 0..top:
    (1 - w) * EMG truncated and renormalised to the window  +  w * uniform.
    Returns None if the EMG has no mass inside the window."""
    k = np.arange(0.0, top + 1.0)
    p = np.exp(_emg_logpdf(k, mu, sigma, tau))
    s = p.sum()
    if not np.isfinite(s) or s <= 0.0:
        return None
    return (1.0 - w) * p / s + w / (top + 1.0)


def _nelder_mead(f, x0, steps, max_iter=600, ftol=1e-7):
    """Plain Nelder-Mead simplex minimiser (no scipy in the applet env)."""
    n = len(x0)
    pts = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += steps[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(max_iter):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < ftol:
            break
        c = sum(pts[:-1]) / n
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = f(xe)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * ((xr if fr < vals[-1] else pts[-1]) - c)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:                                      # shrink toward best
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_emg(x, top):
    """Maximum-likelihood fit of Gaussian-pulse (*) exponential-decay + flat
    background to the photon delays x (ns since gate open) on [0, top].

    Delays are integer ns (click - gate_open in mu, 1 mu = 1 ns on Kasli), so
    the likelihood is a bincount against the per-ns model: the cost does not
    grow with the number of clicks (~25 ms per fit), so no subsampling.

    Free parameters: mu (pulse centre), sigma (pulse rms width), tau (decay
    constant), w (flat background fraction), optimised as (mu, log sigma,
    log tau, logit w) with a Nelder-Mead pass and one restart. Validated on
    synthetic data (40 seeds, N = 1000): unbiased in all four, tau to +-1.2 ns.

    Returns dict(mu, sigma, tau, w, n, top, flat) or None for < 10 clicks.
    `flat` means no decay is resolved (tau comparable to the window or the
    background dominates).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0.0) & (x <= top)]
    if x.size < 10 or not np.isfinite(top) or top <= 0.0:
        return None
    top = float(int(np.ceil(top)))
    counts = np.bincount(np.rint(x).astype(int),
                         minlength=int(top) + 1)[: int(top) + 1].astype(float)
    tau_max = 0.6 * top

    def unpack(th):
        return (th[0], math.exp(th[1]), math.exp(th[2]),
                1.0 / (1.0 + math.exp(-th[3])))

    def nll(th):
        mu, sigma, tau, w = unpack(th)
        if not (-100.0 <= mu <= top and 0.2 <= sigma <= 200.0
                and 2.0 <= tau <= tau_max and w <= 0.6):
            return 1e300
        p = emg_pmf(top, mu, sigma, tau, w)
        if p is None:
            return 1e300
        return float(-(counts * np.log(np.maximum(p, 1e-300))).sum())

    mu0 = float(np.percentile(x, 10.0))
    tail = x[x > mu0]
    tau0 = float(np.clip(tail.mean() - mu0 if tail.size else 30.0,
                         5.0, 0.5 * tau_max))
    th, _ = _nelder_mead(nll, (mu0, math.log(5.0), math.log(tau0),
                               math.log(0.05 / 0.95)), (10.0, 0.7, 0.4, 1.0))
    th, _ = _nelder_mead(nll, th, (3.0, 0.3, 0.15, 0.5))     # restart
    mu, sigma, tau, w = unpack(th)
    return dict(mu=mu, sigma=sigma, tau=tau, w=w, n=int(x.size), top=top,
                flat=bool(tau > 0.5 * top or w > 0.6))


class RateHistograms(pyqtgraph.GraphicsLayoutWidget):
    def __init__(self, args):
        pyqtgraph.GraphicsLayoutWidget.__init__(self)
        self.args = args

        self.p1 = self.addPlot(row=0, col=0)
        self.p1.setLabel("bottom", "SPCM1+2 net counts")
        self.p1.setLabel("left", "shots")

        self.p2 = self.addPlot(row=1, col=0)
        self.p2.setLabel("bottom", "SPCM1+2 counts")
        self.p2.setLabel("left", "checks")

        self.p3 = self.addPlot(row=2, col=0)
        self.p3.setLabel("bottom", "photon delay after gate open [ns]")
        self.p3.setLabel("left", "clicks")

    def _hist(self, plot, vals, thr, brush, title):
        plot.clear()
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            plot.setTitle(title + "  (collecting...)")
            return
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1.0
        # include the threshold in the range so its line is always visible
        if thr is not None and np.isfinite(thr):
            lo = min(lo, float(thr)); hi = max(hi, float(thr))
        nbins = int(max(10, min(60, np.sqrt(v.size))))
        counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
        plot.plot(edges, counts, stepMode=True, fillLevel=0,
                  brush=brush, pen=pyqtgraph.mkPen(brush))
        if thr is not None and np.isfinite(thr):
            plot.addLine(x=float(thr),
                         pen=pyqtgraph.mkPen((255, 60, 60), width=2, style=Qt.DashLine))
            plot.setTitle(f"{title}   (N={v.size}, threshold={float(thr):.0f})")
        else:
            plot.setTitle(f"{title}   (N={v.size})")

    def _decay(self, plot, delays, gate_ns):
        """Panel 3: photon arrival time inside the gate, with the live tau fit."""
        plot.clear()
        d = np.asarray(delays, dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            plot.setTitle("Atom decay profile   (collecting...)")
            return

        # Stale timestamps (see module docstring) are not early photons.
        n_stale = int(np.count_nonzero(d < 0.0))
        x = d[d >= 0.0]
        gate = (float(gate_ns) if gate_ns is not None
                and np.isfinite(gate_ns) and float(gate_ns) > 0.0 else 400.0)
        suffix = f", {n_stale} STALE rejected" if n_stale else ""
        if x.size < 5:
            plot.setTitle(f"Atom decay profile   (collecting... N={x.size}{suffix})")
            return

        top = max(gate, float(x.max()))
        binw = 10.0
        counts, edges = np.histogram(x, bins=np.arange(0.0, top + binw, binw))
        plot.plot(edges, counts, stepMode=True, fillLevel=0,
                  brush=(255, 170, 0, 150), pen=pyqtgraph.mkPen((255, 170, 0)))
        # nominal gate close -- clicks beyond it are the gateware offset
        plot.addLine(x=gate, pen=pyqtgraph.mkPen((255, 60, 60), width=2,
                                                 style=Qt.DashLine))

        fit = fit_emg(x, top)
        title = f"Atom decay profile   N={x.size}{suffix}"
        if fit is None:
            plot.setTitle(title + "   (fitting...)")
            return
        if fit["flat"]:
            plot.setTitle(title + "   FLAT: multi-photon scattering or background")
            return

        mu, sigma, tau, w = fit["mu"], fit["sigma"], fit["tau"], fit["w"]
        pmf = emg_pmf(fit["top"], mu, sigma, tau, w)      # per 1 ns bin
        if pmf is not None:
            k = np.arange(0.0, fit["top"] + 1.0)
            # expected clicks per displayed binw-ns bin
            plot.plot(k, x.size * binw * pmf,
                      pen=pyqtgraph.mkPen((0, 220, 255), width=2))
        verdict = ("~1 photon/attempt" if 20.0 <= tau <= 55.0
                   else "NOT the natural lifetime")
        plot.setTitle(title + f"   pulse {mu:.0f} ns, sigma {sigma:.1f} ns  *  "
                              f"tau = {tau:.1f} ns, bg {100 * w:.0f}%  [{verdict}]")

    def data_changed(self, data, mods, title=None):
        hi = _get(data, self.args.y)
        lo = _get(data, getattr(self.args, "lowtrap", None))
        dly = _get(data, getattr(self.args, "delays", None))
        gate = _get(data, getattr(self.args, "gate", None))
        thr_hi = _get(data, getattr(self.args, "thr_hi", None))
        thr_lo = _get(data, getattr(self.args, "thr_lo", None))

        if hi is not None and len(hi):
            self._hist(self.p1, hi, thr_hi, (0, 150, 255, 150),
                       "High-trap atom readout (load loop)")
        if lo is not None and len(lo):
            self._hist(self.p2, lo, thr_lo, (80, 220, 100, 150),
                       "Low-trap survival readout (recovery)")
        if dly is not None and len(dly):
            self._decay(self.p3, dly, gate)


def main():
    applet = SimpleApplet(RateHistograms)
    applet.add_dataset("y", "high-trap atom readout list (net counts)")
    applet.add_dataset("lowtrap", "low-trap survival readout list (raw counts)",
                       required=False)
    applet.add_dataset("delays", "click_delay_ns list (photon delay after gate open)",
                       required=False)
    applet.add_dataset("gate", "nominal gate length in ns (rate_gate_ns)",
                       required=False)
    # Kept so applet commands saved in the dashboard from before panel 3 became
    # the decay profile still start instead of dying on an unknown argument.
    # No longer plotted (the click-time-vs-run-time profile it fed is gone).
    applet.add_dataset("clicks", "DEPRECATED, unused: click_times_s list",
                       required=False)
    applet.add_dataset("thr_hi", "high-trap threshold scalar (atom_threshold)",
                       required=False)
    applet.add_dataset("thr_lo", "low-trap threshold scalar (atom_threshold_lowtrap)",
                       required=False)
    applet.run()


if __name__ == "__main__":
    main()
