"""Hybrid LSQ → MLE fitting pipeline.

Provides a base class for all fitting models. The pipeline:
  1. LSQ pre-fit via scipy curve_fit for robust initial guess
  2. MLE refinement using binomial NLL with L-BFGS-B
  3. Error extraction from Hessian inverse

Extracted from Microwave_var_scan_analysis_v5_hybrid_MLE.py and
Parity_analysis_w_QWP_scan.py.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize


class HybridFitter:
    """Base class for hybrid LSQ → MLE fitting.

    Subclasses must implement:
      - model(x, *params) -> y  (the fit function)
      - initial_guess(x, y, yerr) -> (p0, bounds_low, bounds_high)
      - param_names  (list of parameter name strings)
    """

    param_names = []

    @staticmethod
    def model(x, *params):
        raise NotImplementedError

    def initial_guess(self, x, y, yerr=None):
        """Return (p0, lower_bounds, upper_bounds) for curve_fit."""
        raise NotImplementedError

    def fit_lsq(self, x, y, yerr=None, p0=None, bounds=None):
        """Stage 1: Least-squares fit via curve_fit.

        Returns dict with 'params', 'errors', 'cov', 'success'.
        """
        if p0 is None or bounds is None:
            p0_auto, lo, hi = self.initial_guess(x, y, yerr)
            if p0 is None:
                p0 = p0_auto
            if bounds is None:
                bounds = (lo, hi)

        sigma = yerr if yerr is not None else None
        absolute_sigma = yerr is not None

        try:
            popt, pcov = curve_fit(self.model, x, y, p0=p0, bounds=bounds,
                                   sigma=sigma, absolute_sigma=absolute_sigma,
                                   maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            return {"params": popt, "errors": perr, "cov": pcov, "success": True}
        except Exception as e:
            print(f"[fit_lsq] curve_fit failed: {e}")
            return {"params": np.array(p0), "errors": np.full(len(p0), np.inf),
                    "cov": None, "success": False}

    def fit_mle_binomial(self, x, k, n, p0=None, bounds=None):
        """Stage 2: MLE refinement using binomial negative log-likelihood.

        Parameters
        ----------
        x : array-like
            Independent variable (scan parameter).
        k : array-like
            Number of successes (e.g., retained atoms) per bin.
        n : array-like
            Number of trials (e.g., loaded atoms) per bin.
        p0 : array-like
            Initial parameter guess (typically from LSQ).
        bounds : list of (lo, hi) tuples
            Parameter bounds for L-BFGS-B.

        Returns dict with 'params', 'errors', 'nll', 'success'.
        """
        x = np.asarray(x, dtype=float)
        k = np.asarray(k, dtype=float)
        n = np.asarray(n, dtype=float)
        eps = 1e-12

        def neg_log_likelihood(params):
            p_model = self.model(x, *params)
            p_model = np.clip(p_model, eps, 1.0 - eps)
            ll = k * np.log(p_model) + (n - k) * np.log(1.0 - p_model)
            return -np.sum(ll)

        if bounds is None:
            bounds = [(None, None)] * len(p0)

        try:
            result = minimize(neg_log_likelihood, p0, method="L-BFGS-B",
                              bounds=bounds)
            # Error from Hessian inverse
            try:
                hess_inv = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') \
                           else result.hess_inv
                perr = np.sqrt(np.abs(np.diag(hess_inv)))
            except Exception:
                perr = np.full(len(p0), np.inf)

            return {"params": result.x, "errors": perr, "nll": result.fun,
                    "success": result.success}
        except Exception:
            return {"params": np.array(p0), "errors": np.full(len(p0), np.inf),
                    "nll": np.inf, "success": False}

    def fit_hybrid(self, x, y, yerr=None, k=None, n=None,
                   p0=None, lsq_bounds=None, mle_bounds=None):
        """Full hybrid pipeline: LSQ → MLE.

        If k and n are provided, MLE uses binomial NLL.
        Otherwise, returns LSQ result only.

        Returns dict with 'params', 'errors', 'method', 'r_squared', 'success'.
        """
        # Stage 1: LSQ
        lsq = self.fit_lsq(x, y, yerr=yerr, p0=p0, bounds=lsq_bounds)

        if k is None or n is None:
            y_fit = self.model(x, *lsq["params"])
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            lsq["method"] = "LSQ"
            lsq["r_squared"] = r2
            return lsq

        # Stage 2: MLE refinement
        mle = self.fit_mle_binomial(x, k, n, p0=lsq["params"], bounds=mle_bounds)

        if mle["success"]:
            y_fit = self.model(x, *mle["params"])
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mle["method"] = "Hybrid MLE"
            mle["r_squared"] = r2
            return mle
        else:
            y_fit = self.model(x, *lsq["params"])
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            lsq["method"] = "LSQ (MLE failed)"
            lsq["r_squared"] = r2
            return lsq

    def evaluate(self, x, params):
        """Evaluate the model at given x with given parameters."""
        return self.model(np.asarray(x), *params)

    def param_dict(self, result):
        """Convert result dict to named parameter dict."""
        return {name: {"value": result["params"][i],
                       "error": result["errors"][i]}
                for i, name in enumerate(self.param_names)}


def binomial_error(p, n):
    """Standard binomial error: sqrt(p(1-p)/n)."""
    p = np.clip(p, 0, 1)
    n = np.maximum(n, 1)
    return np.sqrt(p * (1.0 - p) / n)
