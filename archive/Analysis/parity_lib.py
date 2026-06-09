"""Shared helpers for Bell-state tomography parity analysis."""

from __future__ import annotations

import os
import re
import glob
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit


# ================= STYLE =================

def apply_pub_style() -> None:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['xtick.major.width'] = 1.2
    mpl.rcParams['ytick.major.width'] = 1.2
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['legend.fontsize'] = 14


# ================= AXIS / TEXT LABELS =================

LBL_THETA = r"Analysis angle $\theta$ (deg)"
LBL_PROB = r"Probability"


# ================= MODEL + FIT =================

def cos4_physical(x_deg, C, V, phase_deg, f):
    return C * (1 + V * np.cos(np.deg2rad(f * x_deg - phase_deg)))


def get_fit_point_error(theta, popt, pcov):
    C, V, phi, f = popt
    deg2rad = np.pi / 180.0
    alpha = deg2rad * (f * theta - phi)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    J = np.array([
        1 + V * cos_a,
        C * cos_a,
        C * V * sin_a * deg2rad,
        -C * V * sin_a * theta * deg2rad,
    ])
    var_y = np.dot(J, np.dot(pcov, J))
    return np.sqrt(max(0, var_y))


def fit_cos4(x, y, yerr, fixed_freq=4.0):
    C0 = np.mean(y)
    if C0 <= 1e-9:
        C0 = 0.01
    amp_est = (np.max(y) - np.min(y)) / 2
    V0 = np.clip(amp_est / C0, 0.05, 0.95)

    best_p0 = [C0, V0, 0, fixed_freq if fixed_freq else 4.0]
    best_cost = np.inf
    freq_range = [fixed_freq] if fixed_freq else np.linspace(3.0, 5.0, 5)
    for ph in range(0, 360, 45):
        for ff in freq_range:
            p0_try = [C0, V0, ph, ff]
            model_val = cos4_physical(x, *p0_try)
            cost = np.sum(((model_val - y) / np.maximum(yerr, 1e-9)) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_p0 = p0_try

    f_tol = 1e-7
    if fixed_freq:
        lower_b = [0.0, 0.0, -360, fixed_freq - f_tol]
        upper_b = [np.inf, 1.0, 360, fixed_freq + f_tol]
    else:
        lower_b = [0.0, 0.0, -360, 3.0]
        upper_b = [np.inf, 1.0, 360, 10.0]

    try:
        popt, pcov = curve_fit(
            cos4_physical, x, y, p0=best_p0, sigma=yerr, bounds=(lower_b, upper_b)
        )
        V_err = np.sqrt(pcov[1, 1])
        info = {
            'vis': popt[1],
            'vis_err': V_err,
            'popt': popt,
            'pcov': pcov,
            'y_smooth': cos4_physical(np.linspace(x.min(), x.max(), 500), *popt),
        }
        return popt, info
    except RuntimeError:
        fallback_y = cos4_physical(np.linspace(x.min(), x.max(), 500), *best_p0)
        return best_p0, {
            'vis': 0, 'vis_err': 0,
            'popt': best_p0, 'pcov': np.zeros((4, 4)),
            'y_smooth': fallback_y,
        }


# ================= PER-FILE / PER-QWP ANALYSIS =================

# parity_plus  = p_S0_L + p_S1_E  (indices 1 and 2 in [p_S1_L, p_S0_L, p_S1_E, p_S0_E])
# parity_minus = p_S1_L + p_S0_E  (indices 0 and 3)
IDX_PLUS = (1, 2, 0, 3)   # (big1, big2, small1, small2) when parity_plus dominant
IDX_MINUS = (0, 3, 1, 2)  # ditto when parity_minus dominant


def analyze_qwp_slice(d: pd.DataFrame, hist_thres: int = 15) -> dict | None:
    """Run the parity analysis on a single-QWP slice. d already filtered to one QWP."""
    if len(d) == 0:
        return None

    frac_m2 = np.mean((d['SPCM1'] == -2) | (d['SPCM2'] == -2))
    det0 = (d['SPCM1'] != -2) if frac_m2 > 0.01 else (d['SPCM1'] > 0)
    det1 = (d['SPCM2'] != -2) if frac_m2 > 0.01 else (d['SPCM2'] > 0)
    atom = (d['counts'] >= hist_thres)

    single_click = np.logical_xor(det0, det1)
    valid_mask = single_click & np.isfinite(d['HWP']) & np.isfinite(d['counts'])

    d_valid = d[valid_mask]
    det0, det1, atom = det0[valid_mask], det1[valid_mask], atom[valid_mask]
    hwp = d_valid['HWP'].values

    angles = np.sort(np.unique(hwp))
    if len(angles) < 4:
        return None

    keys = ['p_S0_L', 'p_S1_L', 'p_S0_E', 'p_S1_E',
            'e_S0_L', 'e_S1_L', 'e_S0_E', 'e_S1_E',
            'parity_plus', 'parity_minus', 'e_parity_plus', 'e_parity_minus']
    res = {k: [] for k in keys}

    for ang in angles:
        mask = (hwp == ang)
        N = int(np.sum(mask))
        if N == 0:
            continue
        d0_a, d1_a, a_a = det0[mask], det1[mask], atom[mask]
        k_S0_L = int(np.sum(d0_a & (~a_a)))
        k_S1_L = int(np.sum(d1_a & (~a_a)))
        k_S0_E = int(np.sum(d0_a & a_a))
        k_S1_E = int(np.sum(d1_a & a_a))

        def _pe(k, n):
            p = k / n
            e = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
            return p, max(e, 1e-4)

        p_S0_L, e_S0_L = _pe(k_S0_L, N)
        p_S1_L, e_S1_L = _pe(k_S1_L, N)
        p_S0_E, e_S0_E = _pe(k_S0_E, N)
        p_S1_E, e_S1_E = _pe(k_S1_E, N)

        res['p_S0_L'].append(p_S0_L); res['e_S0_L'].append(e_S0_L)
        res['p_S1_L'].append(p_S1_L); res['e_S1_L'].append(e_S1_L)
        res['p_S0_E'].append(p_S0_E); res['e_S0_E'].append(e_S0_E)
        res['p_S1_E'].append(p_S1_E); res['e_S1_E'].append(e_S1_E)

        res['parity_plus'].append(p_S0_L + p_S1_E)
        res['parity_minus'].append(p_S1_L + p_S0_E)
        res['e_parity_plus'].append(np.sqrt(e_S0_L ** 2 + e_S1_E ** 2))
        res['e_parity_minus'].append(np.sqrt(e_S1_L ** 2 + e_S0_E ** 2))

    for k in res:
        res[k] = np.array(res[k])

    fits = {}
    target_keys = ['p_S0_L', 'p_S1_L', 'p_S0_E', 'p_S1_E', 'parity_plus', 'parity_minus']
    err_keys = ['e_S0_L', 'e_S1_L', 'e_S0_E', 'e_S1_E', 'e_parity_plus', 'e_parity_minus']

    theta_smooth = np.linspace(min(angles), max(angles), 500)
    for k, ek in zip(target_keys, err_keys):
        _, info = fit_cos4(angles, res[k], res[ek], fixed_freq=4.0)
        fits[k] = info

    return {'angles': angles, 'theta_smooth': theta_smooth, 'res': res, 'fits': fits}


def analyze_file(file_path: str, target_qwp: float, hist_thres: int = 15) -> dict | None:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    d = df[df['QWP'] == target_qwp].copy()
    return analyze_qwp_slice(d, hist_thres=hist_thres)


# ================= QUADRATURE EXTRACTION =================

def extract_quadratures_fit(theta, fits):
    keys = ['p_S1_L', 'p_S0_L', 'p_S1_E', 'p_S0_E']
    vals = [max(0, cos4_physical(theta, *fits[k]['popt'])) for k in keys]
    errs = [get_fit_point_error(theta, fits[k]['popt'], fits[k]['pcov']) for k in keys]
    return vals, errs


def extract_quadratures_exp(idx, res):
    keys = ['p_S1_L', 'p_S0_L', 'p_S1_E', 'p_S0_E']
    err_keys = ['e_S1_L', 'e_S0_L', 'e_S1_E', 'e_S0_E']
    vals = [res[k][idx] for k in keys]
    errs = [res[ek][idx] for ek in err_keys]
    return vals, errs


def select_theta_opt(data: dict, use_fits: bool = True):
    """Return (theta_opt, idx, plus_dominant). plus_dominant is True if parity_plus > parity_minus at theta_opt."""
    if use_fits:
        plus = data['fits']['parity_plus']['y_smooth']
        minus = data['fits']['parity_minus']['y_smooth']
        thetas = data['theta_smooth']
    else:
        plus = data['res']['parity_plus']
        minus = data['res']['parity_minus']
        thetas = data['angles']
    idx = int(np.argmax(np.abs(plus - minus)))
    return float(thetas[idx]), idx, bool(plus[idx] > minus[idx])


def select_big_small(plus_dominant: bool) -> tuple[tuple, tuple]:
    """Return (big_idx, small_idx) tuples of length 2 each."""
    full = IDX_PLUS if plus_dominant else IDX_MINUS
    return (full[0], full[1]), (full[2], full[3])


# ================= FIDELITY =================

def fidelity_two_basis(z_vals, z_errs, x_vals, x_errs,
                       *, big_idx_z, small_idx_z, big_idx_x, small_idx_x):
    """Blinov lower bound: F >= 0.5*(Zb1+Zb2+Xb1+Xb2 - Xs1-Xs2 - 2*sqrt(Zs1*Zs2))."""
    Z_b1, eZ_b1 = z_vals[big_idx_z[0]], z_errs[big_idx_z[0]]
    Z_b2, eZ_b2 = z_vals[big_idx_z[1]], z_errs[big_idx_z[1]]
    Z_s1, eZ_s1 = z_vals[small_idx_z[0]], z_errs[small_idx_z[0]]
    Z_s2, eZ_s2 = z_vals[small_idx_z[1]], z_errs[small_idx_z[1]]
    X_b1, eX_b1 = x_vals[big_idx_x[0]], x_errs[big_idx_x[0]]
    X_b2, eX_b2 = x_vals[big_idx_x[1]], x_errs[big_idx_x[1]]
    X_s1, eX_s1 = x_vals[small_idx_x[0]], x_errs[small_idx_x[0]]
    X_s2, eX_s2 = x_vals[small_idx_x[1]], x_errs[small_idx_x[1]]

    F = 0.5 * (Z_b1 + Z_b2 + X_b1 + X_b2 - X_s1 - X_s2 - 2 * np.sqrt(Z_s1 * Z_s2))

    if Z_s1 > 0 and Z_s2 > 0:
        dF_dZs1 = -0.5 * np.sqrt(Z_s2 / Z_s1)
        dF_dZs2 = -0.5 * np.sqrt(Z_s1 / Z_s2)
    else:
        dF_dZs1 = dF_dZs2 = 0.0
    var = (
        (0.5 * eZ_b1) ** 2 + (0.5 * eZ_b2) ** 2
        + (0.5 * eX_b1) ** 2 + (0.5 * eX_b2) ** 2
        + (-0.5 * eX_s1) ** 2 + (-0.5 * eX_s2) ** 2
        + (dF_dZs1 * eZ_s1) ** 2 + (dF_dZs2 * eZ_s2) ** 2
    )
    return F, np.sqrt(var)


def fidelity_three_basis(per_basis: dict) -> tuple[float, float]:
    """F = (1/4)(1 + sum_i (B_i1+B_i2 - S_i1-S_i2)).

    per_basis: {label: {'vals':[4], 'errs':[4], 'big_idx':(i,j), 'small_idx':(i,j)}}.
    Linear in all inputs => Jacobian is +/- 1/4.
    """
    contrib = 0.0
    var = 0.0
    for b in per_basis.values():
        v = np.asarray(b['vals'])
        e = np.asarray(b['errs'])
        ib = b['big_idx']
        is_ = b['small_idx']
        contrib += v[ib[0]] + v[ib[1]] - v[is_[0]] - v[is_[1]]
        var += (e[ib[0]] ** 2 + e[ib[1]] ** 2 + e[is_[0]] ** 2 + e[is_[1]] ** 2) / 16.0
    return 0.25 * (1.0 + contrib), float(np.sqrt(var))


def correlator_per_basis(per_basis: dict) -> dict:
    """Return {label: (B1+B2-S1-S2, sigma)} for each basis — useful for diagnosing which basis drags F down."""
    out = {}
    for k, b in per_basis.items():
        v = np.asarray(b['vals']); e = np.asarray(b['errs'])
        ib = b['big_idx']; is_ = b['small_idx']
        c = v[ib[0]] + v[ib[1]] - v[is_[0]] - v[is_[1]]
        s = np.sqrt(e[ib[0]] ** 2 + e[ib[1]] ** 2 + e[is_[0]] ** 2 + e[is_[1]] ** 2)
        out[k] = (float(c), float(s))
    return out


# ================= PLOTTING =================

def plot_parity_panel(ax, angles, theta_smooth, res, fits, panel_lbl, colors,
                      lbl_a, lbl_b):
    c1, c1_fill, c2, c2_fill = colors
    ax.errorbar(angles, res['parity_plus'], yerr=res['e_parity_plus'], fmt='o',
                color=c1, markerfacecolor=c1_fill, markeredgecolor=c1, capsize=3, label=lbl_a)
    ax.plot(theta_smooth, fits['parity_plus']['y_smooth'], '-', color=c1, linewidth=2)
    ax.errorbar(angles, res['parity_minus'], yerr=res['e_parity_minus'], fmt='o',
                color=c2, markerfacecolor=c2_fill, markeredgecolor=c2, capsize=3, label=lbl_b)
    ax.plot(theta_smooth, fits['parity_minus']['y_smooth'], '-', color=c2, linewidth=2)
    ax.set_xlabel(LBL_THETA)
    ax.set_ylabel(LBL_PROB)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(angles.min() - 5, angles.max() + 5)
    ax.legend(loc='upper right', frameon=True, framealpha=0.92,
              edgecolor='0.5', fontsize=14, handlelength=1.4,
              handletextpad=0.4, borderpad=0.4)
    ax.text(-0.15, 1.05, panel_lbl, transform=ax.transAxes, fontsize=20,
            fontweight='bold', va='top')


def plot_bar_panel(ax, vals, errs, theta_opt, basis, panel_lbl, bar_color,
                   lbl_cats, biggest_outer=False):
    if biggest_outer:
        sorted_idx = np.argsort(vals)
        new_order = [sorted_idx[-1], sorted_idx[0], sorted_idx[1], sorted_idx[-2]]
        vals = [vals[i] for i in new_order]
        errs = [errs[i] for i in new_order]
        cats = [lbl_cats[i] for i in new_order]
    else:
        cats = list(lbl_cats)

    colors = [bar_color] * 4
    rects = ax.bar(np.arange(4), vals, yerr=errs, capsize=3, color=colors,
                   alpha=0.9, edgecolor='black', linewidth=1.2)
    max_height = max(v + e for v, e in zip(vals, errs))
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(cats)
    ax.set_ylim(0, max(0.65, max_height + 0.15))
    ax.set_ylabel(LBL_PROB)
    ax.set_title(f"{basis}-Basis at $\\theta \\approx {theta_opt:.1f}^\\circ$",
                 fontsize=18, pad=8)
    ax.text(-0.15, 1.05, panel_lbl, transform=ax.transAxes, fontsize=20,
            fontweight='bold', va='top')
    for r, v, e in zip(rects, vals, errs):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + e + 0.015,
                f"{v:.3f}\n±{e:.3f}", ha='center', va='bottom', fontsize=10)


# Default per-basis palette. The user's notebook can override.
COLORS_PARITY = {
    'Z': ("#123955", "#7aa3c2", "#0111ff", "#8e96ff"),
    'X': ("#f46363", "#f9b0b0", "#ea0000", "#ff8080"),
    'Y': ("#0d4d0d", "#a3d9a3", "#11a611", "#80d980"),
}
COLORS_BAR = {'Z': "#122255", 'X': "#A81818", 'Y': "#0d4d0d"}


def create_figure(per_basis_data: dict,
                  basis_order: tuple = ('Z', 'X', 'Y'),
                  *,
                  fid: float, fid_err: float, fid_mode: str = 'lower',
                  use_fits: bool = True,
                  filename: str = "parity_figure.svg",
                  show_title: bool = True,
                  biggest_outer: bool = True) -> None:
    """Render the parity figure. fid_mode: 'lower' -> 'F >= ...', 'equal' -> 'F = ...'."""
    bases = [b for b in basis_order if b in per_basis_data]
    n = len(bases)
    if n not in (2, 3):
        raise ValueError(f"create_figure expects 2 or 3 bases; got {n}")

    figsize = (11, 8.5) if n == 2 else (11, 12.0)
    fig, axes = plt.subplots(n, 2, figsize=figsize, constrained_layout=True)

    if show_title:
        prefix = "Fitted" if use_fits else "Experimental"
        rel = r"\geq" if fid_mode == 'lower' else "="
        fig.suptitle(
            f"{prefix} Entanglement Fidelity: $\\mathcal{{F}} {rel} {fid:.3f} \\pm {fid_err:.3f}$",
            fontsize=22, fontweight='bold',
        )

    panel_letters = string.ascii_lowercase
    for row, b in enumerate(bases):
        d = per_basis_data[b]
        a = d['analyze']
        plot_parity_panel(
            axes[row, 0], a['angles'], a['theta_smooth'], a['res'], a['fits'],
            f"({panel_letters[2 * row]})", d['colors_parity'],
            d['lbl_parity_a'], d['lbl_parity_b'],
        )
        plot_bar_panel(
            axes[row, 1], d['vals'], d['errs'], d['theta_opt'], b,
            f"({panel_letters[2 * row + 1]})", d['color_bar'],
            d['lbl_cats'], biggest_outer=biggest_outer,
        )

    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.savefig(filename.replace('.svg', '.pdf'), format='pdf', bbox_inches='tight')
    plt.show()


# ================= FILE METADATA =================

_FN_RX = re.compile(
    r'^(?P<basis>[XYZ])_basis_parity_measurement(?:_(?P<opt>Optimized))?'
    r'.*?_(?P<date>\d{8})_(?P<time>\d{4})\.csv$'
)


def parse_filename(path: str) -> dict:
    name = os.path.basename(path)
    m = _FN_RX.match(name)
    if not m:
        return {'filename': name, 'basis': None, 'date': None,
                'time': None, 'optimized': False}
    d = m.group('date'); t = m.group('time')
    return {
        'filename': name,
        'basis': m.group('basis'),
        'date': f"{d[:4]}-{d[4:6]}-{d[6:8]}",
        'time': f"{t[:2]}:{t[2:]}",
        'optimized': bool(m.group('opt')),
    }


def discover_data_files(data_dir: str = "Data") -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*_basis_*.csv")))
