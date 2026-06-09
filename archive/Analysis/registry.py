"""Per-(file, QWP) fidelity registry.

A flat CSV that tracks every QWP slice we've analyzed, the achievable fidelity
estimate at that QWP, the date/time/basis from the filename, and a
`is_best_in_file` flag for the winning QWP per file. Append-only via upsert
on (filepath, qwp).
"""

from __future__ import annotations

import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from parity_lib import (
    analyze_qwp_slice,
    cos4_physical,
    get_fit_point_error,
    parse_filename,
    discover_data_files,
)


REGISTRY_PATH = Path("registry.csv")

COLUMNS = [
    'filepath', 'filename', 'basis', 'date', 'time', 'optimized',
    'qwp', 'n_qwps_in_file',
    'max_fid', 'max_fid_err',
    'vis_plus', 'vis_minus', 'theta_opt',
    'is_best_in_file',
    'updated_at', 'notes',
]


# ================= ESTIMATOR =================

def qwp_max_fid_estimate(theta_opt: float, fits: dict) -> tuple[float, float]:
    """`F_max ~ 0.5*(A+D+2 sqrt(A D))` where A,D are the top-2 of the 4
    quadrature values at theta_opt. Same estimator parity_osc.ipynb uses.
    """
    keys = ['p_S0_L', 'p_S1_L', 'p_S0_E', 'p_S1_E']
    vals = np.array([max(0.0, cos4_physical(theta_opt, *fits[k]['popt'])) for k in keys])
    errs = np.array([get_fit_point_error(theta_opt, fits[k]['popt'], fits[k]['pcov']) for k in keys])

    order = np.argsort(vals)[::-1]
    iA, iD = int(order[0]), int(order[1])
    A, D = float(vals[iA]), float(vals[iD])
    sA, sD = float(errs[iA]), float(errs[iD])

    F = 0.5 * (A + D + 2.0 * np.sqrt(max(A * D, 0.0)))
    if A > 0 and D > 0:
        dF_dA = 0.5 * (1.0 + np.sqrt(D / A))
        dF_dD = 0.5 * (1.0 + np.sqrt(A / D))
    else:
        dF_dA = dF_dD = 0.5
    sF = float(np.sqrt((dF_dA * sA) ** 2 + (dF_dD * sD) ** 2))
    return float(F), sF


# ================= CSV I/O =================

def load_registry(path: Path = REGISTRY_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def upsert_rows(rows: list[dict], path: Path = REGISTRY_PATH) -> pd.DataFrame:
    """Replace any existing rows matching (filepath, qwp) with the new ones, append the rest."""
    new_df = pd.DataFrame(rows, columns=COLUMNS)
    existing = load_registry(path)
    if len(existing):
        keys_new = set(zip(new_df['filepath'], new_df['qwp']))
        mask_keep = ~existing.apply(lambda r: (r['filepath'], r['qwp']) in keys_new, axis=1)
        merged = pd.concat([existing[mask_keep], new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values(['date', 'time', 'basis', 'qwp']).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


# ================= REGISTER ONE FILE =================

def register_file(file_path: str, hist_thres: int = 15, notes: str = "",
                  path: Path = REGISTRY_PATH) -> pd.DataFrame:
    """Analyze every QWP in `file_path`, compute the per-QWP F_max estimator,
    upsert one row per QWP into the registry, mark the best QWP in this file.
    """
    df = pd.read_csv(file_path)
    qwps = np.sort(df['QWP'].unique())
    meta = parse_filename(file_path)
    rel_path = os.path.relpath(file_path)
    now = dt.datetime.now().isoformat(timespec='seconds')

    rows = []
    for q in qwps:
        slice_df = df[df['QWP'] == q].copy()
        result = analyze_qwp_slice(slice_df, hist_thres=hist_thres)
        if result is None:
            continue
        plus = result['fits']['parity_plus']['y_smooth']
        minus = result['fits']['parity_minus']['y_smooth']
        idx = int(np.argmax(np.abs(plus - minus)))
        theta_opt = float(result['theta_smooth'][idx])
        F, sF = qwp_max_fid_estimate(theta_opt, result['fits'])
        rows.append({
            'filepath': rel_path,
            'filename': meta['filename'],
            'basis': meta['basis'],
            'date': meta['date'],
            'time': meta['time'],
            'optimized': meta['optimized'],
            'qwp': float(q),
            'n_qwps_in_file': int(len(qwps)),
            'max_fid': F,
            'max_fid_err': sF,
            'vis_plus': float(result['fits']['parity_plus']['vis']),
            'vis_minus': float(result['fits']['parity_minus']['vis']),
            'theta_opt': theta_opt,
            'is_best_in_file': False,
            'updated_at': now,
            'notes': notes,
        })

    if rows:
        best_idx = int(np.argmax([r['max_fid'] for r in rows]))
        rows[best_idx]['is_best_in_file'] = True

    return upsert_rows(rows, path=path)


# ================= DISCOVERY =================

def new_files_since_registry(data_dir: str = "Data",
                             path: Path = REGISTRY_PATH) -> list[str]:
    known = set(load_registry(path)['filepath'])
    return [p for p in discover_data_files(data_dir) if os.path.relpath(p) not in known]
