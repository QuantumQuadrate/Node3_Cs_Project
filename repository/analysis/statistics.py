"""Statistical utilities for atom experiment data analysis.

Extracted from Microwave_var_scan_analysis_v5_hybrid_MLE.py.
"""

import numpy as np
import pandas as pd


def binomial_error(p, n):
    """Standard binomial error: sqrt(p*(1-p)/n)."""
    p = np.clip(p, 0, 1)
    n = np.maximum(n, 1)
    return np.sqrt(p * (1.0 - p) / n)


def classify_events(shot1, shot2, background=None, threshold=12.0):
    """Classify atom loading and retention events.

    Parameters
    ----------
    shot1 : array-like
        First imaging shot counts.
    shot2 : array-like
        Second imaging shot counts.
    background : array-like or None
        Background counts to subtract. If None, no subtraction.
    threshold : float
        Count threshold for atom detection.

    Returns
    -------
    dict with keys: loaded, retained, success, shot1_net, shot2_net
    """
    shot1 = np.asarray(shot1, dtype=float)
    shot2 = np.asarray(shot2, dtype=float)

    if background is not None:
        bg = np.asarray(background, dtype=float)
        shot1_net = shot1 - bg
        shot2_net = shot2 - bg
    else:
        shot1_net = shot1
        shot2_net = shot2

    loaded = shot1_net > threshold
    retained = shot2_net > threshold
    success = loaded & retained

    return {
        "loaded": loaded,
        "retained": retained,
        "success": success,
        "shot1_net": shot1_net,
        "shot2_net": shot2_net,
    }


def group_by_scan(scan_values, loaded, retained, n_total_per_point=None):
    """Group events by scan parameter and compute statistics.

    Parameters
    ----------
    scan_values : array-like
        Scan parameter value for each shot.
    loaded : array-like of bool
        Whether atom was loaded.
    retained : array-like of bool
        Whether atom was retained.
    n_total_per_point : int or None
        If set, tile scan values to match data length.

    Returns
    -------
    DataFrame with columns: x, N_total, N_loaded, N_retained,
                            Ret_P, Ret_Err, Load_Rate, Load_Err
    """
    scan_values = np.asarray(scan_values)
    loaded = np.asarray(loaded)
    retained = np.asarray(retained)

    # If scan_values is shorter than data, tile it
    if len(scan_values) < len(loaded):
        n_cycles = len(loaded) // len(scan_values)
        scan_tiled = np.tile(scan_values, n_cycles)[:len(loaded)]
    else:
        scan_tiled = scan_values[:len(loaded)]

    unique_x = np.unique(scan_tiled)
    rows = []

    for x_val in unique_x:
        mask = scan_tiled == x_val
        n_total = np.sum(mask)
        n_loaded = np.sum(loaded[mask])
        n_retained = np.sum(retained[mask])

        load_rate = n_loaded / n_total if n_total > 0 else 0.0
        ret_p = n_retained / n_loaded if n_loaded > 0 else 0.0

        rows.append({
            "x": x_val,
            "N_total": n_total,
            "N_loaded": n_loaded,
            "N_retained": n_retained,
            "Ret_P": ret_p,
            "Ret_Err": binomial_error(ret_p, n_loaded),
            "Load_Rate": load_rate,
            "Load_Err": binomial_error(load_rate, n_total),
        })

    return pd.DataFrame(rows)
