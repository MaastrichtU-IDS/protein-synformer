import numpy as np


def box_from_coords(coords, padding=4.0):
    lo, hi = coords.min(0), coords.max(0)
    center = tuple(float(x) for x in (lo + hi) / 2)
    size = tuple(float(x) for x in (hi - lo) + 2 * padding)
    return center, size


def select_topm(scores, m):
    order = sorted(range(len(scores)), key=lambda i: scores[i])  # ascending = best first
    return order[:m]


def mismatch_summary(score_matrix):
    """Summarise a target-specificity score matrix.

    Parameters
    ----------
    score_matrix:
        N×N array where M[i,j] is the best docking score of target i's
        top-M candidates in pocket j.  May contain NaN (embed/dock failures).

    Returns
    -------
    dict with keys ``own_mean``, ``offdiag_mean``, ``delta``, ``win_rate``.
    All computed NaN-robustly; if no finite values exist the field is NaN.
    ``win_rate`` and ``delta`` count only targets for which BOTH the own
    (diagonal) score AND the off-diagonal mean are finite so that NaN cells
    never silently distort the fraction.
    """
    M = np.asarray(score_matrix, dtype=float)
    n = M.shape[0]
    own = np.diag(M)

    # Per-row off-diagonal mean (NaN-safe: all-NaN row → NaN, not 0 or crash).
    offdiag_rows = []
    for i in range(n):
        row_offdiag = np.array([M[i, j] for j in range(n) if j != i])
        offdiag_rows.append(float(np.nanmean(row_offdiag)) if np.any(np.isfinite(row_offdiag)) else float("nan"))
    offdiag = np.array(offdiag_rows)

    # overall means use nanmean so a single NaN cell doesn't poison the aggregate.
    own_mean = float(np.nanmean(own)) if np.any(np.isfinite(own)) else float("nan")
    offdiag_mean = float(np.nanmean(offdiag)) if np.any(np.isfinite(offdiag)) else float("nan")

    # win_rate and delta: only over targets where BOTH own and offdiag are finite.
    both_finite = ~np.isnan(own) & ~np.isnan(offdiag)
    if not np.any(both_finite):
        win_rate = float("nan")
        delta = float("nan")
    else:
        own_f = own[both_finite]
        offdiag_f = offdiag[both_finite]
        win_rate = float(np.mean(own_f < offdiag_f))
        delta = float(np.mean(own_f - offdiag_f))

    return {
        "own_mean": own_mean,
        "offdiag_mean": offdiag_mean,
        "delta": delta,
        "win_rate": win_rate,
    }
