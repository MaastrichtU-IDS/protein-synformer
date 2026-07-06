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
    M = np.asarray(score_matrix, dtype=float)
    n = M.shape[0]
    own = np.diag(M)
    offdiag = np.array([np.mean([M[i, j] for j in range(n) if j != i]) for i in range(n)])
    return {
        "own_mean": float(own.mean()),
        "offdiag_mean": float(offdiag.mean()),
        "delta": float((own - offdiag).mean()),          # negative = own docks better
        "win_rate": float((own < offdiag).mean()),        # fraction where own strictly better
    }
