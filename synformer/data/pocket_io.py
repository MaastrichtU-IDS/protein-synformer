"""Load SP-A pocket .npz files into an in-memory dict for the datamodule."""
from __future__ import annotations

import glob
import os

import numpy as np


def load_pockets(pocket_dir: str, min_residues: int = 8, max_residues: int = 128) -> dict:
    """target_id -> {ca:(n,3), cb:(n,3), restype:(n,)} float32/int64, n<=max_residues.
    Pockets with < min_residues are dropped; > max_residues keep the residues closest to
    the ligand-pocket centroid (proxy: closest to the pocket's own CA centroid)."""
    out = {}
    for f in glob.glob(os.path.join(pocket_dir, "*.npz")):
        tid = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        ca, cb, rt = d["ca"].astype(np.float32), d["cb"].astype(np.float32), d["restype"].astype(np.int64)
        n = len(rt)
        if n < min_residues:
            continue
        if n > max_residues:
            centroid = ca.mean(0)
            keep = np.argsort(((ca - centroid) ** 2).sum(1))[:max_residues]
            keep.sort()
            ca, cb, rt = ca[keep], cb[keep], rt[keep]
        out[tid] = {"ca": ca, "cb": cb, "restype": rt}
    return out
