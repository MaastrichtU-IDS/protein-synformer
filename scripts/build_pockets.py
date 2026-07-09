"""Build binding-pocket files for every train/val/test protein with a drug-like holo.
For each: pick one single-drug-like-ligand holo PDB (fetch_pdb_ids), fetch it, extract the
pocket (synformer.data.pocket), save data/pockets/<target_id>.npz. Idempotent; logs coverage."""
from __future__ import annotations

import json
import os
import tempfile

import click
import numpy as np
import pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb_io

from synformer.data.pocket import pocket_residues
from scripts.powered_targets import is_single_druglike_ligand, _ligand_atom_counts_from_structure
from scripts.helpers.pdb import fetch_pdb_ids


def target_ids_needing_pockets(pairs_csvs, pool_json) -> list:
    pool = set(json.load(open(pool_json)))
    ids = set()
    for f in pairs_csvs:
        d = pd.read_csv(f)
        ids |= set(d["target_id"].unique())
    return sorted(t for t in ids if t.split("_")[0] in pool)


def _counts_for_pdb(pdb_id: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        fetched = rcsb.fetch(pdb_id.upper(), "pdb", target_path=tmp)
        arr = pdb_io.get_structure(pdb_io.PDBFile.read(fetched), model=1)
    return _ligand_atom_counts_from_structure(arr)


def select_holo_pdb(accession: str, max_pdbs: int = 12):
    """First PDB (from the accession's cross-refs) with exactly one drug-like ligand → (pdb, resname)."""
    _pdb, df = fetch_pdb_ids(accession)
    if df is None or df.empty:
        return None
    for pdb_id in list(df["PDB_ID"])[:max_pdbs]:
        try:
            counts = _counts_for_pdb(pdb_id)
        except Exception:
            continue
        lig = is_single_druglike_ligand(counts)
        if lig:
            return (pdb_id, lig)
    return None


def build_one(target_id: str, out_dir: str, radius: float = 10.0) -> str:
    out = os.path.join(out_dir, f"{target_id}.npz")
    if os.path.exists(out):
        return "skip-exists"
    sel = select_holo_pdb(target_id.split("_")[0])
    if not sel:
        return "no-clean-pdb"
    pdb_id, lig = sel
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fetched = rcsb.fetch(pdb_id.upper(), "pdb", target_path=tmp)
            arr = pdb_io.get_structure(pdb_io.PDBFile.read(fetched), model=1)
        p = pocket_residues(arr, lig, radius=radius)
        np.savez(out, pdb_id=pdb_id, ligand=lig, **p)
        return f"built ({pdb_id}/{lig}, {len(p['restype'])} res)"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


@click.command()
@click.option("--pairs", default="data/protein_molecule_pairs/sp2_train.csv,"
              "data/protein_molecule_pairs/sp2_val.csv,data/protein_molecule_pairs/sp2_test.csv")
@click.option("--pool", default="data/dock/druglike_holo_accs.json")
@click.option("--out-dir", default="data/pockets")
@click.option("--radius", default=10.0, type=float)
@click.option("--shard", default=None, help="'i/n' — build only targets where index%%n==i (parallel).")
def main(pairs, pool, out_dir, radius, shard):
    os.makedirs(out_dir, exist_ok=True)
    tids = target_ids_needing_pockets(pairs.split(","), pool)
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        tids = [t for k, t in enumerate(tids) if k % n == i]
    rows = []
    for j, tid in enumerate(tids, 1):
        status = build_one(tid, out_dir, radius)
        rows.append({"target_id": tid, "status": status})
        if j % 25 == 0 or "built" in status:
            print(f"[{j}/{len(tids)}] {tid}: {status}", flush=True)
    cov = os.path.join(out_dir, f"coverage{('_'+shard.replace('/','of')) if shard else ''}.csv")
    pd.DataFrame(rows).to_csv(cov, index=False)
    built = sum(1 for r in rows if r["status"].startswith("built") or r["status"] == "skip-exists")
    print(f"coverage: {built}/{len(tids)} pockets present; wrote {cov}", flush=True)


if __name__ == "__main__":
    main()
