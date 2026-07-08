"""Select ~20 test-split targets with a clean single drug-like holo pocket for the
powered specificity study. Reuses dock_prepare's drug-like ignore-set + heavy-atom rule."""
from __future__ import annotations

import json
import tempfile

import click
import pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

from scripts.dock_prepare import is_drug_like_ligand  # reuse exact criterion

ORIGINAL_5 = [
    {"target_id": "O43570_WT", "pdb_id": "1JD0", "ligand_resname": "AZM"},
    {"target_id": "P06537_WT", "pdb_id": "3MNP", "ligand_resname": "DEX"},
    {"target_id": "P10721_WT", "pdb_id": "1T46", "ligand_resname": "STI"},
    {"target_id": "P02753_WT", "pdb_id": "1BRP", "ligand_resname": "RTL"},
    {"target_id": "P0C559_WT", "pdb_id": "6Y8O", "ligand_resname": "NOV"},
]


def is_single_druglike_ligand(resname_atomcounts: dict) -> str | None:
    """Return the sole drug-like ligand resname, or None if zero or >=2 qualify.
    Delegates the per-resname drug-like test to dock_prepare.is_drug_like_ligand."""
    dl = [r for r, n in resname_atomcounts.items() if is_drug_like_ligand(r, n)]
    return dl[0] if len(dl) == 1 else None


def known_ligand_counts(sp2_test_csv: str) -> dict:
    df = pd.read_csv(sp2_test_csv)
    df["acc"] = df["target_id"].str.split("_").str[0]
    return df.groupby("acc")["SMILES"].nunique().to_dict()


def _pdb_ligand_atom_counts(pdb_id: str) -> dict:
    """Per-residue-name heavy-atom count of the LARGEST physical (chain_id, res_id)
    copy of each HETATM residue — matches dock_prepare's find_drug_like_ligands
    convention so a partial/alternate copy doesn't wrongly disqualify a real ligand."""
    with tempfile.TemporaryDirectory() as tmp:
        fetched = rcsb.fetch(pdb_id.upper(), "pdb", target_path=tmp)
        arr = pdb_io.get_structure(pdb_io.PDBFile.read(fetched), model=1)
    return _ligand_atom_counts_from_structure(arr)


def _ligand_atom_counts_from_structure(arr: struc.AtomArray) -> dict:
    """Pure, network-free half of _pdb_ligand_atom_counts: given a full-model
    AtomArray, drop amino-acid residues and hydrogens, then count heavy atoms
    of the largest (chain_id, res_id) copy per HETATM resname."""
    het = arr[~struc.filter_amino_acids(arr) & (arr.element != "H")]
    return _atom_counts_by_largest_copy(het.res_name, het.res_id, het.chain_id)


def _atom_counts_by_largest_copy(res_names, res_ids, chain_ids) -> dict:
    """Given parallel per-atom arrays, return {resname: max_copy_atom_count}
    where a "copy" is a unique (chain_id, res_id) group of that resname."""
    from collections import defaultdict

    copy_counts: dict[tuple, int] = defaultdict(int)
    for rn, ri, ci in zip(res_names, res_ids, chain_ids):
        copy_counts[(str(rn), int(ri), str(ci))] += 1

    counts: dict[str, int] = {}
    for (rn, _ri, _ci), n in copy_counts.items():
        if n > counts.get(rn, 0):
            counts[rn] = n
    return counts


@click.command()
@click.option("--pool", default="data/dock/druglike_holo_accs.json")
@click.option("--sp2-test", default="data/protein_molecule_pairs/sp2_test.csv")
@click.option("--out", default="data/dock/powered_targets.json")
@click.option("--n-target", default=20, type=int)
@click.option("--over-select", default=24, type=int)
@click.option("--min-known", default=10, type=int)
def main(pool, sp2_test, out, n_target, over_select, min_known):
    with open(pool) as f:
        pool_accs = set(json.load(f))
    test = pd.read_csv(sp2_test)
    test_accs = {t.split("_")[0] for t in test.target_id.unique()}
    kn = known_ligand_counts(sp2_test)
    orig_accs = {t["target_id"].split("_")[0] for t in ORIGINAL_5}

    # candidate accessions: in pool AND test-split, prefer >=min_known, sorted by known-depth desc
    cands = sorted(pool_accs & test_accs, key=lambda a: -kn.get(a, 0))
    chosen = list(ORIGINAL_5)
    chosen_accs = set(orig_accs)
    for acc in cands:
        if len(chosen) >= over_select:
            break
        if acc in chosen_accs or kn.get(acc, 0) < min_known:
            continue
        tid = f"{acc}_WT"
        # find a PDB with exactly one drug-like ligand (RCSB xref via UniProt)
        from scripts.helpers.pdb import fetch_pdb_ids
        pdb_id, df_pdb = fetch_pdb_ids(acc)
        if df_pdb.empty:
            continue
        picked = None
        for cand_pdb in list(df_pdb["PDB_ID"])[:12]:
            try:
                lig = is_single_druglike_ligand(_pdb_ligand_atom_counts(cand_pdb))
            except Exception:
                continue
            if lig:
                picked = (cand_pdb, lig); break
        if picked:
            chosen.append({"target_id": tid, "pdb_id": picked[0], "ligand_resname": picked[1]})
            chosen_accs.add(acc)
            print(f"  + {tid}: {picked[0]} {picked[1]} (known={kn.get(acc,0)})", flush=True)

    with open(out, "w") as f:
        json.dump(chosen, f, indent=2)
    print(f"chosen {len(chosen)} targets (incl. original 5); wrote {out}")
    print("NOTE: hand-trim to ~%d family-diverse if more than needed." % n_target)


if __name__ == "__main__":
    main()
