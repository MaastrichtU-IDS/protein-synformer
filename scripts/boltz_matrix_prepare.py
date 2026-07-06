"""Assemble Boltz-2 mismatch-matrix inputs: top-1 docking hit SMILES per target
+ each target's holo-construct protein sequence (longest AA chain)."""
from __future__ import annotations

import json
import tempfile

import pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
from biotite.sequence import ProteinSequence


def one_letter_from_residues(res_ids, res_names) -> str:
    """Map (res_ids, res_names) to a one-letter sequence ordered by res_id.
    Unknown 3-letter codes are skipped."""
    order = sorted(range(len(res_ids)), key=lambda i: res_ids[i])
    out = []
    for i in order:
        try:
            letter = ProteinSequence.convert_letter_3to1(res_names[i])
        except KeyError:
            continue
        if letter == "X":
            continue
        out.append(letter)
    return "".join(out)


def pdb_to_sequence(pdb_id: str) -> str:
    """One-letter sequence of the longest amino-acid chain of a holo PDB."""
    with tempfile.TemporaryDirectory() as tmp:
        fetched = rcsb.fetch(pdb_id.upper(), "pdb", target_path=tmp)
        arr = pdb_io.get_structure(pdb_io.PDBFile.read(fetched), model=1)
    prot = arr[struc.filter_amino_acids(arr)]
    best_seq = ""
    for chain_id in set(prot.chain_id):
        chain = prot[prot.chain_id == chain_id]
        res_ids, res_names = struc.get_residues(chain)
        seq = one_letter_from_residues(list(res_ids), list(res_names))
        if len(seq) > len(best_seq):
            best_seq = seq
    if not best_seq:
        raise ValueError(f"No amino-acid sequence extracted from {pdb_id}")
    return best_seq


def top_hits(scores_csv: str, target_ids, k: int = 1) -> dict:
    df = pd.read_csv(scores_csv)
    own = df[(df.target == df.pocket) & (df.source == "candidate")]
    out = {}
    for t in target_ids:
        sub = own[own.target == t].nsmallest(k, "score")
        out[t] = list(sub.molecule)
    return out


def build_matrix_inputs(targets_json: str, scores_csv: str, out_json: str, k: int = 1) -> dict:
    targets = json.load(open(targets_json))
    target_ids = [t["target_id"] for t in targets]
    hits_by_target = top_hits(scores_csv, target_ids, k=k)
    pdb_of = {t["target_id"]: t["pdb_id"] for t in targets}
    hits = []
    for t in target_ids:
        for smi in hits_by_target[t]:
            hits.append({"target_id": t, "smiles": smi})
    proteins = [{"target_id": t, "sequence": pdb_to_sequence(pdb_of[t])} for t in target_ids]
    d = {"hits": hits, "proteins": proteins}
    json.dump(d, open(out_json, "w"), indent=2)
    return d


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--targets", default="data/dock/targets.json")
    @click.option("--scores", default="data/dock/dock_scores.csv")
    @click.option("--out", default="data/boltz/matrix_inputs.json")
    @click.option("--top-k", default=1, type=int)
    def main(targets, scores, out, top_k):
        import os
        os.makedirs(os.path.dirname(out), exist_ok=True)
        d = build_matrix_inputs(targets, scores, out, k=top_k)
        for p in d["proteins"]:
            print(f"{p['target_id']}: seq_len={len(p['sequence'])}")
        print(f"{len(d['hits'])} hits, wrote {out}")

    main()
