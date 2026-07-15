"""Tier: contrastive paralog-discrimination — data assembly.

Build within-family (drug, binder-isoform, non-binder-isoform) triples from routed∩KIBA drugs, split into
TRAIN families (MAPK/CDK/PRKC) and a HELD-OUT family (CSNK1). Pure `binder_label` / `make_within_family_triples`
are unit-tested; `main` writes data/dock/contrastive/{train_triples,heldout_triples,gene2tid}.json.

    .venv/bin/python -m scripts.contrastive_data
"""
import itertools
import json
import os
import re
from pathlib import Path

BIND, NON = 12.1, 11.3
TRAIN_FAMS = set(os.environ.get("TRAIN_FAMS", "MAPK,CDK,PRKC").split(","))
HELDOUT_FAM = os.environ.get("HELDOUT_FAM", "CSNK1")
OUT_DIR = os.environ.get("CONTRASTIVE_DIR", "data/dock/contrastive")
FAM_PATS = {"CDK": r"^CDK\d+$", "JAK": r"^(JAK[123]|TYK2)$", "FGFR": r"^FGFR\d$",
            "ERBB": r"^(EGFR|ERBB\d)$", "MAPK": r"^MAPK\d+$", "AKT": r"^AKT\d$",
            "GSK3": r"^GSK3[AB]$", "PIM": r"^PIM\d$", "CSNK1": r"^CSNK1",
            "PRKC": r"^PRKC[ABDEGHQZ]$", "NEK": r"^NEK\d$", "AURK": r"^AURK[ABC]$"}


def binder_label(kiba, bind=BIND, non=NON):
    if kiba >= bind:
        return "bind"
    if kiba <= non:
        return "non"
    return None


def fam_of(gene):
    for name, p in FAM_PATS.items():
        if re.match(p, str(gene)):
            return name
    return None


def make_within_family_triples(rows, gene2fam, train_fams):
    """rows: list of {smiles, gene, kiba}. Returns (smiles, binder_gene, nonbinder_gene, fam) for
    within-family binder×non-binder gene pairs, restricted to families in `train_fams`."""
    by_mol_fam = {}
    for r in rows:
        fam = gene2fam.get(r["gene"])
        if fam is None:
            continue
        lab = binder_label(r["kiba"])
        if lab is None:
            continue
        by_mol_fam.setdefault((r["smiles"], fam), {"bind": [], "non": []})[lab].append(r["gene"])
    out = []
    for (smi, fam), d in by_mol_fam.items():
        if fam not in train_fams:
            continue
        for bg, ng in itertools.product(d["bind"], d["non"]):
            out.append((smi, bg, ng, fam))
    return out


def _heldout_triples(rows, gene2fam, fam):
    out = []
    by_mol = {}
    for r in rows:
        if gene2fam.get(r["gene"]) != fam:
            continue
        lab = binder_label(r["kiba"])
        if lab:
            by_mol.setdefault(r["smiles"], {"bind": [], "non": []})[lab].append(r["gene"])
    for smi, d in by_mol.items():
        for bg, ng in itertools.product(d["bind"], d["non"]):
            out.append((smi, bg, ng, fam))
    return out


def main():
    import glob
    import torch
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None

    a2g = json.load(open("data/dock/davis/kiba_acc2gene.json"))
    gene2acc = {g: a for a, g in a2g.items()}
    routed = {canon(s) for s in torch.load("data/synthetic_pathways/filtered_pathways_370000.pth",
                                           map_location="cpu")}
    routed.discard(None)
    have_pockets = {f.split("/")[-1].replace("_WT.npz", "") for f in glob.glob("data/pockets/*_WT.npz")}

    df = pd.read_csv("data/dock/davis/kiba_routed.csv")
    df["gene"] = df.Target_ID.map(a2g)
    gene2fam = {g: fam_of(g) for g in df.gene.dropna().unique()}
    # keep rows whose canonical SMILES is routed AND whose gene has a pocket
    rows = []
    for _, r in df.iterrows():
        acc = gene2acc.get(r["gene"])
        c = canon(r["Drug"])
        if c in routed and acc in have_pockets:
            rows.append({"smiles": c, "gene": r["gene"], "kiba": float(r["Y"])})

    train = make_within_family_triples(rows, gene2fam, TRAIN_FAMS)
    held = _heldout_triples(rows, gene2fam, HELDOUT_FAM)
    gene2tid = {g: f"{gene2acc[g]}_WT" for g in {t[1] for t in train + held} | {t[2] for t in train + held}
                if g in gene2acc}
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(out / "train_triples.json", "w"))
    json.dump(held, open(out / "heldout_triples.json", "w"))
    json.dump(gene2tid, open(out / "gene2tid.json", "w"), indent=1)
    print(f"rows(routed∩pocket)={len(rows)} | train triples={len(train)} "
          f"(mols {len({t[0] for t in train})}) | heldout {HELDOUT_FAM} triples={len(held)} "
          f"(mols {len({t[0] for t in held})})", flush=True)


if __name__ == "__main__":
    main()
