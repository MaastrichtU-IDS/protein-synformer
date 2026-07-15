"""Tier-3: load DAVIS, map kinases to our crystal pockets, emit the docking set + measured table.

    .venv/bin/python -m scripts.davis_prep

Writes data/dock/davis/{dock_set.txt, measured_davis.json, kinase_pockets.json}.
"""
import json
import math
import re
from pathlib import Path

GENE_TID = {  # protein kinases (primary) + PIK3CD/RIOK1 (robustness)
    "KIT": "P10721_WT", "JAK3": "P52333_WT", "FGFR1": "P11362_WT", "CDK5": "Q00535_WT",
    "DYRK1A": "Q13627_WT", "CSNK1A1": "P48729_WT", "CSNK1G1": "Q9HCP0_WT", "CSNK1E": "P49674_WT",
    "PHKG1": "Q16816_WT", "STK16": "O75716_WT", "NEK1": "Q96PY6_WT", "CAMK4": "Q16566_WT",
    "DAPK2": "Q9UIK4_WT", "PIK3CD": "O35904_WT", "RIOK1": "Q9BRS2_WT",
}


def base_gene(target_id: str) -> str:
    return re.split(r"[(\-]", str(target_id))[0].strip().upper()


def kd_to_pkd(kd_nM: float) -> float:
    return 9.0 - math.log10(float(kd_nM))


def main():
    import os
    os.makedirs("data/tdc", exist_ok=True)
    cwd = os.getcwd()
    os.chdir("data/tdc")
    from tdc.multi_pred import DTI
    df = DTI(name="DAVIS").get_data()   # Drug_ID, Drug(SMILES), Target_ID(gene), Y(Kd nM)
    os.chdir(cwd)
    df["gene"] = df.Target_ID.map(base_gene)
    df = df[df.gene.isin(GENE_TID)].copy()
    df["pkd"] = df.Y.map(kd_to_pkd)
    agg = df.groupby(["Drug", "gene"]).pkd.median().reset_index()
    measured = {}
    for _, r in agg.iterrows():
        measured.setdefault(r.Drug, {})[r.gene] = float(r.pkd)
    genes_present = sorted({g for m in measured.values() for g in m})
    out = Path("data/dock/davis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dock_set.txt").write_text("\n".join(measured) + "\n")
    json.dump(measured, open(out / "measured_davis.json", "w"))
    json.dump({g: GENE_TID[g] for g in genes_present}, open(out / "kinase_pockets.json", "w"), indent=1)
    print(f"drugs={len(measured)} genes={len(genes_present)}: {genes_present}", flush=True)


if __name__ == "__main__":
    main()
