"""SP-ORACLE Stage-A pre-check (advisor): is a ChEMBL-trained selectivity oracle even applicable to the
GENERATOR's molecules? Computes, per generated molecule, the max Morgan-Tanimoto to the target's ChEMBL
training compounds. If most generated molecules sit below ~0.3 (dissimilar), a QSAR oracle would be
extrapolating on ~all of them AND cannot be validated there (no labels in that region) -> the
oracle-as-reward path is not viable with existing data, and no oracle need be built.

    .venv/bin/python -m scripts.oracle_domain_check
"""
import json
import random
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
NAME_TID = {"KIT": "P10721_WT", "JAK3": "P52333_WT", "CDK5": "Q00535_WT",
            "5HT1A": "P08908_WT", "5HT2A": "P28223_WT", "A1R": "P30542_WT"}


def fps(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m:
            out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    return out


def main(gen_cap=150, chembl_cap=1500):
    print("target | n_gen | median maxTc | frac<0.3 | frac<0.4")
    for name, tid in NAME_TID.items():
        gen = [l.strip() for l in open(f"data/dock/candidates_pocket/{tid}.txt") if l.strip()]
        ch = json.loads(open(f"data/dock/tier2/raw/{name}.json").read())["compounds"]
        ch_smis = [v["smiles"] for v in ch.values()]
        random.Random(42).shuffle(ch_smis)
        gfp = fps(gen[:gen_cap])
        cfp = fps(ch_smis[:chembl_cap])
        maxtc = np.array([max(DataStructs.BulkTanimotoSimilarity(g, cfp)) for g in gfp])
        print(f"{name:6} | {len(gfp):4d} | {np.median(maxtc):.3f} | "
              f"{(maxtc < 0.3).mean():.2f} | {(maxtc < 0.4).mean():.2f}")


if __name__ == "__main__":
    main()
