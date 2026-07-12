"""Tier-1 calibration prep: build three molecule classes per target for the specificity-instrument
calibration — known actives, PROPERTY-MATCHED decoys, and the model's generated candidates — each as a
<tid>.txt SMILES file the docking driver reads.

Property-matched decoys (advisor fix): docking score tracks MW/logP/HBD, so decoys are matched to each
target's actives on those descriptors (and kept topologically dissimilar, Tanimoto < 0.35) so a Tier-1
"actives own-prefer, decoys don't" result cannot be explained by physchem alone.

    .venv/bin/python -m scripts.tier1_prep
"""
import json
import random
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

TARGETS = ["P10721_WT", "P52333_WT", "Q00535_WT",   # kinases (KIT/JAK3/CDK5)
           "P08908_WT", "P28223_WT", "P30542_WT",   # GPCRs (5HT1A/5HT2A/A1R)
           "O43570_WT", "P51151_WT"]                 # CA12 (lyase), RAB9A (GTPase)
N = 25
SEED = 42


def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return m, (Descriptors.MolWt(m), Descriptors.MolLogP(m), Lipinski.NumHDonors(m))


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def main():
    rng = random.Random(SEED)
    test = pd.read_csv("data/protein_molecule_pairs/sp2_test.csv")
    # drug-like decoy pool with descriptors
    pool = []
    for f in ["data/chembl_filtered_1k.txt", "data/enamine_smiles_1k.txt"]:
        for line in open(f):
            s = line.strip()
            if not s:
                continue
            p = props(s)
            if p:
                pool.append((s, p[0], p[1]))  # smi, mol, (mw,logp,hbd)
    print(f"decoy pool: {len(pool)} valid drug-like molecules", flush=True)

    base = Path("data/dock/tier1")
    for cls in ["actives", "decoys", "candidates"]:
        (base / cls).mkdir(parents=True, exist_ok=True)

    for tid in TARGETS:
        # actives
        acts = test[test.target_id == tid].SMILES.dropna().unique().tolist()
        rng.shuffle(acts)
        act_ok = []
        act_props, act_fps = [], []
        for s in acts:
            p = props(s)
            if p and len(act_ok) < N:
                act_ok.append(s); act_props.append(p[1]); act_fps.append(fp(p[0]))
        (base / "actives" / f"{tid}.txt").write_text("\n".join(act_ok) + "\n")

        # property-matched decoys: for each active, find an unused pool mol within MW+-40/logP+-1.0/HBD+-1,
        # Tanimoto<0.35 to every active
        used = set()
        decoys = []
        for (mw, lp, hbd) in act_props:
            cand = None
            order = list(range(len(pool)))
            rng.shuffle(order)
            for i in order:
                s, m, (pmw, plp, phbd) = pool[i]
                if s in used:
                    continue
                if abs(pmw - mw) <= 40 and abs(plp - lp) <= 1.0 and abs(phbd - hbd) <= 1:
                    f = fp(m)
                    if max((DataStructs.TanimotoSimilarity(f, af) for af in act_fps), default=0) < 0.35:
                        cand = s; break
            if cand:
                used.add(cand); decoys.append(cand)
        (base / "decoys" / f"{tid}.txt").write_text("\n".join(decoys) + "\n")

        # candidates: sample from the model's generated pocket pool
        cpool = [l.strip() for l in open(f"data/dock/candidates_pocket/{tid}.txt") if l.strip()]
        rng.shuffle(cpool)
        cands = cpool[:N]
        (base / "candidates" / f"{tid}.txt").write_text("\n".join(cands) + "\n")

        print(f"{tid}: actives={len(act_ok)} decoys={len(decoys)} candidates={len(cands)}", flush=True)

    print("TIER1 PREP DONE", flush=True)


if __name__ == "__main__":
    main()
