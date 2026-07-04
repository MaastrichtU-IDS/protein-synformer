"""Generation-quality metrics for protein-conditioned molecule generation.

Operates on the `infos` structure saved by `scripts/evaluate.py`:
    infos: dict[target_id, dict[gen_idx, {"smiles", "analog", "stack", "ll", "cnt_rxn"}]]
All stored generations are chemically valid by construction (the decoder's stack
built successfully), so validity is measured against the number of attempts.
"""
from collections.abc import Iterable, Sequence

import numpy as np
from rdkit import DataStructs

from synformer.chem.mol import FingerprintOption, Molecule


def _canon(smiles: str) -> str | None:
    m = Molecule(smiles)
    return m.csmiles if m.is_valid else None


def flatten_smiles(infos: dict) -> list[str]:
    return [pred["smiles"] for info in infos.values() for pred in info.values()]


def validity_rate(infos: dict, repeat: int) -> float:
    """Mean fraction of attempts that produced a valid, buildable molecule."""
    return float(np.mean([len(info) / repeat for info in infos.values()]))


def uniqueness(smiles_list: Sequence[str]) -> float:
    canon = [c for c in map(_canon, smiles_list) if c is not None]
    return len(set(canon)) / len(canon) if canon else 0.0


def novelty(smiles_list: Sequence[str], reference_smiles: Iterable[str]) -> float:
    """Fraction of generated molecules not present in the reference (train/known) set."""
    ref = {c for c in map(_canon, reference_smiles) if c is not None}
    canon = [c for c in map(_canon, smiles_list) if c is not None]
    return sum(c not in ref for c in canon) / len(canon) if canon else 0.0


def internal_diversity(mols: Sequence[Molecule], fp_option: FingerprintOption | None = None) -> float:
    """1 - mean pairwise Tanimoto over a set of molecules."""
    fp_option = fp_option or FingerprintOption.morgan_for_tanimoto_similarity()
    fps = [m.get_fingerprint(fp_option, as_bitvec=True) for m in mols if m.is_valid]
    if len(fps) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(sims)
        count += len(sims)
    return 1.0 - (total / count)


def per_target_internal_diversity(infos: dict) -> float:
    """Mean internal diversity of the molecules generated for each protein."""
    vals = []
    for info in infos.values():
        mols = [Molecule(p["smiles"]) for p in info.values()]
        if sum(m.is_valid for m in mols) >= 2:
            vals.append(internal_diversity(mols))
    return float(np.mean(vals)) if vals else 0.0


def scaffold_diversity(mols: Sequence[Molecule]) -> float:
    """Unique Bemis-Murcko scaffolds / number of valid molecules."""
    valid = [m for m in mols if m.is_valid]
    scaffolds = {m.scaffold.csmiles for m in valid}
    return len(scaffolds) / len(valid) if valid else 0.0
