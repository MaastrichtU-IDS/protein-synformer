# SP-ORACLE (Stage A): killed by the applicability-domain pre-check

**Date:** 2026-07-14 · Branch `powered-specificity` · Spec:
`docs/superpowers/specs/2026-07-14-selectivity-oracle-design.md` · Check: `scripts/oracle_domain_check.py`

## What we were going to build

A ChEMBL-trained structure→affinity **selectivity oracle** to use as a reward for a grounded generator
retry (Stage B), gated on it beating docking (Tier-2 ρ 0.245) at predicting held-out measured selectivity.

## The pre-check that stopped it (advisor)

Before building anything, we asked the decisive question: **is a ChEMBL-trained oracle even applicable to
the molecules the generator produces?** Per generated molecule (SP-C pocket candidates), max Morgan-Tanimoto
to that target's ChEMBL training compounds:

| target | median max-Tanimoto (generated → nearest ChEMBL) | frac < 0.3 | frac < 0.4 |
|---|---|---|---|
| KIT | 0.26 | 75% | 97% |
| JAK3 | 0.27 | 75% | 97% |
| CDK5 | 0.30 | 52% | 88% |
| 5-HT1A | 0.27 | 67% | 95% |
| 5-HT2A | 0.28 | 56% | 88% |
| A1R | 0.27 | 71% | 95% |

**The generator's molecules sit at median Tanimoto ~0.27 to their nearest ChEMBL neighbor — 52–75% below
0.3, ~90% below 0.4.** They live almost entirely **outside** the chemical region the oracle would be
trained on.

## Verdict — the oracle-as-reward path is not viable with existing data

- A QSAR oracle would be **extrapolating on essentially every molecule** it must score in Stage B.
- Worse, it **cannot be validated in that regime**: there is no measured selectivity data for the
  generator's chemotypes (held-out ChEMBL compounds are still ChEMBL med-chem, far more similar to training
  than the generator's outputs). The scaffold-split gate we designed would have reported an
  **in-distribution** number — likely a pass — that says nothing about the out-of-distribution use Stage B
  requires. Building `oracle_train.py` would have produced a misleading green light.
- So we **did not build the oracle.** The pre-check (10 minutes, existing data) is the answer.

## The unifying insight

This is the **same wall** as the five generator-side nulls, seen from the other direction:

> Target selectivity is a property of **labeled data in the relevant chemical region**, and that data does
> not exist for the synthesizable, novel chemotypes the generator explores. You cannot *condition* a
> generator into selectivity its training never contained (SP2/SP-C/SP-L/SP-F/SP-DPO), and you cannot
> *reward* it with a learned oracle whose training data doesn't cover its outputs (SP-ORACLE). Both fail
> for the one reason: **no selectivity signal exists in the generator's distribution.**

Docking-selection works at all (weakly, kinase-only — Tier-2 ρ 0.245) precisely *because* it is
physics-based and needs no in-distribution training data — but it is weak and class-limited.

## What would actually move this (each needs new data or a distribution change)

1. **Measured selectivity on generator-like molecules** — synthesize + assay a batch of the generator's
   REAL-space outputs across a target panel (wet-lab). This is the only way to get labels in the right
   region; it would ground both an oracle and a DPO retry.
2. **Constrain the generator to a drug-like / ChEMBL-covered region** so the oracle applies — but this
   fights SynFormer's synthesis-first design and collapses the diversity that motivated it (and just
   retrieves near-known chemotypes).
3. **Physics that needs no in-distribution labels** — better docking/FEP or Boltz-2 affinity as the
   scorer, and **allosteric-pocket** targeting for stronger paralog selectivity. This is the only
   compute-only path left, and it is where the remaining signal (kinase docking) lives.

## Reproduce

```
.venv/bin/python -m scripts.oracle_domain_check
```
