# Protein-conditioned SynFormer for target-specific generation — Capstone Synthesis

**Last updated 2026-07-14.** This is the one-document read of the whole project. Detailed evidence lives in
[FINDINGS](FINDINGS.md) and the per-study results docs it links; this is the narrative and the thesis.

---

## The thesis (one paragraph)

We set out to make a synthesis-aware generator produce **target-specific** (selective) molecules by
conditioning on, re-biasing toward, or selecting against a protein target. After ~a dozen sub-studies the
answer is clear and unusually well-evidenced:

> **Target selectivity is a property of labeled data in the relevant chemical region, and that data does
> not exist for the synthesizable, novel chemotypes this generator explores.** You cannot *condition* a
> generator into a selectivity signal its training never contained (five generator-side nulls), you cannot
> *reward* it with a learned oracle whose training data doesn't cover its outputs (oracle path, killed on a
> domain pre-check), and you cannot validate an *allosteric* alternative without measured allosteric-ligand
> selectivity (anchor-walled). The **only** signal reachable is a **weak, physics-based, kinase-specific
> docking-selection signal (ρ ≈ 0.25 vs measured selectivity)** — reachable precisely because physics needs
> no in-distribution labels. Everything else is walled by the same missing data.

The generator is a good **synthesizable-diversity engine**; it makes molecules that dock as well as real
drugs. Targeting was never the axis it could reach.

---

## The arc, and what each step showed

**1. Generation-side conditioning/re-biasing — five nulls.**
Sequence conditioning (SP2), 3D-pocket conditioning (SP-C: paired pocket−sequence Δ −0.10 [−0.49,+0.31],
CI incl. 0), motif-enrichment of a frozen model (SP-L), fragment-seeded hill-climbing (SP-F: a smina
"win" that Boltz refuted as a hacking artifact), and **weight updates via DPO (SP-DPO)** — all failed to
confer targeting. SP-DPO is the sharpest statement: DPO **fit** the per-molecule specificity preference
*in-sample* (train margin 2.97→3.49) but it **did not transfer** — held-out own-preference DPO−base was
+0.08 [−0.22,+0.41], ns. The generator can be trained to prefer specific molecules on seen pockets; it does
not generalize to making raw samples target-specific on unseen ones.

**2. Selection against the 3D pocket — the one positive.**
Docking-selection (own-vs-mismatch normalized delta) gave a modest, *replicated* specificity
(SP-SC: N=41 delta −0.80; independent 21-target replication −0.64 [−1.09,−0.20]). Real and reproducible —
as a *relative same-scorer* smina/shape-fit phenomenon.

**3. Scorers disagree.**
Boltz-2 (competent: known-vs-random AUROC 0.95) did **not** corroborate the docking specificity at the
molecule level (Boltz own-vs-mismatch delta −0.04 vs docking −1.74). Consensus averaging didn't beat Boltz.
Single-proxy *absolute* candidate quality is unreliable; the specificity was method-dependent.

**4. Calibration — the pivotal late move: we had never checked the instrument.**
Every result above rested on the own-vs-mismatch docking metric, never validated against measured
selectivity. Calibrating it:
- **Tier-1** (known actives vs property-matched decoys vs candidates): after de-confounding a
  family-clustered-panel artifact, the metric carries a **real, modest cross-family** signal (own-vs-cross
  actives −0.107 vs decoys +0.146, diff −0.25 [−0.44,−0.08]) riding on a family-dependent **affinity**
  signal (own-pocket actives>decoys AUROC 0.66; strong kinases, chance for CA/GTPase).
- **Tier-2** (docked selectivity vs **measured** ΔpChEMBL, 460 compounds, 530 triples — the ground-truth
  test): the metric tracks measured selectivity **weakly and only for kinases** — within-kinase paralog
  ρ **+0.245** [+0.13,+0.35] (all three kinase pairs positive, 0.17–0.34), within-GPCR ρ +0.05 (ns). ≈6%
  of variance.

So the central positive is **neither an artifact nor strong targeting** — it is a **weak, kinase-biased**
selectivity signal, flattered by a kinase-heavy corpus. Docking-selection enriches *binders* well and
*selective* binders weakly, kinases only.

**4b. Drug-likeness.** ADMET (admet-ai) over 5,701 generated molecules: only **5.4% pass** a
safety+absorption guard (DILI 83% flagged, hERG 59%) — the QED/SA gates badly overestimated druggability.

**5. Learned selectivity oracle (SP-ORACLE) — killed by a 10-minute pre-check.**
The plan: train an oracle on the assembled ChEMBL selectivity data and use it as a grounded reward. The
pre-check: the generator's molecules sit at **median Tanimoto ~0.27** to their nearest ChEMBL training
compound (52–75% < 0.3). They are **out of any ChEMBL-trained oracle's domain, and unvalidatable there** —
no measured selectivity exists for those chemotypes. No oracle was built.

**6. Allosteric-pocket targeting — anchor-walled.**
The idea: select against the divergent allosteric pocket, where paralog selectivity is physically
achievable. But the only ground-truth signal (ChEMBL) is for **orthosteric** binders; docking them into an
allosteric pocket is meaningless, so the calibration machinery can't validate it — and measured
*allosteric-ligand* selectivity doesn't exist as a usable dataset. The premise (allosteric regions diverge)
is textbook-true and thus low-information to "confirm." Not pursued.

---

## What is settled

- Generation-side levers (conditioning, enrichment, local search, **weight updates**) do **not** confer
  transferable targeting. Five independent nulls, the strongest a true held-out generalization test.
- Docking-selection specificity is **real but weak and kinase-specific** (measured-selectivity ρ ≈ 0.25),
  and method-dependent (Boltz doesn't corroborate — consistent with its weakness).
- The generated pool is synthesizable and physchem-reasonable but **~95% ADMET-liable**.
- Two would-be rescue paths (learned oracle, allosteric) are **blocked by the same missing data**, and we
  proved each cheaply *before* building, via a pre-check.

## Why it's all one wall

Selectivity must be *learned from* or *validated against* labels in the chemical region you operate in.
- The generator operates in a **novel, synthesizable region with no selectivity labels** → can't condition
  or reward it there.
- ChEMBL has labels but in a **different (med-chem, orthosteric) region** → an oracle trained there
  extrapolates on the generator's molecules and can't be validated on them.
- Physics (docking) needs **no** in-distribution labels → it's the only thing that works at all — but it's
  weak and, here, kinase-only.

## What would actually unlock it

1. **Wet-lab measured selectivity on generator-like molecules** (synthesize + assay a batch of the
   generator's REAL-space outputs across a target panel). The one thing that puts labels in the right
   region; it would ground both an oracle and a grounded DPO retry. *The real unlock; out of scope for
   compute.*
2. **Better physics that needs no labels** — FEP or Boltz-2 affinity as a *selector* (not just validator),
   and validated allosteric structures for a paralog family with measured allosteric-ligand selectivity.
3. Accept the boundary: use the generator for **synthesizable diversity** and apply the weak
   docking-selection filter only where it earns its keep — **kinases**.

## Methodological note (the throughline that made this trustworthy)

The project's most valuable habit was **validating the instrument before building on it**. The calibration
revealed the central positive was weak and kinase-only (not the strong targeting the raw −0.8 delta
implied); two cheap pre-checks (the ChEMBL known/random controls' absence, then the oracle Tanimoto-domain
check) killed doomed builds in minutes; and adversarial review repeatedly caught over-reads before they
became the record (a "falsified" Tier-1 that was a confound; a DPO eval design that would have forced a
structural-zero; an oracle gate that measured the wrong distribution). The negative results here are strong
*because* the measurements were audited, not assumed.

---

*Detailed evidence and reproduce commands: [FINDINGS](FINDINGS.md) and the per-study results docs it links.*
