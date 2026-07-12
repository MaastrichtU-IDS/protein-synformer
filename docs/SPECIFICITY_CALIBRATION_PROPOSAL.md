# Proposal: Calibrate the specificity instrument before any further method work

**Date:** 2026-07-12 · Status: proposal (pre-review) · Author context: SP-DPO just closed the 5th
generator-side null; this proposal steps back and questions the measurement all results rest on.

## 1. The problem this addresses

Every conclusion in the project — the five generator-side nulls **and** the one positive result
(selection against the pocket confers specificity) — is read off a single instrument: the
**own-vs-mismatch smina shape-fit** delta. That instrument has never been calibrated against *measured*
selectivity. Two facts make this urgent:

- The powered study docked known actives (530) and random decoys (600) **only into their own pocket,
  never into mismatch pockets** — so the specificity delta was only ever computed for the model's own
  candidates. There is no decoy control on the specificity axis.
- Our one competent, validated independent scorer (Boltz-2 affinity head; known-vs-random AUROC 0.95)
  sees the own-vs-mismatch delta as **−0.04 (chance)**. It does not corroborate the smina specificity.

So the load-bearing signal may be shape-fit/pocket-stickiness noise, and we cannot currently tell.
**No new scorer, training-data, or architecture work is worth doing until this reads out**, because the
outcome changes whether "selection works" is a real finding or an artifact.

## 2. Central question

> Does the own-vs-mismatch docking axis track **real molecular selectivity** — i.e. does a molecule
> that measurably binds target A more tightly than off-target B also score better (own-preferring) for A
> than B in our pipeline?

## 3. Design — three staged tiers: cheapest *falsifier* first, decisive calibration (Tier 2) as the main event

### Tier 1 — Decoy + candidate control (fast; mostly existing SMILES). *Can only FALSIFY, not confirm.*
Dock the **existing** known actives, random decoys, **and the model's generated candidates** into the
**same own + mismatch panel** (reuse `powered_run` / the docking driver). Compute **the metric the
positive result actually used** — the **z-per-pocket-column delta** (`_delta_win_from_matrix`) — as the
primary readout (report raw-d secondarily; note SP-DPO's eval used raw-d, the powered positive used
z-delta, so calibrate the z-delta here). Split by class.

- **Falsification signature:** if random decoys are *just as own-preferring* as known actives, the
  "specificity" is **pocket-stickiness** — not molecule selectivity.
- **Sharper, free comparison (advisor):** put the **generated candidates** on the same axis. If the
  model's candidates own-prefer *as much as or more than real known actives*, the metric is rewarding a
  generation artifact, not selectivity — this directly interrogates the "selection works" result and may
  be more diagnostic than the decoy control.
- **Property-match confound (must fix):** docking score tracks MW/logP/HBD, and `source=random` are
  random REAL-space molecules, not property-matched. Build a **property-matched decoy set** (MW/logP/HBD
  bins, DUD-E style) or covary those descriptors out — otherwise a Tier-1 "pass" is confounded by
  physchem, not selectivity.
- **Readout:** AUROC of the z-delta separating (known actives of A) from (matched decoys) at pocket A,
  plus the actives-vs-candidates contrast, per target and pooled; bootstrap CI.
- **Interpretation limit (load-bearing):** per-column z *already* differences out additive
  pocket-stickiness by construction, so a clean decoy-null on the z-metric is partly guaranteed and is
  **weak** confirmation. **Tier 1 can falsify (decoys/candidates own-prefer as much as actives) but
  cannot confirm.** Tier 2 is the actual calibration.

### Tier 2 — Measured-selectivity correlation (the decisive calibration; needs ChEMBL).
For target **pairs** (A, B) drawn from the 41 (rich within-family: kinase×kinase, GPCR×GPCR), pull from
ChEMBL/BindingDB compounds with **measured** pKi/pIC50 on **both** A and B. For each such (compound, A, B):

- **measured selectivity** = pAff(A) − pAff(B).
- **docked selectivity** = z(score in B) − z(score in A), where **z is against a common reference library
  docked into every pocket** (advisor — must specify): dock one fixed diverse reference set into all
  pockets, and z-score each test compound's score against *that pocket's reference distribution*. This is
  what makes A and B comparable — comparing one compound's raw score in A vs B conflates selectivity with
  pocket depth. The reference library is the per-pocket scale.
- **Test:** Spearman ρ between measured and docked selectivity across all triples (pooled and per
  family). This directly asks "does the docking axis track measured selectivity?" — no arbitrary
  selective/promiscuous binarization.

**Pre-check before designing Tier 2 (advisor, ~10 min):** verify ChEMBL coverage for *these* UniProt IDs —
pick 3 kinase pairs from the set and confirm compounds with pChEMBL measured on both. "Pan-kinase data
exists" is true in general but unverified for our exact targets; de-risk before committing.
Start with the kinase subset (KIT/JAK3/CDK5/FGFR1/DYRK1A/… — the hardest and most data-rich case); expand
to GPCRs and CA if coverage and signal warrant.

### Tier 3 — Cross-scorer (which oracle, if any, is real).
Run the **same** Tier-1/Tier-2 compound sets through **Boltz-2's affinity head** into the own+mismatch
panel (cap the set for GPU cost). Compare smina vs Boltz on the Tier-2 correlation. Whichever scorer
tracks measured selectivity becomes the selection oracle going forward; if **neither** does, that is the
finding, and the answer is a learned selectivity model (Tier-2 data becomes its training set).

## 4. Pre-committed decision criteria (guard against overreach)

| Tier-1 decoy control | Tier-2 measured correlation | Conclusion → action |
|---|---|---|
| actives own-prefer, decoys don't | ρ meaningfully > 0 (CI excludes 0) | **Axis is REAL.** Scale selection; the generator-null stands as a real result; build the selection pipeline on the validated scorer (Tier-3 winner). |
| decoys own-prefer too | (any) | **Pocket-stickiness artifact.** The "selection works" positive is **downgraded**; honest project conclusion changes. Pivot to a learned selectivity oracle. |
| actives own-prefer, decoys don't | ρ ≈ 0 | Axis separates binders from non-binders but **not** degrees of selectivity → it's an affinity proxy, not a *selectivity* proxy. Selection can enrich binders but not selective ones → oracle work needed for selectivity. |

**No "the metric is validated" claim unless BOTH the decoy control passes AND Tier-2 ρ excludes 0.** A
null in Tier-2 at small n is *inconclusive*, not proof the axis is noise (same discipline as SP-DPO).

## 5. What each outcome buys

- **Rescue:** if the axis is real, the whole "selection > generation" thesis is validated on ground
  truth, and the project has a defensible selection pipeline — the strongest possible version of the
  current story.
- **Reframe:** if it's an artifact, we've avoided building more methods on sand, and the Tier-2 ChEMBL
  data is exactly the training set for a **learned selectivity oracle** — which then becomes the reward
  for a *properly-grounded* DPO/selection retry (the one that could actually work, because it's trained
  on a real selectivity signal).

## 6. Scope / cost

- **Tier 1:** ~530 knowns + ~600 decoys × (own + ~8 sampled mismatch) ≈ **~10k docks ≈ ~overnight** on
  the cap-4 `smina.static` driver. Reuses existing SMILES + harness. **Do this first.**
- **Tier 2:** ChEMBL/BindingDB data engineering (query these targets, extract multi-target compound
  activities, build (compound, A, B) triples) — the real *work* is data assembly, docking is modest
  (~a few thousand). ~1–2 days incl. data.
- **Tier 3:** Boltz on a capped set — GPU, ~a day.

## 7. Risks / caveats

- ChEMBL activity data is heterogeneous (assay types, Ki vs IC50); needs careful standardization
  (pChEMBL, assay filtering). This is the main Tier-2 risk.
- Docking A and B pockets uses the crystal orthosteric site; if a compound's real selectivity comes from
  an allosteric site we don't model, the correlation will be attenuated (relevant to the separate
  "allosteric pocket" lever).
- Same-scorer relative contrast (own-vs-mismatch) partly differences out smina's absolute unreliability —
  so a *positive* Tier-1/2 is strong; a *negative* must be read with the caveats above, not as proof.

## 8. Recommendation

**One-line summary (advisor):** *Tier 1 can only falsify; Tier 2 against measured affinity is the actual
calibration — resource it as the main event, not the follow-up.*

Run **Tier 1 first** (cheap, uses existing data + a property-matched decoy set + candidates on the same
axis). It can **falsify** the central positive within a day — if decoys or candidates own-prefer as much
as real actives on the z-metric, we have the answer and stop. But a clean Tier-1 result is only weak
support (per-column z pre-differences out stickiness), so it does **not** license "the metric is
validated." **Tier 2 (measured-selectivity correlation, with the common-reference per-pocket
normalization) is the decisive calibration and should be resourced as the main event.** Do the ~10-min
ChEMBL coverage pre-check before committing to the Tier-2 design.
