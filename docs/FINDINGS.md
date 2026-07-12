# FINDINGS — protein-conditioned SynFormer for target-specific generation

**Umbrella synthesis.** Last updated 2026-07-12. Links every sub-study's results doc. The one-line
conclusion:

> **Conditioning or re-biasing the *generator* on the target does not confer targeting; *selecting*
> against the 3D pocket does — and that specificity replicates at scale, because it is a *relative*
> pocket-discrimination that survives the single-scorer confound which makes *absolute* candidate quality
> unreliable (smina and Boltz-2 disagree on generated molecules).**

The project splits into two levers (make the generator target-aware vs select against the pocket) plus a
cross-cutting scorer question (is the docking proxy trustworthy?).

---

## A. Generator-side conditioning/re-biasing — consistently null or non-robust

| study | what | result | doc |
|---|---|---|---|
| SP2 | sequence (ESM) conditioning | weak / no robust targeting | [SP2_RESULTS](SP2_RESULTS.md), [SP2_ENCODER_COLLAPSE](SP2_ENCODER_COLLAPSE.md) |
| SP3 | richer encoders | help generic quality, not targeting | [SP3_RESULTS](SP3_RESULTS.md) |
| SP-C | 3D pocket conditioning | **honest null** — paired pocket−sequence Δ −0.10 [−0.49,+0.31] (CI incl. 0); broader chemistry, no targeting | [POCKET_CONDITIONED_RESULTS](POCKET_CONDITIONED_RESULTS.md) |
| SP-L | enrichment loop: re-bias frozen model toward docking-winners | **null** — winners use the generator's *modal* motifs; nothing distinctive to amplify (inert even at 30× weights) | [SP_L_RESULTS](SP_L_RESULTS.md) |
| SP-F | fragment-seeding hill-climb: analog-search around best dockers | **smina win 4/5 but NOT robust** — Boltz disagreed on 3/5; the biggest smina win (P10721 −2.34) was a co-folding-refuted, diversity-collapsed **smina-hacking artifact** | [SP_F_RESULTS](SP_F_RESULTS.md) |

**→ No generation-side lever (conditioning, motif-enrichment, or local search) confers robust targeting.**

## B. Selection against the 3D pocket — the positive result

| study | what | result | doc |
|---|---|---|---|
| Docking-selection | rigid-pocket shape-fit selection | first modest specificity signal | [DOCKING_SELECTION_RESULTS](DOCKING_SELECTION_RESULTS.md) |
| Powered study (N=20) | own-vs-mismatch normalized delta, all-pairs + AF arm | modest specificity, delta ≈ **−0.62/−0.71**, structure-origin-independent | [POWERED_SPECIFICITY_RESULTS](POWERED_SPECIFICITY_RESULTS.md), [TARGET_SPECIFICITY](TARGET_SPECIFICITY.md) |
| SP-SC | scale to N=41 (sampled mismatch) | **replicates on 21 *independent* targets: Δ −0.64, CI [−1.09,−0.20] excludes 0** (combined N=41 −0.80). Robust because a *relative same-scorer* contrast differences out uniform smina bias. **Family-stratified: the signal is fine-grained** — holds *within* the 16-kinase family (own vs sibling-kinase −0.77 ≈ own vs cross-family −0.75), i.e. paralog-level discrimination, not coarse family-appropriateness | [POWERED_SCALE_RESULTS](POWERED_SCALE_RESULTS.md) |

**→ Selecting against the pocket confers a modest, real, independently-replicated specificity.**

## C. Scorers — is the docking proxy trustworthy? (smina vs Boltz-2)

| study | what | result | doc |
|---|---|---|---|
| Affinity proxy | early ML affinity readout | scorer artifact, not real signal | [AFFINITY_TOOLS_RESEARCH](AFFINITY_TOOLS_RESEARCH.md) |
| Boltz-2 validation | co-folding as independent scorer | Boltz **competent** (known-vs-random AUROC 0.95) but does **not** corroborate docking's specificity → method-dependent | [BOLTZ_VALIDATION_RESULTS](BOLTZ_VALIDATION_RESULTS.md) |
| SP-CS | consensus scorer (smina+Boltz averaging) | **score-averaging doesn't beat Boltz**; averaging in the weaker proxy *dilutes*. Prefer **Boltz-as-validator** | [SP_CS_RESULTS](SP_CS_RESULTS.md) |
| SP-CC | candidate-regime scorer agreement | smina & Boltz **pick different candidate top-5** (Jaccard ≤0.25); the "candidates disagree more" contrast was a **bimodality artifact** (within-class agreement ≈ equal) | [SP_CC_RESULTS](SP_CC_RESULTS.md) |

**→ Single-proxy *absolute* candidate quality is unreliable (smina/Boltz disagree on generated
molecules). Boltz is the more trustworthy scorer, but its role so far has been *validation*, never
*selection*.**

---

## Reconciling the two facts about the specificity: robust *within* smina, method-dependent *across* methods

Two things are both true and must be stated together:

- **Robust within smina (relative signal).** SP-F/SP-CC show smina is unreliable for *absolute* candidate
  quality, yet SP-SC's specificity replicates on independent targets. The reason: the delta is a
  **relative, same-scorer** contrast (own pocket vs mismatch, all smina), so uniform smina bias
  *differences out* — pocket-discrimination is far less vulnerable to smina's absolute unreliability. The
  own-pocket-preference signal is real and reproducible **as a smina/rigid-shape-fit phenomenon.**
- **Method-dependent across methods (not corroborated by co-folding).** The scorer-independence question
  is **answered** (BOLTZ_VALIDATION): the *same* smina-selected molecules, co-folded 5×5, give a Boltz
  own-vs-mismatch delta of **−0.04 (chance)** vs docking's **−1.74** — Boltz does **not** see the
  own-vs-mismatch preference (and Boltz is competent here, known-vs-random AUROC 0.95, so this is an
  informative null). This was a *clean* test (molecules smina-selected, so no Boltz winner's-curse); a
  "select-by-Boltz, read-by-Boltz" design would instead be confounded and was **not** pursued.

**Honest synthesis:** the docking-selection specificity is a **real, replicated, relative smina
(rigid-pocket shape-fit) signal that co-folding does not corroborate at the molecule level** — i.e.
method-dependent. It is the project's strongest positive precisely because it is *relative and
replicated*, but it should be reported as shape-fit specificity, not as validated binding specificity.

## Open frontier — what we have NOT tried

**Already answered (do not re-run):**
- **Scorer-independence of the specificity** — Boltz does *not* see the own-vs-mismatch preference on
  smina-selected molecules (BOLTZ_VALIDATION, delta −0.04 vs docking −1.74). The docking specificity is
  method-dependent. A "select-by-Boltz, read-by-Boltz" experiment would be winner's-curse-confounded and
  adds nothing to this.

**Highest value, genuinely untried:**
- **Reward fine-tuning / DPO** of the generator (SP-L/SP-F were frozen-model — weight-update loops
  untested; the one lever that could make the *generator* itself target-aware).
- **Controlled Boltz-*selection* (different question):** does selecting *by* Boltz change/help which
  candidates you get, vs random and vs smina — with a random-selection baseline + same-molecule
  cross-scoring (winner's-curse-aware). Likely null, lower value; only worth it to close the loop.
- **Molecule-level absolute quality:** which scorer to trust for picking real binders remains open; the
  honest path is a held-out third scorer (below), not smina or Boltz alone.

**Encoder ablations (deferred):** hybrid pocket+sequence (`encoder_type: dual`); stronger pocket encoder
(`pocket_cb` Cβ orientation is extracted but unused; equivariant GNN / ProteinMPNN / ESM-IF features).

**Scale/robustness:** SP-SC was 1 seed, crystal-only, N=41 — multi-seed, the AF (flexible) arm at scale,
and the full 76 targets are untried at scale. A held-out **third scorer** to break smina/Boltz ties
(ML rescorer, not yet set up on the box).

**Validation:** no within-target potency positive (PBCNet2.0 env unbuilt); no experimental/wet-lab
validation or actual synthesis of the REAL-space routes — everything to date is proxy-based.

---

*Setup/infra: [DOCKING_SETUP](DOCKING_SETUP.md), [BOLTZ_SETUP](BOLTZ_SETUP.md). Session/ops context:
[SESSION_HANDOFF](SESSION_HANDOFF.md). Reproduce commands live in each results doc.*
