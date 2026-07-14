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
| SP-DPO | **weight-update** lever: DPO fine-tune SP-C on per-molecule own-vs-mismatch specificity pairs (pilot) | **null (pilot, underpowered\*)** — the strongest-*designed* test (true held-out generalization): DPO **fit the preference in-sample** (train margin 2.97→3.49) but it **did not transfer** — held-out DPO−base own-preference split 2/2, pooled +0.08 [−0.22,+0.41] ns. ADMET unchanged. \*n=4/1-seed → inconclusive, not "cannot work" | [SP_DPO_RESULTS](SP_DPO_RESULTS.md) |

**→ No generation-side lever — conditioning, motif-enrichment, local search, *or weight updates* — confers
robust targeting.** The sharpest statement of the wall is SP-DPO's: the generator **can be trained to fit
the specificity preference in-sample, but it does not generalize** to making raw samples target-specific on
unseen pockets. (SP-DPO is a pilot — underpowered, not a proof of impossibility.)

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

## D. ADMET of generated molecules — synthesizable & QED-reasonable, but liable

| study | what | result | doc |
|---|---|---|---|
| SP-AD | ML-ADMET (`admet-ai`, ~40 endpoints) over 5,701 generated pocket candidates | **only 5.4% pass** a safety+absorption guard; driven by **DILI (83% flagged), hERG (59%)**, high CYP, poor solubility (p25) — yet QED/Lipinski/HIA look drug-like. The QED/SA gates used earlier **overestimate** drug-likeness; real ADMET is worse (ML-proxy caveat: DILI models over-flag) | [SP_AD_RESULTS](SP_AD_RESULTS.md) |

**→ The pipeline generates synthesizable, physchem-reasonable molecules with pervasive predicted safety/
metabolic liabilities — a real drug-likeness gap the specificity work never surfaced.**

## E. Calibration — is the specificity metric measuring real selectivity? (Tier-1)

| study | what | result | doc |
|---|---|---|---|
| Tier-1 | dock **known actives**, **property-matched decoys**, and generated **candidates** into a shared panel; does the own-vs-mismatch z-delta separate real binders from decoys? | **the axis is real but modest, and only demonstrated cross-family.** Family-stratified (docking-failure pockets CA12/GTPase dropped): real actives prefer own over **cross-family** pockets more than matched decoys (Δ −0.25, CI [−0.44,−0.08] excl. 0; AUROC 0.58); **own-vs-same-family is ns** (real actives genuinely cross-react at the paralog level). smina also has a family-dependent **own-pocket affinity** signal (actives>decoys AUROC 0.66; strong kinases, chance for CA/GTPase). | [TIER1_CALIBRATION_RESULTS](TIER1_CALIBRATION_RESULTS.md) |

**→ Tier-1: the docking-selection specificity is NOT a pure normalization artifact — it carries a real,
modest *cross-family* signal riding on a real affinity signal.**
*(A first Tier-1 read wrongly concluded "falsified" from a family-clustered-panel + top-M confound;
corrected via stratification — see the doc's correction note.)*

| study | what | result | doc |
|---|---|---|---|
| Tier-2 | docked selectivity vs **measured** ΔpChEMBL (460 compounds, 530 target-pair triples) — the ground-truth calibration | **real but WEAK and kinase-specific.** All three kinase paralog pairs track measured selectivity (KIT/JAK3 ρ+0.34, KIT/CDK5 +0.32, JAK3/CDK5 +0.17; within-kinase pooled ρ **+0.245**, compound-clustered CI [+0.13,+0.35]); the one testable aminergic-GPCR pair (5-HT1A/5-HT2A) does **not** (ρ+0.05, ns). ρ≈0.25 ≈ ~6% of variance. | [TIER2_CALIBRATION_RESULTS](TIER2_CALIBRATION_RESULTS.md) |

**→ Calibrated against measured affinity, the docking specificity metric is a *real but weak, target-class-
dependent* selectivity signal: it tracks kinase paralog selectivity (consistently across all 3 kinase
pairs, ρ 0.17–0.34) and misses 5-HT-receptor subtype selectivity. The project's central positive is thus
neither an artifact nor strong targeting — it is a weak (ρ≈0.25) kinase-biased signal, plausibly helped by
the kinase-heavy corpus. Real targeting needs a *learned selectivity oracle* (measured-selectivity data now
assembled: 1,732 multi-target compounds) and/or allosteric-pocket targeting; docking-selection alone gives
weak kinase selectivity and nothing for the tested GPCR pair.**

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

**Piloted since (underpowered null):**
- **Reward fine-tuning / DPO** of the generator — **done as a pilot (SP-DPO): null but underpowered**
  (n=4 held-out, 1 seed). DPO fit the specificity preference in-sample but did not transfer to held-out
  raw samples. A full study (more targets/seeds, hyperparameter sweep, and a Boltz gate on any win) is the
  remaining way to firm this from "no pilot signal" to a real negative — but the pilot showed no signal
  worth that ~7-day cost.

**Tried since (killed by a pre-check):**
- **Learned selectivity oracle as reward (SP-ORACLE):** **not viable with existing data** — an
  applicability-domain pre-check showed the generator's molecules sit at median Tanimoto ~0.27 to the
  nearest ChEMBL training compound (52–75% < 0.3), i.e. **out of any ChEMBL-trained oracle's domain and
  unvalidatable there**. No oracle built. This is the **same data wall** as the generator-side nulls from
  the other side: you can neither *condition* nor *reward* a generator into a selectivity signal that does
  not exist in its own chemical region. See [SP_ORACLE_RESULTS](SP_ORACLE_RESULTS.md).

**Highest value, genuinely untried:**
- **Tier-2 selectivity calibration (decisive; the real next step):** correlate docked own-vs-mismatch
  Δscore against *measured* Δaffinity (ChEMBL/BindingDB) across target pairs, **including within-family
  (paralog) pairs**. This settles whether the docking specificity tracks real selectivity beyond the
  modest cross-family signal Tier-1 found, and whether paralog-level selectivity is reachable at all. If
  it does not, a **learned selectivity oracle** (trained on the Tier-2 data) becomes the credible path,
  and a properly-grounded selection/DPO retry could finally have a real target signal. See
  [SPECIFICITY_CALIBRATION_PROPOSAL](SPECIFICITY_CALIBRATION_PROPOSAL.md).
- **Allosteric/regulatory-pocket conditioning:** all work conditioned on the (conserved, orthosteric)
  catalytic pocket — the *hardest* place to be paralog-selective. The divergent allosteric site is where
  selectivity is physically achievable; untried.
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
