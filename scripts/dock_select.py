"""Docking-based selection + mismatch-control pipeline.

Usage (dry-run, 1 target × 5 candidates):
    ./.venv/bin/python -m scripts.dock_select \\
        --targets data/dock/targets.json --n-candidates 5 --limit-targets 1

Full run:
    caffeinate -i ./.venv/bin/python -m scripts.dock_select \\
        --targets data/dock/targets.json --n-candidates 150 --n-refs 30
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pathlib
import random
import sys
from typing import Any

import click
import numpy as np
import pandas as pd
import torch

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dock_select")

# ── project root on sys.path (needed when run with -m) ────────────────────────
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ── lazy imports (heavy; deferred until actually needed) ──────────────────────
def _import_dock():
    from synformer.dock.dock import dock  # noqa: PLC0415
    from synformer.dock.receptor import prepare_target  # noqa: PLC0415
    from synformer.dock.geometry import select_topm, mismatch_summary  # noqa: PLC0415
    return dock, prepare_target, select_topm, mismatch_summary


def _import_load_model():
    from scripts.sample_helpers import load_model  # noqa: PLC0415
    return load_model


# ── checkpoints / data paths ──────────────────────────────────────────────────
MASKED_CKPT = (
    _root
    / "logs_gate/sp2_masked"
    / "2607051705-22f3794@sp2-protein-conditioning"
    / "2026_07_06__00_18_30/checkpoints/last.ckpt"
)

SP2_TEST_CSV = _root / "data/protein_molecule_pairs/sp2_test.csv"
CANDIDATES_DIR = _root / "data/dock/candidates"


def set_candidates_dir(path):
    """Override the directory _load_candidates reads (used by the pocket arm to avoid clobbering
    the sequence baseline in data/dock/candidates/)."""
    global CANDIDATES_DIR
    CANDIDATES_DIR = pathlib.Path(path)

RECEPTORS_DIR = _root / "data/dock/receptors"
OUTPUT_DIR = _root / "data/dock"
SCORES_CSV = OUTPUT_DIR / "dock_scores.csv"
SUMMARY_CSV = OUTPUT_DIR / "dock_select_summary.csv"

SCORES_COLS = ["target", "pocket", "molecule", "source", "score"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_existing_scores(path: pathlib.Path) -> set[tuple[str, str]]:
    """Return set of (molecule, pocket) pairs already scored."""
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add((row["molecule"], row["pocket"]))
    return done


def _load_scores_table(path: pathlib.Path) -> dict[tuple[str, str], float]:
    """Load all (molecule, pocket) → score from the scores CSV into memory.

    Returns an empty dict if the file does not exist.  Only the first
    occurrence of each (molecule, pocket) pair is kept (matches the
    append-only CSV semantics where earlier rows win on restart).
    """
    table: dict[tuple[str, str], float] = {}
    if not path.exists():
        return table
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["molecule"], row["pocket"])
            if key not in table:
                try:
                    v = float(row["score"])
                except (ValueError, KeyError):
                    v = float("nan")
                table[key] = v
    return table


def _append_score_row(
    fh,
    writer,
    target: str,
    pocket: str,
    molecule: str,
    source: str,
    score: float,
) -> None:
    writer.writerow(
        {
            "target": target,
            "pocket": pocket,
            "molecule": molecule,
            "source": source,
            "score": f"{score:.4f}",
        }
    )
    fh.flush()


def _safe_mean(values: list[float]) -> float:
    """NaN-safe mean; returns NaN if empty or all-NaN."""
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def select_topm_for_target(scores_by_smiles: dict[str, float], m: int) -> list[str]:
    """Return the top-M SMILES by lowest (best) docking score, NaN excluded.

    Parameters
    ----------
    scores_by_smiles:
        Mapping SMILES → own-pocket score (may contain NaN).
    m:
        Maximum number to select.

    Returns
    -------
    List of up to ``m`` SMILES in ascending score order (best first).
    """
    finite_pairs = [(smi, sc) for smi, sc in scores_by_smiles.items() if not math.isnan(sc)]
    finite_pairs.sort(key=lambda x: x[1])
    return [smi for smi, _ in finite_pairs[:m]]


def _stable_seed(seed: int, target_id: str) -> int:
    """Stable per-target seed independent of PYTHONHASHSEED."""
    import hashlib
    digest = hashlib.md5(target_id.encode()).hexdigest()
    return seed ^ (int(digest[:8], 16) & 0xFFFFFFFF)


def _load_known_ligands(target_id: str, n_refs: int, seed: int) -> list[str]:
    """Return up to n_refs unique known SMILES for target_id, seeded."""
    df = pd.read_csv(SP2_TEST_CSV)
    smiles_pool = df[df["target_id"] == target_id]["SMILES"].unique().tolist()
    available = len(smiles_pool)
    k = min(n_refs, available)
    rng = random.Random(_stable_seed(seed, target_id))
    selected = rng.sample(smiles_pool, k)
    log.info("  known ligands for %s: %d available, using %d", target_id, available, k)
    return selected, available, k


def _load_random_real(n_refs: int, seed: int, target_idx: int, fpindex) -> list[str]:
    """Return n_refs randomly sampled SMILES from the fpindex molecule pool.

    Uses a per-target seed so different targets get different random baselines.
    """
    molecules = fpindex.molecules  # tuple of Molecule objects
    pool_size = len(molecules)
    k = min(n_refs, pool_size)
    # Per-target seed: mix global seed with target index to avoid identical pools
    per_target_seed = seed ^ (0xDEADBEEF + target_idx * 0x9E3779B9 & 0xFFFFFFFF)
    rng = random.Random(per_target_seed)
    indices = rng.sample(range(pool_size), k)
    return [molecules[i].smiles for i in indices]


def _load_candidates(target_id: str, n_candidates: int) -> list[str]:
    """Load up to n_candidates from candidates/<target_id>.txt."""
    cand_file = CANDIDATES_DIR / f"{target_id}.txt"
    if not cand_file.exists():
        log.warning("Candidates file not found: %s", cand_file)
        return []
    lines = cand_file.read_text().splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    return lines[:n_candidates]


# ── main entry ────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--targets",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to targets.json",
)
@click.option(
    "--n-candidates",
    default=150,
    show_default=True,
    type=int,
    help="Max candidates per target to dock",
)
@click.option(
    "--n-refs",
    default=30,
    show_default=True,
    type=int,
    help="Max known ligands AND random molecules per target",
)
@click.option(
    "--top-m",
    default=10,
    show_default=True,
    type=int,
    help="Top-M selected candidates for mismatch-control cross-docking",
)
@click.option(
    "--limit-targets",
    default=None,
    type=int,
    help="Limit number of targets processed (useful for dry-run)",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Global random seed",
)
@click.option(
    "--output-dir",
    default=str(OUTPUT_DIR),
    show_default=True,
    type=click.Path(),
    help="Directory for dock_scores.csv and dock_select_summary.csv",
)
def main(
    targets: str,
    n_candidates: int,
    n_refs: int,
    top_m: int,
    limit_targets: int | None,
    seed: int,
    output_dir: str,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "dock_scores.csv"
    summary_path = out_dir / "dock_select_summary.csv"
    mismatch_path = out_dir / "dock_mismatch_summary.csv"

    dock, prepare_target, _select_topm_unused, mismatch_summary = _import_dock()

    # ── load model / fpindex for random-REAL pool ─────────────────────────────
    log.info("Loading model + fpindex from %s …", MASKED_CKPT)
    device = torch.device("cpu")
    load_model = _import_load_model()
    _model, fpindex, _rxn_matrix = load_model(MASKED_CKPT, None, device)
    log.info("fpindex has %d molecules", len(fpindex.molecules))

    # ── load targets ──────────────────────────────────────────────────────────
    all_targets: list[dict[str, Any]] = json.loads(pathlib.Path(targets).read_text())
    if limit_targets is not None:
        all_targets = all_targets[:limit_targets]
        log.info("--limit-targets %d → processing %d target(s)", limit_targets, len(all_targets))

    # ── load or create scores CSV ─────────────────────────────────────────────
    # D2 fix: check exists BEFORE stat() to avoid FileNotFoundError
    existing_done = _load_existing_scores(scores_path)
    log.info("%d (molecule, pocket) pairs already in %s — will skip", len(existing_done), scores_path)

    # Load entire on-disk scores table into memory once (Perf fix: avoid per-row re-reads)
    scores_table: dict[tuple[str, str], float] = _load_scores_table(scores_path)

    scores_fh = scores_path.open("a", newline="")
    scores_writer = csv.DictWriter(scores_fh, fieldnames=SCORES_COLS)
    # Write the CSV header exactly once: on a new or empty file.  open("a") does
    # not truncate, so tell()==0 is the single reliable indicator that nothing has
    # been written yet (covers both "file did not exist" and "file was empty").
    if scores_fh.tell() == 0:
        scores_writer.writeheader()
        scores_fh.flush()

    # ── per-target: prepare receptor + score all molecules ───────────────────
    # spec_cache[target_id] = ReceptorSpec | None (None = failed)
    spec_cache: dict[str, Any] = {}
    # per_target_scores: target_id → {source: [score, ...]}
    per_target_scores: dict[str, dict[str, list[float]]] = {}
    # top_m_smiles: target_id → list of SMILES for top-M selected candidates
    top_m_smiles: dict[str, list[str]] = {}
    # known counts for summary
    known_counts: dict[str, dict[str, int]] = {}  # target_id → {available, used}

    for tgt_idx, tgt in enumerate(all_targets):
        target_id: str = tgt["target_id"]
        pdb_id: str = tgt["pdb_id"]
        ligand_resname: str = tgt["ligand_resname"]
        chain: str | None = tgt.get("chain", None)

        log.info("=" * 60)
        log.info("TARGET: %s  (PDB: %s  ligand: %s)", target_id, pdb_id, ligand_resname)

        # ── prepare receptor ──────────────────────────────────────────────────
        rec_dir = RECEPTORS_DIR / target_id
        try:
            spec = prepare_target(
                pdb_id,
                out_dir=str(rec_dir),
                chain=chain,
                ligand_resname=ligand_resname,
            )
            spec_cache[target_id] = spec
            log.info("  ReceptorSpec ready: %s", rec_dir)
        except Exception as exc:
            log.error("  prepare_target FAILED for %s: %s — skipping", target_id, exc)
            spec_cache[target_id] = None
            continue

        spec = spec_cache[target_id]
        pocket_id = target_id  # pocket identifier = target_id (its own pocket)

        # ── load molecules to dock ────────────────────────────────────────────
        candidates = _load_candidates(target_id, n_candidates)
        known_smiles, avail, used = _load_known_ligands(target_id, n_refs, seed)
        known_counts[target_id] = {"available": avail, "used": used}
        # Per-target random pool (minor fix: use tgt_idx to vary seed per target)
        random_smiles = _load_random_real(n_refs, seed, tgt_idx, fpindex)

        log.info(
            "  molecules: %d candidates, %d known (%d available), %d random",
            len(candidates),
            len(known_smiles),
            avail,
            len(random_smiles),
        )

        molecule_batches: list[tuple[str, str]] = (
            [(smi, "candidate") for smi in candidates]
            + [(smi, "known") for smi in known_smiles]
            + [(smi, "random") for smi in random_smiles]
        )

        scores_by_source: dict[str, list[float]] = {
            "candidate": [],
            "known": [],
            "random": [],
        }

        for smi, source in molecule_batches:
            pair_key = (smi, pocket_id)
            if pair_key in existing_done:
                log.debug("  skip already-scored: %s in %s", smi[:40], pocket_id)
                # Score already in scores_table from the initial load — no re-read needed
                continue

            score = dock(spec, smi, seed=seed)
            if not math.isnan(score):
                _append_score_row(scores_fh, scores_writer, target_id, pocket_id, smi, source, score)
                # Update both the in-memory table and the per-source lists
                scores_table[pair_key] = score
                scores_by_source[source].append(score)
                existing_done.add(pair_key)
                log.info("  docked %-10s  score=%.3f  src=%s", smi[:35], score, source)
            else:
                log.warning("  NaN score for %s (src=%s) — dropped", smi[:40], source)

        # ── fill per_target_scores from authoritative in-memory scores_table ─
        # Rebuild from scores_table by MOLECULE identity so that two molecules
        # with the same score are both counted (the old dedup-by-value was a bias
        # that caused n_known_used / n_random_docked to be under-counted).
        # This covers both fresh and restart cases uniformly (D1 fix core idea).
        scores_by_source = {"candidate": [], "known": [], "random": []}
        seen: dict[str, set[str]] = {"candidate": set(), "known": set(), "random": set()}
        for smi, source in molecule_batches:
            if smi in seen[source]:
                continue
            seen[source].add(smi)
            pair_key = (smi, pocket_id)
            v = scores_table.get(pair_key, float("nan"))
            if not math.isnan(v):
                scores_by_source[source].append(v)

        per_target_scores[target_id] = scores_by_source

        # ── select top-M candidates (D1 fix) ─────────────────────────────────
        # Build the complete own-pocket scores dict from the authoritative
        # in-memory table, covering BOTH freshly-docked and already-on-disk rows.
        own_pocket_scores: dict[str, float] = {}
        for smi in candidates:
            pair_key = (smi, pocket_id)
            v = scores_table.get(pair_key, float("nan"))
            own_pocket_scores[smi] = v

        selected = select_topm_for_target(own_pocket_scores, top_m)
        top_m_smiles[target_id] = selected
        if selected:
            best = min(own_pocket_scores[s] for s in selected if not math.isnan(own_pocket_scores[s]))
            log.info("  top-%d selected (best score=%.3f)", len(selected), best)
        else:
            log.warning("  No finite candidate scores for %s", target_id)

    # ── mismatch-control cross-docking ────────────────────────────────────────
    successful_targets = [
        tid for tid in [t["target_id"] for t in all_targets]
        if spec_cache.get(tid) is not None
    ]
    log.info("Successful targets for mismatch control: %s", successful_targets)

    mismatch_matrix_data: dict[tuple[str, str], float] = {}  # (target_i, pocket_j) → best score

    if len(successful_targets) >= 2:
        log.info("Building mismatch score matrix (%d × %d)…", len(successful_targets), len(successful_targets))

        for i, target_i in enumerate(successful_targets):
            cands_i = top_m_smiles.get(target_i, [])
            if not cands_i:
                log.warning("  No top-M candidates for %s — skipping row", target_i)
                continue

            for j, target_j in enumerate(successful_targets):
                spec_j = spec_cache[target_j]

                # D3 fix: diagonal uses the SAME fixed cands_i molecule set.
                # Look up their own-pocket scores from the authoritative scores_table.
                if i == j:
                    own_scores_cands_i = [
                        scores_table.get((smi, target_i), float("nan"))
                        for smi in cands_i
                    ]
                    finite_own = [v for v in own_scores_cands_i if not math.isnan(v)]
                    diag_val = min(finite_own) if finite_own else float("nan")
                    mismatch_matrix_data[(target_i, target_j)] = diag_val
                    continue

                # Off-diagonal: dock target_i's top-M into target_j's pocket
                cross_scores: list[float] = []
                for smi in cands_i:
                    pair_key = (smi, target_j)
                    if pair_key in existing_done:
                        # Look up from the in-memory table (Perf fix: no CSV re-read)
                        v = scores_table.get(pair_key, float("nan"))
                        if not math.isnan(v):
                            cross_scores.append(v)
                        continue
                    score = dock(spec_j, smi, seed=seed)
                    if not math.isnan(score):
                        _append_score_row(
                            scores_fh, scores_writer,
                            target_i, target_j, smi, "candidate", score
                        )
                        scores_table[pair_key] = score
                        cross_scores.append(score)
                        existing_done.add(pair_key)
                    else:
                        log.warning(
                            "  NaN cross-dock score: %s in %s — dropped",
                            smi[:40], target_j
                        )

                if cross_scores:
                    mismatch_matrix_data[(target_i, target_j)] = min(cross_scores)
                    log.info(
                        "  cross-dock %s → %s: best=%.3f over %d scores",
                        target_i, target_j, min(cross_scores), len(cross_scores)
                    )
                else:
                    log.warning("  No finite cross-dock scores for %s → %s", target_i, target_j)
                    mismatch_matrix_data[(target_i, target_j)] = float("nan")

        # Build matrix
        score_matrix = []
        for i, ti in enumerate(successful_targets):
            row = []
            for j, tj in enumerate(successful_targets):
                key = (ti, tj)
                if key in mismatch_matrix_data:
                    row.append(mismatch_matrix_data[key])
                else:
                    row.append(float("nan"))
            score_matrix.append(row)

        # Compute summary only if diagonal is mostly filled
        diag_filled = sum(
            1 for i, ti in enumerate(successful_targets)
            if not math.isnan(score_matrix[i][i])
        )
        if diag_filled > 0:
            mm_summary = mismatch_summary(score_matrix)
            log.info("Mismatch summary: %s", mm_summary)
        else:
            mm_summary = {"own_mean": float("nan"), "offdiag_mean": float("nan"),
                          "delta": float("nan"), "win_rate": float("nan")}
    else:
        log.info("Fewer than 2 successful targets — skipping mismatch matrix (dry-run OK)")
        mm_summary = None
        score_matrix = None

    scores_fh.close()

    # ── write per-target summary CSV ──────────────────────────────────────────
    # D4 fix: selected_mean uses ONLY the top-M selected candidates' own-pocket scores.
    summary_rows: list[dict] = []
    for tgt in all_targets:
        target_id = tgt["target_id"]
        if target_id not in per_target_scores:
            continue
        sc = per_target_scores[target_id]
        kc = known_counts.get(target_id, {"available": 0, "used": 0})
        # Compute selected_mean from the actual top-M SMILES own-pocket scores
        pocket_id = target_id
        topm = top_m_smiles.get(target_id, [])
        topm_scores = [
            scores_table.get((smi, pocket_id), float("nan"))
            for smi in topm
        ]
        summary_rows.append(
            {
                "target": target_id,
                "n_candidates_docked": len(sc["candidate"]),
                "n_known_available": kc["available"],
                "n_known_used": kc["used"],
                "n_random_docked": len(sc["random"]),
                "selected_mean": _safe_mean(topm_scores),
                "known_mean": _safe_mean(sc["known"]),
                "random_mean": _safe_mean(sc["random"]),
            }
        )

    # Minor fix: write per-target table and overall mismatch to SEPARATE files
    with summary_path.open("w", newline="") as sfh:
        fieldnames = [
            "target", "n_candidates_docked", "n_known_available", "n_known_used",
            "n_random_docked", "selected_mean", "known_mean", "random_mean",
        ]
        writer = csv.DictWriter(sfh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    log.info("Per-target summary written to %s", summary_path)

    if mm_summary is not None:
        with mismatch_path.open("w", newline="") as mfh:
            writer = csv.DictWriter(mfh, fieldnames=["own_mean", "offdiag_mean", "delta", "win_rate"])
            writer.writeheader()
            writer.writerow(mm_summary)
        log.info("Mismatch summary written to %s", mismatch_path)
        log.info(
            "Mismatch: own=%.3f offdiag=%.3f delta=%.3f win_rate=%.2f",
            mm_summary["own_mean"],
            mm_summary["offdiag_mean"],
            mm_summary["delta"],
            mm_summary["win_rate"],
        )

    log.info("Scores written to %s", scores_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
