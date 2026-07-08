"""Docking-selection POC — Task 4: target selection + candidate generation.

Two independent modes so selection and generation can be re-run separately in a
terminal:

  scan     : scan test proteins (those with ESM embeddings), fetch their PDB
             entries, inspect HETATM residues, and print PDBs that carry a clean
             drug-like co-crystal ligand. Human-in-the-loop shortlisting step.
  generate : read data/dock/targets.json and, for each target, sample ~150
             synthesizable candidates from the SP2 `masked` checkpoint, dedup
             valid SMILES, and write data/dock/candidates/<target_id>.txt.

targets.json is written by hand (or by the scan's --out flag) after
reviewing the scan output; generate consumes it.

Run from repo root with `.venv/bin/python scripts/dock_prepare.py <mode> ...`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import click

# Repo root on sys.path so `scripts.*` / `synformer.*` import when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("dock_prepare")

# --- Paths --------------------------------------------------------------------
EMB_PATH = "data/protein_embeddings/embeddings_selection_float16_4973.pth"
TEST_CSV = "data/protein_molecule_pairs/sp2_test.csv"
CKPT_PATH = (
    "logs_gate/sp2_masked/2607051705-22f3794@sp2-protein-conditioning/"
    "2026_07_06__00_18_30/checkpoints/last.ckpt"
)
DOCK_DIR = Path("data/dock")
TARGETS_JSON = DOCK_DIR / "targets.json"
CANDIDATES_DIR = DOCK_DIR / "candidates"

# --- Drug-like ligand filter (pure logic — unit tested) -----------------------
# Water / ions / crystallization additives / common sugars & buffers to ignore.
LIGAND_IGNORE_SET = {
    "HOH", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "SO4",
    "PO4", "GOL", "EDO", "PEG", "PG4", "DMS", "ACT", "FMT", "EPE", "TRS", "IMD",
    "MPD", "BME", "CO3", "NO3", "IOD", "BR", "CD", "NI", "HG", "ACE", "NAG",
    "BMA", "MAN", "FUC", "CIT", "TLA",
}
MIN_HEAVY_ATOMS = 12  # a single residue copy must have >= this many atoms


def is_drug_like_ligand(resname: str, atom_count: int) -> bool:
    """Return True if a HETATM residue looks like a drug-like co-crystal ligand.

    A ligand qualifies when its residue name is NOT in the ignore set (water,
    ions, buffers, crystallization additives, common sugars) AND a single
    residue copy has at least MIN_HEAVY_ATOMS atoms.

    Parameters
    ----------
    resname:  3-(or fewer)-letter HETATM residue name (case-insensitive).
    atom_count:  number of atoms in ONE copy of that residue.
    """
    if resname is None:
        return False
    if resname.strip().upper() in LIGAND_IGNORE_SET:
        return False
    return atom_count >= MIN_HEAVY_ATOMS


def find_drug_like_ligands(res_names, res_ids, chain_ids):
    """Given parallel per-atom arrays for HETATM (non-protein, non-solvent)
    atoms, return a list of (resname, max_copy_atom_count) for residues that
    pass is_drug_like_ligand.

    A "copy" is a unique (chain_id, res_id, res_name) group; we take the largest
    copy's atom count so partial/alternate copies don't disqualify a ligand.
    """
    from collections import defaultdict

    copy_counts: dict[tuple, int] = defaultdict(int)
    for rn, ri, ci in zip(res_names, res_ids, chain_ids):
        copy_counts[(str(rn), int(ri), str(ci))] += 1

    # Largest copy per resname.
    best_per_resname: dict[str, int] = {}
    for (rn, _ri, _ci), n in copy_counts.items():
        if n > best_per_resname.get(rn, 0):
            best_per_resname[rn] = n

    out = []
    for rn, n in best_per_resname.items():
        if is_drug_like_ligand(rn, n):
            out.append((rn, n))
    return sorted(out, key=lambda t: -t[1])


# --- Network helpers with retry/backoff ---------------------------------------
def _with_retry(fn, *args, tries=3, base_delay=1.5, what="", **kwargs):
    """Call fn with simple exponential backoff; return None on final failure."""
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - network errors are broad
            if attempt == tries:
                log.warning("give up %s after %d tries: %s", what, tries, e)
                return None
            delay = base_delay * (2 ** (attempt - 1))
            log.info("retry %s (attempt %d/%d) after %.1fs: %s", what, attempt, tries, delay, e)
            time.sleep(delay)
    return None


def _load_test_targets_with_embeddings():
    """Return sorted list of target_ids present in both the test set and the
    embedding dict, plus the embedding dict itself."""
    import pandas as pd
    import torch

    emb = torch.load(EMB_PATH, map_location="cpu")
    emb_keys = set(emb.keys())
    test = pd.read_csv(TEST_CSV)
    test_ids = set(test["target_id"].unique())
    both = sorted(test_ids & emb_keys)
    log.info(
        "test targets: %d, with embeddings: %d", len(test_ids), len(both)
    )
    return both, emb


def _inspect_pdb_ligands(pdb_id: str):
    """Fetch a PDB with biotite and return (drug_like_list, method, resolution).

    drug_like_list is [(resname, atom_count), ...]. Returns None on fetch/parse
    failure so the caller can skip.
    """
    import biotite.database.rcsb as rcsb
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io

    with tempfile.TemporaryDirectory() as tmp:
        fetched = _with_retry(
            rcsb.fetch, pdb_id.upper(), "pdb", target_path=tmp,
            what=f"fetch {pdb_id}",
        )
        if fetched is None:
            return None
        try:
            arr = pdb_io.get_structure(pdb_io.PDBFile.read(fetched), model=1)
        except Exception as e:  # noqa: BLE001
            log.warning("parse fail %s: %s", pdb_id, e)
            return None

    prot_mask = struc.filter_amino_acids(arr)
    het = arr[~prot_mask]
    if len(het) == 0:
        return []
    drug_like = find_drug_like_ligands(het.res_name, het.res_id, het.chain_id)
    return drug_like


# --- Modes --------------------------------------------------------------------
@click.group()
def cli():
    """Docking-selection prep: target selection (scan) + candidate generation."""


@cli.command()
@click.option("--limit", type=int, default=None, help="Max test targets to scan.")
@click.option("--max-pdbs", type=int, default=6, help="PDBs to inspect per target.")
@click.option("--out", type=click.Path(), default=None, help="Write scan results as JSON here.")
def scan(limit, max_pdbs, out):
    """Scan test proteins and print PDBs bearing a clean drug-like ligand."""
    from scripts.helpers.pdb import fetch_pdb_ids

    both, _emb = _load_test_targets_with_embeddings()
    if limit:
        both = both[:limit]

    results = []
    for target_id in both:
        acc = target_id.split("_")[0]
        pair = _with_retry(fetch_pdb_ids, acc, what=f"pdb ids {acc}")
        if pair is None:
            log.info("skip %s: no PDB lookup", target_id)
            continue
        _first, df = pair
        if df is None or df.empty:
            log.info("skip %s (%s): no PDB entries", target_id, acc)
            continue

        pdb_ids = list(df["PDB_ID"])[:max_pdbs]
        for pdb_id in pdb_ids:
            drug_like = _inspect_pdb_ligands(pdb_id)
            if not drug_like:
                continue
            # Prefer PDBs with exactly one drug-like ligand.
            row = df[df["PDB_ID"] == pdb_id].iloc[0].to_dict()
            method = row.get("Method", "?")
            resolution = row.get("Resolution", "?")
            best_lig, best_n = drug_like[0]
            entry = {
                "target_id": target_id,
                "uniprot": acc,
                "pdb_id": pdb_id,
                "method": method,
                "resolution": resolution,
                "n_drug_like": len(drug_like),
                "ligands": drug_like,
                "best_ligand": best_lig,
                "best_ligand_atoms": best_n,
            }
            results.append(entry)
            flag = "SINGLE" if len(drug_like) == 1 else f"{len(drug_like)}-lig"
            click.echo(
                f"{target_id}\t{pdb_id}\t{method}\t{resolution}\t{flag}\t"
                f"best={best_lig}({best_n})\tall={drug_like}"
            )

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info("wrote %d scan hits to %s", len(results), out)


@cli.command()
@click.option("--repeat", type=int, default=64, help="Batch size per sample() call.")
@click.option("--n-calls", type=int, default=3, help="sample() calls per target.")
@click.option("--target-min", type=int, default=150, help="Stop early once this many unique valid.")
@click.option("--seed", type=int, default=42)
@click.option("--only", default=None, help="Comma-separated target_ids to (re)generate; default all.")
@click.option("--targets", "targets_path", default=None,
              help="Targets JSON to read (default data/dock/targets.json).")
def generate(repeat, n_calls, target_min, seed, only, targets_path):
    """Sample candidate SMILES for each target in the targets JSON.

    Skips any target whose candidate file already exists (idempotent/resumable, and
    preserves candidate pools already docked in dock_scores.csv)."""
    import torch

    from scripts.sample_helpers import load_model, load_protein_embeddings, sample

    tj = Path(targets_path) if targets_path else TARGETS_JSON
    if not tj.exists():
        raise click.ClickException(f"{tj} not found — run scan + write targets.json first")
    targets = json.loads(tj.read_text())
    target_ids = [t["target_id"] for t in targets]
    if only:
        wanted = {s.strip() for s in only.split(",")}
        target_ids = [t for t in target_ids if t in wanted]
    log.info("generating for %d targets from %s: %s", len(target_ids), tj, target_ids)

    device = torch.device("mps")
    torch.manual_seed(seed)
    protein_embeddings = load_protein_embeddings(EMB_PATH)
    model, fpindex, rxn_matrix = load_model(CKPT_PATH, None, device)
    log.info("model loaded on %s", device)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    for target_id in target_ids:
        if (CANDIDATES_DIR / f"{target_id}.txt").exists():
            log.info("skip %s: candidate file already exists", target_id)
            continue
        if target_id not in protein_embeddings:
            log.warning("skip %s: not in embeddings", target_id)
            continue
        t0 = time.time()
        smiles_set: set[str] = set()
        for call in range(n_calls):
            info, _ = sample(
                target_id, model, fpindex, rxn_matrix,
                protein_embeddings, device, repeat=repeat,
            )
            for _i, d in info.items():
                smi = d.get("smiles")
                if smi:
                    smiles_set.add(smi)
            log.info(
                "%s call %d/%d: %d unique valid so far (%.1fs)",
                target_id, call + 1, n_calls, len(smiles_set), time.time() - t0,
            )
            if len(smiles_set) >= target_min:
                break
        out_path = CANDIDATES_DIR / f"{target_id}.txt"
        with open(out_path, "w") as f:
            for smi in sorted(smiles_set):
                f.write(smi + "\n")
        log.info(
            "WROTE %s: %d unique valid SMILES (%.1fs total)",
            out_path, len(smiles_set), time.time() - t0,
        )


if __name__ == "__main__":
    cli()
