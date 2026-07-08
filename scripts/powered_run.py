"""Powered specificity run: crystal + AlphaFold docking arms over ~20 targets, using a
full N×N all-pairs matrix (every target's top-M docked into EVERY prepped pocket) for the
mismatch control. Reuses scripts.dock_select helpers and synformer.dock.
Idempotent/resumable; intended for an in-session background run."""
from __future__ import annotations

import csv
import json
import os
import pathlib

import click
import pandas as pd
import torch

from scripts.dock_select import (
    MASKED_CKPT,
    _import_dock,
    _import_load_model,
    _load_candidates,
    _load_known_ligands,
    _load_random_real,
    _load_scores_table,
    select_topm_for_target,
)
from synformer.dock.af_receptor import prepare_af_target

COLUMNS = ["target", "pocket", "molecule", "source", "score"]


def _done_pairs(path: str) -> set:
    """(molecule, pocket) pairs already scored — the idempotency key. Built directly from
    the CSV so it does not depend on dock_select's internal key ordering."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    df = pd.read_csv(path)
    return set(zip(df.molecule, df.pocket)) if len(df) else set()


def _append(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in COLUMNS})
        fh.flush()


def _dock_into(dock_fn, spec, smiles, seed, target, pocket, source, scores_csv, done):
    if (smiles, pocket) in done:
        return
    score = dock_fn(spec, smiles, seed=seed)
    _append(
        scores_csv,
        {"target": target, "pocket": pocket, "molecule": smiles, "source": source, "score": score},
    )
    done.add((smiles, pocket))


@click.command()
@click.option("--targets", default="data/dock/powered_targets.json")
@click.option("--scores", default="data/dock/dock_scores.csv")
@click.option("--af-scores", default="data/dock/dock_scores_af.csv")
@click.option("--matrix-out", default="data/dock/matrix_targets.json")
@click.option("--af-quality-out", default="data/dock/af_quality.json")
@click.option("--n-candidates", default=150, type=int)
@click.option("--n-refs", default=30, type=int)
@click.option("--top-m", default=10, type=int)
@click.option("--seed", default=42, type=int)
@click.option(
    "--limit-targets",
    default=None,
    type=int,
    help="Limit number of SOURCE targets/pockets actually prepped/docked this run (dry-run). "
    "A genuine full run is invoked with no limit.",
)
def main(targets, scores, af_scores, matrix_out, af_quality_out, n_candidates, n_refs, top_m, seed, limit_targets):
    dock_fn, prepare_target, _stm, _mm = _import_dock()
    device = torch.device("cpu")
    # The generation model + fingerprint index are ONLY needed to enumerate the random-REAL
    # pool for own-pocket docking. If the checkpoint is unavailable (e.g. a remote box where
    # own-pocket docking is already cached and only the mismatch/AF phases remain), skip the
    # load; the mismatch + AF phases and top-M selection need only the candidate files + the
    # score CSV. `fpindex is None` disables random-REAL enumeration (own-pocket must be cached).
    fpindex = None
    try:
        load_model = _import_load_model()
        _model, fpindex, _rxn = load_model(MASKED_CKPT, None, device)
    except (FileNotFoundError, OSError, ImportError) as e:
        # FileNotFoundError/OSError: checkpoint/fpindex absent. ImportError: the generation
        # model stack (omegaconf/torch-lightning/synformer.models) isn't installed (e.g. a
        # remote docking-only box). Either way, fall back to cached mode — the own-pocket
        # docking must already be present (guarded below), and mismatch + AF need no model.
        print(f"generation model unavailable ({type(e).__name__}: {e}); own-pocket docking "
              f"assumed cached — running mismatch + AF phases only (random-REAL disabled)", flush=True)

    tgts_all = json.load(open(targets))
    tgts = tgts_all[:limit_targets] if limit_targets else tgts_all
    if limit_targets:
        print(f"--limit-targets {limit_targets} -> processing {len(tgts)} target(s) this run", flush=True)

    # crystal receptor prep for the targets processed this run (own pockets == mismatch pockets)
    holo = {}
    for t in tgts:
        tid = t["target_id"]
        try:
            holo[tid] = prepare_target(
                t["pdb_id"], f"boltz_out/pw/holo/{tid}", ligand_resname=t["ligand_resname"]
            )
        except Exception as e:
            print(f"  prepare_target FAILED {tid}: {e} — skip", flush=True)
            holo[tid] = None
    ok = [t for t in tgts if holo[t["target_id"]] is not None]
    ok_ids = [t["target_id"] for t in ok]
    json.dump({"targets": ok_ids, "mode": "all_pairs", "seed": seed}, open(matrix_out, "w"), indent=2)
    print(f"all-pairs matrix targets (N={len(ok_ids)}): {ok_ids}", flush=True)

    # ---- crystal own-pocket docking + top-M selection ----
    done = _done_pairs(scores)
    top_m_smiles = {}
    for i, t in enumerate(ok):
        tid = t["target_id"]
        spec = holo[tid]
        cands = _load_candidates(tid, n_candidates)
        if fpindex is not None:
            # full own-pocket docking (candidates + known + random)
            knowns, _avail, _used = _load_known_ligands(tid, n_refs, seed)
            randoms = _load_random_real(n_refs, seed, i, fpindex)
            for smi in cands:
                _dock_into(dock_fn, spec, smi, seed, tid, tid, "candidate", scores, done)
            for smi in knowns:
                _dock_into(dock_fn, spec, smi, seed, tid, tid, "known", scores, done)
            for smi in randoms:
                _dock_into(dock_fn, spec, smi, seed, tid, tid, "random", scores, done)
        else:
            # cached mode (no generation model): own-pocket must already be docked
            if not any((smi, tid) in done for smi in cands):
                raise RuntimeError(
                    f"{tid}: no cached own-pocket candidate scores and no model to generate — "
                    f"cannot compute top-M. Provide the checkpoint or a populated dock_scores.csv."
                )
        tbl = _load_scores_table(pathlib.Path(scores))
        own = {smi: tbl.get((smi, tid), float("nan")) for smi in cands}
        top_m_smiles[tid] = select_topm_for_target(own, top_m)
        print(f"  {tid}: own-pocket ready, top-{len(top_m_smiles[tid])} selected", flush=True)

    # ---- crystal all-pairs mismatch: every target's top-M into EVERY prepped pocket ----
    # (pk == tid, i.e. the diagonal, is already covered by own-pocket docking above and
    # will idempotent-skip via `done` — kept simple rather than special-cased.)
    for t in ok:
        tid = t["target_id"]
        for pk in ok_ids:
            spec_pk = holo[pk]
            for smi in top_m_smiles[tid]:
                _dock_into(dock_fn, spec_pk, smi, seed, tid, pk, "candidate", scores, done)
        print(f"  {tid}: crystal all-pairs mismatch done", flush=True)

    # ---- AF arm: AF-render ALL prepped pockets, dock every source's top-M into every AF
    # pocket whose prep succeeded ----
    af_done = _done_pairs(af_scores)
    af_pockets = ok_ids
    af_spec = {}
    for pk in af_pockets:
        acc = pk.split("_")[0]
        try:
            r = prepare_af_target(acc, holo[pk], f"boltz_out/pw/af/{pk}")
            af_spec[pk] = r
            print(f"  AF {pk}: CA-RMSD {r.ca_rmsd:.2f} pocket-pLDDT {r.pocket_plddt:.1f}", flush=True)
        except Exception as e:
            print(f"  AF prep FAILED {pk}: {e} — skip pocket", flush=True)
            af_spec[pk] = None
    af_ok_pockets = [pk for pk in af_pockets if af_spec.get(pk) is not None]
    for t in ok:
        tid = t["target_id"]
        for pk in af_ok_pockets:
            for smi in top_m_smiles[tid]:
                _dock_into(dock_fn, af_spec[pk].spec, smi, seed, tid, pk, "candidate", af_scores, af_done)
        print(f"  {tid}: AF arm done", flush=True)

    # record AF prep quality
    json.dump(
        {
            pk: (None if r is None else {"ca_rmsd": r.ca_rmsd, "pocket_plddt": r.pocket_plddt})
            for pk, r in af_spec.items()
        },
        open(af_quality_out, "w"),
        indent=2,
    )
    print("done.", flush=True)


if __name__ == "__main__":
    main()
