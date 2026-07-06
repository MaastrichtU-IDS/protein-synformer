# Docking setup (Task 1 — retired 2026-07-06)

**Tool: smina** 2020.12.10 (conda-forge, based on AutoDock Vina 1.1.2), at
`~/miniforge3/envs/dock/bin/smina`.

## How it was installed (no Homebrew/conda previously on this arm64 Mac)
```bash
curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh
bash miniforge.sh -b -p "$HOME/miniforge3"          # batch, isolated, no shell-init changes
"$HOME/miniforge3/bin/conda" create -y -n dock -c conda-forge smina
```
(Prior attempts that FAILED on this env, for the record: no smina pip wheel; AutoDock Vina
python bindings won't build — Boost libs absent + setup.py ignores env overrides; Meeko
`mk_prepare_receptor` fails on H-placement. A prebuilt **Vina 1.2.6 arm64 CLI binary** also
runs, kept at `.tools/vina` as a fallback, but Vina needs PDBQT + manual box.)

## Why smina (vs Vina) — the clean path
smina reads **PDB/SDF directly (no PDBQT prep)** and **autoboxes from a reference ligand**,
so no receptor PDBQT prep and no manual box math. This is the plan's original design.

## Working command template
```bash
$SMINA -r receptor.pdb -l ligand.sdf --autobox_ligand ref_ligand.pdb \
       --exhaustiveness 8 --seed <S> -o out.sdf
# best affinity is mode 1 in the printed table (kcal/mol, lower = stronger)
```
- Receptor: protein ATOM records from the holo PDB → `receptor.pdb`.
- `ref_ligand`: the co-crystal ligand (HETATM) → defines the autobox.
- Ligand to dock: RDKit-embedded 3D SDF from a SMILES (`AddHs`+`EmbedMolecule`+MMFF).

## Redock sanity
Biotin into the 1STP (streptavidin) pocket, smina autobox, exhaustiveness 8:
**affinity −7.4 kcal/mol**, ran end-to-end. Pose-RMSD vs crystal was NOT validated (the
quick check hit an atom-count mismatch; biotin's flexible tail is also a hard redock) →
**Task 3 should add a proper `Chem.rdMolAlign.GetBestRMS` redock check on a cleaner target.**

## Implications for the plan
- Reverts to **smina autobox** (Task 3 uses this template, not Vina/PDBQT). `box_from_coords`
  (Task 2) is not needed for smina autobox — keep it only as a tested utility.
- `data/**` and `.tools/` are gitignored; smina lives outside the repo (conda env).
