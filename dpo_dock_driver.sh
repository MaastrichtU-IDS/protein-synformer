#!/usr/bin/env bash
# SP-DPO pilot docking driver. Adapts pocket_dock_driver.sh for the per-molecule
# specificity pairs: dock ALL 48 generated mols per train target into own + every
# other train pocket (all-pairs within the 10 train targets), candidates only, no AF.
# Phase 1: own-pocket per target (parallel) -> top-M=48 = all mols.
# Phase 2: per-source mismatch, seeded with merged own scores (own loop idempotent-skips).
set -u
cd ~/pw
export CUDA_VISIBLE_DEVICES=""
export SMINA="$(pwd)/smina.static"
PY=.venv/bin/python
CAND=data/dock/dpo/candidates
TJSON=data/dock/dpo/train10.json
WORK=work_dpo
CONC=4                       # concurrent docking workers (4 x ~7 cores ~= 28/32)
NCAND=48                     # dock all 48 generated mols
TOPM=48                      # keep all for mismatch (per-molecule specificity needs every mol)
mkdir -p "$WORK" logs/dpo

TIDS=$($PY -c "import json;print(' '.join(t['target_id'] for t in json.load(open('$TJSON'))))")
echo "targets: $TIDS"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done; }

# ---------- Phase 1: own-pocket per target ----------
echo "=== PHASE 1: own-pocket (parallel, cap $CONC) $(date) ==="
for tid in $TIDS; do
  throttle
  (
    j="$WORK/$tid"; mkdir -p "$j"
    $PY -c "import json;d=[t for t in json.load(open('$TJSON')) if t['target_id']=='$tid'];json.dump(d,open('$j/tgt.json','w'))"
    $PY -m scripts.powered_run --targets "$j/tgt.json" --candidates-dir "$CAND" \
      --scores "$j/own.csv" --matrix-out "$j/m.json" \
      --n-candidates $NCAND --top-m $TOPM --n-refs 0 --skip-af --seed 42 \
      --work-dir "$j/wd" \
      > "logs/dpo/own_$tid.log" 2>&1
  ) &
done
wait
echo "=== PHASE 1 done $(date) ==="

OWN=data/dock/dpo/own_master.csv
$PY - <<PY
import glob, os, pandas as pd
fs=[f for f in glob.glob("$WORK/*/own.csv") if os.path.getsize(f)>0]
df=pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["molecule","pocket"])
df.to_csv("$OWN", index=False)
print("own_master rows:", len(df), "from", len(fs), "targets")
PY

# ---------- Phase 2: per-source mismatch (all other train pockets) ----------
echo "=== PHASE 2: mismatch per source (parallel, cap $CONC) $(date) ==="
for tid in $TIDS; do
  throttle
  (
    j="$WORK/$tid"
    cp "$OWN" "$j/mis.csv"                 # seed own-pocket -> own loop idempotent-skips
    $PY -m scripts.powered_run --targets "$TJSON" --candidates-dir "$CAND" \
      --sources "$tid" --scores "$j/mis.csv" --matrix-out "$j/m2.json" \
      --n-candidates $NCAND --top-m $TOPM --n-refs 0 --skip-af --seed 42 \
      --work-dir "$j/wd" \
      > "logs/dpo/mis_$tid.log" 2>&1
  ) &
done
wait
echo "=== PHASE 2 done $(date) ==="

$PY - <<PY
import glob, os, pandas as pd
fs=[f for f in glob.glob("$WORK/*/mis.csv") if os.path.getsize(f)>0]
df=pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["molecule","pocket"])
df.to_csv("data/dock/dpo/dpo_dock_scores.csv", index=False)
print("dpo_dock_scores rows:", len(df), "from", len(fs), "files")
PY
echo "=== ALL DONE $(date) ==="
