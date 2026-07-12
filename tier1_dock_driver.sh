#!/usr/bin/env bash
# Tier-1 calibration docking: dock 3 classes (actives/decoys/candidates) x 8 sources into the shared
# 8-pocket panel (own + 7 mismatch). One powered_run per (class,source), parallel cap 4.
set -u
cd ~/pw
export CUDA_VISIBLE_DEVICES=""
export SMINA="$(pwd)/smina.static"
PY=.venv/bin/python
TJSON=data/dock/tier1/panel8.json
WORK=work_tier1
CONC=4
mkdir -p "$WORK" logs/tier1
TIDS=$($PY -c "import json;print(' '.join(t['target_id'] for t in json.load(open('$TJSON'))))")
throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done; }

echo "=== TIER1 DOCK (3 classes x 8 sources, cap $CONC) $(date) ==="
for cls in actives decoys candidates; do
  for tid in $TIDS; do
    throttle
    (
      j="$WORK/${cls}_${tid}"; mkdir -p "$j"
      $PY -m scripts.powered_run --targets "$TJSON" --candidates-dir "data/dock/tier1/$cls" \
        --sources "$tid" --scores "$j/s.csv" --matrix-out "$j/m.json" \
        --n-candidates 25 --top-m 25 --n-refs 0 --skip-af --seed 42 --work-dir "$j/wd" \
        > "logs/tier1/${cls}_${tid}.log" 2>&1
    ) &
  done
done
wait
echo "=== TIER1 DOCK done $(date) ==="
$PY - <<PY
import glob, os, pandas as pd
for cls in ["actives","decoys","candidates"]:
    fs=[f for f in glob.glob(f"$WORK/{cls}_*/s.csv") if os.path.getsize(f)>0]
    df=pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["molecule","pocket"])
    df["cls"]=cls
    df.to_csv(f"data/dock/tier1/{cls}_scores.csv", index=False)
    print(cls, len(df), "rows from", len(fs), "files")
PY
echo "=== TIER1 ALL DONE $(date) ==="
