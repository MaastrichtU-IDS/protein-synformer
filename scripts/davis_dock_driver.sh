#!/usr/bin/env bash
# Tier-3 DAVIS docking: dock the 68 DAVIS drugs into the 15 kinase pockets, 4 shards parallel.
set -u
cd ~/pw
export CUDA_VISIBLE_DEVICES=""
export SMINA="$(pwd)/smina.static"
PY=.venv/bin/python
TJSON=data/dock/davis/panelN.json
WORK=work_davis
mkdir -p "$WORK" logs/davis
echo "=== DAVIS DOCK (4 shards x 15 pockets) $(date) ==="
for i in 0 1 2 3; do
  (
    j="$WORK/shard$i"; mkdir -p "$j"
    $PY -m scripts.powered_run --targets "$TJSON" --candidates-dir "data/dock/davis/shard$i/cand" \
      --sources "Q16566_WT" --scores "$j/s.csv" --matrix-out "$j/m.json" \
      --n-candidates 200 --top-m 200 --n-refs 0 --skip-af --seed 42 --work-dir "$j/wd" \
      > "logs/davis/shard$i.log" 2>&1
  ) &
done
wait
echo "=== DAVIS DOCK done $(date) ==="
$PY - <<PY
import glob,os,pandas as pd
fs=[f for f in glob.glob("$WORK/shard*/s.csv") if os.path.getsize(f)>0]
df=pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["molecule","pocket"])
df.to_csv("data/dock/davis/dock_scores.csv",index=False)
print("dock_scores rows:",len(df),"from",len(fs),"shards")
PY
echo "=== DAVIS DOCK ALL DONE $(date) ==="
