#!/usr/bin/env bash
# Tier-2 docking: dock the measured-selectivity compound set into all 6 pockets, 4 shards parallel.
# Each shard = a slice of dock_set as KIT's candidate file; source=KIT docks it into own(KIT)+5 mismatch
# = all 6 pockets. Origin/measured values recovered later by SMILES in tier2_analyze.
set -u
cd ~/pw
export CUDA_VISIBLE_DEVICES=""
export SMINA="$(pwd)/smina.static"
PY=.venv/bin/python
TJSON=data/dock/tier2/panel6.json
WORK=work_tier2
mkdir -p "$WORK" logs/tier2b
echo "=== TIER2 DOCK (4 shards x 6 pockets) $(date) ==="
for i in 0 1 2 3; do
  (
    j="$WORK/shard$i"; mkdir -p "$j"
    $PY -m scripts.powered_run --targets "$TJSON" --candidates-dir "data/dock/tier2/shard$i/cand" \
      --sources "P10721_WT" --scores "$j/s.csv" --matrix-out "$j/m.json" \
      --n-candidates 200 --top-m 200 --n-refs 0 --skip-af --seed 42 --work-dir "$j/wd" \
      > "logs/tier2b/shard$i.log" 2>&1
  ) &
done
wait
echo "=== TIER2 DOCK done $(date) ==="
$PY - <<PY
import glob,os,pandas as pd
fs=[f for f in glob.glob("$WORK/shard*/s.csv") if os.path.getsize(f)>0]
df=pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["molecule","pocket"])
df.to_csv("data/dock/tier2/dock_scores.csv",index=False)
print("dock_scores rows:",len(df),"from",len(fs),"shards")
PY
echo "=== TIER2 DOCK ALL DONE $(date) ==="
