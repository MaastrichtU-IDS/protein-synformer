#!/usr/bin/env bash
# SP-DPO held-out eval docking. For each held-out target, dock the UNION of its base+DPO
# generated pools into a SHARED panel = its own pocket + the 10 TRAIN pockets (mismatch).
# One powered_run process per held-out source (parallel, cap 4). Origin (base vs DPO) is
# recovered later by dpo_eval via per-pool .smi membership, NOT the source column.
set -u
cd ~/pw
export CUDA_VISIBLE_DEVICES=""
export SMINA="$(pwd)/smina.static"
PY=.venv/bin/python
CAND=data/dock/dpo/heldout/candidates
TRAIN=data/dock/dpo/train10.json
WORK=work_dpo_eval
CONC=4
mkdir -p "$WORK" logs/dpo/eval

HELD="O75716_WT P28223_WT P15090_WT P0C559_WT"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done; }

echo "=== EVAL DOCK: held-out own + 10 train panel (parallel, cap $CONC) $(date) ==="
for tid in $HELD; do
  throttle
  (
    j="$WORK/$tid"; mkdir -p "$j"
    # 11-target json: this held-out target + the 10 train pockets (== own + mismatch panel)
    $PY -c "import json;t=[x for x in json.load(open('data/dock/dpo/heldout4.json')) if x['target_id']=='$tid'];t+=json.load(open('$TRAIN'));json.dump(t,open('$j/tgt.json','w'))"
    $PY -m scripts.powered_run --targets "$j/tgt.json" --candidates-dir "$CAND" \
      --sources "$tid" --scores "$j/eval.csv" --matrix-out "$j/m.json" \
      --n-candidates 200 --top-m 200 --n-refs 0 --skip-af --seed 42 \
      --work-dir "$j/wd" \
      > "logs/dpo/eval/$tid.log" 2>&1
  ) &
done
wait
echo "=== EVAL DOCK done $(date) ==="

$PY - <<PY
import glob, os, pandas as pd
fs=[f for f in glob.glob("$WORK/*/eval.csv") if os.path.getsize(f)>0]
df=pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["molecule","pocket"])
df.to_csv("data/dock/dpo/heldout/eval_scores.csv", index=False)
print("eval_scores rows:", len(df), "from", len(fs), "files")
PY
echo "=== EVAL ALL DONE $(date) ==="
