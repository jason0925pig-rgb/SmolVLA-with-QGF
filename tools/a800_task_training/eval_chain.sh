#!/bin/bash
# Offline validation chain for one task on a800new.
#   $1 = gpu   $2 = run_dir_name   $3 = ds_dir   $4 = repo_id   $5 = train_tag
# Light eval on 20 checkpoints -> replay on the 4 milestones + top-3 light ->
# lexicographic selection. GPU is released between phases by separate calls.
set -u
GPU=$1; RUNDIR=$2; DSDIR=$3; REPO=$4; TAG=$5
R=$HOME/parcel_smolvla/$RUNDIR
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mark() { echo "$(date '+%m-%d %H:%M:%S') $1" >> "$R/PIPELINE.log"; }

CKPTS=$(ls $R/outputs/train_$TAG/checkpoints/ 2>/dev/null | grep -E '^[0-9]{6}$' | sort)
[ -z "$CKPTS" ] && { mark "EVAL_NO_CKPT"; exit 1; }
mark "EVAL_LIGHT_START n=$(echo $CKPTS | wc -w)"

ARGS=""
for c in $CKPTS; do ARGS="$ARGS --ckpt $c"; done
CUDA_VISIBLE_DEVICES=$GPU $HOME/parcel_env/bin/python $HOME/offline_eval.py \
  --run-dir $R --ds-name $DSDIR --repo-id $REPO --train-tag $TAG \
  --mode light $ARGS > $R/eval_light.log 2>&1
[ $? -ne 0 ] && { mark "EVAL_LIGHT_FAIL"; tail -5 $R/eval_light.log; exit 1; }
mark "EVAL_LIGHT_DONE"

# pick replay candidates: 4 milestones + best-3 by light flow loss
CAND=$($HOME/parcel_env/bin/python - "$R/validation_reports" <<'PY'
import json, sys
from pathlib import Path
rep = Path(sys.argv[1])
rows = []
for f in rep.glob("light_*.json"):
    d = json.loads(f.read_text())
    rows.append((d.get("val_flow_loss", 9e9), d["checkpoint"]))
rows.sort()
best3 = [s for _, s in rows[:3]]
miles = [s for s in ["005000", "010000", "015000", "020000"]
         if (rep / f"light_{s}.json").exists()]
print(" ".join(sorted(set(best3 + miles))))
PY
)
mark "EVAL_REPLAY_START cands=$CAND"
ARGS=""
for c in $CAND; do ARGS="$ARGS --ckpt $c"; done
CUDA_VISIBLE_DEVICES=$GPU $HOME/parcel_env/bin/python $HOME/offline_eval.py \
  --run-dir $R --ds-name $DSDIR --repo-id $REPO --train-tag $TAG \
  --mode replay $ARGS > $R/eval_replay.log 2>&1
[ $? -ne 0 ] && { mark "EVAL_REPLAY_FAIL"; tail -5 $R/eval_replay.log; exit 1; }
mark "EVAL_REPLAY_DONE"

$HOME/parcel_env/bin/python $HOME/select_ckpt.py $R/validation_reports > $R/select.log 2>&1
BEST=$($HOME/parcel_env/bin/python -c "import json;print(json.load(open('$R/validation_reports/selected.json'))['best'])" 2>/dev/null)
mark "EVAL_CHAIN_DONE best=$BEST"
echo "BEST=$BEST"
