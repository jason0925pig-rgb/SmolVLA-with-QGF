#!/bin/bash
# Handoff section 11: single-critic visual IQL, TASK-PARAMETERIZED.
#
# Task identity comes from the environment; there is no default task.  Every
# hyperparameter below is the water-bottle configuration fixed by handoff
# section 11 and is NOT parameterized: the only per-task inputs are the paths,
# the run id and the episode range.
#
# Required environment (fail loudly, never guessed):
#   QGF_SSD_ROOT       e.g. /opt/qgf_real_robot
#   QGF_TASK_KEY       e.g. red_parcel                     (provenance only)
#   QGF_RUN_ID         e.g. red_parcel_single_q_45_5_20260902
#   QGF_EPISODE_FIRST  e.g. 0
#   QGF_EPISODE_LAST   e.g. 49
# Optional:
#   QGF_FORCE_RETRAIN=1        allow overwriting an existing critic in this run
#   QGF_GPU_PROBE_TIMEOUT=1800 seconds to wait for the training PID to appear on a GPU
#
# Handoff iron rule 1: physical GPU 1 only, always explicit, and the binding is
# proven by an nvidia-smi PID -> GPU UUID -> index snapshot taken WHILE training
# runs, not by the environment variable alone.
set -eu

# ---------------------------------------------------------------- environment
MISSING=0
for _v in QGF_SSD_ROOT QGF_TASK_KEY QGF_RUN_ID QGF_EPISODE_FIRST QGF_EPISODE_LAST; do
  if [ -z "${!_v-}" ]; then
    echo "FATAL: required environment variable $_v is unset or empty." >&2
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "FATAL: refusing to guess a task.  Export the variables above and re-run." >&2
  exit 1
fi

case "$QGF_EPISODE_FIRST" in
  ''|*[!0-9]*) echo "FATAL: QGF_EPISODE_FIRST='$QGF_EPISODE_FIRST' is not a non-negative integer." >&2; exit 1 ;;
esac
case "$QGF_EPISODE_LAST" in
  ''|*[!0-9]*) echo "FATAL: QGF_EPISODE_LAST='$QGF_EPISODE_LAST' is not a non-negative integer." >&2; exit 1 ;;
esac
if [ "$QGF_EPISODE_LAST" -lt "$QGF_EPISODE_FIRST" ]; then
  echo "FATAL: QGF_EPISODE_LAST ($QGF_EPISODE_LAST) < QGF_EPISODE_FIRST ($QGF_EPISODE_FIRST)." >&2
  exit 1
fi
N_EPISODES=$(( QGF_EPISODE_LAST - QGF_EPISODE_FIRST + 1 ))

# ---------------------------------------------------------------- fixed values
# Handoff sections 8 and 11.  Do not parameterize these away.
TRAIN_EPISODES=45
VAL_EPISODES=5
EPOCHS=80
BATCH_SIZE=16
LR=3e-4
WEIGHT_DECAY=1e-4
GAMMA=0.99
EXPECTILE=0.7
POLYAK=0.005
D_MODEL=256
LAYERS=3
HEADS=4
DROPOUT=0.1
SEED=20260814
ENSEMBLE_SIZE=1
PHYSICAL_GPU=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# nvidia-smi must always see BOTH physical cards, so every call below strips the
# CUDA masking variables.  Otherwise "index 1" could silently mean the masked
# logical device and the whole GPU proof would be circular.
smi() { env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER nvidia-smi "$@"; }

# ---------------------------------------------------------------- paths
E="$QGF_SSD_ROOT/envs/visual_iql_py310"
PY="$E/bin/python"
QGF_REPO="$QGF_SSD_ROOT/repos/SmolVLA-with-QGF"
TRAIN_PY="$QGF_REPO/qgf/scripts/train_real_robot_visual_iql.py"
QGF_RUN="$QGF_SSD_ROOT/runs/$QGF_RUN_ID"
FEATURES="$QGF_RUN/features"
SPLIT_FILE="$QGF_RUN/manifest/episode_split_${TRAIN_EPISODES}_${VAL_EPISODES}.json"
OUT="$QGF_RUN/outputs/single_qcritic"

echo "task              : $QGF_TASK_KEY"
echo "run               : $QGF_RUN"
echo "episodes          : ${QGF_EPISODE_FIRST}..${QGF_EPISODE_LAST}  (${N_EPISODES})"
echo "split             : ${TRAIN_EPISODES} train / ${VAL_EPISODES} val"
echo "physical GPU      : $PHYSICAL_GPU  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo

# --- the split is 45/5, so the cohort must be exactly 50 episodes -------------
if [ "$N_EPISODES" -ne $(( TRAIN_EPISODES + VAL_EPISODES )) ]; then
  echo "FATAL: episode range gives $N_EPISODES episodes but the fixed split needs $(( TRAIN_EPISODES + VAL_EPISODES ))." >&2
  exit 1
fi

# --- the pieces produced by the earlier stages must already be there ----------
for p in "$PY" "$TRAIN_PY"; do
  if [ ! -f "$p" ]; then
    echo "FATAL: missing $p" >&2
    exit 1
  fi
done
if [ ! -x "$PY" ]; then
  echo "FATAL: $PY is not executable." >&2
  exit 1
fi
if [ ! -d "$QGF_RUN" ]; then
  echo "FATAL: run directory does not exist: $QGF_RUN" >&2
  echo "       (a wrong QGF_RUN_ID must not silently create a fresh empty run)" >&2
  exit 1
fi
if [ ! -f "$SPLIT_FILE" ]; then
  echo "FATAL: missing split file: $SPLIT_FILE" >&2
  exit 1
fi

# --- the 45/5 compatibility patch must be in the checked-out code -------------
# Handoff section 11: the two --expected-* arguments require the section 8 patch
# to be committed.  Never delete the split assertion to work around a missing patch.
if ! grep -q -- "--expected-train-episodes" "$TRAIN_PY"; then
  echo "FATAL: $TRAIN_PY has no --expected-train-episodes; the section 8 patch is not applied." >&2
  echo "       Apply patch_45_5.py and commit it.  Do NOT drop the split check instead." >&2
  exit 1
fi

# --- refuse to start on a split that cannot select a Q checkpoint -------------
"$PY" - "$SPLIT_FILE" "$TRAIN_EPISODES" "$VAL_EPISODES" <<'PYGATE'
import io
import json
import sys

split = json.load(io.open(sys.argv[1], encoding="utf-8"))
want_tr, want_va = int(sys.argv[2]), int(sys.argv[3])
tr = [int(x) for x in split["train_episode_indices"]]
va = [int(x) for x in split["val_episode_indices"]]
tc = split.get("train_outcome_counts", {})
vc = split.get("val_outcome_counts", {})
bad = []
if len(tr) != want_tr:
    bad.append("train episodes %d != %d" % (len(tr), want_tr))
if len(va) != want_va:
    bad.append("val episodes %d != %d" % (len(va), want_va))
shared = sorted(set(tr) & set(va))
if shared:
    bad.append("train/val share episodes %s" % shared)
if len(set(tr) | set(va)) != want_tr + want_va:
    bad.append("union covers %d distinct episodes" % len(set(tr) | set(va)))
for side, counts in (("train", tc), ("val", vc)):
    for outcome in ("success", "failure"):
        if int(counts.get(outcome, 0)) <= 0:
            bad.append("%s has no %s episodes (counts=%s)" % (side, outcome, counts))
print("  split strategy : %s  seed %s" % (split.get("strategy"), split.get("seed")))
print("  train outcomes : %s" % tc)
print("  val   outcomes : %s" % vc)
print("  val episodes   : %s" % sorted(va))
if bad:
    print("FATAL: split is unusable:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("  split gate     : OK")
PYGATE

# --- refuse to start if the feature cache is incomplete -----------------------
if [ ! -d "$FEATURES" ]; then
  echo "FATAL: feature directory does not exist: $FEATURES" >&2
  exit 1
fi
N=$(find "$FEATURES" -maxdepth 1 -type f -name '*.pt' | wc -l)
if [ "$N" -ne "$N_EPISODES" ]; then
  echo "FATAL: expected $N_EPISODES feature caches, found $N.  Not starting training." >&2
  exit 1
fi
i="$QGF_EPISODE_FIRST"
while [ "$i" -le "$QGF_EPISODE_LAST" ]; do
  f=$(printf '%s/episode_%06d.pt' "$FEATURES" "$i")
  if [ ! -f "$f" ]; then
    echo "FATAL: missing feature cache $f" >&2
    exit 1
  fi
  i=$(( i + 1 ))
done
echo "feature caches: $N/$N_EPISODES  (episode_%06d.pt complete over ${QGF_EPISODE_FIRST}..${QGF_EPISODE_LAST})"

# --- refuse to share the card with LingBot ------------------------------------
if pgrep -af lingbot | grep -v pgrep > /dev/null 2>&1; then
  echo "FATAL: a lingbot process is running.  Stop it or re-check the GPU policy." >&2
  pgrep -af lingbot | grep -v pgrep
  exit 1
fi

# --- physical GPU 1 must exist ------------------------------------------------
GPU1_UUID=$(smi --query-gpu=uuid --format=csv,noheader --id="$PHYSICAL_GPU" 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$GPU1_UUID" ]; then
  echo "FATAL: nvidia-smi does not report a physical GPU $PHYSICAL_GPU." >&2
  exit 1
fi
echo "physical GPU $PHYSICAL_GPU uuid: $GPU1_UUID"

# --- never silently overwrite a finished critic -------------------------------
mkdir -p "$QGF_RUN"/{outputs,logs,commands,environment}
mkdir -p "$OUT"
EXISTING=$(find "$OUT" -maxdepth 1 -type f -name 'critic_member_*.pt' | wc -l)
if [ "$EXISTING" -ne 0 ] && [ "${QGF_FORCE_RETRAIN:-0}" != "1" ]; then
  echo "FATAL: $OUT already holds $EXISTING critic_member_*.pt." >&2
  echo "       Handoff section 12 forbids overwriting a previous run's outputs." >&2
  echo "       Use a new QGF_RUN_ID, or export QGF_FORCE_RETRAIN=1 to redo this run deliberately." >&2
  exit 1
fi

# ---------------------------------------------------------------- command
CMD=(
  "$PY" "$TRAIN_PY"
  --data-dir "$FEATURES"
  --split-file "$SPLIT_FILE"
  --output-dir "$OUT"
  --ensemble-size "$ENSEMBLE_SIZE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --gamma "$GAMMA"
  --expectile "$EXPECTILE"
  --polyak "$POLYAK"
  --d-model "$D_MODEL"
  --layers "$LAYERS"
  --heads "$HEADS"
  --dropout "$DROPOUT"
  --seed "$SEED"
  --device cuda
  --expected-train-episodes "$TRAIN_EPISODES"
  --expected-val-episodes "$VAL_EPISODES"
)

{
  echo "# executed $(date -u +%FT%TZ) on $(hostname)"
  echo "# task=$QGF_TASK_KEY run=$QGF_RUN_ID episodes=${QGF_EPISODE_FIRST}..${QGF_EPISODE_LAST}"
  echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (physical GPU $PHYSICAL_GPU, uuid $GPU1_UUID)"
  echo "# HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  echo "# checkpoint selection = lowest validation TD loss, never simply epoch $EPOCHS"
  echo "# uncertainty gate disabled by ensemble-size $ENSEMBLE_SIZE (uncertainty_scale=0.0 at deploy)"
  printf '%q ' "${CMD[@]}"
  echo
} > "$QGF_RUN/commands/train_single_qcritic.cmd"

LOG="$QGF_RUN/logs/train_single_qcritic.log"
BEFORE="$QGF_RUN/environment/nvidia_smi_train_before.txt"
DURING="$QGF_RUN/environment/nvidia_smi_train_during.txt"
AFTER="$QGF_RUN/environment/nvidia_smi_train_after.txt"
SAMPLES="$QGF_RUN/environment/nvidia_smi_train_during_samples.txt"
BINDING="$QGF_RUN/environment/train_gpu_binding.txt"
VIOLATION="$QGF_RUN/environment/train_gpu_violation.flag"
rm -f "$VIOLATION"

{
  echo "# before training $(date -u +%FT%TZ)"
  smi
  echo
  smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
  echo
  smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
} > "$BEFORE"

# ---------------------------------------------------------------- run
echo "=== training start $(date -u +%FT%TZ) ==="
printf '%q ' "${CMD[@]}"; echo

: > "$LOG"   # pre-create so the follower below cannot lose the race
"${CMD[@]}" > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "training PID: $TRAIN_PID"

TAIL_PID=""
WATCH_PID=""
cleanup() {
  if [ -n "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null || true; fi
  if [ -n "$WATCH_PID" ]; then kill "$WATCH_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

tail -n +1 -f --pid="$TRAIN_PID" "$LOG" &
TAIL_PID=$!

# Resolve the physical GPU index a compute-app PID is sitting on.  Prints the
# index, or nothing when the PID currently owns no CUDA context.
gpu_index_of() {
  local pid="$1"
  local line app_pid uuid idx
  while IFS= read -r line; do
    app_pid=$(printf '%s' "$line" | awk -F, '{gsub(/ /,"",$1); print $1}')
    uuid=$(printf '%s' "$line" | awk -F, '{gsub(/ /,"",$NF); print $NF}')
    case "$app_pid" in ''|*[!0-9]*) continue ;; esac
    if [ "$app_pid" = "$pid" ]; then
      idx=$(smi --query-gpu=index,uuid --format=csv,noheader \
            | awk -F, -v u="$uuid" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2==u) print $1}')
      printf '%s' "$idx"
      return 0
    fi
  done < <(smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null || true)
  return 0
}

# Is $1 the training process or one of its descendants?
is_ours() {
  local pid="$1"
  local hops=0
  local ppid
  while [ "$pid" -gt 1 ] && [ "$hops" -lt 24 ]; do
    if [ "$pid" = "$TRAIN_PID" ]; then
      return 0
    fi
    ppid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null || true)
    if [ -z "$ppid" ]; then
      return 1
    fi
    pid="$ppid"
    hops=$(( hops + 1 ))
  done
  return 1
}

# Snapshot every CUDA process, marking ours vs other, with the physical index.
snapshot_binding() {
  local label="$1"
  local line app_pid uuid idx
  {
    echo "# $label $(date -u +%FT%TZ)"
    while IFS= read -r line; do
      app_pid=$(printf '%s' "$line" | awk -F, '{gsub(/ /,"",$1); print $1}')
      uuid=$(printf '%s' "$line" | awk -F, '{gsub(/ /,"",$NF); print $NF}')
      case "$app_pid" in ''|*[!0-9]*) continue ;; esac
      idx=$(smi --query-gpu=index,uuid --format=csv,noheader \
            | awk -F, -v u="$uuid" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2==u) print $1}')
      if is_ours "$app_pid"; then
        echo "OURS   pid=$app_pid gpu_index=$idx uuid=$uuid"
      else
        echo "other  pid=$app_pid gpu_index=$idx uuid=$uuid"
      fi
    done < <(smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null || true)
  } >> "$BINDING"
}

# --- wait for the training process to actually take a CUDA context -----------
PROBE_TIMEOUT="${QGF_GPU_PROBE_TIMEOUT:-1800}"
WAITED=0
SEEN_INDEX=""
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  SEEN_INDEX=$(gpu_index_of "$TRAIN_PID")
  if [ -n "$SEEN_INDEX" ]; then
    break
  fi
  if [ "$WAITED" -ge "$PROBE_TIMEOUT" ]; then
    break
  fi
  sleep 10
  WAITED=$(( WAITED + 10 ))
done

GPU_PROOF=fail
if [ -n "$SEEN_INDEX" ]; then
  {
    echo "# during training $(date -u +%FT%TZ)  training PID $TRAIN_PID"
    smi
    echo
    smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
    echo
    smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
  } > "$DURING"
  snapshot_binding "first GPU sighting, training PID $TRAIN_PID"
  if [ "$SEEN_INDEX" = "$PHYSICAL_GPU" ]; then
    echo "GPU binding OK: PID $TRAIN_PID -> uuid -> physical GPU index $SEEN_INDEX"
    echo "VERDICT first-sighting: PID $TRAIN_PID on physical GPU index $SEEN_INDEX (expected $PHYSICAL_GPU) OK" >> "$BINDING"
    GPU_PROOF=ok
  else
    echo "FATAL: training PID $TRAIN_PID is on physical GPU index $SEEN_INDEX, expected $PHYSICAL_GPU." >&2
    echo "VERDICT first-sighting: PID $TRAIN_PID on physical GPU index $SEEN_INDEX (expected $PHYSICAL_GPU) VIOLATION" >> "$BINDING"
    touch "$VIOLATION"
    kill "$TRAIN_PID" 2>/dev/null || true
  fi
else
  echo "WARNING: never observed PID $TRAIN_PID on any GPU (waited ${WAITED}s)." >&2
  echo "VERDICT first-sighting: PID $TRAIN_PID never appeared in nvidia-smi compute apps" >> "$BINDING"
fi

# --- keep watching: a late allocation on GPU 0 must not go unnoticed ---------
if [ "$GPU_PROOF" = ok ]; then
  (
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      sleep 300
      kill -0 "$TRAIN_PID" 2>/dev/null || break
      {
        echo "# sample $(date -u +%FT%TZ)"
        smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
      } >> "$SAMPLES"
      idx=$(gpu_index_of "$TRAIN_PID")
      if [ -n "$idx" ] && [ "$idx" != "$PHYSICAL_GPU" ]; then
        echo "VERDICT watchdog $(date -u +%FT%TZ): PID $TRAIN_PID moved to physical GPU index $idx VIOLATION" >> "$BINDING"
        touch "$VIOLATION"
        kill "$TRAIN_PID" 2>/dev/null || true
        break
      fi
    done
  ) &
  WATCH_PID=$!
fi

RC=0
wait "$TRAIN_PID" || RC=$?
cleanup
trap - EXIT

{
  echo "# after training $(date -u +%FT%TZ) rc=$RC"
  smi
  echo
  smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
  echo
  smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
} > "$AFTER"

echo "=== training end $(date -u +%FT%TZ) rc=$RC ==="
ls -la "$OUT" 2>/dev/null | head -20

# ---------------------------------------------------------------- verdict
if [ -f "$VIOLATION" ]; then
  echo "FATAL: the training process used a GPU other than physical GPU $PHYSICAL_GPU; it was killed." >&2
  echo "       See $BINDING" >&2
  exit 1
fi
if [ "$RC" -ne 0 ]; then
  echo "training exited rc=$RC; see $LOG" >&2
  exit "$RC"
fi
if [ "$GPU_PROOF" != ok ]; then
  echo "FATAL: no during-training nvidia-smi proof that the training PID sat on physical GPU $PHYSICAL_GPU." >&2
  echo "       Handoff section 11 requires that evidence; treating this run as unaccepted." >&2
  exit 1
fi
if [ ! -f "$OUT/critic_member_00.pt" ]; then
  echo "FATAL: training returned 0 but $OUT/critic_member_00.pt does not exist." >&2
  exit 1
fi
if [ ! -f "$OUT/training_summary.json" ] || [ ! -f "$OUT/training_input_summary.json" ]; then
  echo "FATAL: training returned 0 but training_summary.json / training_input_summary.json are missing." >&2
  exit 1
fi

echo
echo "=== selected checkpoint (lowest validation TD loss, not simply epoch $EPOCHS) ==="
"$PY" - "$OUT/training_summary.json" <<'PYSUM'
import io
import json
import sys

ts = json.load(io.open(sys.argv[1], encoding="utf-8"))
for m in ts.get("members", []):
    print("  member %s: %s  selected_epoch=%s  selected_val_td_loss=%s" % (
        m.get("member_index"), m.get("path"),
        m.get("selected_epoch"), m.get("selected_val_td_loss")))
PYSUM

echo
echo "training done.  Now run the section 12 acceptance:"
echo "  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU $PY $(dirname "$0")/accept_single_q_task.py"
exit 0
