#!/bin/bash
# Handoff sections 8 + 9: build the aligned manifest and the stratified 45/5
# episode split.  TASK-PARAMETERIZED version of run_build_manifest.sh: every
# task-specific path comes from the environment, nothing is hardcoded to a task.
# Runs ON THE 4090 (192.168.2.110).  Reads local NVMe only; never touches the
# robot and never decodes anything over the network.
#
# Required environment (NO defaults - the script refuses to guess a task):
#   QGF_SSD_ROOT       /opt/qgf_real_robot
#   QGF_TASK_KEY       red_parcel
#   QGF_DATASET_ID     red_parcel_baseline50_20260902
#   QGF_RUN_ID         red_parcel_single_q_45_5_20260902
#   QGF_EPISODE_FIRST  0
#   QGF_EPISODE_LAST   49
#
# Optional overrides (derived from QGF_SSD_ROOT when unset):
#   QGF_ENV_PREFIX     python env prefix   (default $QGF_SSD_ROOT/envs/visual_iql_py310)
#   QGF_REPO_DIR       repo snapshot       (default $QGF_SSD_ROOT/repos/SmolVLA-with-QGF)
#   QGF_STAGING_DIR    dir holding verify_split_45_5.py (default: this script's dir)
#   QGF_FORCE_REBUILD=1  allow rebuilding over an existing manifest in this run dir
#
# Fixed by the handoff, NOT parameterized:
#   CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only, handoff rule 1)
#   45 train / 5 validation, --split-seed 20260814
#   --action-horizon 50 --policy-hz 15 --max-transition-gap-seconds 0.1
set -euo pipefail

FATAL() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------- environment
REQUIRED_VARS=(QGF_SSD_ROOT QGF_TASK_KEY QGF_DATASET_ID QGF_RUN_ID QGF_EPISODE_FIRST QGF_EPISODE_LAST)
MISSING=""
for v in "${REQUIRED_VARS[@]}"; do
  [ -n "${!v-}" ] || MISSING="$MISSING $v"
done
if [ -n "$MISSING" ]; then
  echo "FATAL: required environment variable(s) unset or empty:$MISSING" >&2
  echo "This script is task-parameterized and will not guess a task." >&2
  echo "Example (handoff sections 4 and 9):" >&2
  echo "  export QGF_SSD_ROOT=/opt/qgf_real_robot" >&2
  echo "  export QGF_TASK_KEY=red_parcel" >&2
  echo "  export QGF_DATASET_ID=red_parcel_baseline50_20260902" >&2
  echo "  export QGF_RUN_ID=red_parcel_single_q_45_5_20260902" >&2
  echo "  export QGF_EPISODE_FIRST=0" >&2
  echo "  export QGF_EPISODE_LAST=49" >&2
  exit 2
fi

case "$QGF_SSD_ROOT" in
  /*) ;;
  *) FATAL "QGF_SSD_ROOT must be an absolute path (got '$QGF_SSD_ROOT')" ;;
esac
for v in QGF_TASK_KEY QGF_DATASET_ID QGF_RUN_ID; do
  case "${!v}" in
    *[!A-Za-z0-9._-]*) FATAL "$v='${!v}' may only contain [A-Za-z0-9._-]; it is used as a directory name" ;;
  esac
done
for v in QGF_EPISODE_FIRST QGF_EPISODE_LAST; do
  case "${!v}" in
    ''|*[!0-9]*) FATAL "$v='${!v}' must be a non-negative integer" ;;
  esac
done
[ "$QGF_EPISODE_LAST" -ge "$QGF_EPISODE_FIRST" ] \
  || FATAL "QGF_EPISODE_LAST ($QGF_EPISODE_LAST) < QGF_EPISODE_FIRST ($QGF_EPISODE_FIRST)"

# ------------------------------------------------- fixed, non-negotiable knobs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1        # handoff rule 1: physical GPU 1 ONLY
export HF_HUB_OFFLINE=1
EXPECT_TRAIN=45
EXPECT_VAL=5
EXPECT_EPISODES=$(( EXPECT_TRAIN + EXPECT_VAL ))
ACTION_HORIZON=50
POLICY_HZ=15
MAX_GAP=0.1
SPLIT_SEED=20260814
SPLIT_NAME="episode_split_${EXPECT_TRAIN}_${EXPECT_VAL}.json"

EP_COUNT=$(( QGF_EPISODE_LAST - QGF_EPISODE_FIRST + 1 ))
[ "$EP_COUNT" -eq "$EXPECT_EPISODES" ] || FATAL \
  "episode range ${QGF_EPISODE_FIRST}..${QGF_EPISODE_LAST} yields $EP_COUNT episodes, but the frozen procedure is a ${EXPECT_TRAIN}/${EXPECT_VAL} split over exactly $EXPECT_EPISODES episodes (handoff section 8)"

# --------------------------------------------------------------------- layout
E=${QGF_ENV_PREFIX:-$QGF_SSD_ROOT/envs/visual_iql_py310}
PY=$E/bin/python
QGF_REPO=${QGF_REPO_DIR:-$QGF_SSD_ROOT/repos/SmolVLA-with-QGF}
BUILDER=$QGF_REPO/qgf/scripts/build_real_robot_visual_iql_manifest.py
DATASET_DIR=$QGF_SSD_ROOT/datasets/$QGF_DATASET_ID
QGF_DATA=$DATASET_DIR/raw_episodes
SRC_MAP=$DATASET_DIR/source_episode_map.json
QGF_RUN=$QGF_SSD_ROOT/runs/$QGF_RUN_ID
SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGING=${QGF_STAGING_DIR:-$SELF_DIR}
VERIFY_SPLIT=$STAGING/verify_split_45_5.py
[ -f "$VERIFY_SPLIT" ] || VERIFY_SPLIT=$QGF_REPO/tools/qgf_4090_staging/verify_split_45_5.py

[ -x "$PY" ]        || FATAL "python not found or not executable: $PY (handoff section 7)"
[ -f "$BUILDER" ]   || FATAL "manifest builder missing: $BUILDER"
[ -d "$QGF_DATA" ]  || FATAL "raw episode root missing: $QGF_DATA (stage the cohort first, handoff sections 5-6)"
[ -f "$SRC_MAP" ]   || FATAL "provenance missing: $SRC_MAP (handoff section 5 requires source_episode_map.json)"
[ -f "$VERIFY_SPLIT" ] || FATAL "verify_split_45_5.py not found next to this script nor at $QGF_REPO/tools/qgf_4090_staging/; set QGF_STAGING_DIR"

echo "=== config ==="
echo "  task key    : $QGF_TASK_KEY"
echo "  dataset     : $DATASET_DIR"
echo "  run dir     : $QGF_RUN"
echo "  episodes    : ${QGF_EPISODE_FIRST}..${QGF_EPISODE_LAST}  ($EP_COUNT)"
echo "  split       : ${EXPECT_TRAIN}/${EXPECT_VAL}  seed $SPLIT_SEED  -> $SPLIT_NAME"
echo "  python      : $PY"
echo "  repo        : $QGF_REPO"
echo "  GPU pin     : CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (physical GPU 1)"
echo

# ------------------------------------------- run dir ownership / no clobbering
mkdir -p "$QGF_RUN/manifest" "$QGF_RUN/logs" "$QGF_RUN/commands" \
         "$QGF_RUN/environment" "$QGF_RUN/checksums"
MARKER=$QGF_RUN/environment/qgf_task.env
if [ -f "$MARKER" ]; then
  PREV_TASK=$(sed -n 's/^QGF_TASK_KEY=//p' "$MARKER" | head -1)
  if [ -n "$PREV_TASK" ] && [ "$PREV_TASK" != "$QGF_TASK_KEY" ]; then
    FATAL "run dir $QGF_RUN already belongs to task '$PREV_TASK'; refusing to mix tasks (handoff section 12: never overwrite a previous Q run)"
  fi
fi
if [ -f "$QGF_RUN/manifest/manifest_summary.json" ] && [ "${QGF_FORCE_REBUILD:-0}" != "1" ]; then
  FATAL "a manifest already exists in $QGF_RUN/manifest; re-run with QGF_FORCE_REBUILD=1 only if you really mean to rebuild it"
fi
{
  echo "QGF_TASK_KEY=$QGF_TASK_KEY"
  echo "QGF_DATASET_ID=$QGF_DATASET_ID"
  echo "QGF_RUN_ID=$QGF_RUN_ID"
  echo "QGF_EPISODE_FIRST=$QGF_EPISODE_FIRST"
  echo "QGF_EPISODE_LAST=$QGF_EPISODE_LAST"
  echo "QGF_SSD_ROOT=$QGF_SSD_ROOT"
} > "$MARKER"

if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi > "$QGF_RUN/environment/nvidia_smi_build_manifest_start.txt" 2>&1 || true
  nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.used --format=csv \
    > "$QGF_RUN/environment/nvidia_smi_physical_gpu1.txt" 2>&1 || true
fi

# ------------------------------------------- section 5 preflight on the cohort
echo "=== preflight: cohort completeness, outcomes, prompt, provenance ==="
QGF_DATA="$QGF_DATA" SRC_MAP="$SRC_MAP" \
QGF_EPISODE_FIRST="$QGF_EPISODE_FIRST" QGF_EPISODE_LAST="$QGF_EPISODE_LAST" \
QGF_TASK_KEY="$QGF_TASK_KEY" EXPECT_EPISODES="$EXPECT_EPISODES" \
"$PY" - <<'PY'
import io, json, os, sys
from collections import Counter
from pathlib import Path

root = Path(os.environ["QGF_DATA"])
first = int(os.environ["QGF_EPISODE_FIRST"])
last = int(os.environ["QGF_EPISODE_LAST"])
expect = int(os.environ["EXPECT_EPISODES"])
task_key = os.environ["QGF_TASK_KEY"]
src_map_path = Path(os.environ["SRC_MAP"])

REQUIRED = [
    "episode_metadata.json",
    "transitions.parquet",
    "normalized_policy_chunks.parquet",
    "policy_observations.parquet",
    "chest.mp4",
    "wrist_right.mp4",
]

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)
        print("  FAIL  " + msg)


indices = list(range(first, last + 1))
check(len(indices) == expect, f"episode range covers {len(indices)} episodes, expected {expect}")

outcomes, prompts = {}, {}
for i in indices:
    d = root / f"episode_{i:06d}"
    if not d.is_dir():
        check(False, f"episode_{i:06d}: directory missing under {root}")
        continue
    for name in REQUIRED:
        p = d / name
        if not p.is_file() or p.stat().st_size == 0:
            check(False, f"episode_{i:06d}: required file missing/empty: {name}")
    mp = d / "episode_metadata.json"
    if not mp.is_file():
        continue
    m = json.load(io.open(mp, encoding="utf-8"))
    oc = m.get("outcome")
    outcomes[i] = oc
    prompts[i] = m.get("task_prompt")
    check(
        oc in ("success", "failure"),
        f"episode_{i:06d}: outcome {oc!r} is not 'success' or 'failure' (handoff section 5)",
    )

oc_counts = Counter(outcomes.values())
print(f"  outcomes: {dict(oc_counts)}")
check(oc_counts.get("success", 0) > 0, "cohort has no success episode")
check(oc_counts.get("failure", 0) > 0, "cohort has no failure episode")
check(
    sum(oc_counts.values()) == expect,
    f"outcome-bearing episodes {sum(oc_counts.values())} != {expect}",
)

uniq_prompts = sorted({p for p in prompts.values() if p is not None})
print(f"  distinct task prompts: {len(uniq_prompts)}")
check(len(uniq_prompts) == 1, f"task prompt is identical across the cohort (got {uniq_prompts[:4]})")
check(bool(uniq_prompts) and bool(str(uniq_prompts[0]).strip()), "task prompt is non-empty")
if uniq_prompts:
    print(f"  prompt: {uniq_prompts[0]!r}")

smap = json.load(io.open(src_map_path, encoding="utf-8"))
eps = smap.get("episodes", [])
check(len(eps) == expect, f"source_episode_map.json lists {len(eps)} episodes, expected {expect}")
map_task = smap.get("task")
if map_task is not None:
    check(
        str(map_task) == task_key,
        f"source_episode_map.json task {map_task!r} != QGF_TASK_KEY {task_key!r}",
    )
dest = {
    int(e["dest_episode_index"]): e.get("outcome")
    for e in eps
    if e.get("dest_episode_index") is not None
}
check(
    set(dest) == set(indices),
    "source map dest indices match the requested range "
    f"(only in map: {sorted(set(dest) - set(indices))[:5]}, "
    f"only in range: {sorted(set(indices) - set(dest))[:5]})",
)
mismatch = [i for i in indices if i in dest and i in outcomes and dest[i] != outcomes[i]]
check(not mismatch, f"outcome mismatch between source map and episode_metadata.json for {mismatch[:5]}")

print()
if fail:
    print(f"PREFLIGHT FAILED: {len(fail)} check(s)")
    sys.exit(1)
print("PREFLIGHT PASSED")
PY
echo

# ------------------------------------------------------------------- build it
CMD=(
  "$PY" "$BUILDER"
  --raw-episodes-root "$QGF_DATA"
  --output-dir "$QGF_RUN/manifest"
  --episode-first "$QGF_EPISODE_FIRST"
  --episode-last "$QGF_EPISODE_LAST"
  --action-horizon "$ACTION_HORIZON"
  --policy-hz "$POLICY_HZ"
  --max-transition-gap-seconds "$MAX_GAP"
  --val-count "$EXPECT_VAL"
  --split-seed "$SPLIT_SEED"
)

{
  echo "# executed $(date -u +%FT%TZ) on $(hostname)"
  echo "# task=$QGF_TASK_KEY dataset=$QGF_DATASET_ID run=$QGF_RUN_ID"
  echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1"
  printf '%q ' "${CMD[@]}"
  echo
} > "$QGF_RUN/commands/build_manifest.cmd"

echo "=== running ==="
printf '%q ' "${CMD[@]}"; echo
set +e
"${CMD[@]}" 2>&1 | tee "$QGF_RUN/logs/build_manifest.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || FATAL "build_real_robot_visual_iql_manifest.py exited $RC (see $QGF_RUN/logs/build_manifest.log)"

# ---------------------------------------------------------- section 8/9 gates
MANIFEST=$QGF_RUN/manifest/aligned_normalized_chunks.parquet
SUMMARY=$QGF_RUN/manifest/manifest_summary.json
SPLIT=$QGF_RUN/manifest/$SPLIT_NAME

[ -s "$MANIFEST" ] || FATAL "aligned manifest missing or empty: $MANIFEST"
[ -s "$SUMMARY" ]  || FATAL "manifest summary missing or empty: $SUMMARY"
if [ ! -s "$SPLIT" ]; then
  if [ -f "$QGF_RUN/manifest/episode_split_90_10.json" ]; then
    FATAL "the builder wrote episode_split_90_10.json instead of $SPLIT_NAME: the handoff section 8 patch (tools/qgf_4090_staging/patch_45_5.py) is NOT applied in $QGF_REPO"
  fi
  FATAL "split file missing: $SPLIT"
fi

echo
echo "=== manifest summary checks (handoff section 9) ==="
SUMMARY="$SUMMARY" SPLIT_NAME="$SPLIT_NAME" EXPECT_EPISODES="$EXPECT_EPISODES" \
QGF_EPISODE_FIRST="$QGF_EPISODE_FIRST" QGF_EPISODE_LAST="$QGF_EPISODE_LAST" \
ACTION_HORIZON="$ACTION_HORIZON" \
"$PY" - <<'PY'
import io, json, os, sys

m = json.load(io.open(os.environ["SUMMARY"], encoding="utf-8"))
expect = int(os.environ["EXPECT_EPISODES"])
first = int(os.environ["QGF_EPISODE_FIRST"])
last = int(os.environ["QGF_EPISODE_LAST"])
horizon = int(os.environ["ACTION_HORIZON"])
split_name = os.environ["SPLIT_NAME"]

fail = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


check(
    m.get("raw_episode_count") == expect,
    f"raw_episode_count == {expect} (got {m.get('raw_episode_count')})",
)
check(
    list(m.get("raw_episode_range", [])) == [first, last],
    f"raw_episode_range == [{first}, {last}] (got {m.get('raw_episode_range')})",
)
check(
    int(m.get("aligned_chunk_count", 0)) > 0,
    f"aligned_chunk_count > 0 (got {m.get('aligned_chunk_count')})",
)
check(
    list(m.get("action_chunk_shape", [])) == [horizon, 8],
    f"action_chunk_shape == [{horizon}, 8] (got {m.get('action_chunk_shape')})",
)
check(int(m.get("state_dim", 0)) == 8, f"state_dim == 8 (got {m.get('state_dim')})")
check(
    m.get("split_file") == split_name,
    f"split_file == {split_name} (got {m.get('split_file')})",
)
oc = m.get("outcome_counts", {})
check(oc.get("success", 0) > 0, f"cohort has successes ({oc.get('success', 0)})")
check(oc.get("failure", 0) > 0, f"cohort has failures ({oc.get('failure', 0)})")
check(sum(oc.values()) == expect, f"outcome counts sum to {expect} (got {sum(oc.values())})")
check(set(oc) <= {"success", "failure"}, f"no unexpected outcome labels (got {sorted(oc)})")
print(f"  aligned chunks : {m.get('aligned_chunk_count')}")
print(f"  skip accounting: {m.get('alignment', {}).get('skipped')}")
print()
if fail:
    print(f"FAILED {len(fail)} check(s)")
    sys.exit(1)
print("MANIFEST SUMMARY CHECKS PASSED")
PY

echo
echo "=== section 8 split acceptance (gate: the pipeline STOPS on failure) ==="
set +e
"$PY" "$VERIFY_SPLIT" "$SPLIT" "$SRC_MAP" 2>&1 \
  | tee "$QGF_RUN/logs/verify_split_${EXPECT_TRAIN}_${EXPECT_VAL}.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || FATAL "split acceptance failed (rc=$RC).  Handoff section 8: a single-outcome validation set cannot select a Q checkpoint.  Re-do the stratified split; do not bypass this check."

# ----------------------------------------------------------------- provenance
( cd "$QGF_RUN/manifest" && sha256sum \
    "aligned_normalized_chunks.parquet" "manifest_summary.json" "$SPLIT_NAME" ) \
  > "$QGF_RUN/checksums/manifest_SHA256SUMS"

echo
echo "=== manifest dir ==="
ls -la "$QGF_RUN/manifest"
echo
echo "=== checksums ==="
cat "$QGF_RUN/checksums/manifest_SHA256SUMS"
echo
echo "OK: manifest + ${EXPECT_TRAIN}/${EXPECT_VAL} split built and accepted for task '$QGF_TASK_KEY'"
