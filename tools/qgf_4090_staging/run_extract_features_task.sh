#!/bin/bash
# Handoff section 10: dual-camera SmolVLA visual feature extraction.
# TASK-PARAMETERIZED version of run_extract_features.sh: every task-specific
# path comes from the environment, nothing is hardcoded to a task.
# Runs ON THE 4090 (192.168.2.110).  Reads only local NVMe; never decodes Orin
# MP4 over the network and never touches the robot.
#
# Required environment (NO defaults - the script refuses to guess a task):
#   QGF_SSD_ROOT       /opt/qgf_real_robot
#   QGF_TASK_KEY       red_parcel
#   QGF_DATASET_ID     red_parcel_baseline50_20260902
#   QGF_RUN_ID         red_parcel_single_q_45_5_20260902
#   QGF_BUNDLE_NAME    red_parcel_clean      (dir under $QGF_SSD_ROOT/policy_bundles)
#   QGF_EPISODE_FIRST  0
#   QGF_EPISODE_LAST   49
#
# Optional overrides:
#   QGF_ENV_PREFIX          python env prefix (default $QGF_SSD_ROOT/envs/visual_iql_py310)
#   QGF_REPO_DIR            repo snapshot     (default $QGF_SSD_ROOT/repos/SmolVLA-with-QGF)
#   QGF_STAGING_DIR         dir holding verify_features.py (default: this script's dir)
#   QGF_POLICY_CHECKPOINT   explicit SmolVLA checkpoint dir
#                           (default $QGF_SSD_ROOT/policy_bundles/$QGF_BUNDLE_NAME/checkpoint)
#   QGF_EXTRACT_OVERWRITE=1 re-extract episodes that already have a .pt cache
#   QGF_ALLOW_SHARED_GPU=1  proceed even if another compute process holds GPU 1
#
# Fixed by the handoff, NOT parameterized:
#   CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only, handoff rule 1)
#   --device cuda --batch-size 32, offline HF/transformers
set -euo pipefail

FATAL() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------- environment
REQUIRED_VARS=(QGF_SSD_ROOT QGF_TASK_KEY QGF_DATASET_ID QGF_RUN_ID QGF_BUNDLE_NAME QGF_EPISODE_FIRST QGF_EPISODE_LAST)
MISSING=""
for v in "${REQUIRED_VARS[@]}"; do
  [ -n "${!v-}" ] || MISSING="$MISSING $v"
done
if [ -n "$MISSING" ]; then
  echo "FATAL: required environment variable(s) unset or empty:$MISSING" >&2
  echo "This script is task-parameterized and will not guess a task." >&2
  echo "Example (handoff sections 4, 6 and 10):" >&2
  echo "  export QGF_SSD_ROOT=/opt/qgf_real_robot" >&2
  echo "  export QGF_TASK_KEY=red_parcel" >&2
  echo "  export QGF_DATASET_ID=red_parcel_baseline50_20260902" >&2
  echo "  export QGF_RUN_ID=red_parcel_single_q_45_5_20260902" >&2
  echo "  export QGF_BUNDLE_NAME=red_parcel_clean" >&2
  echo "  export QGF_EPISODE_FIRST=0" >&2
  echo "  export QGF_EPISODE_LAST=49" >&2
  exit 2
fi

case "$QGF_SSD_ROOT" in
  /*) ;;
  *) FATAL "QGF_SSD_ROOT must be an absolute path (got '$QGF_SSD_ROOT')" ;;
esac
for v in QGF_TASK_KEY QGF_DATASET_ID QGF_RUN_ID QGF_BUNDLE_NAME; do
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
export TRANSFORMERS_OFFLINE=1
EXPECT_TRAIN=45
EXPECT_VAL=5
EXPECT_EPISODES=$(( EXPECT_TRAIN + EXPECT_VAL ))
BATCH_SIZE=32
DEVICE=cuda
VISUAL_TOKENS=128
VISUAL_DIM=960

EP_COUNT=$(( QGF_EPISODE_LAST - QGF_EPISODE_FIRST + 1 ))
[ "$EP_COUNT" -eq "$EXPECT_EPISODES" ] || FATAL \
  "episode range ${QGF_EPISODE_FIRST}..${QGF_EPISODE_LAST} yields $EP_COUNT episodes, but the frozen procedure covers exactly $EXPECT_EPISODES episodes (handoff sections 5 and 8)"

# --------------------------------------------------------------------- layout
E=${QGF_ENV_PREFIX:-$QGF_SSD_ROOT/envs/visual_iql_py310}
PY=$E/bin/python
QGF_REPO=${QGF_REPO_DIR:-$QGF_SSD_ROOT/repos/SmolVLA-with-QGF}
EXTRACTOR=$QGF_REPO/qgf/scripts/extract_smolvla_visual_features.py
DATASET_DIR=$QGF_SSD_ROOT/datasets/$QGF_DATASET_ID
QGF_DATA=$DATASET_DIR/raw_episodes
QGF_RUN=$QGF_SSD_ROOT/runs/$QGF_RUN_ID
BUNDLE_DIR=$QGF_SSD_ROOT/policy_bundles/$QGF_BUNDLE_NAME
QGF_POLICY=${QGF_POLICY_CHECKPOINT:-$BUNDLE_DIR/checkpoint}
MANIFEST=$QGF_RUN/manifest/aligned_normalized_chunks.parquet
SUMMARY=$QGF_RUN/manifest/manifest_summary.json
FEATURES=$QGF_RUN/features
SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGING=${QGF_STAGING_DIR:-$SELF_DIR}
VERIFY_FEATURES=$STAGING/verify_features.py
[ -f "$VERIFY_FEATURES" ] || VERIFY_FEATURES=$QGF_REPO/tools/qgf_4090_staging/verify_features.py

[ -x "$PY" ]         || FATAL "python not found or not executable: $PY (handoff section 7)"
[ -f "$EXTRACTOR" ]  || FATAL "feature extractor missing: $EXTRACTOR"
[ -d "$QGF_DATA" ]   || FATAL "raw episode root missing: $QGF_DATA (stage the cohort first, handoff sections 5-6)"
[ -s "$MANIFEST" ]   || FATAL "aligned manifest missing: $MANIFEST (run run_build_manifest_task.sh first, handoff section 9)"
[ -s "$SUMMARY" ]    || FATAL "manifest summary missing: $SUMMARY"
[ -f "$VERIFY_FEATURES" ] || FATAL "verify_features.py not found next to this script nor at $QGF_REPO/tools/qgf_4090_staging/; set QGF_STAGING_DIR"
if [ ! -d "$QGF_POLICY" ]; then
  echo "policy bundle dir contents ($BUNDLE_DIR):" >&2
  ls -la "$BUNDLE_DIR" >&2 2>/dev/null || echo "  (bundle dir does not exist)" >&2
  FATAL "SmolVLA checkpoint dir missing: $QGF_POLICY (handoff section 6: features MUST come from the bundle that produced these rollouts; set QGF_POLICY_CHECKPOINT if the bundle layout differs)"
fi
[ -f "$QGF_POLICY/config.json" ] \
  || FATAL "$QGF_POLICY does not look like a SmolVLA checkpoint (no config.json)"

echo "=== config ==="
echo "  task key    : $QGF_TASK_KEY"
echo "  dataset     : $QGF_DATA"
echo "  run dir     : $QGF_RUN"
echo "  episodes    : ${QGF_EPISODE_FIRST}..${QGF_EPISODE_LAST}  ($EP_COUNT)"
echo "  checkpoint  : $QGF_POLICY"
echo "  manifest    : $MANIFEST"
echo "  batch size  : $BATCH_SIZE   device: $DEVICE"
echo "  overwrite   : ${QGF_EXTRACT_OVERWRITE:-0}"
echo "  GPU pin     : CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (physical GPU 1)"
echo

mkdir -p "$FEATURES" "$QGF_RUN/logs" "$QGF_RUN/commands" \
         "$QGF_RUN/environment" "$QGF_RUN/checksums"

# ----------------------------------------------- run dir belongs to this task
MARKER=$QGF_RUN/environment/qgf_task.env
if [ -f "$MARKER" ]; then
  PREV_TASK=$(sed -n 's/^QGF_TASK_KEY=//p' "$MARKER" | head -1)
  if [ -n "$PREV_TASK" ] && [ "$PREV_TASK" != "$QGF_TASK_KEY" ]; then
    FATAL "run dir $QGF_RUN already belongs to task '$PREV_TASK'; refusing to mix tasks (handoff section 12)"
  fi
else
  {
    echo "QGF_TASK_KEY=$QGF_TASK_KEY"
    echo "QGF_DATASET_ID=$QGF_DATASET_ID"
    echo "QGF_RUN_ID=$QGF_RUN_ID"
    echo "QGF_EPISODE_FIRST=$QGF_EPISODE_FIRST"
    echo "QGF_EPISODE_LAST=$QGF_EPISODE_LAST"
    echo "QGF_SSD_ROOT=$QGF_SSD_ROOT"
  } > "$MARKER"
fi
PREV_BUNDLE=$(sed -n 's/^QGF_BUNDLE_NAME=//p' "$MARKER" | head -1)
if [ -n "$PREV_BUNDLE" ] && [ "$PREV_BUNDLE" != "$QGF_BUNDLE_NAME" ]; then
  FATAL "run dir $QGF_RUN was started with bundle '$PREV_BUNDLE', now asked for '$QGF_BUNDLE_NAME'; handoff section 6 forbids mixing bundles inside one run"
fi
if [ -z "$PREV_BUNDLE" ]; then
  {
    echo "QGF_BUNDLE_NAME=$QGF_BUNDLE_NAME"
    echo "QGF_POLICY_CHECKPOINT=$QGF_POLICY"
  } >> "$MARKER"
fi

# ------------------------------------------------ handoff rule 1: physical GPU 1
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.used --format=csv \
    > "$QGF_RUN/environment/nvidia_smi_physical_gpu1.txt" 2>&1 || true
  BUSY=$(nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
  if [ -n "$BUSY" ]; then
    echo "physical GPU 1 already has compute processes:" >&2
    echo "$BUSY" >&2
    [ "${QGF_ALLOW_SHARED_GPU:-0}" = "1" ] \
      || FATAL "refusing to share physical GPU 1 (handoff section 11: do not share the card).  Stop the other process, or set QGF_ALLOW_SHARED_GPU=1 deliberately."
  fi
fi
"$PY" - <<'PY' | tee "$QGF_RUN/environment/gpu_pin_check_extract.txt"
import torch

assert torch.cuda.is_available(), "CUDA is not available with CUDA_VISIBLE_DEVICES=1"
assert torch.cuda.device_count() == 1, (
    f"expected exactly 1 visible GPU (the physical GPU 1 pin), got {torch.cuda.device_count()}"
)
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
PY

# ---------------------------- preflight: manifest, disk headroom, stale caches
echo
echo "=== preflight: manifest coverage, disk headroom, existing caches ==="
QGF_SSD_ROOT="$QGF_SSD_ROOT" MANIFEST="$MANIFEST" SUMMARY="$SUMMARY" \
FEATURES="$FEATURES" QGF_POLICY="$QGF_POLICY" \
QGF_EPISODE_FIRST="$QGF_EPISODE_FIRST" QGF_EPISODE_LAST="$QGF_EPISODE_LAST" \
EXPECT_EPISODES="$EXPECT_EPISODES" VISUAL_TOKENS="$VISUAL_TOKENS" \
VISUAL_DIM="$VISUAL_DIM" QGF_EXTRACT_OVERWRITE="${QGF_EXTRACT_OVERWRITE:-0}" \
"$PY" - <<'PY'
import io, json, os, sys
from pathlib import Path

import pyarrow.parquet as pq

manifest = Path(os.environ["MANIFEST"])
summary = json.load(io.open(os.environ["SUMMARY"], encoding="utf-8"))
features = Path(os.environ["FEATURES"])
policy = os.environ["QGF_POLICY"]
first = int(os.environ["QGF_EPISODE_FIRST"])
last = int(os.environ["QGF_EPISODE_LAST"])
expect = int(os.environ["EXPECT_EPISODES"])
tokens = int(os.environ["VISUAL_TOKENS"])
dim = int(os.environ["VISUAL_DIM"])
overwrite = os.environ["QGF_EXTRACT_OVERWRITE"] == "1"
ssd_root = os.environ["QGF_SSD_ROOT"]

fail = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


table = pq.read_table(manifest, columns=["episode_index", "task"])
episodes = sorted({int(x) for x in table.column("episode_index").to_pylist()})
tasks = sorted({str(x) for x in table.column("task").to_pylist()})
rows = table.num_rows
wanted = list(range(first, last + 1))

check(episodes == wanted,
      f"manifest covers episodes {first}..{last} ({len(episodes)} present, "
      f"missing {sorted(set(wanted) - set(episodes))[:5]}, "
      f"extra {sorted(set(episodes) - set(wanted))[:5]})")
check(len(episodes) == expect, f"manifest episode count == {expect} (got {len(episodes)})")
check(len(tasks) == 1, f"single task prompt in the manifest (got {len(tasks)}: {tasks[:3]})")
check(rows == int(summary.get("aligned_chunk_count", -1)),
      f"manifest rows {rows} == manifest_summary aligned_chunk_count "
      f"{summary.get('aligned_chunk_count')}")
check(rows > 0, f"manifest has samples (got {rows})")
if tasks:
    print(f"  task prompt: {tasks[0]!r}")

# handoff section 2 headroom: the cache stores z AND z' as [tokens, dim] BF16.
per_sample = 2 * tokens * dim * 2
need_bytes = rows * per_sample + 20 * (1 << 30)
st = os.statvfs(ssd_root)
free_bytes = st.f_bavail * st.f_frsize
print(f"  estimated feature cache: {rows * per_sample / 2**30:.2f} GiB "
      f"({rows} samples x {per_sample / 2**20:.2f} MiB)")
print(f"  free on {ssd_root}: {free_bytes / 2**30:.2f} GiB, "
      f"need >= {need_bytes / 2**30:.2f} GiB (cache + 20 GiB margin)")
check(free_bytes >= need_bytes,
      "sufficient free space (handoff section 2: stop and report, do NOT delete Orin "
      "data and do NOT write to /home)")

existing = sorted(features.glob("episode_*.pt"))
print(f"  existing feature caches: {len(existing)}")
old_summary = features / "visual_feature_summary.json"
if old_summary.is_file():
    old = json.load(io.open(old_summary, encoding="utf-8"))
    old_ckpt = str(old.get("checkpoint", ""))
    print(f"  previous run checkpoint: {old_ckpt}")
    check(old_ckpt == policy or overwrite,
          f"existing cache was built from {old_ckpt!r}, this run uses {policy!r} "
          "(handoff section 6 forbids mixing bundles; set QGF_EXTRACT_OVERWRITE=1 to re-extract)")
elif existing and not overwrite:
    print("  NOTE: caches exist without visual_feature_summary.json; they will be reused "
          "as-is unless QGF_EXTRACT_OVERWRITE=1")

print()
if fail:
    print(f"PREFLIGHT FAILED: {len(fail)} check(s)")
    sys.exit(1)
print("PREFLIGHT PASSED")
PY
echo

# ----------------------------------------------------------------- extract it
CMD=(
  "$PY" "$EXTRACTOR"
  --raw-episodes-root "$QGF_DATA"
  --aligned-manifest "$MANIFEST"
  --output-dir "$FEATURES"
  --checkpoint "$QGF_POLICY"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
)
if [ "${QGF_EXTRACT_OVERWRITE:-0}" = "1" ]; then
  CMD+=(--overwrite)
fi

{
  echo "# executed $(date -u +%FT%TZ) on $(hostname)"
  echo "# task=$QGF_TASK_KEY dataset=$QGF_DATASET_ID run=$QGF_RUN_ID bundle=$QGF_BUNDLE_NAME"
  echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  printf '%q ' "${CMD[@]}"
  echo
} > "$QGF_RUN/commands/extract_visual_features.cmd"

if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi > "$QGF_RUN/environment/nvidia_smi_extract_start.txt" 2>&1 || true
  # handoff rule 1 needs a PID snapshot proving the work lands on physical GPU 1
  (
    sleep 120
    {
      date -u +%FT%TZ
      nvidia-smi
      echo "--- compute apps on physical GPU 1 ---"
      nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv
    } > "$QGF_RUN/environment/nvidia_smi_extract_running.txt" 2>&1
  ) &
  WATCHER=$!
else
  WATCHER=""
fi

echo "=== running ==="
printf '%q ' "${CMD[@]}"; echo
echo "=== extraction start $(date -u +%FT%TZ) ==="
set +e
"${CMD[@]}" 2>&1 | tee "$QGF_RUN/logs/extract_visual_features.log"
RC=${PIPESTATUS[0]}
set -e
echo "=== extraction end $(date -u +%FT%TZ) rc=$RC ==="
if [ -n "$WATCHER" ]; then
  kill "$WATCHER" 2>/dev/null || true
  wait "$WATCHER" 2>/dev/null || true
fi
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi > "$QGF_RUN/environment/nvidia_smi_extract_end.txt" 2>&1 || true
fi
[ "$RC" -eq 0 ] || FATAL "extract_smolvla_visual_features.py exited $RC (see $QGF_RUN/logs/extract_visual_features.log)"

# ------------------------------------------------------- section 10 acceptance
N=$(find "$FEATURES" -maxdepth 1 -name 'episode_*.pt' -type f | wc -l)
echo
echo "feature caches: $N/$EXPECT_EPISODES"
[ "$N" -eq "$EXPECT_EPISODES" ] \
  || FATAL "expected $EXPECT_EPISODES per-episode caches, found $N in $FEATURES"

echo
echo "=== extractor summary checks (handoff section 10) ==="
FEATURES="$FEATURES" SUMMARY="$SUMMARY" QGF_POLICY="$QGF_POLICY" \
EXPECT_EPISODES="$EXPECT_EPISODES" VISUAL_TOKENS="$VISUAL_TOKENS" \
VISUAL_DIM="$VISUAL_DIM" \
"$PY" - <<'PY'
import io, json, os, sys
from pathlib import Path

features = Path(os.environ["FEATURES"])
vsum_path = features / "visual_feature_summary.json"
msum = json.load(io.open(os.environ["SUMMARY"], encoding="utf-8"))
policy = os.environ["QGF_POLICY"]
expect = int(os.environ["EXPECT_EPISODES"])
tokens = int(os.environ["VISUAL_TOKENS"])
dim = int(os.environ["VISUAL_DIM"])

fail = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


check(vsum_path.is_file(), f"visual_feature_summary.json exists ({vsum_path})")
if not vsum_path.is_file():
    sys.exit(1)
v = json.load(io.open(vsum_path, encoding="utf-8"))

check(str(v.get("checkpoint")) == policy,
      f"features extracted from THIS task's bundle (summary {v.get('checkpoint')!r} "
      f"vs {policy!r}, handoff section 6)")
check(str(v.get("device")) == "cuda", f"device == cuda (got {v.get('device')!r})")
check(int(v.get("episode_count", -1)) == expect,
      f"episode_count == {expect} (got {v.get('episode_count')})")

eps = v.get("episodes", [])
reused = [e for e in eps if e.get("status") == "existing"]
if reused:
    print(f"  NOTE: {len(reused)} episode(s) reused a pre-existing cache; "
          "shapes/totals are verified below by verify_features.py")
else:
    check(int(v.get("sample_count", -1)) == int(msum.get("aligned_chunk_count", -2)),
          f"sample_count {v.get('sample_count')} == manifest aligned_chunk_count "
          f"{msum.get('aligned_chunk_count')}")
bad = [
    e for e in eps
    if e.get("visual_shape") is not None and list(e["visual_shape"])[1:] != [tokens, dim]
]
check(not bad, f"every reported visual_shape is [N,{tokens},{dim}] (offenders: "
               f"{[e.get('path') for e in bad][:5]})")
print()
if fail:
    print(f"FAILED {len(fail)} check(s)")
    sys.exit(1)
print("EXTRACTOR SUMMARY CHECKS PASSED")
PY

echo
echo "=== section 10 feature-cache acceptance (gate: pipeline STOPS on failure) ==="
REPORT=$QGF_RUN/manifest/feature_cache_report.json
set +e
"$PY" "$VERIFY_FEATURES" "$QGF_RUN" "$REPORT" 2>&1 \
  | tee "$QGF_RUN/logs/verify_features.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || FATAL "feature cache acceptance failed (rc=$RC).  Handoff section 10: shapes [N,128,960]/[N,50,8]/[N,8], no NaN/Inf, totals must match the manifest.  Do not start training."

# ----------------------------------------------------------------- provenance
REPORT="$REPORT" OUT="$QGF_RUN/checksums/feature_cache_SHA256SUMS" "$PY" - <<'PY'
import io, json, os

report = json.load(io.open(os.environ["REPORT"], encoding="utf-8"))
lines = [f"{e['sha256']}  features/{e['file']}\n" for e in report["episodes"]]
io.open(os.environ["OUT"], "w", encoding="utf-8", newline="\n").writelines(lines)
print(f"{len(lines)} per-episode SHA256 recorded -> {os.environ['OUT']}")
print(f"total bytes: {report['total_bytes']} "
      f"({report['total_bytes'] / 2**30:.2f} GiB), samples: {report['total_samples']}")
PY

echo
echo "=== features ==="
ls "$FEATURES" | head -5
find "$FEATURES" -maxdepth 1 -name 'episode_*.pt' -type f | wc -l
du -sh "$FEATURES"
echo
echo "OK: visual feature cache extracted and accepted for task '$QGF_TASK_KEY' "
echo "    ($N/$EXPECT_EPISODES caches, report: $REPORT)"
