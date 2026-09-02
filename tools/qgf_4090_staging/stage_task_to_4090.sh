#!/bin/bash
# Stage ONE task's baseline rollouts from the Orin to the RTX4090 SSD.
# Task-parameterized replacement for stage_mug_to_4090.sh.
# Runs ON THE ORIN.  Read-only against the source; never touches robot control.
# Per docs/qgf/RTX4090_SINGLE_Q_CRITIC_TRAINING_HANDOFF.md sections 2, 3, 5, 6.
#
#   ./stage_task_to_4090.sh <episode_list_file> [--bundle] [--verify-only]
#
# episode_list_file: one episode dir name per line (e.g. episode_000002), the
# frozen cohort.  Lines starting with # are ignored.
#
# Required environment (no defaults -- the script refuses to guess a task):
#   QGF_SSD_ROOT       /opt/qgf_real_robot                (on the 4090)
#   QGF_TASK_KEY       red_parcel
#   QGF_DATASET_ID     red_parcel_baseline50_20260902
#   QGF_ORIN_EPISODES  /home/nvidia/work/telop/red_parcel_real_rollouts/episodes
#   QGF_ORIN_BUNDLE    /home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean
#   QGF_BUNDLE_NAME    red_parcel_clean
#   QGF_EPISODE_FIRST  0
#   QGF_EPISODE_LAST   49
#
# Optional environment:
#   QGF_RUN_ID              recorded in transfer.log when set
#   QGF_EXPECTED_EPISODES   cohort size override; default LAST-FIRST+1.  Use it
#                           ONLY for a deliberately non-contiguous frozen cohort.
#   QGF_ORIN_PYTHON         Orin python3 with pyarrow
#   QGF_SOURCE_MAP_PY       path to task_source_map.py
#   BWLIMIT                 rsync KB/s cap, default 12000
#
# The destination host is the Orin-side ssh alias walle4090 and is NOT
# configurable.  walle@192.168.2.110 bypasses ~/.ssh/config and therefore the
# BindAddress 192.168.2.171 /32 interface that handoff section 3 requires to
# keep traffic off the robot-control port when two robots share 192.168.2.x.
set -euo pipefail

fatal() { echo "FATAL: $*" >&2; exit 1; }

req() {
  local n=$1 v
  v=${!n-}
  [ -n "${v}" ] || {
    echo "FATAL: required environment variable $n is unset or empty." >&2
    echo "  stage_task_to_4090.sh is task-parameterized; it refuses to guess a task." >&2
    exit 2
  }
  printf '%s' "$v"
}

SSD=$(req QGF_SSD_ROOT)
TASK_KEY=$(req QGF_TASK_KEY)
DSID=$(req QGF_DATASET_ID)
SRC=$(req QGF_ORIN_EPISODES)
BUNDLE_SRC=$(req QGF_ORIN_BUNDLE)
BUNDLE_NAME=$(req QGF_BUNDLE_NAME)
EP_FIRST=$(req QGF_EPISODE_FIRST)
EP_LAST=$(req QGF_EPISODE_LAST)
RUN_ID=${QGF_RUN_ID:-}

case "$EP_FIRST" in ''|*[!0-9]*) fatal "QGF_EPISODE_FIRST must be a non-negative integer, got '$EP_FIRST'";; esac
case "$EP_LAST"  in ''|*[!0-9]*) fatal "QGF_EPISODE_LAST must be a non-negative integer, got '$EP_LAST'";; esac
EP_FIRST=$((10#$EP_FIRST)); EP_LAST=$((10#$EP_LAST))
[ "$EP_LAST" -ge "$EP_FIRST" ] || fatal "QGF_EPISODE_LAST ($EP_LAST) < QGF_EPISODE_FIRST ($EP_FIRST)"
RANGE_N=$(( EP_LAST - EP_FIRST + 1 ))
if [ -n "${QGF_EXPECTED_EPISODES:-}" ]; then
  EXPECTED_N=$QGF_EXPECTED_EPISODES
  EXPECTED_OVERRIDE=1
  EXPECTED_SRC="QGF_EXPECTED_EPISODES override"
else
  EXPECTED_N=$RANGE_N
  EXPECTED_OVERRIDE=0
  EXPECTED_SRC="QGF_EPISODE_LAST - QGF_EPISODE_FIRST + 1"
fi
case "$EXPECTED_N" in ''|*[!0-9]*) fatal "QGF_EXPECTED_EPISODES must be a positive integer";; esac
[ "$EXPECTED_N" -gt 0 ] || fatal "expected cohort size must be > 0"

# the freeze/provenance helpers are separate processes; make sure they see the
# same contract even if the caller only set (and did not export) the variables.
export QGF_SSD_ROOT="$SSD" QGF_TASK_KEY="$TASK_KEY" QGF_DATASET_ID="$DSID" \
       QGF_ORIN_EPISODES="$SRC" QGF_ORIN_BUNDLE="$BUNDLE_SRC" \
       QGF_BUNDLE_NAME="$BUNDLE_NAME" QGF_EPISODE_FIRST="$EP_FIRST" \
       QGF_EPISODE_LAST="$EP_LAST"
[ -z "$RUN_ID" ] || export QGF_RUN_ID="$RUN_ID"
[ "$EXPECTED_OVERRIDE" -eq 0 ] || export QGF_EXPECTED_EPISODES="$EXPECTED_N"

PY=${QGF_ORIN_PYTHON:-/home/nvidia/work/telop/venvs/smolvla-orin/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE_MAP_PY=${QGF_SOURCE_MAP_PY:-}
if [ -z "$SOURCE_MAP_PY" ]; then
  if [ -f "$HERE/task_source_map.py" ]; then SOURCE_MAP_PY="$HERE/task_source_map.py"
  else SOURCE_MAP_PY=/tmp/task_source_map.py; fi
fi

DEST_HOST=walle4090             # handoff section 3: alias only, never walle@IP
DEST=$SSD/datasets/$DSID
BUNDLE_DEST=$SSD/policy_bundles/$BUNDLE_NAME
BWLIMIT=${BWLIMIT:-12000}          # KB/s; keep headroom for the live recorder
FILES=(episode_metadata.json transitions.parquet normalized_policy_chunks.parquet policy_observations.parquet chest.mp4 wrist_right.mp4)

LIST_FILE=${1:?usage: $0 <episode_list_file> [--bundle] [--verify-only]}
shift || true
DO_BUNDLE=0
VERIFY_ONLY=0
for a in "$@"; do
  case "$a" in
    --bundle) DO_BUNDLE=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

[ -f "$LIST_FILE" ] || fatal "episode list file not found: $LIST_FILE"
[ -d "$SRC" ]       || fatal "episodes root not found on this Orin: $SRC"
[ -x "$PY" ]        || fatal "Orin python not executable: $PY (set QGF_ORIN_PYTHON)"
[ -f "$SOURCE_MAP_PY" ] || fatal "task_source_map.py not found: $SOURCE_MAP_PY (set QGF_SOURCE_MAP_PY)"
[ $DO_BUNDLE -eq 0 ] || [ -d "$BUNDLE_SRC" ] || fatal "bundle source not found: $BUNDLE_SRC"

echo "=== task=$TASK_KEY dataset=$DSID run=${RUN_ID:-<unset>} ==="
echo "=== src=$SRC -> $DEST_HOST:$DEST/raw_episodes ==="

mapfile -t EPS < <(grep -vE '^[[:space:]]*(#|$)' "$LIST_FILE" | tr -d '\r' | sort -u)
[ ${#EPS[@]} -gt 0 ] || { echo "empty episode list"; exit 2; }
echo "=== cohort: ${#EPS[@]} episodes: ${EPS[0]} .. ${EPS[${#EPS[@]}-1]} ==="

# ---------- 0. cohort shape (handoff section 5: do not take "whatever N dirs") ----------
[ ${#EPS[@]} -eq "$EXPECTED_N" ] || fatal \
  "frozen list has ${#EPS[@]} episodes, expected $EXPECTED_N (from $EXPECTED_SRC)"
IDXS=()
for e in "${EPS[@]}"; do
  [[ "$e" =~ ^episode_([0-9]+)$ ]] || fatal "episode name '$e' is not episode_<digits>"
  i=$((10#${BASH_REMATCH[1]}))
  [ "$i" -ge "$EP_FIRST" ] && [ "$i" -le "$EP_LAST" ] || fatal \
    "episode '$e' (index $i) is outside [$EP_FIRST,$EP_LAST]; the manifest builder selects by --episode-first/--episode-last and would disagree with this transfer"
  IDXS+=("$i")
done
if [ $EXPECTED_OVERRIDE -eq 0 ]; then
  WANT=$(seq "$EP_FIRST" "$EP_LAST" | LC_ALL=C sort)
  GOT=$(printf '%s\n' "${IDXS[@]}" | LC_ALL=C sort)
  [ "$WANT" = "$GOT" ] || fatal \
    "frozen list is not exactly the contiguous range [$EP_FIRST,$EP_LAST]; set QGF_EXPECTED_EPISODES deliberately and record why (handoff section 5)"
fi
echo "cohort indices verified within [$EP_FIRST,$EP_LAST]"

# ---------- 0b. ssh alias sanity (handoff section 3) ----------
# 'ssh -G' only resolves ~/.ssh/config locally; it opens no connection.
SSHCFG=$(ssh -G "$DEST_HOST" 2>/dev/null || true)
cfg_val() { printf '%s\n' "$SSHCFG" | awk -v k="$1" 'tolower($1)==k {print $2; exit}'; }
if [ -z "$SSHCFG" ]; then
  echo "WARNING: 'ssh -G $DEST_HOST' produced no output; cannot verify the /32 BindAddress."
else
  H=$(cfg_val hostname); B=$(cfg_val bindaddress); I=$(cfg_val identityfile)
  echo "ssh alias $DEST_HOST -> hostname=$H bindaddress=${B:-<none>} identityfile=${I:-<none>}"
  [ "$H" = "192.168.2.110" ] || fatal "alias $DEST_HOST resolves to '$H', expected 192.168.2.110"
  [ "$B" = "192.168.2.171" ] || fatal \
    "alias $DEST_HOST has BindAddress '${B:-<none>}', expected 192.168.2.171 (the /32 eth0 interface). Fix ~/.ssh/config; do NOT fall back to walle@192.168.2.110."
  case "$I" in
    *qgf_4090_transfer_ed25519) : ;;
    *) echo "WARNING: identityfile is '${I:-<none>}', expected the dedicated ~/.ssh/qgf_4090_transfer_ed25519 key" ;;
  esac
fi

# ---------- 1. source-side preflight ----------
TOTAL=0
for e in "${EPS[@]}"; do
  for f in "${FILES[@]}"; do
    p="$SRC/$e/$f"
    [ -s "$p" ] || { echo "FATAL missing/empty source file: $p"; exit 1; }
    TOTAL=$(( TOTAL + $(stat -c%s "$p") ))
  done
done
NFILES=$(( ${#EPS[@]} * ${#FILES[@]} ))
TOTAL_GIB=$(awk -v b=$TOTAL 'BEGIN{printf "%.2f", b/1073741824}')
echo "training input: $NFILES files, ${TOTAL_GIB} GiB"

# ---------- 2. destination preflight (handoff section 2 formula) ----------
# free >= input + 2 x expected feature cache + 20 GiB
# feature cache: per chunk a [128,960] BF16 visual block = 240 KiB; training
# stores z and z' -> 2 blocks per sample.
CHUNKS=0
for e in "${EPS[@]}"; do
  n=$("$PY" -c "import pyarrow.parquet as pq,sys;print(pq.read_metadata(sys.argv[1]).num_rows)" "$SRC/$e/normalized_policy_chunks.parquet")
  CHUNKS=$(( CHUNKS + n ))
done
FEAT_GIB=$(awk -v c=$CHUNKS 'BEGIN{printf "%.2f", c*2*128*960*2/1073741824}')
NEED_GIB=$(awk -v t=$TOTAL_GIB -v f=$FEAT_GIB 'BEGIN{printf "%.2f", t + 2*f + 20}')
echo "chunks total: $CHUNKS  -> feature cache est ${FEAT_GIB} GiB  -> need >= ${NEED_GIB} GiB free"

ssh -o BatchMode=yes "$DEST_HOST" "mkdir -p '$DEST/raw_episodes' '$BUNDLE_DEST' '$SSD/runs' '$SSD/repos'"
FREE_GIB=$(ssh -o BatchMode=yes "$DEST_HOST" "df -BG --output=avail '$SSD' | tail -1 | tr -dc '0-9'")
echo "destination free: ${FREE_GIB} GiB on $SSD"
if ! awk -v f="$FREE_GIB" -v n="$NEED_GIB" 'BEGIN{exit !(f+0 >= n+0)}'; then
  echo "FATAL: insufficient space on $DEST_HOST ($SSD): free ${FREE_GIB} GiB < need ${NEED_GIB} GiB"
  echo "Per handoff section 2: stop and report.  Do NOT delete Orin data, do NOT use /home."
  exit 1
fi

LOG=$DEST/transfer.log
ssh -o BatchMode=yes "$DEST_HOST" "touch '$LOG'"
say() {
  local msg=${*//\'/}
  echo "$*"
  ssh -o BatchMode=yes "$DEST_HOST" "printf '%s %s\n' \"\$(date -u +%FT%TZ)\" '$msg' >> '$LOG'"
}
say "START task=$TASK_KEY dataset=$DSID run=${RUN_ID:-none} cohort=${#EPS[@]} range=[$EP_FIRST,$EP_LAST] files=$NFILES bytes=$TOTAL src=$SRC bwlimit=${BWLIMIT}KBps"

# ---------- 3. transfer ----------
if [ $VERIFY_ONLY -eq 0 ]; then
  INC=()
  for e in "${EPS[@]}"; do
    INC+=(--include="$e/")
    for f in "${FILES[@]}"; do INC+=(--include="$e/$f"); done
  done
  nice -n 19 ionice -c3 rsync -a --partial --append-verify --info=progress2 \
    --prune-empty-dirs --bwlimit="$BWLIMIT" \
    "${INC[@]}" --exclude='*' \
    "$SRC/" "$DEST_HOST:$DEST/raw_episodes/"
  say "RSYNC_DONE episodes"

  if [ $DO_BUNDLE -eq 1 ]; then
    echo "=== copying the SmolVLA bundle that produced these rollouts ==="
    nice -n 19 ionice -c3 rsync -a --partial --append-verify --info=progress2 \
      --bwlimit="$BWLIMIT" "$BUNDLE_SRC/" "$DEST_HOST:$BUNDLE_DEST/"
    say "RSYNC_DONE bundle $BUNDLE_SRC to $BUNDLE_DEST"
  fi
fi

# ---------- 3b. destination file count + total bytes (handoff section 6) ----------
DST_FILES=$(ssh -o BatchMode=yes "$DEST_HOST" "cd '$DEST/raw_episodes' && find . -type f | wc -l" | tr -dc '0-9')
DST_BYTES=$(ssh -o BatchMode=yes "$DEST_HOST" "cd '$DEST/raw_episodes' && find . -type f -printf '%s\n' | awk -v OFMT=%.0f -v CONVFMT=%.0f '{s+=\$1} END{print s+0}'" | tr -dc '0-9')
echo "destination: $DST_FILES files, $DST_BYTES bytes (source: $NFILES files, $TOTAL bytes)"
if [ "$DST_FILES" != "$NFILES" ] || [ "$DST_BYTES" != "$TOTAL" ]; then
  say "VERIFY_FAIL count/bytes src=$NFILES/$TOTAL dst=$DST_FILES/$DST_BYTES"
  fatal "destination file count or total bytes disagree with the frozen cohort (src $NFILES files/$TOTAL B vs dst $DST_FILES files/$DST_BYTES B)"
fi

# ---------- 4. SHA256 both sides ----------
echo "=== hashing source (nice/ionice) ==="
SRCSUM=/tmp/${TASK_KEY}_src_SHA256SUMS
: > "$SRCSUM"
for e in "${EPS[@]}"; do
  for f in "${FILES[@]}"; do
    ( cd "$SRC" && nice -n 19 ionice -c3 sha256sum "$e/$f" ) >> "$SRCSUM"
  done
done
echo "=== hashing destination ==="
DSTSUM=/tmp/${TASK_KEY}_dst_SHA256SUMS
ssh -o BatchMode=yes "$DEST_HOST" "cd '$DEST/raw_episodes' && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs -r sha256sum" > "$DSTSUM"

TMP_S=/tmp/.qgf_${TASK_KEY}_s.$$
TMP_D=/tmp/.qgf_${TASK_KEY}_d.$$
sort -k2 "$SRCSUM" > "$TMP_S"
sort -k2 "$DSTSUM" > "$TMP_D"
if diff -q "$TMP_S" "$TMP_D" >/dev/null; then
  echo "SHA256 MATCH: $(wc -l < "$TMP_S") files identical on both sides"
  say "VERIFY_OK files=$(wc -l < "$TMP_S") bytes=$TOTAL"
else
  echo "SHA256 MISMATCH -- differing entries:"
  diff "$TMP_S" "$TMP_D" | head -40
  say "VERIFY_FAIL"
  rm -f "$TMP_S" "$TMP_D"
  exit 1
fi
rm -f "$TMP_S" "$TMP_D"
scp -q -o BatchMode=yes "$SRCSUM" "$DEST_HOST:$DEST/source_SHA256SUMS"

# ---------- 4b. bundle verification (handoff section 6: same bundle, byte-identical) ----------
if [ $DO_BUNDLE -eq 1 ]; then
  echo "=== verifying the copied SmolVLA bundle ==="
  B_SFILES=$(find "$BUNDLE_SRC" -type f | wc -l | tr -dc '0-9')
  B_SBYTES=$(find "$BUNDLE_SRC" -type f -printf '%s\n' | awk -v OFMT=%.0f -v CONVFMT=%.0f '{s+=$1} END{print s+0}' | tr -dc '0-9')
  B_DFILES=$(ssh -o BatchMode=yes "$DEST_HOST" "find '$BUNDLE_DEST' -type f | wc -l" | tr -dc '0-9')
  B_DBYTES=$(ssh -o BatchMode=yes "$DEST_HOST" "find '$BUNDLE_DEST' -type f -printf '%s\n' | awk -v OFMT=%.0f -v CONVFMT=%.0f '{s+=\$1} END{print s+0}'" | tr -dc '0-9')
  echo "bundle src: $B_SFILES files / $B_SBYTES B   dest: $B_DFILES files / $B_DBYTES B"
  if [ "$B_SFILES" != "$B_DFILES" ] || [ "$B_SBYTES" != "$B_DBYTES" ]; then
    say "BUNDLE_VERIFY_FAIL src=$B_SFILES/$B_SBYTES dst=$B_DFILES/$B_DBYTES"
    fatal "bundle file count or total bytes disagree between $BUNDLE_SRC and $DEST_HOST:$BUNDLE_DEST"
  fi
  if [ -f "$BUNDLE_SRC/SHA256SUMS" ]; then
    ssh -o BatchMode=yes "$DEST_HOST" "cd '$BUNDLE_DEST' && sha256sum -c SHA256SUMS" >/dev/null \
      || { say "BUNDLE_VERIFY_FAIL sha256sum -c"; fatal "bundle SHA256SUMS check failed on $DEST_HOST:$BUNDLE_DEST"; }
    echo "bundle SHA256SUMS: all OK on $DEST_HOST"
  else
    echo "NOTE: bundle carries no SHA256SUMS; verified by file count + total bytes only."
  fi
  say "BUNDLE_VERIFY_OK files=$B_DFILES bytes=$B_DBYTES name=$BUNDLE_NAME src=$BUNDLE_SRC"
fi

# ---------- 5. provenance ----------
MAPJSON=/tmp/${TASK_KEY}_source_episode_map.json
"$PY" "$SOURCE_MAP_PY" "$LIST_FILE" "$SRC" > "$MAPJSON"
[ -s "$MAPJSON" ] || fatal "task_source_map.py produced an empty $MAPJSON"
"$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert len(d['episodes'])==int(sys.argv[2]), (len(d['episodes']), sys.argv[2]); print('source_episode_map.json ok:', len(d['episodes']), 'episodes,', d['outcome_counts'])" "$MAPJSON" "${#EPS[@]}"
scp -q -o BatchMode=yes "$MAPJSON" "$DEST_HOST:$DEST/source_episode_map.json"
scp -q -o BatchMode=yes "$LIST_FILE" "$DEST_HOST:$DEST/frozen_episode_list.txt"
say "PROVENANCE_WRITTEN source_episode_map.json source_SHA256SUMS frozen_episode_list.txt"

echo
echo "=== done ==="
ssh -o BatchMode=yes "$DEST_HOST" "du -sh '$DEST'; ls -la '$DEST'; df -h '$SSD' | tail -1"
