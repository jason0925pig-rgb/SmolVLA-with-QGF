#!/bin/bash
# Stage mug baseline rollouts from Orin to the RTX4090 SSD.
# Runs ON THE ORIN.  Read-only against the source; never touches robot control.
# Per docs/qgf/RTX4090_SINGLE_Q_CRITIC_TRAINING_HANDOFF.md sections 2, 5, 6.
#
#   ./stage_mug_to_4090.sh <episode_list_file> [--bundle] [--verify-only]
#
# episode_list_file: one episode dir name per line (e.g. episode_000002), the
# frozen cohort.  Lines starting with # are ignored.
set -euo pipefail

SRC=/home/nvidia/work/telop/mug_purple_box_real_rollouts/episodes
BUNDLE_SRC=/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box
PY=/home/nvidia/work/telop/venvs/smolvla-orin/bin/python3
DEST_HOST=walle4090
SSD=/opt/qgf_real_robot
DSID=mug_purple_box_baseline50_20260829
DEST=$SSD/datasets/$DSID
BUNDLE_DEST=$SSD/policy_bundles/mug_purple_box
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

mapfile -t EPS < <(grep -vE '^[[:space:]]*(#|$)' "$LIST_FILE" | tr -d '\r' | sort -u)
[ ${#EPS[@]} -gt 0 ] || { echo "empty episode list"; exit 2; }
echo "=== cohort: ${#EPS[@]} episodes: ${EPS[0]} .. ${EPS[${#EPS[@]}-1]} ==="

# ---------- 1. source-side preflight ----------
TOTAL=0
for e in "${EPS[@]}"; do
  for f in "${FILES[@]}"; do
    p="$SRC/$e/$f"
    [ -s "$p" ] || { echo "FATAL missing/empty source file: $p"; exit 1; }
    TOTAL=$(( TOTAL + $(stat -c%s "$p") ))
  done
done
TOTAL_GIB=$(awk -v b=$TOTAL 'BEGIN{printf "%.2f", b/1073741824}')
echo "training input: $(( ${#EPS[@]} * ${#FILES[@]} )) files, ${TOTAL_GIB} GiB"

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
  echo "$*"
  ssh -o BatchMode=yes "$DEST_HOST" "printf '%s %s\n' \"\$(date -u +%FT%TZ)\" '$*' >> '$LOG'"
}
say "START cohort=${#EPS[@]} bytes=$TOTAL src=$SRC bwlimit=${BWLIMIT}KBps"

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

# ---------- 4. SHA256 both sides ----------
echo "=== hashing source (nice/ionice) ==="
SRCSUM=/tmp/mug_src_SHA256SUMS
: > "$SRCSUM"
for e in "${EPS[@]}"; do
  for f in "${FILES[@]}"; do
    ( cd "$SRC" && nice -n 19 ionice -c3 sha256sum "$e/$f" ) >> "$SRCSUM"
  done
done
echo "=== hashing destination ==="
DSTSUM=/tmp/mug_dst_SHA256SUMS
ssh -o BatchMode=yes "$DEST_HOST" "cd '$DEST/raw_episodes' && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs -r sha256sum" > "$DSTSUM"

sort -k2 "$SRCSUM" > /tmp/.mugs.$$
sort -k2 "$DSTSUM" > /tmp/.mugd.$$
if diff -q /tmp/.mugs.$$ /tmp/.mugd.$$ >/dev/null; then
  echo "SHA256 MATCH: $(wc -l < /tmp/.mugs.$$) files identical on both sides"
  say "VERIFY_OK files=$(wc -l < /tmp/.mugs.$$) bytes=$TOTAL"
else
  echo "SHA256 MISMATCH -- differing entries:"
  diff /tmp/.mugs.$$ /tmp/.mugd.$$ | head -40
  say "VERIFY_FAIL"
  rm -f /tmp/.mugs.$$ /tmp/.mugd.$$
  exit 1
fi
rm -f /tmp/.mugs.$$ /tmp/.mugd.$$
scp -q -o BatchMode=yes "$SRCSUM" "$DEST_HOST:$DEST/source_SHA256SUMS"

# ---------- 5. provenance ----------
"$PY" /tmp/mug_source_map.py "$LIST_FILE" "$SRC" > /tmp/mug_source_episode_map.json
scp -q -o BatchMode=yes /tmp/mug_source_episode_map.json "$DEST_HOST:$DEST/source_episode_map.json"
scp -q -o BatchMode=yes "$LIST_FILE" "$DEST_HOST:$DEST/frozen_episode_list.txt"
say "PROVENANCE_WRITTEN source_episode_map.json source_SHA256SUMS frozen_episode_list.txt"

echo
echo "=== done ==="
ssh -o BatchMode=yes "$DEST_HOST" "du -sh '$DEST'; ls -la '$DEST'; df -h '$SSD' | tail -1"
