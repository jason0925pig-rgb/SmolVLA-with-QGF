#!/bin/bash
# Handoff section 13, TASK PARAMETERISED: pull one task's Q critic from the
# 4090 onto the Orin.  Task-generic successor of deploy_critic_to_orin.sh.
#
# Runs ON THE ORIN.  Pure file transfer - no robot command of any kind: it does
# not power on, enable, enter servo mode, actuate the gripper, or send any arm
# motion.  It only reads and writes files under models/qgf and calls rsync/ssh.
#
# Required environment (no defaults - an unset variable is a hard failure so a
# half-configured shell can never deploy the wrong task):
#
#   QGF_SSD_ROOT         /opt/qgf_real_robot                    (path ON THE 4090)
#   QGF_RUN_ID           red_parcel_single_q_45_5_20260902
#   QGF_ORIN_DEPLOY_DIR  /home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260902
#
# The 4090 is always reached through the ssh alias `walle4090`, which carries
# IdentityFile ~/.ssh/qgf_4090_transfer_ed25519 and BindAddress 192.168.2.171
# (the /32 interface the handoff requires).  Never use walle@192.168.2.110
# directly: that bypasses the config and the bind address.
set -euo pipefail

: "${QGF_SSD_ROOT:?FATAL: QGF_SSD_ROOT is unset. Export the task contract (QGF_SSD_ROOT, QGF_RUN_ID, QGF_ORIN_DEPLOY_DIR) before running.}"
: "${QGF_RUN_ID:?FATAL: QGF_RUN_ID is unset. Export the task contract (QGF_SSD_ROOT, QGF_RUN_ID, QGF_ORIN_DEPLOY_DIR) before running.}"
: "${QGF_ORIN_DEPLOY_DIR:?FATAL: QGF_ORIN_DEPLOY_DIR is unset. Export the task contract (QGF_SSD_ROOT, QGF_RUN_ID, QGF_ORIN_DEPLOY_DIR) before running.}"

REMOTE=walle4090                                     # fixed by the handoff; never an IP
SRC=${QGF_SSD_ROOT%/}/runs/${QGF_RUN_ID}/deployment_bundle
DST=${QGF_ORIN_DEPLOY_DIR%/}
QGF_ROOT=/home/nvidia/work/telop/models/qgf

# The water-bottle critic and its pinned directory digest.  This constant is the
# value recorded during the mug and stapler deployments; it is how those two runs
# proved they changed nothing.  It is NOT task data and must never be edited to
# make a failing run pass.
OLD=$QGF_ROOT/real_17_116_single_qcritic
OLD_HASH_BEFORE=ea97f51a67de5d7509128f10e5e8bbce

WANT_FILES="SHA256SUMS
critic_member_00.pt
episode_split_45_5.json
training_input_summary.json
training_provenance.json
training_summary.json"

TMP=$(mktemp -d /tmp/qgf_deploy_XXXXXX)
trap 'rm -rf "$TMP"' EXIT

dir_digest() {   # digest of a whole directory tree (paths + contents), or of one file
  if [ -d "$1" ]; then
    find "$1" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-64
  else
    sha256sum "$1" | cut -c1-64
  fi
}

snapshot() {     # $1 = output file; one "<digest>  <path>" line per existing entry
  : > "$1"
  for entry in "$QGF_ROOT"/*; do
    [ -e "$entry" ] || continue
    printf '%s  %s\n' "$(dir_digest "$entry")" "$entry" >> "$1"
  done
  LC_ALL=C sort -o "$1" "$1"
}

echo "=== task contract ==="
echo "  QGF_SSD_ROOT        $QGF_SSD_ROOT   (on the 4090)"
echo "  QGF_RUN_ID          $QGF_RUN_ID"
echo "  QGF_ORIN_DEPLOY_DIR $DST"
echo "  source              $REMOTE:$SRC"
echo

# ---------- 0. target sanity: never overwrite an existing critic --------------
if [ "$(dirname "$DST")" != "$QGF_ROOT" ]; then
  echo "FATAL: $DST is not directly under $QGF_ROOT."
  exit 1
fi
if [ "$(basename "$DST")" != "$QGF_RUN_ID" ]; then
  echo "FATAL: the Orin directory name $(basename "$DST") does not equal QGF_RUN_ID=$QGF_RUN_ID."
  exit 1
fi
if [ "$DST" = "$OLD" ]; then
  echo "FATAL: refusing to deploy into the preserved water-bottle critic directory."
  exit 1
fi
if [ -e "$DST" ]; then
  echo "FATAL: $DST already exists.  Refusing to overwrite."
  exit 1
fi
mkdir -p "$QGF_ROOT"

# ---------- 1. protected state BEFORE -----------------------------------------
echo "=== every pre-existing critic under $QGF_ROOT (must not change) ==="
snapshot "$TMP/before"
if [ ! -s "$TMP/before" ]; then
  echo "  (none - this is the first critic on this Orin)"
else
  sed 's/^/  /' "$TMP/before"
fi

echo
echo "=== pinned water-bottle critic hash BEFORE (must not change) ==="
if [ ! -d "$OLD" ]; then
  echo "FATAL: $OLD is missing.  The handoff requires it to be preserved."
  exit 1
fi
H0=$(dir_digest "$OLD" | cut -c1-32)
echo "  $H0"
if [ "$H0" != "$OLD_HASH_BEFORE" ]; then
  echo "FATAL: the old critic already differs from what was recorded earlier."
  echo "       expected $OLD_HASH_BEFORE"
  exit 1
fi

# ---------- 2. source preflight on the 4090 -----------------------------------
echo
echo "=== source bundle on the 4090 ==="
if ! ssh -o BatchMode=yes "$REMOTE" "test -d '$SRC'"; then
  echo "FATAL: $REMOTE:$SRC does not exist.  Build it first with build_deployment_bundle_task.py."
  exit 1
fi
ssh -o BatchMode=yes "$REMOTE" "cd '$SRC' && find . -type f -printf '%P\n' | LC_ALL=C sort" > "$TMP/src_files"
sed 's/^/  /' "$TMP/src_files"
printf '%s\n' "$WANT_FILES" | LC_ALL=C sort > "$TMP/want_files"
if ! diff -q "$TMP/want_files" "$TMP/src_files" > /dev/null; then
  echo "FATAL: the remote bundle is not exactly the six section-13 files:"
  diff "$TMP/want_files" "$TMP/src_files" || true
  exit 1
fi
echo "  exactly the six section-13 files are present"

# ---------- 3. copy ------------------------------------------------------------
mkdir -p "$DST"
echo
echo "=== pulling from the 4090 ==="
rsync -a --info=progress2 "$REMOTE":"$SRC/" "$DST/"

echo
echo "=== the local copy is exactly the six section-13 files ==="
( cd "$DST" && find . -type f -printf '%P\n' | LC_ALL=C sort ) > "$TMP/dst_files"
sed 's/^/  /' "$TMP/dst_files"
if ! diff -q "$TMP/want_files" "$TMP/dst_files" > /dev/null; then
  echo "FATAL: the deployed directory is not exactly the six section-13 files:"
  diff "$TMP/want_files" "$TMP/dst_files" || true
  exit 1
fi

# ---------- 4. per-file SHA256 on both ends ------------------------------------
echo
echo "=== per-file SHA256, both sides ==="
ssh -o BatchMode=yes "$REMOTE" "cd '$SRC' && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs sha256sum" > "$TMP/src_sums"
( cd "$DST" && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs sha256sum ) > "$TMP/dst_sums"
if diff -q "$TMP/src_sums" "$TMP/dst_sums" > /dev/null; then
  echo "  SHA256 MATCH: $(wc -l < "$TMP/src_sums") files identical on both sides"
  sed 's/^/    /' "$TMP/dst_sums"
else
  echo "  SHA256 MISMATCH:"
  diff "$TMP/src_sums" "$TMP/dst_sums" || true
  exit 1
fi

echo
echo "=== the bundle's own SHA256SUMS still verifies here ==="
( cd "$DST" && sha256sum -c SHA256SUMS ) | sed 's/^/  /'

# ---------- 5. the bundle really was built for THIS directory ------------------
echo
echo "=== the provenance was built for this deployment directory ==="
if ! grep -qF "\"$DST/critic_member_00.pt\"" "$DST/training_provenance.json"; then
  echo "FATAL: training_provenance.json does not name $DST/critic_member_00.pt as"
  echo "       SMOLVLA_QGF_CRITIC_PATH.  This bundle was built for another target."
  exit 1
fi
if ! grep -qF "\"$QGF_RUN_ID\"" "$DST/training_provenance.json"; then
  echo "FATAL: training_provenance.json does not carry run_id $QGF_RUN_ID."
  exit 1
fi
echo "  provenance run_id and SMOLVLA_QGF_CRITIC_PATH both match this directory"

# ---------- 6. protected state AFTER -------------------------------------------
echo
echo "=== every pre-existing critic AFTER (must be byte-identical) ==="
snapshot "$TMP/after_all"
# each line is "<64-char digest><2 spaces><path>", so the path starts at column 67;
# compare the whole path, never a prefix, or a sibling like "<DST>_old" would be
# silently excluded from the unchanged-check.
awk -v d="$DST" 'substr($0, 67) != d' "$TMP/after_all" > "$TMP/after"
if diff -q "$TMP/before" "$TMP/after" > /dev/null; then
  echo "  UNCHANGED: $(wc -l < "$TMP/before") pre-existing entries are byte-identical"
  sed 's/^/    /' "$TMP/after"
else
  echo "FATAL: a pre-existing critic changed during deployment:"
  diff "$TMP/before" "$TMP/after" || true
  exit 1
fi

NEW_ENTRIES=$(( $(wc -l < "$TMP/after_all") - $(wc -l < "$TMP/before") ))
if [ "$NEW_ENTRIES" -ne 1 ]; then
  echo "FATAL: expected exactly one new entry under $QGF_ROOT, found $NEW_ENTRIES."
  exit 1
fi
echo "  exactly one new entry was created: $DST"

echo
echo "=== pinned water-bottle critic hash AFTER (must be unchanged) ==="
H1=$(dir_digest "$OLD" | cut -c1-32)
echo "  $H1"
if [ "$H0" != "$H1" ]; then
  echo "FATAL: the old critic changed during deployment."
  exit 1
fi
echo "  unchanged"

# ---------- 7. what is on disk now ---------------------------------------------
echo
echo "=== what is now under models/qgf ==="
ls -la "$QGF_ROOT"/
echo
ls -la "$DST"
echo
echo "no robot service was called: no power-on, no enable, no servo, no gripper"
echo "command, no arm motion.  File transfer and hashing only."
echo
echo "next: run orin_offline_smoke_task.py on this Orin before any powered test"
echo "DEPLOY OK"
