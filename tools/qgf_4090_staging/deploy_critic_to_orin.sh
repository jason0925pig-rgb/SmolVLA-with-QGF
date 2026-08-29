#!/bin/bash
# Handoff section 13: pull the mug Q critic from the 4090 onto the Orin.
# Runs ON THE ORIN.  Pure file transfer - no robot command of any kind.
set -eu
SRC=/opt/qgf_real_robot/runs/mug_purple_box_single_q_45_5_20260829/deployment_bundle
DST=/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829
OLD=/home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic
OLD_HASH_BEFORE=ea97f51a67de5d7509128f10e5e8bbce

echo "=== old water-bottle critic hash BEFORE (must not change) ==="
H0=$(find "$OLD" -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-32)
echo "  $H0"
if [ "$H0" != "$OLD_HASH_BEFORE" ]; then
  echo "FATAL: the old critic already differs from what was recorded earlier."
  exit 1
fi

if [ -e "$DST" ]; then
  echo "FATAL: $DST already exists.  Refusing to overwrite."
  exit 1
fi
mkdir -p "$DST"

echo
echo "=== pulling from the 4090 ==="
rsync -a --info=progress2 walle4090:"$SRC/" "$DST/"

echo
echo "=== per-file SHA256, both sides ==="
ssh -o BatchMode=yes walle4090 "cd '$SRC' && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs sha256sum" > /tmp/critic_src_sums
( cd "$DST" && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs sha256sum ) > /tmp/critic_dst_sums
if diff -q /tmp/critic_src_sums /tmp/critic_dst_sums > /dev/null; then
  echo "  SHA256 MATCH: $(wc -l < /tmp/critic_src_sums) files identical on both sides"
  cat /tmp/critic_dst_sums | sed 's/^/    /'
else
  echo "  SHA256 MISMATCH:"
  diff /tmp/critic_src_sums /tmp/critic_dst_sums
  exit 1
fi

echo
echo "=== the bundle's own SHA256SUMS still verifies here ==="
( cd "$DST" && sha256sum -c SHA256SUMS ) | sed 's/^/  /'

echo
echo "=== old water-bottle critic hash AFTER (must be unchanged) ==="
H1=$(find "$OLD" -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-32)
echo "  $H1"
if [ "$H0" != "$H1" ]; then
  echo "FATAL: the old critic changed during deployment."
  exit 1
fi
echo "  unchanged"

echo
echo "=== what is now under models/qgf ==="
ls -la /home/nvidia/work/telop/models/qgf/
echo
ls -la "$DST"
echo
echo "DEPLOY OK"
