#!/bin/bash
# Lay down a clean snapshot of SmolVLA-with-QGF @ origin/main on the 4090 SSD
# and turn it into a real git repo so the 45/5 patch is a recorded commit.
# Offline; handoff section 7.
set -eu
UPSTREAM_SHA=ed0108f9544c33b0166b5b17a338bdb1ea502bb7
R=/opt/qgf_real_robot/repos/SmolVLA-with-QGF
rm -rf "$R"
mkdir -p "$R"
tar -xf /tmp/qgf_main.tar -C "$R"
cd "$R"

git init -q .
git config user.email "chenchaosheng24@gmail.com"
git config user.name "SCC"
git add -A
git commit -q -m "snapshot: SmolVLA-with-QGF upstream origin/main ${UPSTREAM_SHA}

Clean tree exported with git archive from the laptop clone (that clone is
shallow, so a full-history bundle was not possible).  Upstream SHA above is
authoritative for provenance."
echo "$UPSTREAM_SHA" > /opt/qgf_real_robot/upstream_git_commit.txt

echo "=== local snapshot commit ==="
git rev-parse HEAD
echo "=== upstream SHA (authoritative) ==="
cat /opt/qgf_real_robot/upstream_git_commit.txt
echo "=== worktree clean check ==="
s=$(git status --porcelain)
if [ -z "$s" ]; then echo "clean"; else echo "$s" | head -5; fi

echo "=== required files ==="
for f in \
  qgf/scripts/build_real_robot_visual_iql_manifest.py \
  qgf/scripts/extract_smolvla_visual_features.py \
  qgf/scripts/train_real_robot_visual_iql.py \
  qgf/src/guided_action_flow/critics/visual_transformer_critic.py \
  qgf/src/guided_action_flow/critics/checkpoint.py \
  qgf/pyproject.toml
do
  printf "  %-68s " "$f"
  if [ -f "$f" ]; then echo OK; else echo MISSING; fi
done

echo
echo "=== trap 1: manifest builder hardcodes 90_10 in the filename ==="
grep -n "90_10" qgf/scripts/build_real_robot_visual_iql_manifest.py | head -6
echo
echo "=== trap 2: trainer demands exactly 90 train / 10 val ==="
grep -nE "90|10" qgf/scripts/train_real_robot_visual_iql.py | grep -iE "train|valid|episode|expect|assert|raise" | head -12
