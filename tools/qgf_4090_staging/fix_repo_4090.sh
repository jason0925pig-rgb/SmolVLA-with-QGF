#!/bin/bash
# Normalise line endings in the exported snapshot and record it as a git commit.
# Offline; handoff section 7.
set -eu
UPSTREAM_SHA=ed0108f9544c33b0166b5b17a338bdb1ea502bb7
R=/opt/qgf_real_robot/repos/SmolVLA-with-QGF
cd "$R"

echo "=== normalising CRLF -> LF on text sources ==="
n=0
while IFS= read -r -d '' f; do
  if file "$f" | grep -q CRLF; then
    sed -i 's/\r$//' "$f"
    n=$((n + 1))
  fi
done < <(find . -path ./.git -prune -o -type f \
          \( -name '*.py' -o -name '*.sh' -o -name '*.toml' -o -name '*.cfg' \
             -o -name '*.txt' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) -print0)
echo "converted $n files"

echo "=== git init / commit ==="
rm -rf .git
git init -q .
git config user.email "chenchaosheng24@gmail.com"
git config user.name "SCC"
git config core.autocrlf false
git add -A 2>/dev/null
git commit -q -m "snapshot: SmolVLA-with-QGF upstream origin/main ${UPSTREAM_SHA} (LF-normalised)"
echo "$UPSTREAM_SHA" > /opt/qgf_real_robot/upstream_git_commit.txt

echo "--- snapshot commit ---"
git rev-parse HEAD
echo "--- upstream SHA (authoritative) ---"
cat /opt/qgf_real_robot/upstream_git_commit.txt
echo "--- worktree ---"
s=$(git status --porcelain)
if [ -z "$s" ]; then echo clean; else echo "$s" | head -5; fi
file qgf/scripts/train_real_robot_visual_iql.py

echo
echo "=== trap 1: manifest builder hardcodes 90_10 ==="
grep -n "90_10" qgf/scripts/build_real_robot_visual_iql_manifest.py | head -6
echo
echo "=== trap 2: trainer demands exactly 90 train / 10 val ==="
grep -nE "expected|== ?90|== ?10|!= ?90|!= ?10" qgf/scripts/train_real_robot_visual_iql.py | head -14
