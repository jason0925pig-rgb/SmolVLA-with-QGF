#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs/_logs

setsid bash scripts/remote_bootstrap_a800_libero.sh \
  > runs/_logs/bootstrap_a800_libero.log \
  2>&1 \
  < /dev/null &

pid=$!
echo "$pid" > runs/_logs/bootstrap_a800_libero.pid
echo "$pid"
