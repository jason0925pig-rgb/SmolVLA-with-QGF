#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

clone_repo() {
  local url="$1"
  local dir="$2"
  local commit="${3:-}"

  if [ -d "$dir" ] && [ ! -d "$dir/.git" ]; then
    echo "Using existing source tree at $dir"
    return
  fi

  if [ ! -d "$dir/.git" ]; then
    git clone --filter=blob:none --no-checkout "$url" "$dir"
  fi

  if [ -n "$commit" ]; then
    if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
      git -C "$dir" fetch --depth 1 origin "$commit"
    fi
    git -C "$dir" checkout "$commit"
  else
    git -C "$dir" checkout main 2>/dev/null || git -C "$dir" checkout master
  fi
}

clone_repo \
  https://github.com/huggingface/lerobot.git \
  third_party/lerobot \
  "${LEROBOT_COMMIT:-}"

clone_repo \
  https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  third_party/LIBERO \
  "${LIBERO_COMMIT:-}"

echo "Third-party checkouts are ready under third_party/."

