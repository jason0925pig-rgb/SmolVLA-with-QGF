#!/bin/bash
# 1. Point the copied SmolVLA bundle at its own vlm/ directory (the config still
#    carries the Orin absolute path, which does not exist here).  Weights are
#    NOT touched; only two JSON path fields.  Original hashes recorded first.
# 2. Write the section 7 / section 12 environment provenance.
set -eu
E=/opt/qgf_real_robot/envs/visual_iql_py310
B=/opt/qgf_real_robot/policy_bundles/mug_purple_box
RUN=/opt/qgf_real_robot/runs/mug_purple_box_single_q_45_5_20260829
ENVD=$RUN/environment
mkdir -p "$ENVD"

echo "=== weight hash BEFORE any edit (must stay constant) ==="
sha256sum "$B/checkpoint/model.safetensors" | tee "$ENVD/bundle_model_safetensors_sha256.txt"

echo
echo "=== recording the as-received config files ==="
cp "$B/checkpoint/config.json" "$ENVD/bundle_config.json.as_received"
cp "$B/checkpoint/policy_preprocessor.json" "$ENVD/bundle_policy_preprocessor.json.as_received"

echo "=== localising vlm paths in the COPY (Orin original untouched) ==="
"$E/bin/python" - "$B" <<'PY'
import io, json, sys
base = sys.argv[1]
VLM = base + "/vlm"
changed = {"config.json": 0, "policy_preprocessor.json": 0}

p = f"{base}/checkpoint/config.json"
c = json.load(io.open(p, encoding="utf-8"))
if c.get("vlm_model_name") != VLM:
    print("  config.json vlm_model_name:", c.get("vlm_model_name"), "->", VLM)
    c["vlm_model_name"] = VLM
    changed["config.json"] += 1
c["load_vlm_weights"] = True
json.dump(c, io.open(p, "w", encoding="utf-8"), indent=2)

# tokenizer_name lives inside a LIST under "steps" - a dict-only walk silently
# changes nothing, which is exactly how this was missed once before.
q = f"{base}/checkpoint/policy_preprocessor.json"
d = json.load(io.open(q, encoding="utf-8"))
n = [0]
def walk(o):
    if isinstance(o, dict):
        for k, v in list(o.items()):
            if k == "tokenizer_name" and isinstance(v, str) and v != VLM:
                print("  policy_preprocessor tokenizer_name:", v, "->", VLM)
                o[k] = VLM
                n[0] += 1
            else:
                walk(v)
    elif isinstance(o, list):
        for x in o:
            walk(x)
walk(d)
json.dump(d, io.open(q, "w", encoding="utf-8"), indent=2)
changed["policy_preprocessor.json"] = n[0]
print("  edits:", changed)
if changed["policy_preprocessor.json"] == 0:
    print("  NOTE: tokenizer_name already local or absent")
PY

echo "=== verify no Orin path survives anywhere in the bundle configs ==="
if grep -rl "/home/nvidia" "$B"/checkpoint*/ 2>/dev/null; then
  echo "FATAL: an Orin path is still present above"; exit 1
fi
echo "  clean"

echo
echo "=== weight hash AFTER (must be identical) ==="
sha256sum "$B/checkpoint/model.safetensors"

echo
echo "=== environment provenance (handoff 7 / 12) ==="
"$E/bin/pip" freeze > "$ENVD/pip_freeze.txt"
~/miniconda3/bin/conda env export -p "$E" > "$ENVD/conda_env_export.yml" 2>/dev/null || echo "(conda export skipped)"
{
  echo "captured_at_utc: $(date -u +%FT%TZ)"
  echo "host: $(hostname)"
  echo "os: $(lsb_release -ds 2>/dev/null)"
  echo "arch: $(uname -m)"
  echo "env_prefix: $E"
  echo "python: $("$E/bin/python" --version 2>&1)"
  "$E/bin/python" -c "import torch;print('torch:',torch.__version__);print('torch_cuda:',torch.version.cuda);print('cudnn:',torch.backends.cudnn.version())"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpu_policy: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 (physical GPU 0)"
  echo "gpu_policy_note: the handoff mandates GPU 1, but LingBot's own env_gpu1.sh pins CUDA_VISIBLE_DEVICES=1, so GPU 0 was chosen by the user to avoid contending with LingBot inference."
  echo "upstream_git_commit: $(cat /opt/qgf_real_robot/upstream_git_commit.txt)"
  echo "snapshot_commit: $(cd /opt/qgf_real_robot/repos/SmolVLA-with-QGF && git rev-parse HEAD)"
  echo "policy_bundle: $B"
  echo "policy_bundle_source: /home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box (Orin)"
  echo "policy_bundle_note: this is the bundle that produced the 50 rollouts; it carries a SUPERSEDED.txt from an earlier session preferring smolvla_20260827_mug_50 (step 015000). Verified by weight SHA256 that this bundle is training step 020000. Kept deliberately: visual tokens must come from the bundle that generated the data."
} > "$ENVD/environment.txt"

nvidia-smi > "$ENVD/nvidia_smi_before_training.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$ENVD/nvidia_smi_gpus.csv"

echo "--- environment.txt ---"
cat "$ENVD/environment.txt"
echo "--- pip freeze lines: $(wc -l < "$ENVD/pip_freeze.txt") ---"
ls "$ENVD"
