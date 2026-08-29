"""Assemble the section 13 deployment bundle on the 4090.

Produces, under runs/<run>/deployment_bundle/:
  critic_member_00.pt
  training_summary.json
  training_input_summary.json
  episode_split_45_5.json
  training_provenance.json
  SHA256SUMS

training_provenance.json is authored here because the handoff requires it but
nothing in the pipeline writes it.
"""
import hashlib
import io
import json
import shutil
import subprocess
import time
from pathlib import Path

SSD = Path("/opt/qgf_real_robot")
RUN = SSD / "runs/mug_purple_box_single_q_45_5_20260829"
OUT = RUN / "outputs/single_qcritic"
DEP = RUN / "deployment_bundle"
DS = SSD / "datasets/mug_purple_box_baseline50_20260829"
BUNDLE = SSD / "policy_bundles/mug_purple_box"
REPO = SSD / "repos/SmolVLA-with-QGF"
DEP.mkdir(parents=True, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def jload(p):
    return json.load(io.open(p, encoding="utf-8"))


def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


# ---- copy the four artefacts ------------------------------------------------
for src in (
    OUT / "critic_member_00.pt",
    OUT / "training_summary.json",
    OUT / "training_input_summary.json",
    RUN / "manifest/episode_split_45_5.json",
):
    if not src.is_file():
        raise SystemExit(f"missing artefact: {src}")
    shutil.copy2(src, DEP / src.name)
    print("copied", src.name)

# ---- provenance -------------------------------------------------------------
import torch

ck = torch.load(OUT / "critic_member_00.pt", map_location="cpu", weights_only=False)
tis = jload(OUT / "training_input_summary.json")
smap = jload(DS / "source_episode_map.json")
msum = jload(RUN / "manifest/manifest_summary.json")
hist = ck.get("history", [])
sel = ck.get("selected_epoch")
sel_entry = next((h for h in hist if h.get("epoch") == sel), {})

outcomes = {e["dest_episode_index"]: e["outcome"] for e in smap["episodes"]}
lighting = {}
for e in smap["episodes"]:
    import re

    m = re.search(r"lighting=(\w+)", str(e.get("notes", "")))
    lighting.setdefault(m.group(1) if m else "untagged", []).append(e["dest_episode_index"])

prov = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_id": RUN.name,
    "task": "mug_purple_box",
    "task_prompt": "把水杯放到紫色的箱子上",

    "machine": {
        "host": sh("hostname"),
        "os": sh("lsb_release", "-ds"),
        "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()[:1],
        "gpu_policy": "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 (physical GPU 0)",
        "gpu_policy_deviation": (
            "The handoff mandates physical GPU 1. GPU 0 was used at the user's explicit "
            "direction. Independently, LingBot's own env_gpu1.sh pins CUDA_VISIBLE_DEVICES=1, "
            "so training on GPU 0 avoids contending with LingBot inference."
        ),
        "python": sh(str(SSD / "envs/visual_iql_py310/bin/python"), "--version"),
    },

    "code": {
        "upstream_git_commit": io.open(SSD / "upstream_git_commit.txt").read().strip(),
        "snapshot_commit": sh("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "patch": "45/5 split support; see environment/patch_45_5.diff. Defaults remain 90/10.",
        "repo": str(REPO),
    },

    "policy_bundle": {
        "path_on_4090": str(BUNDLE),
        "source_on_orin": "/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box",
        "model_safetensors_sha256": sha256(BUNDLE / "checkpoint/model.safetensors"),
        "training_step": "020000 (verified by weight SHA256 against the a800 training outputs)",
        "note": (
            "This bundle carries a SUPERSEDED.txt from an earlier session preferring "
            "smolvla_20260827_mug_50 (step 015000). It is used here deliberately: it is the "
            "bundle that generated all 50 rollouts, and visual tokens must come from the "
            "policy that produced the data. The v2 replay margin between 015000 and 020000 "
            "was within noise on a 5-episode validation set."
        ),
        "config_localised": (
            "vlm_model_name and tokenizer_name were rewritten from the Orin absolute path to "
            "the 4090 path. Weights untouched (hash above matches the Orin copy)."
        ),
    },

    "dataset": {
        "path_on_4090": str(DS),
        "source_on_orin": smap["source_root"],
        "episodes": 50,
        "outcome_counts": {
            "success": sum(1 for v in outcomes.values() if v == "success"),
            "failure": sum(1 for v in outcomes.values() if v == "failure"),
        },
        "lighting_groups": {k: sorted(v) for k, v in lighting.items()},
        "lighting_caveat": (
            "Lighting is perfectly confounded with session time: normal covers ep2-31, "
            "medium covers ep0-1 and ep32-49. The observed baseline success-rate difference "
            "(normal 70.0% vs medium 40.0%) must NOT be read as a lighting effect."
        ),
        "transfer_verification": "300 files, 18812927981 bytes, per-file SHA256 identical on Orin and the 4090",
        "not_copied": smap.get("not_copied_non_training_files", []),
    },

    "manifest": {
        "aligned_chunk_count": msum.get("aligned_chunk_count"),
        "skipped": msum.get("alignment", {}).get("skipped"),
        "action_chunk_shape": msum.get("action_chunk_shape"),
        "state_dim": msum.get("state_dim"),
        "split_seed": 20260814,
        "split_strategy": "episode-level stratified by recorded outcome",
    },

    "training": {
        "train_episodes": sorted(tis["train_episode_indices"]),
        "val_episodes": sorted(tis["val_episode_indices"]),
        "train_samples": tis.get("train_samples"),
        "val_samples": tis.get("val_samples"),
        "train_positive_rewards": tis.get("train_positive_rewards"),
        "val_positive_rewards": tis.get("val_positive_rewards"),
        "args": tis.get("training_args"),
        "epochs_run": len(hist),
        "selected_epoch": sel,
        "selected_val_td_loss": ck.get("selected_val_td_loss"),
        "selection_rule": "lowest validation TD loss, not the final epoch",
        "at_selected_epoch": {
            "val_td_loss": sel_entry.get("val_td_loss"),
            "val_q_mean": sel_entry.get("val_q_mean"),
            "val_q_success_mean": sel_entry.get("val_q_success_mean"),
            "val_q_failure_mean": sel_entry.get("val_q_failure_mean"),
            "val_q_success_failure_gap": sel_entry.get("val_q_success_failure_gap"),
        },
        "metric_caveat": (
            "val_q_success_mean / val_q_failure_mean / gap are Q-value separations on 5 "
            "held-out episodes. They are NOT a success rate and must not be reported as one."
        ),
        "full_history_location": "critic_member_00.pt['history'] and logs/train_single_qcritic.log",
    },

    "critic": {
        "critic_arch": ck.get("critic_arch"),
        "critic_config": ck.get("critic_config"),
        "ensemble_size": 1,
        "uncertainty_gate": "disabled; uncertainty_scale=0.0 at deployment",
    },

    "deployment": {
        "orin_target": "/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829/",
        "preserved_untouched": "/home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic/",
        "runtime_env": {
            "SMOLVLA_QGF_CRITIC_PATH": "/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829/critic_member_00.pt",
            "QGF_RUN_MODE": "qgf",
            "QGF_BETA": "<positive number, chosen by the user>",
            "guidance_coefficient": "1 / beta",
            "grad_clip_norm": 1.0,
            "uncertainty_scale": 0.0,
            "SMOLVLA_ORIN_BUNDLE": "/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box",
            "task_prompt": "把水杯放到紫色的箱子上",
        },
    },

    "safety_declaration": (
        "No power-on, enable, servo, gripper, or arm motion command was issued to any robot "
        "at any point during data transfer, environment setup, feature extraction, training, "
        "or deployment preparation. All work was file and GPU operations only."
    ),
}

(DEP / "training_provenance.json").write_text(
    json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("wrote training_provenance.json")

# ---- SHA256SUMS -------------------------------------------------------------
lines = []
for p in sorted(DEP.iterdir()):
    if p.name == "SHA256SUMS" or not p.is_file():
        continue
    lines.append(f"{sha256(p)}  {p.name}")
(DEP / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("=== deployment bundle ===")
for line in lines:
    h, n = line.split("  ")
    print(f"  {n:34s} {(DEP / n).stat().st_size:>12,} B  {h[:16]}")
print(f"  {'SHA256SUMS':34s} {(DEP / 'SHA256SUMS').stat().st_size:>12,} B")
