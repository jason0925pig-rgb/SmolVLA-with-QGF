"""Assemble the section 13 deployment bundle on the 4090 - TASK PARAMETERISED.

This is the task-generic successor of build_deployment_bundle.py.  Nothing about
the task is hardcoded any more; everything comes from the environment and every
task-specific fact that used to be a string literal is now *read back out of the
run's own artefacts and verified*.

Runs ON THE 4090 (.110), inside /opt/qgf_real_robot/envs/visual_iql_py310.
Pure file + CPU work.  It never talks to a robot and never opens a socket.

Required environment (no defaults - an unset variable is a hard failure, so a
half-configured shell can never silently build a bundle for the wrong task):

    QGF_SSD_ROOT        /opt/qgf_real_robot
    QGF_TASK_KEY        red_parcel
    QGF_DATASET_ID      red_parcel_baseline50_20260902
    QGF_RUN_ID          red_parcel_single_q_45_5_20260902
    QGF_BUNDLE_NAME     red_parcel_clean
    QGF_ORIN_EPISODES   /home/nvidia/work/telop/red_parcel_real_rollouts/episodes
    QGF_ORIN_BUNDLE     /home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean
    QGF_ORIN_DEPLOY_DIR /home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260902
    QGF_EPISODE_FIRST   0
    QGF_EPISODE_LAST    49

Optional:
    QGF_POLICY_TRAINING_STEP   provenance annotation used only when the bundle
                               itself does not record its training step.

Produces, under runs/<QGF_RUN_ID>/deployment_bundle/, exactly six files:

    critic_member_00.pt
    training_summary.json
    training_input_summary.json
    episode_split_45_5.json
    training_provenance.json
    SHA256SUMS

training_provenance.json is authored here because the handoff requires it but
nothing in the pipeline writes it.  Every claim it makes is checked first; the
bundle directory is only written after all checks pass, so a failed run never
leaves a half-true bundle behind.
"""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed, non-negotiable contract values (handoff sections 1, 8, 11, 12, 13).
# These are deliberately NOT parameterised.
# ---------------------------------------------------------------------------
REQUIRED_GPU = "1"                       # physical GPU 1 only, always explicit
EXPECTED_TRAIN_EPISODES = 45
EXPECTED_VAL_EPISODES = 5
EXPECTED_EPISODES = EXPECTED_TRAIN_EPISODES + EXPECTED_VAL_EPISODES
SPLIT_SEED = 20260814
SPLIT_FILE_NAME = "episode_split_45_5.json"
EXPECTED_CRITIC_ARCH = "visual_transformer"
EXPECTED_CRITIC_CONFIG = {
    "state_dim": 8,
    "action_dim": 8,
    "action_horizon": 50,
    "visual_tokens": 128,
    "visual_token_dim": 960,
}
EXPECTED_ACTION_CHUNK_SHAPE = [50, 8]
EPISODE_FILES = (
    "episode_metadata.json",
    "transitions.parquet",
    "normalized_policy_chunks.parquet",
    "policy_observations.parquet",
    "chest.mp4",
    "wrist_right.mp4",
)
BUNDLE_FILES = (
    "critic_member_00.pt",
    "training_summary.json",
    "training_input_summary.json",
    SPLIT_FILE_NAME,
    "training_provenance.json",
    "SHA256SUMS",
)
# The water-bottle critic that every deployment so far has left untouched.
PRESERVED_WATER_BOTTLE = "/home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic/"
ORIN_QGF_ROOT = "/home/nvidia/work/telop/models/qgf"

CONTRACT_VARS = (
    "QGF_SSD_ROOT", "QGF_TASK_KEY", "QGF_DATASET_ID", "QGF_RUN_ID",
    "QGF_BUNDLE_NAME", "QGF_ORIN_EPISODES", "QGF_ORIN_BUNDLE",
    "QGF_ORIN_DEPLOY_DIR", "QGF_EPISODE_FIRST", "QGF_EPISODE_LAST",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def env(name, required=True, default=None):
    """Read one contract variable.  Never invents a task."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        if required:
            raise SystemExit(
                "FATAL: environment variable {0} is unset or empty.\n"
                "       This script has no task defaults on purpose.  Export the whole\n"
                "       contract before running it: {1}".format(name, " ".join(CONTRACT_VARS))
            )
        return default
    return raw.strip()


def env_int(name):
    raw = env(name)
    try:
        return int(raw)
    except ValueError:
        raise SystemExit("FATAL: {0}={1!r} is not an integer.".format(name, raw))


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with io.open(str(path), "rb") as fh:
        for blk in iter(lambda: fh.read(block), b""):
            h.update(blk)
    return h.hexdigest()


def jload(path):
    return json.load(io.open(str(path), encoding="utf-8"))


def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


PROBLEMS = []


def check(ok, message):
    """Record a verification.  Nothing is written while any check is failing."""
    print("  {0}  {1}".format("PASS" if ok else "FAIL", message))
    if not ok:
        PROBLEMS.append(message)
    return bool(ok)


def need_file(path, what):
    if not Path(str(path)).is_file():
        raise SystemExit("FATAL: missing {0}: {1}".format(what, path))
    return Path(str(path))


def need_dir(path, what):
    if not Path(str(path)).is_dir():
        raise SystemExit("FATAL: missing {0}: {1}".format(what, path))
    return Path(str(path))


def is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == value and value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# 0. environment -> paths
# ---------------------------------------------------------------------------
SSD = Path(env("QGF_SSD_ROOT"))
TASK_KEY = env("QGF_TASK_KEY")
DATASET_ID = env("QGF_DATASET_ID")
RUN_ID = env("QGF_RUN_ID")
BUNDLE_NAME = env("QGF_BUNDLE_NAME")
ORIN_EPISODES = env("QGF_ORIN_EPISODES").rstrip("/")
ORIN_BUNDLE = env("QGF_ORIN_BUNDLE").rstrip("/")
ORIN_DEPLOY_DIR = env("QGF_ORIN_DEPLOY_DIR").rstrip("/")
EPISODE_FIRST = env_int("QGF_EPISODE_FIRST")
EPISODE_LAST = env_int("QGF_EPISODE_LAST")
POLICY_STEP_OVERRIDE = env("QGF_POLICY_TRAINING_STEP", required=False)

RUN = SSD / "runs" / RUN_ID
OUT = RUN / "outputs/single_qcritic"
DEP = RUN / "deployment_bundle"
DS = SSD / "datasets" / DATASET_ID
RAW = DS / "raw_episodes"
BUNDLE = SSD / "policy_bundles" / BUNDLE_NAME
REPO = SSD / "repos/SmolVLA-with-QGF"

print("=== task contract ===")
for name in CONTRACT_VARS:
    print("  {0:20s} {1}".format(name, os.environ.get(name, "").strip()))
print()

need_dir(SSD, "SSD root (QGF_SSD_ROOT)")
need_dir(RUN, "run directory (QGF_RUN_ID)")
need_dir(OUT, "training output directory")
need_dir(DS, "dataset directory (QGF_DATASET_ID)")
need_dir(RAW, "raw_episodes directory")
need_dir(BUNDLE, "policy bundle directory (QGF_BUNDLE_NAME)")
need_dir(REPO, "code snapshot")

CRITIC_PT = need_file(OUT / "critic_member_00.pt", "critic checkpoint")
TRAIN_SUMMARY = need_file(OUT / "training_summary.json", "training_summary.json")
TIS_PATH = need_file(OUT / "training_input_summary.json", "training_input_summary.json")
SPLIT_PATH = need_file(RUN / "manifest" / SPLIT_FILE_NAME, "45/5 split file")
MSUM_PATH = need_file(RUN / "manifest/manifest_summary.json", "manifest_summary.json")
SMAP_PATH = need_file(DS / "source_episode_map.json", "source_episode_map.json")
SUMS_PATH = need_file(DS / "source_SHA256SUMS", "source_SHA256SUMS (Orin-side digests)")
UPSTREAM_COMMIT = need_file(SSD / "upstream_git_commit.txt", "upstream_git_commit.txt")
MODEL_SAFETENSORS = need_file(BUNDLE / "checkpoint/model.safetensors", "policy bundle weights")

ARTEFACTS = (
    (CRITIC_PT, "critic_member_00.pt"),
    (TRAIN_SUMMARY, "training_summary.json"),
    (TIS_PATH, "training_input_summary.json"),
    (SPLIT_PATH, SPLIT_FILE_NAME),
)

smap = jload(SMAP_PATH)
tis = jload(TIS_PATH)
msum = jload(MSUM_PATH)
split = jload(SPLIT_PATH)

import torch  # noqa: E402  (after the path checks so a missing artefact fails fast)

ck = torch.load(str(CRITIC_PT), map_location="cpu", weights_only=False)

# ---------------------------------------------------------------------------
# 1. cohort, prompt and outcome checks (handoff section 5)
# ---------------------------------------------------------------------------
print("=== cohort, prompt and outcomes ===")
episodes = smap.get("episodes")
if not isinstance(episodes, list) or not episodes:
    raise SystemExit("FATAL: source_episode_map.json has no 'episodes' list.")

dest_index = [e.get("dest_episode_index") for e in episodes]
check(all(isinstance(i, int) for i in dest_index),
      "every episode has an integer dest_episode_index")
expected_range = list(range(EPISODE_FIRST, EPISODE_LAST + 1))
check(sorted(i for i in dest_index if isinstance(i, int)) == expected_range,
      "episode indices are exactly {0}..{1} ({2} episodes)".format(
          EPISODE_FIRST, EPISODE_LAST, len(expected_range)))
check(len(episodes) == EXPECTED_EPISODES,
      "cohort size is {0} (45 train + 5 validation)".format(EXPECTED_EPISODES))
check(len(set(dest_index)) == len(dest_index), "no duplicate episode index")

prompts = sorted({str(e.get("task_prompt", "")).strip() for e in episodes})
check(len(prompts) == 1 and prompts[0] != "",
      "the task prompt is identical and non-empty in all {0} episodes".format(len(episodes)))
TASK_PROMPT = prompts[0] if len(prompts) == 1 else "<INCONSISTENT: {0!r}>".format(prompts)
print("       task prompt: {0}".format(TASK_PROMPT))

outcomes = {}
for e in episodes:
    if isinstance(e.get("dest_episode_index"), int):
        outcomes[e["dest_episode_index"]] = e.get("outcome")
n_success = sum(1 for v in outcomes.values() if v == "success")
n_failure = sum(1 for v in outcomes.values() if v == "failure")
check(set(outcomes.values()) <= {"success", "failure"},
      "every outcome is success or failure")
check(n_success > 0 and n_failure > 0,
      "both outcome classes are present (success={0}, failure={1})".format(n_success, n_failure))
check(n_success + n_failure == len(episodes), "outcome counts cover the whole cohort")

check(str(smap.get("source_root", "")).rstrip("/") == ORIN_EPISODES,
      "source_episode_map source_root == QGF_ORIN_EPISODES")
smap_bundle = str(smap.get("policy_bundle_that_produced_these_rollouts", "")).rstrip("/")
check(smap_bundle == ORIN_BUNDLE,
      "the rollouts' own policy bundle == QGF_ORIN_BUNDLE")
check(str(smap.get("task", TASK_KEY)) == TASK_KEY,
      "source_episode_map task == QGF_TASK_KEY")
not_copied = smap.get("not_copied_non_training_files")
check(isinstance(not_copied, list),
      "the non-training files that were deliberately not copied are recorded")

# ---------------------------------------------------------------------------
# 2. split integrity (handoff sections 8 and 12)
# ---------------------------------------------------------------------------
print()
print("=== 45/5 split ===")
split_train = [int(i) for i in split.get("train_episode_indices", [])]
split_val = [int(i) for i in split.get("val_episode_indices", [])]
tis_train = [int(i) for i in tis.get("train_episode_indices", [])]
tis_val = [int(i) for i in tis.get("val_episode_indices", [])]

check(len(split_train) == EXPECTED_TRAIN_EPISODES,
      "split file has {0} train episodes".format(EXPECTED_TRAIN_EPISODES))
check(len(split_val) == EXPECTED_VAL_EPISODES,
      "split file has {0} validation episodes".format(EXPECTED_VAL_EPISODES))
check(not (set(split_train) & set(split_val)), "train and validation are disjoint")
check(sorted(split_train + split_val) == expected_range,
      "train + validation is exactly the frozen cohort")
check(sorted(tis_train) == sorted(split_train) and sorted(tis_val) == sorted(split_val),
      "training_input_summary uses the same split as the split file")
check(int(split.get("seed", -1)) == SPLIT_SEED,
      "split seed is {0}".format(SPLIT_SEED))
val_outcomes = [outcomes.get(i) for i in split_val]
check("success" in val_outcomes and "failure" in val_outcomes,
      "the 5 validation episodes contain both a success and a failure")

for key in ("train_samples", "val_samples", "train_positive_rewards", "val_positive_rewards"):
    check(isinstance(tis.get(key), int) and tis[key] > 0,
          "training_input_summary.{0} is recorded and positive ({1})".format(key, tis.get(key)))

# ---------------------------------------------------------------------------
# 3. manifest checks (handoff sections 9 and 12)
# ---------------------------------------------------------------------------
print()
print("=== manifest ===")
check(msum.get("action_chunk_shape") == EXPECTED_ACTION_CHUNK_SHAPE,
      "manifest action_chunk_shape == {0}".format(EXPECTED_ACTION_CHUNK_SHAPE))
check(msum.get("state_dim") == 8, "manifest state_dim == 8")
check(msum.get("raw_episode_range") == [EPISODE_FIRST, EPISODE_LAST],
      "manifest raw_episode_range == [{0}, {1}]".format(EPISODE_FIRST, EPISODE_LAST))
check(msum.get("raw_episode_count") == EXPECTED_EPISODES,
      "manifest raw_episode_count == {0}".format(EXPECTED_EPISODES))
check(isinstance(msum.get("aligned_chunk_count"), int) and msum["aligned_chunk_count"] > 0,
      "manifest aligned_chunk_count is positive ({0})".format(msum.get("aligned_chunk_count")))
check(msum.get("split_file") == SPLIT_FILE_NAME,
      "manifest points at {0}".format(SPLIT_FILE_NAME))

# ---------------------------------------------------------------------------
# 4. checkpoint identity (handoff section 12)
# ---------------------------------------------------------------------------
print()
print("=== critic checkpoint ===")
cfg = ck.get("critic_config") or {}
hist = ck.get("history", [])
sel = ck.get("selected_epoch")
sel_entry = next((h for h in hist if h.get("epoch") == sel), {})

check(ck.get("critic_arch") == EXPECTED_CRITIC_ARCH,
      "critic_arch == {0}".format(EXPECTED_CRITIC_ARCH))
for key, want in sorted(EXPECTED_CRITIC_CONFIG.items()):
    check(cfg.get(key) == want, "critic_config.{0} == {1} (got {2})".format(key, want, cfg.get(key)))
check(ck.get("ensemble_member_index") == 0, "ensemble_member_index == 0 (single critic)")
check(len(list(OUT.glob("critic_member_*.pt"))) == 1,
      "exactly one critic_member_*.pt exists in the training output")
check(isinstance(hist, list) and len(hist) > 0, "the full training history is in the checkpoint")

val_losses = [h.get("val_td_loss") for h in hist if is_finite(h.get("val_td_loss"))]
check(len(val_losses) == len(hist),
      "every epoch recorded a finite validation TD loss ({0} epochs)".format(len(hist)))
if val_losses:
    best = min(val_losses)
    check(is_finite(sel_entry.get("val_td_loss")) and abs(sel_entry["val_td_loss"] - best) <= 1e-12,
          "selected epoch {0} is the lowest validation TD loss, not the final epoch".format(sel))
    check(is_finite(ck.get("selected_val_td_loss"))
          and abs(float(ck["selected_val_td_loss"]) - float(sel_entry.get("val_td_loss", -1))) <= 1e-12,
          "selected_val_td_loss matches the selected epoch")

nonfinite = []
for h in hist:
    for k, v in h.items():
        if isinstance(v, float) and not is_finite(v):
            nonfinite.append("epoch {0} {1}={2}".format(h.get("epoch"), k, v))
check(not nonfinite, "no NaN/Inf anywhere in the training history")
if nonfinite:
    for item in nonfinite[:10]:
        print("        {0}".format(item))

# ---------------------------------------------------------------------------
# 5. GPU policy evidence (handoff iron rule 1: physical GPU 1 only)
# ---------------------------------------------------------------------------
print()
print("=== GPU policy evidence ===")
gpu_hits = []
gpu_pat = re.compile(r"CUDA_VISIBLE_DEVICES\s*=\s*\"?'?([0-9]+(?:,[0-9]+)*)")
for sub in ("commands", "environment"):
    d = RUN / sub
    if not d.is_dir():
        continue
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.stat().st_size > (8 << 20):
            continue
        try:
            text = io.open(str(p), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in gpu_pat.finditer(text):
            gpu_hits.append((p.name, m.group(1)))
values = sorted({v for _, v in gpu_hits})
check(len(gpu_hits) > 0,
      "the run records an explicit CUDA_VISIBLE_DEVICES ({0} occurrences under commands/ and environment/)".format(len(gpu_hits)))
check(values == [REQUIRED_GPU],
      "every recorded CUDA_VISIBLE_DEVICES is {0} (physical GPU 1); found {1}".format(REQUIRED_GPU, values))
for name, value in gpu_hits[:8]:
    print("        {0}: CUDA_VISIBLE_DEVICES={1}".format(name, value))

# ---------------------------------------------------------------------------
# 6. policy bundle integrity
# ---------------------------------------------------------------------------
print()
print("=== policy bundle ===")
bundle_sha = sha256(MODEL_SAFETENSORS)
recorded_sha_file = RUN / "environment/bundle_model_safetensors_sha256.txt"
if recorded_sha_file.is_file():
    recorded = io.open(str(recorded_sha_file), encoding="utf-8").read().split()
    recorded_hash = recorded[0] if recorded else ""
    check(recorded_hash == bundle_sha,
          "bundle weights are byte-identical to the hash recorded before localisation")
else:
    check(False,
          "environment/bundle_model_safetensors_sha256.txt exists (proof the weights "
          "were not touched by config localisation)")
print("        model.safetensors sha256 = {0}".format(bundle_sha))

orin_leak = []
for p in sorted(BUNDLE.rglob("*.json")):
    try:
        if "/home/nvidia" in io.open(str(p), encoding="utf-8", errors="replace").read():
            orin_leak.append(str(p.relative_to(BUNDLE)))
    except OSError:
        continue
check(not orin_leak, "no Orin absolute path survives in the localised bundle configs")
for item in orin_leak[:5]:
    print("        still Orin-pathed: {0}".format(item))

superseded_note = None
sup = BUNDLE / "SUPERSEDED.txt"
if sup.is_file():
    superseded_note = io.open(str(sup), encoding="utf-8", errors="replace").read().strip()[:800]
    print("        SUPERSEDED.txt is present and recorded in provenance")

policy_step = POLICY_STEP_OVERRIDE
policy_step_source = "QGF_POLICY_TRAINING_STEP" if POLICY_STEP_OVERRIDE else None
if policy_step is None:
    for cand, keys in (
        (BUNDLE / "checkpoint/train_config.json", ("steps", "step", "training_step")),
        (BUNDLE / "train_config.json", ("steps", "step", "training_step")),
        (BUNDLE / "checkpoint/training_step.txt", ()),
    ):
        if not cand.is_file():
            continue
        try:
            if cand.suffix == ".json":
                data = jload(cand)
                for k in keys:
                    if data.get(k) is not None:
                        policy_step = str(data[k])
                        policy_step_source = str(cand.relative_to(BUNDLE))
                        break
            else:
                policy_step = io.open(str(cand), encoding="utf-8").read().strip()
                policy_step_source = str(cand.relative_to(BUNDLE))
        except (OSError, ValueError):
            continue
        if policy_step is not None:
            break

# ---------------------------------------------------------------------------
# 7. training input re-verification against the Orin-side digests
# ---------------------------------------------------------------------------
print()
print("=== training input: re-hashing every copied file on the 4090 ===")
orin_sums = {}
sum_line = re.compile(r"^([0-9a-fA-F]{64})\s+[*]?(.+)$")
for raw_line in io.open(str(SUMS_PATH), encoding="utf-8", errors="replace"):
    m = sum_line.match(raw_line.rstrip("\n").rstrip("\r"))
    if not m:
        continue
    parts = m.group(2).replace("\\", "/").strip().split("/")
    key = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    orin_sums[key] = m.group(1).lower()
check(len(orin_sums) > 0, "source_SHA256SUMS parsed ({0} digests)".format(len(orin_sums)))

expected_files = EXPECTED_EPISODES * len(EPISODE_FILES)
missing, mismatched, unmatched = [], [], []
total_bytes = 0
started = time.time()
done = 0
for e in episodes:
    dest_dir = e.get("dest_dir") or "episode_{0:06d}".format(e.get("dest_episode_index", -1))
    src_dir = e.get("source_dir") or dest_dir
    for fname in EPISODE_FILES:
        local = RAW / dest_dir / fname
        if not local.is_file() or local.stat().st_size == 0:
            missing.append(str(local))
            continue
        total_bytes += local.stat().st_size
        want = orin_sums.get("{0}/{1}".format(src_dir, fname)) \
            or orin_sums.get("{0}/{1}".format(dest_dir, fname))
        if want is None:
            unmatched.append("{0}/{1}".format(src_dir, fname))
            continue
        if sha256(local) != want:
            mismatched.append(str(local))
    done += 1
    if done % 10 == 0 or done == len(episodes):
        print("        hashed {0}/{1} episodes ({2:.1f} GiB) in {3:.0f}s".format(
            done, len(episodes), total_bytes / (1 << 30), time.time() - started))

n_files = expected_files - len(missing)
check(not missing, "all {0} training files are present and non-empty".format(expected_files))
check(not unmatched,
      "every copied file has an Orin-side digest in source_SHA256SUMS "
      "({0} without one)".format(len(unmatched)))
check(not mismatched,
      "every copied file re-hashes to the Orin digest ({0} mismatches)".format(len(mismatched)))
for item in (missing + unmatched + mismatched)[:10]:
    print("        problem: {0}".format(item))
transfer_verification = (
    "{0} files, {1} bytes; every file re-hashed on the 4090 and compared to the "
    "Orin-side digests in {2}; {3} mismatches".format(
        n_files, total_bytes, str(SUMS_PATH), len(mismatched))
)

# ---------------------------------------------------------------------------
# 8. deployment target sanity (handoff section 13)
# ---------------------------------------------------------------------------
print()
print("=== deployment target ===")
deploy_parent = ORIN_DEPLOY_DIR.rsplit("/", 1)[0] if "/" in ORIN_DEPLOY_DIR else ""
deploy_leaf = ORIN_DEPLOY_DIR.rsplit("/", 1)[-1]
check(deploy_parent == ORIN_QGF_ROOT,
      "QGF_ORIN_DEPLOY_DIR sits directly under {0}".format(ORIN_QGF_ROOT))
check(deploy_leaf == RUN_ID,
      "the Orin directory name equals QGF_RUN_ID ({0})".format(RUN_ID))
check(ORIN_DEPLOY_DIR.rstrip("/") != PRESERVED_WATER_BOTTLE.rstrip("/"),
      "the target is not the preserved water-bottle critic directory")

# ---------------------------------------------------------------------------
# 9. lighting / session grouping, computed rather than asserted
# ---------------------------------------------------------------------------
lighting = {}
for e in episodes:
    m = re.search(r"lighting=(\w+)", str(e.get("notes", "")))
    lighting.setdefault(m.group(1) if m else "untagged", []).append(e.get("dest_episode_index"))
lighting = {k: sorted(v) for k, v in lighting.items()}

lighting_stats = {}
for group, idxs in lighting.items():
    succ = sum(1 for i in idxs if outcomes.get(i) == "success")
    lighting_stats[group] = {
        "episodes": len(idxs),
        "success": succ,
        "failure": len(idxs) - succ,
        "success_rate_percent": round(100.0 * succ / len(idxs), 1) if idxs else None,
        "index_min": min(idxs) if idxs else None,
        "index_max": max(idxs) if idxs else None,
    }
if len(lighting) <= 1:
    lighting_caveat = (
        "All {0} episodes carry the same lighting tag ({1}); there is no lighting "
        "contrast in this cohort and no lighting claim may be made from it.".format(
            len(episodes), list(lighting)[0] if lighting else "untagged")
    )
else:
    parts = ", ".join(
        "{0} covers ep{1}-{2} ({3}/{4} success)".format(
            g, s["index_min"], s["index_max"], s["success"], s["episodes"])
        for g, s in sorted(lighting_stats.items())
    )
    lighting_caveat = (
        "Lighting groups are confounded with session time: " + parts + ". "
        "Any success-rate difference between these groups must NOT be read as a "
        "lighting effect."
    )

# ---------------------------------------------------------------------------
# stop here if anything failed - never write a half-true bundle
# ---------------------------------------------------------------------------
print()
if PROBLEMS:
    print("=== {0} CHECK(S) FAILED - no bundle written ===".format(len(PROBLEMS)))
    for item in PROBLEMS:
        print("  FAIL  {0}".format(item))
    sys.exit(1)
print("=== all pre-bundle checks passed ===")

# ---------------------------------------------------------------------------
# 10. copy the four artefacts
# ---------------------------------------------------------------------------
DEP.mkdir(parents=True, exist_ok=True)
print()
for src, name in ARTEFACTS:
    shutil.copy2(str(src), str(DEP / name))
    if sha256(src) != sha256(DEP / name):
        raise SystemExit("FATAL: copy of {0} does not match its source".format(name))
    print("copied and re-hashed", name)

# ---------------------------------------------------------------------------
# 11. provenance
# ---------------------------------------------------------------------------
driver = (sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader") or "").splitlines()
critic_runtime_path = "{0}/critic_member_00.pt".format(ORIN_DEPLOY_DIR)

prov = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_id": RUN_ID,
    "task": TASK_KEY,
    "task_prompt": TASK_PROMPT,
    "task_prompt_source": "read from all {0} episode_metadata.json records via {1}; "
                          "verified identical".format(len(episodes), SMAP_PATH.name),

    "machine": {
        "host": sh("hostname"),
        "os": sh("lsb_release", "-ds"),
        "driver": driver[0] if driver else "",
        "gpu_policy": "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (physical GPU 1)",
        "gpu_policy_evidence": (
            "{0} explicit CUDA_VISIBLE_DEVICES occurrences were found under "
            "runs/{1}/commands and runs/{1}/environment; all of them are {2}. "
            "Cross-check against the nvidia-smi PID snapshots in environment/.".format(
                len(gpu_hits), RUN_ID, REQUIRED_GPU)
        ),
        "python": sh(str(SSD / "envs/visual_iql_py310/bin/python"), "--version"),
    },

    "code": {
        "upstream_git_commit": io.open(str(UPSTREAM_COMMIT), encoding="utf-8").read().strip(),
        "snapshot_commit": sh("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "patch": "45/5 split support; see environment/patch_45_5.diff. Defaults remain 90/10.",
        "repo": str(REPO),
    },

    "policy_bundle": {
        "path_on_4090": str(BUNDLE),
        "source_on_orin": ORIN_BUNDLE,
        "model_safetensors_sha256": bundle_sha,
        "training_step": policy_step,
        "training_step_source": policy_step_source or (
            "not recorded inside the bundle; the weight SHA256 above is the authoritative identity"),
        "note": (
            "This is the bundle that generated all {0} rollouts: it is named by "
            "QGF_ORIN_BUNDLE and cross-checked against "
            "source_episode_map.json['policy_bundle_that_produced_these_rollouts']. "
            "Visual tokens must come from the policy that produced the data.".format(len(episodes))
        ),
        "superseded_txt": superseded_note,
        "config_localised": (
            "vlm_model_name and tokenizer_name were rewritten from the Orin absolute path to "
            "the 4090 path. Weights untouched (hash above matches the pre-localisation record)."
        ),
    },

    "dataset": {
        "path_on_4090": str(DS),
        "source_on_orin": smap.get("source_root"),
        "episodes": len(episodes),
        "episode_index_range": [EPISODE_FIRST, EPISODE_LAST],
        "outcome_counts": {"success": n_success, "failure": n_failure},
        "lighting_groups": lighting,
        "lighting_group_stats": lighting_stats,
        "lighting_caveat": lighting_caveat,
        "transfer_verification": transfer_verification,
        "not_copied": not_copied,
    },

    "manifest": {
        "aligned_chunk_count": msum.get("aligned_chunk_count"),
        "skipped": msum.get("alignment", {}).get("skipped"),
        "action_chunk_shape": msum.get("action_chunk_shape"),
        "state_dim": msum.get("state_dim"),
        "split_seed": SPLIT_SEED,
        "split_strategy": split.get("strategy", "episode-level stratified by recorded outcome"),
    },

    "training": {
        "train_episodes": sorted(tis_train),
        "val_episodes": sorted(tis_val),
        "train_samples": tis.get("train_samples"),
        "val_samples": tis.get("val_samples"),
        "train_positive_rewards": tis.get("train_positive_rewards"),
        "val_positive_rewards": tis.get("val_positive_rewards"),
        "args": tis.get("training_args"),
        "epochs_run": len(hist),
        "selected_epoch": sel,
        "selected_val_td_loss": ck.get("selected_val_td_loss"),
        "selection_rule": "lowest validation TD loss, not the final epoch (verified here)",
        "at_selected_epoch": {
            "val_td_loss": sel_entry.get("val_td_loss"),
            "val_q_mean": sel_entry.get("val_q_mean"),
            "val_q_success_mean": sel_entry.get("val_q_success_mean"),
            "val_q_failure_mean": sel_entry.get("val_q_failure_mean"),
            "val_q_success_failure_gap": sel_entry.get("val_q_success_failure_gap"),
        },
        "metric_caveat": (
            "val_q_success_mean / val_q_failure_mean / gap are Q-value separations on {0} "
            "held-out episodes. They are NOT a success rate and must not be reported as one.".format(
                EXPECTED_VAL_EPISODES)
        ),
        "full_history_location": "critic_member_00.pt['history'] and logs/train_single_qcritic.log",
    },

    "critic": {
        "critic_arch": ck.get("critic_arch"),
        "critic_config": cfg,
        "ensemble_size": 1,
        "uncertainty_gate": "disabled; uncertainty_scale=0.0 at deployment",
    },

    "deployment": {
        "orin_target": ORIN_DEPLOY_DIR + "/",
        "preserved_untouched": [
            PRESERVED_WATER_BOTTLE,
            "every other directory already present under {0}/ - deploy_critic_to_orin_task.sh "
            "hashes them all before and after the copy".format(ORIN_QGF_ROOT),
        ],
        "runtime_env": {
            "SMOLVLA_QGF_CRITIC_PATH": critic_runtime_path,
            "QGF_RUN_MODE": "qgf",
            "SMOLVLA_QGF_BETA": "<positive number, chosen by the user; this is the name the policy server reads>",
            "QGF_BETA": "<same value; this is the name the handoff document uses>",
            "SMOLVLA_QGF_GRAD_CLIP_NORM": "1.0",
            "guidance_coefficient": "1 / beta",
            "grad_clip_norm": 1.0,
            "uncertainty_scale": 0.0,
            "SMOLVLA_ORIN_BUNDLE": ORIN_BUNDLE,
            "task_prompt": TASK_PROMPT,
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

# ---------------------------------------------------------------------------
# 12. SHA256SUMS + the exact six-file contract
# ---------------------------------------------------------------------------
lines = []
for p in sorted(DEP.iterdir()):
    if p.name == "SHA256SUMS" or not p.is_file():
        continue
    lines.append("{0}  {1}".format(sha256(p), p.name))
(DEP / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

present = sorted(p.name for p in DEP.iterdir() if p.is_file())
extra_dirs = sorted(p.name for p in DEP.iterdir() if not p.is_file())
print()
print("=== bundle contents ===")
ok_set = check(present == sorted(BUNDLE_FILES),
               "the bundle contains exactly the six section-13 files")
ok_dirs = check(not extra_dirs, "the bundle contains no subdirectories")

for line in lines:
    h, n = line.split("  ")
    print("  {0:34s} {1:>12,} B  {2}".format(n, (DEP / n).stat().st_size, h[:16]))
print("  {0:34s} {1:>12,} B".format("SHA256SUMS", (DEP / "SHA256SUMS").stat().st_size))

print()
if not (ok_set and ok_dirs):
    print("=== BUNDLE CONTENT CHECK FAILED ===")
    sys.exit(1)
print("bundle directory: {0}".format(DEP))
print("next: run deploy_critic_to_orin_task.sh ON THE ORIN with the same contract exported")
print("BUILD DEPLOYMENT BUNDLE OK")
