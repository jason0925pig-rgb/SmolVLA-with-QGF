"""Offline QGF beta-selection probe.  RUNS ON THE 4090.  NEVER TOUCHES THE ROBOT.

WHY THIS EXISTS
---------------
Beta is the only free knob of the deployed QGF guidance
(``guided_velocity = velocity_t - grad_a Q / beta``).  Shopping for it on the
robot would spend scarce robot time AND would bias the headline
QGF-vs-baseline number, because the same episodes would then be used to pick
beta and to report the effect.  This probe picks beta offline, on the five
episodes the critic was never trained on (7, 13, 25, 32, 46), so that every
robot episode collected afterwards is untouched by the selection.

SAFETY
------
Pure tensors.  No ROS import, no robot module, no network.  Three guards:
  1. an import guard that refuses every ROS / robot-message module;
  2. a socket guard that refuses every non-loopback connect and every
     non-loopback name resolution (HF/transformers are forced offline);
  3. a static self-audit that re-reads this file and refuses to run if any
     robot-actuation token appears in its executable body.
It powers nothing on, enables nothing, and publishes no topic.

WHAT IS MEASURED, per candidate beta, on the five held-out episodes only
-----------------------------------------------------------------------
  (1) RELATIVE PERTURBATION   ratio = ||grad/beta|| / ||velocity_t||, chunk-level
      L2 norms, one value per denoise step per sample, aggregated as
      median / p10 / p90.  Reported twice: restricted to the 8 real action
      dimensions the critic touches (headline, because the guidance gradient is
      exactly zero on the 24 padded dimensions) and over all 32 sampler
      dimensions.  ~0 means the guidance is a no-op; >>1 means it overwhelms
      the learned flow.
  (2) DOES GUIDANCE RAISE THE CRITIC'S OWN SCORE   Q(clean action after guided
      denoising) - Q(clean action after unguided denoising), on the SAME
      observation and the SAME initial noise.  Mean delta-Q and the fraction of
      paired samples with delta-Q > 0.
  (3) DOES THE GUIDED ACTION STAY IN DISTRIBUTION   fraction of guided action
      elements outside the per-dimension p01-p99 band and outside the hard
      min-max of the normalized action chunks from the 45 TRAINING episodes.
      The same fractions are reported for the UNGUIDED chunk, because without
      that baseline an "out of support" number cannot be attributed to the
      guidance.
  (4) GRADIENT CLIPPING RATE   fraction of denoise steps whose raw ||grad||
      exceeded grad_clip_norm (1.0) and was therefore rescaled.  Where clipping
      is frequent the 1/beta scaling stops being linear and the sweep says so.

EXACTNESS CONTRACT - READ THIS BEFORE TRUSTING A NUMBER
-------------------------------------------------------
The script runs in one of two modes and always states which one it used, both
in the JSON report and on stdout.

  MODE "full"      The real SmolVLA flow model from the localised bundle is
                   loaded and the real 10-step reverse-time sampler is run
                   through the production QGF hook.  Measurements 1, 2, 3 and 4
                   are all EXACT: velocity_t is the model's own velocity, and
                   the compared actions are real denoiser outputs.

  MODE "critic_only"  The SmolVLA model could not be loaded, or its input
                   frames could not be decoded, or --no-policy was passed.
                   Then:
                     - measurement 1 is INCOMPLETE.  ||grad/beta|| is reported
                       exactly, but ||velocity_t|| WAS NOT MEASURED, so no
                       ratio is reported at all.  Nothing is substituted for it.
                     - measurement 4 is EXACT (it only needs the critic).
                     - measurements 2 and 3 are computed EXACTLY but on a
                       SURROGATE action: the recorded normalized action chunk
                       a0 from the rollout, displaced by one aggregated
                       guidance step  a_beta = a0 + clip(grad_a Q(a0))/beta.
                       That displacement is the exact net drift the guided
                       sampler would add if the gradient were constant across
                       the 10 steps (sum over steps of -dt*grad/beta with
                       dt = -1/10).  It is a SURROGATE for the guided denoiser
                       output, not the guided denoiser output.

The critic-only surrogate block is computed in BOTH modes, so that in "full"
mode it can be compared against the exact numbers.  Nothing in this script ever
substitutes a proxy silently.

THE STATE-CONVENTION PROBLEM (READ THIS TOO)
--------------------------------------------
The critic was trained on the RAW proprioceptive state cached in
features/episode_XXXXXX.pt (physical joint radians plus a 0/1 gripper flag).
The deployed QGF processor feeds the critic ``batch["observation.state"]``,
which at that point in the policy server has already been MEAN_STD-normalized
by the SmolVLA preprocessor.  Those are not the same vector.  This probe does
not fix that; it measures both conventions side by side:

  --critic-state raw        what the critic was trained on
  --critic-state deployed   what the installed QGF processor feeds it today
  --critic-state both       default; runs the whole sweep twice

If the two tables disagree, the "deployed" table is the one that describes what
the robot would actually do today, and the disagreement itself should be
resolved before any robot time is spent.

HONEST CAVEAT ABOUT WHAT THIS PROBE CAN AND CANNOT TELL YOU
-----------------------------------------------------------
Every criterion here is scored against the critic's own opinion.  A
miscalibrated critic can be maximised beautifully while the real robot gets no
better, or gets worse.  "Held out" here means held out from CRITIC TRAINING; it
does not make these episodes a substitute for a robot comparison.  The output
of this script is a recommendation to a human, not a decision.

USAGE
-----
    CUDA_VISIBLE_DEVICES=1 \
    /opt/qgf_real_robot/envs/visual_iql_py310/bin/python \
        probe_beta_offline.py

Everything has a default pointing at the red-parcel run; every default can be
overridden.  See --help.
"""

# --- BEGIN SAFETY DECLARATION AND TOKEN LIST ---
# The static self-audit deliberately skips this block, otherwise the safety
# vocabulary itself would trip the scan.  The end marker is built by
# concatenation so the literal marker does not appear before the real one.
FORBIDDEN_CALL_TOKENS = (
    "power_on", "poweron", "set_enabled", "enable_service", "servo",
    "gripper_command", "executed_gripper_command", "teleop_joint_command",
    "motion_enabled", "stop_request", "JointTrajectory", "JointState",
    "SetBool", "Trigger", "rclpy", "create_publisher", "create_client",
    "create_service", "/right_arm/", "TelemetryPolicyServer", "QGFPolicyServer",
    "PolicyServer",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "rclpy", "rmw", "rosidl", "ros2", "ros2cli", "std_msgs", "std_srvs",
    "sensor_msgs", "trajectory_msgs", "geometry_msgs", "control_msgs",
    "lerobot_robot_armstrong_ros2", "lerobot.async_inference",
)
MARK_END = "# --- END SAFETY DECLARATION" + " AND TOKEN LIST ---"
# --- END SAFETY DECLARATION AND TOKEN LIST ---

import argparse
import io
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()


# ---------------------------------------------------------------------------
# guard 1: no ROS / robot module may be imported by this process
# ---------------------------------------------------------------------------
def _is_forbidden_module(fullname):
    for prefix in FORBIDDEN_IMPORT_PREFIXES:
        if fullname == prefix or fullname.startswith(prefix + "."):
            return True
    return False


class _ImportGuard(object):
    def find_spec(self, fullname, path=None, target=None):
        if _is_forbidden_module(fullname):
            raise ImportError(
                "beta probe refused to import {0!r}: this script must not touch "
                "the robot stack".format(fullname)
            )
        return None

    def find_module(self, fullname, path=None):
        if _is_forbidden_module(fullname):
            raise ImportError("beta probe refused to import {0!r}".format(fullname))
        return None


sys.meta_path.insert(0, _ImportGuard())

# ---------------------------------------------------------------------------
# guard 2: loopback sockets only.  Every attempt is recorded either way.
# ---------------------------------------------------------------------------
NET_ALLOWED = []
NET_DENIED = []
_real_socket = socket.socket
_real_getaddrinfo = socket.getaddrinfo


def _is_loopback_host(host):
    if host is None:
        return True
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except Exception:
            return False
    if not isinstance(host, str):
        return False
    if host in ("", "localhost", "localhost.localdomain", "127.0.0.1", "::1"):
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except Exception:
        return False


def _is_loopback_address(address):
    if isinstance(address, tuple) and address:
        return _is_loopback_host(address[0])
    return False


class _LoopbackOnlySocket(_real_socket):
    def _probe_gate(self, address, how):
        if _is_loopback_address(address):
            NET_ALLOWED.append((how, repr(address)))
            return
        NET_DENIED.append((how, repr(address)))
        raise RuntimeError("network access attempted: {0} {1!r}".format(how, address))

    def connect(self, address, *a, **k):
        self._probe_gate(address, "connect")
        return _real_socket.connect(self, address, *a, **k)

    def connect_ex(self, address, *a, **k):
        self._probe_gate(address, "connect_ex")
        return _real_socket.connect_ex(self, address, *a, **k)


def _guarded_getaddrinfo(host, *a, **k):
    if not _is_loopback_host(host):
        NET_DENIED.append(("getaddrinfo", repr(host)))
        raise RuntimeError("network name resolution attempted: {0!r}".format(host))
    NET_ALLOWED.append(("getaddrinfo", repr(host)))
    return _real_getaddrinfo(host, *a, **k)


socket.socket = _LoopbackOnlySocket
socket.getaddrinfo = _guarded_getaddrinfo


# ---------------------------------------------------------------------------
# guard 3: static self-audit of this file's executable body
# ---------------------------------------------------------------------------
def _self_audit():
    text = io.open(SCRIPT_PATH, encoding="utf-8").read()
    end = text.find(MARK_END)
    if end < 0:
        return ["self-audit could not find its own end marker"]
    body = text[end + len(MARK_END):]
    hits = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for token in FORBIDDEN_CALL_TOKENS:
            if token in stripped:
                hits.append("{0!r} in: {1}".format(token, stripped[:110]))
    return hits


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
DEFAULT_RUN_DIR = "/opt/qgf_real_robot/runs/red_parcel_single_q_45_5_20260902"
DEFAULT_RAW_EPISODES = (
    "/opt/qgf_real_robot/datasets/red_parcel_baseline50_20260902/raw_episodes"
)
DEFAULT_BUNDLE_CKPT = "/opt/qgf_real_robot/policy_bundles/red_parcel_clean/checkpoint"
DEFAULT_REPO = "/opt/qgf_real_robot/repos/SmolVLA-with-QGF"
DEFAULT_VAL_EPISODES = (7, 13, 25, 32, 46)
DEFAULT_BETAS = (0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0)

OBS_STATE_KEY = "observation.state"
CHEST_KEY = "observation.images.chest"
WRIST_KEY = "observation.images.wrist_right"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--features-dir", default=None, help="default <run-dir>/features")
    p.add_argument(
        "--critic",
        default=None,
        help="default <run-dir>/outputs/single_qcritic/critic_member_00.pt",
    )
    p.add_argument(
        "--split-file",
        default=None,
        help="default <run-dir>/manifest/episode_split_45_5.json",
    )
    p.add_argument("--raw-episodes-root", default=DEFAULT_RAW_EPISODES)
    p.add_argument("--policy-checkpoint", default=DEFAULT_BUNDLE_CKPT)
    p.add_argument("--repo", default=DEFAULT_REPO, help="repo whose qgf/src is imported")
    p.add_argument("--val-episodes", nargs="+", default=None)
    p.add_argument("--train-episodes", nargs="+", default=None)
    p.add_argument(
        "--betas",
        nargs="+",
        default=None,
        help="candidate betas; space or comma separated. default: "
        + " ".join(str(b) for b in DEFAULT_BETAS),
    )
    p.add_argument("--samples-per-episode", type=int, default=10)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--noise-seed", type=int, default=20260903)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--critic-state",
        choices=("raw", "deployed", "both"),
        default="both",
        help="which state vector to hand the critic; see the module docstring",
    )
    p.add_argument("--no-policy", action="store_true", help="force critic_only mode")
    p.add_argument("--output", default=None, help="default <run-dir>/outputs/beta_probe/...")
    p.add_argument(
        "--allow-any-gpu",
        action="store_true",
        help="bypass the project rule that requires CUDA_VISIBLE_DEVICES=1",
    )
    # recommendation gates - declared, printed, and overridable on purpose
    p.add_argument("--gate-ratio-floor", type=float, default=0.02)
    p.add_argument("--gate-ratio-ceiling", type=float, default=0.35)
    p.add_argument("--gate-dq-positive-frac", type=float, default=0.70)
    p.add_argument("--gate-ood-slack", type=float, default=0.005)
    p.add_argument("--gate-clip-rate", type=float, default=0.05)
    return p.parse_args(argv)


def _flat_floats(raw, fallback):
    if raw is None:
        return list(fallback)
    out = []
    for item in raw:
        for piece in str(item).replace(",", " ").split():
            out.append(float(piece))
    return out


def _flat_ints(raw, fallback):
    if raw is None:
        return list(fallback)
    out = []
    for item in raw:
        for piece in str(item).replace(",", " ").split():
            out.append(int(piece))
    return out


# ---------------------------------------------------------------------------
# small statistics helpers (torch only, no numpy dependency)
# ---------------------------------------------------------------------------
def summarise(values, torch):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "p10": float(torch.quantile(t, 0.10)),
        "median": float(torch.quantile(t, 0.50)),
        "p90": float(torch.quantile(t, 0.90)),
        "max": float(t.max()),
    }


def fmt(value, width=9, digits=4):
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, float) and not math.isfinite(value):
        return "inf".rjust(width)
    return ("{0:." + str(digits) + "f}").format(value).rjust(width)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_episode(features_dir, episode_index, torch):
    path = Path(features_dir) / "episode_{0:06d}.pt".format(int(episode_index))
    if not path.exists():
        raise FileNotFoundError(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=False)


def training_action_support(features_dir, train_episodes, torch):
    """Per-dimension support of the normalized chunks the critic trained on."""
    chunks = []
    used = []
    for episode_index in train_episodes:
        try:
            payload = load_episode(features_dir, episode_index, torch)
        except FileNotFoundError:
            continue
        chunks.append(payload["action_chunk"].reshape(-1, payload["action_chunk"].shape[-1]))
        used.append(int(episode_index))
        del payload
    if not chunks:
        raise RuntimeError("no training episodes could be read for the action support")
    stacked = torch.cat(chunks, dim=0).double()
    support = {
        "episodes": used,
        "episode_count": len(used),
        "value_count_per_dim": int(stacked.shape[0]),
        "action_dim": int(stacked.shape[1]),
        "min": stacked.min(dim=0).values.tolist(),
        "max": stacked.max(dim=0).values.tolist(),
        "p01": torch.quantile(stacked, 0.01, dim=0).tolist(),
        "p99": torch.quantile(stacked, 0.99, dim=0).tolist(),
        "note": "aggregated over samples AND over the 50 horizon steps, per action dimension",
    }
    return support


def out_of_support_fractions(chunk, support, torch):
    """chunk: [H, A] tensor in the critic's normalized action space."""
    a = chunk.detach().to("cpu").double()
    lo_band = torch.tensor(support["p01"], dtype=torch.float64)
    hi_band = torch.tensor(support["p99"], dtype=torch.float64)
    lo_hard = torch.tensor(support["min"], dtype=torch.float64)
    hi_hard = torch.tensor(support["max"], dtype=torch.float64)
    outside_band = ((a < lo_band) | (a > hi_band)).double().mean()
    outside_hard = ((a < lo_hard) | (a > hi_hard)).double().mean()
    return float(outside_band), float(outside_hard)


def load_state_normalizer(checkpoint_dir):
    """Read observation.state MEAN_STD stats straight out of the preprocessor.

    Returns (mean, std, eps) as plain python lists / float, or None when the
    preprocessor does not normalize the state.
    """
    from safetensors.torch import load_file

    cfg_path = Path(checkpoint_dir) / "policy_preprocessor.json"
    cfg = json.loads(io.open(cfg_path, encoding="utf-8").read())
    for step in cfg.get("steps", []):
        if step.get("registry_name") != "normalizer_processor":
            continue
        norm_map = step.get("config", {}).get("norm_map", {})
        if norm_map.get("STATE") != "MEAN_STD":
            return None
        state_file = step.get("state_file")
        if not state_file:
            return None
        tensors = load_file(str(Path(checkpoint_dir) / state_file))
        mean = tensors["observation.state.mean"].flatten().tolist()
        std = tensors["observation.state.std"].flatten().tolist()
        eps = float(step.get("config", {}).get("eps", 1.0e-8))
        return {"mean": mean, "std": std, "eps": eps}
    return None


def normalize_state(raw_state, normalizer, torch):
    mean = torch.tensor(normalizer["mean"], dtype=raw_state.dtype, device=raw_state.device)
    std = torch.tensor(normalizer["std"], dtype=raw_state.dtype, device=raw_state.device)
    return (raw_state - mean) / (std + normalizer["eps"])


def decode_frames(video_path, indices, torch):
    """Decode exactly the requested frame numbers as CHW uint8 RGB tensors."""
    import av

    wanted = set(int(i) for i in indices)
    if not wanted:
        return {}
    last = max(wanted)
    decoded = {}
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                decoded[frame_index] = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            if frame_index >= last:
                break
    missing = sorted(wanted - set(decoded))
    if missing:
        raise RuntimeError(
            "{0}: frames not decodable: {1}".format(video_path, missing[:10])
        )
    return decoded


def pick_sample_indices(count, wanted):
    if wanted >= count:
        return list(range(count))
    if wanted <= 1:
        return [count // 2]
    step = (count - 1) / float(wanted - 1)
    picked = sorted({int(round(i * step)) for i in range(wanted)})
    return picked


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    started = time.time()

    audit_hits = _self_audit()
    print("=" * 78)
    print("QGF OFFLINE BETA-SELECTION PROBE")
    print("=" * 78)
    print("safety self-audit: {0}".format("CLEAN" if not audit_hits else "FAILED"))
    for hit in audit_hits:
        print("   " + hit)
    if audit_hits:
        print("refusing to run: this file contains robot-actuation vocabulary")
        return 2

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.device.startswith("cuda") and visible != "1" and not args.allow_any_gpu:
        print(
            "FATAL: project rule is physical GPU 1 only. Re-run with\n"
            "       CUDA_VISIBLE_DEVICES=1 ... (or pass --allow-any-gpu).\n"
            "       CUDA_VISIBLE_DEVICES is currently {0!r}.".format(visible)
        )
        return 2

    run_dir = Path(args.run_dir)
    features_dir = Path(args.features_dir) if args.features_dir else run_dir / "features"
    critic_path = (
        Path(args.critic)
        if args.critic
        else run_dir / "outputs" / "single_qcritic" / "critic_member_00.pt"
    )
    split_path = (
        Path(args.split_file)
        if args.split_file
        else run_dir / "manifest" / "episode_split_45_5.json"
    )
    output_path = (
        Path(args.output)
        if args.output
        else run_dir / "outputs" / "beta_probe" / "beta_probe_report.json"
    )

    repo_src = Path(args.repo) / "qgf" / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))

    import torch

    from guided_action_flow.critics.checkpoint import load_action_chunk_critic
    from guided_action_flow.guidance.qgf import (
        QGuidanceConfig,
        q_guided_velocity_smolvla_reverse_time,
    )

    betas = sorted(set(_flat_floats(args.betas, DEFAULT_BETAS)))
    if any(b <= 0 for b in betas):
        print("FATAL: every beta must be positive.")
        return 2

    split = None
    if split_path.exists():
        split = json.loads(io.open(split_path, encoding="utf-8").read())
    if args.val_episodes is not None:
        val_episodes = _flat_ints(args.val_episodes, DEFAULT_VAL_EPISODES)
    elif split is not None:
        val_episodes = [int(x) for x in split["val_episode_indices"]]
    else:
        val_episodes = list(DEFAULT_VAL_EPISODES)
    if args.train_episodes is not None:
        train_episodes = _flat_ints(args.train_episodes, ())
    elif split is not None:
        train_episodes = [int(x) for x in split["train_episode_indices"]]
    else:
        train_episodes = [
            int(p.stem.split("_")[-1])
            for p in sorted(Path(features_dir).glob("episode_*.pt"))
            if int(p.stem.split("_")[-1]) not in val_episodes
        ]
    overlap = sorted(set(train_episodes) & set(val_episodes))
    if overlap:
        print("FATAL: train/val overlap {0}; refusing to select beta on training data.".format(overlap))
        return 2

    device = torch.device(args.device)
    critic, critic_ckpt = load_action_chunk_critic(str(critic_path), device=device)
    critic_cfg = critic_ckpt.get("critic_config", {})
    critic_action_dim = int(critic_cfg.get("action_dim", 8))
    critic_horizon = int(critic_cfg.get("action_horizon", 50))

    print()
    print("run dir          : {0}".format(run_dir))
    print("features         : {0}".format(features_dir))
    print("critic           : {0}".format(critic_path))
    print("critic arch      : {0}  cfg={1}".format(critic_ckpt.get("critic_arch"), critic_cfg))
    print("split file       : {0}".format(split_path if split is not None else "<absent>"))
    print("train episodes   : {0} ({1})".format(len(train_episodes), train_episodes))
    print("HELD-OUT episodes: {0}".format(val_episodes))
    print("betas            : {0}".format(betas))
    print("grad_clip_norm   : {0}".format(args.grad_clip_norm))
    print("device           : {0}  CUDA_VISIBLE_DEVICES={1!r}".format(device, visible))
    print()

    # -- (3) support of the normalized action chunks the critic trained on ----
    print("building training action support from the 45 training episodes ...")
    support = training_action_support(features_dir, train_episodes, torch)
    print(
        "  {0} episodes, {1} values per dimension, action_dim={2}".format(
            support["episode_count"], support["value_count_per_dim"], support["action_dim"]
        )
    )

    normalizer = load_state_normalizer(args.policy_checkpoint)
    print(
        "  state normalizer : {0}".format(
            "MEAN_STD from policy_preprocessor.json" if normalizer else "NOT FOUND"
        )
    )

    # -- gather held-out samples ---------------------------------------------
    samples = []
    for episode_index in val_episodes:
        payload = load_episode(features_dir, episode_index, torch)
        count = int(payload["state"].shape[0])
        picked = pick_sample_indices(count, args.samples_per_episode)
        frame_pairs = payload.get("provenance", {}).get("current_frame_indices")
        for row in picked:
            samples.append(
                {
                    "episode_index": int(episode_index),
                    "episode_name": str(payload.get("episode_name", "")),
                    "episode_outcome": str(payload.get("episode_outcome", "")),
                    "task": str(payload.get("task", "")),
                    "row": int(row),
                    "state_raw": payload["state"][row].clone(),
                    "action_chunk": payload["action_chunk"][row].clone(),
                    "visual_cached": payload["visual_features"][row].clone(),
                    "frames": (
                        [int(frame_pairs[row][0]), int(frame_pairs[row][1])]
                        if frame_pairs is not None
                        else None
                    ),
                }
            )
        del payload
    print("held-out samples : {0} ({1} per episode requested)".format(len(samples), args.samples_per_episode))

    state_modes = ["raw", "deployed"] if args.critic_state == "both" else [args.critic_state]
    if "deployed" in state_modes and normalizer is None:
        print("  WARNING: no state normalizer found; dropping the 'deployed' convention.")
        state_modes = [m for m in state_modes if m != "deployed"] or ["raw"]

    report = {
        "format": "qgf-offline-beta-probe-v1",
        "generated_unix": int(started),
        "script": str(SCRIPT_PATH),
        "run_dir": str(run_dir),
        "features_dir": str(features_dir),
        "critic_path": str(critic_path),
        "critic_arch": critic_ckpt.get("critic_arch"),
        "critic_config": critic_cfg,
        "critic_selected_epoch": critic_ckpt.get("selected_epoch"),
        "critic_selected_val_td_loss": critic_ckpt.get("selected_val_td_loss"),
        "policy_checkpoint": str(args.policy_checkpoint),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "betas": betas,
        "grad_clip_norm": args.grad_clip_norm,
        "noise_seed": args.noise_seed,
        "samples_per_episode": args.samples_per_episode,
        "sample_count": len(samples),
        "state_modes": state_modes,
        "training_action_support": support,
        "state_normalizer_present": normalizer is not None,
        "gates": {
            "ratio_floor_median": args.gate_ratio_floor,
            "ratio_ceiling_p90": args.gate_ratio_ceiling,
            "delta_q_positive_fraction": args.gate_dq_positive_frac,
            "ood_slack_over_unguided": args.gate_ood_slack,
            "clip_rate_max": args.gate_clip_rate,
        },
    }

    # -----------------------------------------------------------------------
    # try to bring up the real SmolVLA denoiser
    # -----------------------------------------------------------------------
    policy = None
    preprocessor = None
    processor = None
    adapter = None
    mode = "critic_only"
    mode_reason = ""
    if args.no_policy:
        mode_reason = "--no-policy was passed"
    else:
        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            from guided_action_flow.policies.smolvla_qgf import (
                SmolVLAVisualCriticAdapter,
                encode_smolvla_visual_tokens,
                install_smolvla_qgf,
            )

            print()
            print("loading SmolVLA from {0} ...".format(args.policy_checkpoint))
            policy = SmolVLAPolicy.from_pretrained(args.policy_checkpoint)
            policy.eval()
            policy.to(device)
            preprocessor, _unused_post = make_pre_post_processors(
                policy.config,
                args.policy_checkpoint,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
                postprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            adapter = SmolVLAVisualCriticAdapter(critic)
            processor = install_smolvla_qgf(
                policy,
                critic=adapter,
                config=QGuidanceConfig(
                    beta=betas[0],
                    grad_clip_norm=args.grad_clip_norm,
                    uncertainty_scale=0.0,
                    min_gate=0.0,
                ),
                critic_action_dim=critic_action_dim,
                task_feature_dim=0,
                task_feature_source="none",
            )
            mode = "full"
            print("SmolVLA + production QGF hook installed; mode = full")
        except Exception as exc:  # noqa: BLE001 - the fallback is the point
            mode = "critic_only"
            mode_reason = "{0}: {1}".format(type(exc).__name__, exc)
            print("could not bring up the SmolVLA denoiser -> mode = critic_only")
            print("   reason: {0}".format(mode_reason))
            policy = None

    exact_results = {}
    if mode == "full":
        try:
            exact_results = run_full_sweep(
                torch=torch,
                policy=policy,
                preprocessor=preprocessor,
                processor=processor,
                adapter=adapter,
                encode_visual=encode_smolvla_visual_tokens,
                QGuidanceConfig=QGuidanceConfig,
                critic=critic,
                samples=samples,
                betas=betas,
                state_modes=state_modes,
                normalizer=normalizer,
                support=support,
                args=args,
                device=device,
                critic_action_dim=critic_action_dim,
                critic_horizon=critic_horizon,
            )
        except Exception as exc:  # noqa: BLE001
            mode = "critic_only"
            mode_reason = "full sweep failed: {0}: {1}".format(type(exc).__name__, exc)
            print("FULL SWEEP FAILED -> falling back to critic_only")
            print("   reason: {0}".format(mode_reason))
            exact_results = {}

    surrogate_results = run_surrogate_sweep(
        torch=torch,
        guidance_fn=q_guided_velocity_smolvla_reverse_time,
        QGuidanceConfig=QGuidanceConfig,
        critic=critic,
        samples=samples,
        betas=betas,
        state_modes=state_modes,
        normalizer=normalizer,
        support=support,
        args=args,
        device=device,
        critic_action_dim=critic_action_dim,
    )

    report["mode"] = mode
    report["mode_reason"] = mode_reason
    report["measurement_status"] = measurement_status(mode)
    report["exact"] = exact_results
    report["surrogate"] = surrogate_results
    report["network_allowed"] = NET_ALLOWED[-20:]
    report["network_denied"] = NET_DENIED[-20:]

    print_report(report, args, mode)

    recommendation = recommend(report, args)
    report["recommendation"] = recommendation
    print_recommendation(recommendation, report, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.open(output_path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    )
    print()
    print("JSON report written to: {0}".format(output_path))
    print("elapsed: {0:.1f}s".format(time.time() - started))
    if NET_DENIED:
        print("NOTE: {0} non-loopback network attempt(s) were BLOCKED.".format(len(NET_DENIED)))
    return 0


def measurement_status(mode):
    if mode == "full":
        return {
            "mode": "full",
            "1_relative_perturbation": "EXACT (velocity_t is the SmolVLA model's own velocity)",
            "2_delta_q": "EXACT (guided vs unguided real denoiser output, shared initial noise)",
            "3_in_distribution": "EXACT (real guided denoiser output vs training support)",
            "4_clip_rate": "EXACT",
            "surrogate_block": "also computed, for cross-checking only",
        }
    return {
        "mode": "critic_only",
        "1_relative_perturbation": (
            "INCOMPLETE - ||grad/beta|| is exact, ||velocity_t|| WAS NOT MEASURED, "
            "so no ratio is reported. Nothing was substituted for it."
        ),
        "2_delta_q": (
            "EXACT computation on a SURROGATE action: a_beta = a0 + clip(grad_a Q(a0))/beta, "
            "where a0 is the recorded normalized chunk. This is the net drift the guided "
            "sampler would add if the gradient were constant over the 10 denoise steps. "
            "It is NOT the guided denoiser output."
        ),
        "3_in_distribution": "EXACT computation on the same SURROGATE action",
        "4_clip_rate": "EXACT (single gradient evaluation per sample, not per denoise step)",
        "surrogate_block": "this IS the surrogate block; there is no exact block",
    }


# ---------------------------------------------------------------------------
# the exact sweep: real SmolVLA denoiser, production QGF hook
# ---------------------------------------------------------------------------
def run_full_sweep(
    *,
    torch,
    policy,
    preprocessor,
    processor,
    adapter,
    encode_visual,
    QGuidanceConfig,
    critic,
    samples,
    betas,
    state_modes,
    normalizer,
    support,
    args,
    device,
    critic_action_dim,
    critic_horizon,
):
    """Run the real 10-step sampler with and without guidance on shared noise."""

    raw_root = Path(args.raw_episodes_root)

    # --- decode every frame we need, one pass per video ---------------------
    frame_cache = {}
    by_episode = {}
    for sample in samples:
        if sample["frames"] is None:
            raise RuntimeError(
                "feature cache has no provenance.current_frame_indices; cannot "
                "reconstruct the policy observation for the exact sweep"
            )
        by_episode.setdefault(sample["episode_name"], []).append(sample)
    for episode_name, rows in by_episode.items():
        episode_dir = raw_root / episode_name
        chest_indices = [r["frames"][0] for r in rows]
        wrist_indices = [r["frames"][1] for r in rows]
        print("decoding {0}: {1} frame pair(s) ...".format(episode_name, len(rows)))
        chest = decode_frames(episode_dir / "chest.mp4", chest_indices, torch)
        wrist = decode_frames(episode_dir / "wrist_right.mp4", wrist_indices, torch)
        frame_cache[episode_name] = (chest, wrist)

    # --- instrument the production denoise hook -----------------------------
    control = {"bypass": True, "steps": []}
    original_denoise_step = processor.denoise_step
    original_set_batch_features = processor.set_batch_features
    state_override = {"tensor": None}

    def instrumented_set_batch_features(batch):
        original_set_batch_features(batch)
        if state_override["tensor"] is not None:
            processor.obs_features = state_override["tensor"]

    def instrumented_denoise_step(
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time_value,
        original_denoise_step_partial,
        execution_horizon=None,
        **extra
    ):
        captured = {}

        def capture(input_x_t):
            velocity = original_denoise_step_partial(input_x_t)
            captured["v"] = velocity
            return velocity

        if control["bypass"]:
            guided = capture(x_t)
            diagnostics = None
        else:
            before = len(processor.diagnostics)
            guided = original_denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time_value,
                original_denoise_step_partial=capture,
                execution_horizon=execution_horizon,
            )
            diagnostics = (
                processor.diagnostics[-1] if len(processor.diagnostics) > before else None
            )

        velocity = captured["v"].detach()
        delta = (velocity - guided).detach()
        act = slice(0, critic_action_dim)
        record = {
            "t": float(time_value),
            "v_norm_full": float(velocity.reshape(velocity.shape[0], -1).norm(dim=-1).mean()),
            "v_norm_act": float(
                velocity[..., act].reshape(velocity.shape[0], -1).norm(dim=-1).mean()
            ),
            "delta_norm": float(delta.reshape(delta.shape[0], -1).norm(dim=-1).mean()),
            "delta_norm_pad": float(
                delta[..., critic_action_dim:].reshape(delta.shape[0], -1).norm(dim=-1).mean()
            ),
        }
        if diagnostics is not None:
            record["grad_norm_raw"] = float(diagnostics["q_grad_norm_raw_mean"])
            record["grad_norm_clipped"] = float(diagnostics["q_grad_norm_mean"])
            record["q_value"] = float(diagnostics["q_value_mean"])
        control["steps"].append(record)
        return guided

    # keyword-only in production; accept the positional name the sampler uses
    def denoise_step_shim(
        x_t=None,
        prev_chunk_left_over=None,
        inference_delay=None,
        time=None,
        original_denoise_step_partial=None,
        execution_horizon=None,
        **extra
    ):
        return instrumented_denoise_step(
            x_t,
            prev_chunk_left_over,
            inference_delay,
            time,
            original_denoise_step_partial,
            execution_horizon,
            **extra
        )

    processor.denoise_step = denoise_step_shim
    processor.set_batch_features = instrumented_set_batch_features

    chunk_size = int(policy.config.chunk_size)
    max_action_dim = int(policy.config.max_action_dim)
    num_steps = int(policy.config.num_steps)

    def run_once(batch, noise, beta):
        control["bypass"] = beta is None
        control["steps"] = []
        processor.diagnostics = []
        policy.reset()
        if beta is not None:
            processor.config = QGuidanceConfig(
                beta=float(beta),
                grad_clip_norm=args.grad_clip_norm,
                uncertainty_scale=0.0,
                min_gate=0.0,
            )
        chunk = policy.predict_action_chunk(dict(batch), noise)
        return chunk.detach(), list(control["steps"])

    results = {}
    token_diffs = []
    determinism_checks = []

    for state_mode in state_modes:
        print()
        print("--- EXACT sweep, critic-state convention = {0} ---".format(state_mode))
        per_beta = {
            beta: {
                "ratio_act": [],
                "ratio_full": [],
                "delta_norm": [],
                "grad_norm_raw": [],
                "grad_norm_clipped": [],
                "clipped_steps": 0,
                "total_steps": 0,
                "delta_q": [],
                "q_guided": [],
                "q_base": [],
                "ood_band": [],
                "ood_hard": [],
                "chunk_shift": [],
                "max_abs_shift": [],
                "ideal_shift": [],
                "realized_vs_ideal": [],
            }
            for beta in betas
        }
        base_stats = {"ood_band": [], "ood_hard": [], "q": [], "v_norm_act": [], "v_norm_full": []}

        for ordinal, sample in enumerate(samples):
            chest_map, wrist_map = frame_cache[sample["episode_name"]]
            chest = chest_map[sample["frames"][0]].to(torch.float32) / 255.0
            wrist = wrist_map[sample["frames"][1]].to(torch.float32) / 255.0
            observation = {
                CHEST_KEY: chest,
                WRIST_KEY: wrist,
                OBS_STATE_KEY: sample["state_raw"].clone(),
                "task": sample["task"],
            }
            batch = preprocessor(observation)

            raw_state = sample["state_raw"].to(device).reshape(1, -1)
            deployed_state = batch[OBS_STATE_KEY].detach().to(device).reshape(1, -1)
            state_override["tensor"] = raw_state if state_mode == "raw" else None
            critic_state = raw_state if state_mode == "raw" else deployed_state

            visual_tokens = encode_visual(policy, dict(batch)).detach()
            cached_tokens = sample["visual_cached"].to(device=device, dtype=torch.float32).unsqueeze(0)
            denom = cached_tokens.abs().mean().clamp(min=1e-9)
            token_diffs.append(float((visual_tokens.float() - cached_tokens).abs().mean() / denom))

            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.noise_seed) + 1000003 * ordinal)
            noise = torch.randn(
                (1, chunk_size, max_action_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )

            base_chunk, base_steps = run_once(batch, noise.clone(), None)
            if ordinal == 0:
                repeat_chunk, _ = run_once(batch, noise.clone(), None)
                determinism_checks.append(
                    float((repeat_chunk - base_chunk).abs().max())
                )
            adapter.set_visual_tokens(visual_tokens)
            with torch.no_grad():
                q_base = float(critic.module(critic_state, visual_tokens, base_chunk).mean())
            band, hard = out_of_support_fractions(base_chunk[0], support, torch)
            base_stats["ood_band"].append(band)
            base_stats["ood_hard"].append(hard)
            base_stats["q"].append(q_base)
            for step in base_steps:
                base_stats["v_norm_act"].append(step["v_norm_act"])
                base_stats["v_norm_full"].append(step["v_norm_full"])

            for beta in betas:
                guided_chunk, guided_steps = run_once(batch, noise.clone(), beta)
                bucket = per_beta[beta]
                for step in guided_steps:
                    bucket["total_steps"] += 1
                    if step["v_norm_act"] > 0:
                        bucket["ratio_act"].append(step["delta_norm"] / step["v_norm_act"])
                    if step["v_norm_full"] > 0:
                        bucket["ratio_full"].append(step["delta_norm"] / step["v_norm_full"])
                    bucket["delta_norm"].append(step["delta_norm"])
                    if "grad_norm_raw" in step:
                        bucket["grad_norm_raw"].append(step["grad_norm_raw"])
                        bucket["grad_norm_clipped"].append(step["grad_norm_clipped"])
                        if step["grad_norm_raw"] > args.grad_clip_norm * (1.0 - 1e-6):
                            bucket["clipped_steps"] += 1
                adapter.set_visual_tokens(visual_tokens)
                with torch.no_grad():
                    q_guided = float(
                        critic.module(critic_state, visual_tokens, guided_chunk).mean()
                    )
                bucket["q_guided"].append(q_guided)
                bucket["q_base"].append(q_base)
                bucket["delta_q"].append(q_guided - q_base)
                band_g, hard_g = out_of_support_fractions(guided_chunk[0], support, torch)
                bucket["ood_band"].append(band_g)
                bucket["ood_hard"].append(hard_g)
                shift = (guided_chunk - base_chunk).detach()
                realized = float(shift.norm())
                bucket["chunk_shift"].append(realized)
                bucket["max_abs_shift"].append(float(shift.abs().max()))
                # ideal drift: sum_k (-dt) * grad_k/beta, magnitude bound when the
                # per-step guidance vectors are perfectly aligned.  -dt = 1/num_steps,
                # and step["delta_norm"] already is ||grad_k||/beta.
                step_deltas = [s["delta_norm"] for s in guided_steps]
                if step_deltas:
                    ideal = sum(step_deltas) / float(len(step_deltas))
                    bucket["ideal_shift"].append(ideal)
                    if ideal > 0:
                        bucket["realized_vs_ideal"].append(realized / ideal)

            if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(samples):
                print("   {0}/{1} samples".format(ordinal + 1, len(samples)))

        pooled_grad = []
        for beta in betas:
            pooled_grad.extend(per_beta[beta]["grad_norm_clipped"])
        pooled_grad_summary = summarise(pooled_grad, torch)
        v_median = (summarise(base_stats["v_norm_act"], torch) or {}).get("median")
        implied = {}
        if pooled_grad_summary and v_median:
            for target in (args.gate_ratio_floor, 0.05, 0.10, 0.20):
                implied["{0:g}".format(target)] = (
                    pooled_grad_summary["median"] / (target * v_median)
                )

        results[state_mode] = {
            "unguided": {
                "q": summarise(base_stats["q"], torch),
                "velocity_norm_action_dims": summarise(base_stats["v_norm_act"], torch),
                "velocity_norm_all_dims": summarise(base_stats["v_norm_full"], torch),
                "out_of_p01_p99_fraction": summarise(base_stats["ood_band"], torch),
                "out_of_min_max_fraction": summarise(base_stats["ood_hard"], torch),
            },
            "pooled_grad_norm_clipped": pooled_grad_summary,
            "implied_beta_for_median_ratio": implied,
            "implied_beta_note": (
                "beta = median||clipped grad|| / (target_ratio * median||velocity_t||) on the "
                "8 action dims. Valid only while the clip rate stays 0, and it predicts the "
                "PER-STEP velocity perturbation, not the realized action-chunk shift, which "
                "the denoiser partly absorbs (see realized_vs_ideal_shift_ratio)."
            ),
            "per_beta": {},
        }
        for beta in betas:
            bucket = per_beta[beta]
            positive = [1.0 for v in bucket["delta_q"] if v > 0]
            results[state_mode]["per_beta"][str(beta)] = {
                "beta": beta,
                "ratio_action_dims": summarise(bucket["ratio_act"], torch),
                "ratio_all_dims": summarise(bucket["ratio_full"], torch),
                "guidance_norm": summarise(bucket["delta_norm"], torch),
                "grad_norm_raw": summarise(bucket["grad_norm_raw"], torch),
                "grad_norm_clipped": summarise(bucket["grad_norm_clipped"], torch),
                "clip_rate": (
                    bucket["clipped_steps"] / float(bucket["total_steps"])
                    if bucket["total_steps"]
                    else None
                ),
                "denoise_steps_seen": bucket["total_steps"],
                "delta_q": summarise(bucket["delta_q"], torch),
                "delta_q_positive_fraction": (
                    len(positive) / float(len(bucket["delta_q"])) if bucket["delta_q"] else None
                ),
                "q_guided": summarise(bucket["q_guided"], torch),
                "out_of_p01_p99_fraction": summarise(bucket["ood_band"], torch),
                "out_of_min_max_fraction": summarise(bucket["ood_hard"], torch),
                "chunk_shift_l2": summarise(bucket["chunk_shift"], torch),
                "chunk_shift_max_abs": summarise(bucket["max_abs_shift"], torch),
                "ideal_shift_l2": summarise(bucket["ideal_shift"], torch),
                "realized_vs_ideal_shift_ratio": summarise(bucket["realized_vs_ideal"], torch),
            }

    processor.denoise_step = original_denoise_step
    processor.set_batch_features = original_set_batch_features

    return {
        "sampler": {
            "num_denoise_steps": num_steps,
            "chunk_size": chunk_size,
            "max_action_dim": max_action_dim,
            "dt": -1.0 / num_steps,
        },
        "noise_mechanism": (
            "explicit injection: torch.Generator seeded per sample, the SAME noise "
            "tensor object value is passed to policy.predict_action_chunk(batch, noise) "
            "for the unguided arm and for every guided arm. torch.manual_seed was NOT "
            "relied upon, and SmolVLA's own sample_noise was never called."
        ),
        "shared_noise_verified": True,
        "unguided_repeat_max_abs_diff": determinism_checks,
        "visual_token_recompute_rel_diff_vs_cache": summarise(token_diffs, torch),
        "by_state_mode": results,
    }


# ---------------------------------------------------------------------------
# the critic-only surrogate sweep (always runs; the only sweep in fallback mode)
# ---------------------------------------------------------------------------
def run_surrogate_sweep(
    *,
    torch,
    guidance_fn,
    QGuidanceConfig,
    critic,
    samples,
    betas,
    state_modes,
    normalizer,
    support,
    args,
    device,
    critic_action_dim,
):
    """a_beta = a0 + clip(grad_a Q(a0)) / beta, evaluated exactly.

    This is NOT the guided denoiser output.  It is the aggregate drift the
    guided sampler adds over its 10 steps when the gradient is treated as
    constant:  sum_k (-dt) * grad/beta  with dt = -1/num_steps.
    """
    results = {}
    base_q_by_mode = {}
    for state_mode in state_modes:
        if state_mode == "deployed" and normalizer is None:
            continue
        per_beta = {
            beta: {
                "delta_q": [],
                "ood_band": [],
                "ood_hard": [],
                "shift_l2": [],
                "max_abs_shift": [],
            }
            for beta in betas
        }
        base = {"q": [], "ood_band": [], "ood_hard": [], "grad_raw": [], "grad_clipped": []}
        clipped = 0
        for sample in samples:
            action = sample["action_chunk"].to(device).unsqueeze(0)
            visual = sample["visual_cached"].to(device=device, dtype=torch.float32).unsqueeze(0)
            raw_state = sample["state_raw"].to(device).reshape(1, -1)
            if state_mode == "deployed":
                critic_state = normalize_state(raw_state, normalizer, torch)
            else:
                critic_state = raw_state

            with torch.enable_grad():
                leaf = action.detach().clone().requires_grad_(True)
                q_value = critic.module(critic_state, visual, leaf)
                grad = torch.autograd.grad(q_value.sum(), leaf)[0]
            raw_norm = float(grad.reshape(1, -1).norm())
            scale = min(1.0, args.grad_clip_norm / (raw_norm + 1e-6))
            grad_clipped = grad.detach() * scale
            clipped_norm = float(grad_clipped.reshape(1, -1).norm())
            if raw_norm > args.grad_clip_norm * (1.0 - 1e-6):
                clipped += 1
            base["grad_raw"].append(raw_norm)
            base["grad_clipped"].append(clipped_norm)
            q_base = float(q_value.detach().mean())
            base["q"].append(q_base)
            band, hard = out_of_support_fractions(action[0], support, torch)
            base["ood_band"].append(band)
            base["ood_hard"].append(hard)

            for beta in betas:
                shifted = action.detach() + grad_clipped / float(beta)
                with torch.no_grad():
                    q_shift = float(critic.module(critic_state, visual, shifted).mean())
                bucket = per_beta[beta]
                bucket["delta_q"].append(q_shift - q_base)
                band_g, hard_g = out_of_support_fractions(shifted[0], support, torch)
                bucket["ood_band"].append(band_g)
                bucket["ood_hard"].append(hard_g)
                bucket["shift_l2"].append(float((shifted - action).norm()))
                bucket["max_abs_shift"].append(float((shifted - action).abs().max()))

        base_q_by_mode[state_mode] = list(base["q"])
        block = {
            "recorded_action": {
                "q": summarise(base["q"], torch),
                "out_of_p01_p99_fraction": summarise(base["ood_band"], torch),
                "out_of_min_max_fraction": summarise(base["ood_hard"], torch),
                "grad_norm_raw": summarise(base["grad_raw"], torch),
                "grad_norm_clipped": summarise(base["grad_clipped"], torch),
                "clip_rate": clipped / float(len(samples)) if samples else None,
            },
            "per_beta": {},
        }
        for beta in betas:
            bucket = per_beta[beta]
            positive = [1.0 for v in bucket["delta_q"] if v > 0]
            block["per_beta"][str(beta)] = {
                "beta": beta,
                "delta_q": summarise(bucket["delta_q"], torch),
                "delta_q_positive_fraction": (
                    len(positive) / float(len(bucket["delta_q"])) if bucket["delta_q"] else None
                ),
                "out_of_p01_p99_fraction": summarise(bucket["ood_band"], torch),
                "out_of_min_max_fraction": summarise(bucket["ood_hard"], torch),
                "shift_l2": summarise(bucket["shift_l2"], torch),
                "shift_max_abs": summarise(bucket["max_abs_shift"], torch),
            }
        results[state_mode] = block

    sensitivity = None
    if "raw" in base_q_by_mode and "deployed" in base_q_by_mode:
        raw_q = base_q_by_mode["raw"]
        dep_q = base_q_by_mode["deployed"]
        if len(raw_q) == len(dep_q) and raw_q:
            diffs = [abs(a - b) for a, b in zip(raw_q, dep_q)]
            spread = summarise(raw_q, torch)
            sensitivity = {
                "mean_abs_q_difference_raw_vs_deployed": summarise(diffs, torch)["mean"],
                "q_std_across_samples_raw": spread["std"],
                "relative": (
                    summarise(diffs, torch)["mean"] / spread["std"]
                    if spread["std"] > 0
                    else None
                ),
                "note": (
                    "Replacing the ENTIRE state vector with a different convention should "
                    "move Q a lot if the critic uses the state. A relative value near 0 "
                    "means the critic is close to state-blind, and then the raw/deployed "
                    "mismatch matters less for Q, but so does the state itself."
                ),
            }

    return {
        "definition": "a_beta = a0 + clip(grad_a Q(a0), 1.0) / beta on the RECORDED chunk a0",
        "is_surrogate": True,
        "not_the_guided_denoiser_output": True,
        "state_convention_sensitivity": sensitivity,
        "by_state_mode": results,
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def print_report(report, args, mode):
    print()
    print("=" * 78)
    print("MEASUREMENT STATUS  (mode = {0})".format(mode))
    print("=" * 78)
    for key, value in report["measurement_status"].items():
        print("  {0:<26} {1}".format(key, value))
    if report.get("mode_reason"):
        print("  reason                     {0}".format(report["mode_reason"]))

    exact = report.get("exact") or {}
    if exact:
        print()
        print("noise mechanism: {0}".format(exact["noise_mechanism"]))
        repeats = exact.get("unguided_repeat_max_abs_diff") or []
        if repeats:
            print(
                "determinism check: the unguided arm was run twice on identical noise; "
                "max |difference| = {0:.3e} (0 means the two arms differ only by guidance)".format(
                    max(repeats)
                )
            )
        tok = exact.get("visual_token_recompute_rel_diff_vs_cache")
        if tok:
            print(
                "recomputed visual tokens vs cached bf16 tokens: mean relative |diff| "
                "median {0:.5f}, max {1:.5f}".format(tok["median"], tok["max"])
            )
        for state_mode, block in exact["by_state_mode"].items():
            print()
            print("-" * 78)
            print("EXACT SWEEP   critic-state convention = {0}".format(state_mode))
            print("-" * 78)
            unguided = block["unguided"]
            print(
                "unguided baseline:  Q mean {0}   ||v_t|| (8 action dims) median {1}   "
                "||v_t|| (all 32) median {2}".format(
                    fmt(unguided["q"]["mean"]),
                    fmt(unguided["velocity_norm_action_dims"]["median"]),
                    fmt(unguided["velocity_norm_all_dims"]["median"]),
                )
            )
            print(
                "unguided baseline:  outside p01-p99 {0}   outside min-max {1}".format(
                    fmt(unguided["out_of_p01_p99_fraction"]["mean"]),
                    fmt(unguided["out_of_min_max_fraction"]["mean"]),
                )
            )
            print()
            header = (
                "  beta |  ratio p10   median      p90 | mean dQ  frac dQ>0 | "
                "OOD band   OOD hard | clip rate | shift L2  realized/ideal"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for beta in report["betas"]:
                row = block["per_beta"][str(beta)]
                ratio = row["ratio_action_dims"] or {}
                dq = row["delta_q"] or {}
                print(
                    "  {0:>4} | {1} {2} {3} | {4} {5} | {6} {7} | {8} | {9} {10}".format(
                        beta,
                        fmt(ratio.get("p10")),
                        fmt(ratio.get("median")),
                        fmt(ratio.get("p90")),
                        fmt(dq.get("mean"), 8, 5),
                        fmt(row.get("delta_q_positive_fraction"), 10, 3),
                        fmt((row["out_of_p01_p99_fraction"] or {}).get("mean")),
                        fmt((row["out_of_min_max_fraction"] or {}).get("mean")),
                        fmt(row.get("clip_rate"), 9, 3),
                        fmt((row["chunk_shift_l2"] or {}).get("mean"), 8, 4),
                        fmt((row["realized_vs_ideal_shift_ratio"] or {}).get("mean"), 14, 3),
                    )
                )
            print(
                "  ratio = ||grad/beta|| / ||velocity_t||, restricted to the 8 real action "
                "dimensions.\n"
                "  OOD columns are element fractions of the guided chunk outside the "
                "training support.\n"
                "  realized/ideal = realized chunk shift divided by the aligned-drift bound\n"
                "  sum_k (-dt)*||grad_k||/beta. Well under 1 means the flow model pulls the\n"
                "  chunk back toward its own mode, so beta is not the whole story. Above 1 at\n"
                "  weak guidance means the sampler amplifies a tiny nudge over its 10 steps,\n"
                "  which is chaos in the sampler, not guidance strength."
            )
            implied = block.get("implied_beta_for_median_ratio") or {}
            if implied:
                print(
                    "  extrapolated beta needed for a given median ratio "
                    "(valid while clip rate stays 0):"
                )
                for target in sorted(implied, key=float):
                    print(
                        "     median ratio {0:>5} -> beta ~ {1:.4g}".format(
                            target, implied[target]
                        )
                    )

    surrogate = report.get("surrogate") or {}
    sensitivity = surrogate.get("state_convention_sensitivity")
    if sensitivity:
        print()
        print("-" * 78)
        print("CRITIC SENSITIVITY TO THE STATE CONVENTION")
        print("-" * 78)
        print(
            "  mean |Q(raw state) - Q(deployed state)| = {0}\n"
            "  std of Q across held-out samples        = {1}\n"
            "  ratio                                   = {2}".format(
                fmt(sensitivity["mean_abs_q_difference_raw_vs_deployed"], 9, 6),
                fmt(sensitivity["q_std_across_samples_raw"], 9, 6),
                fmt(sensitivity["relative"], 9, 4),
            )
        )
        print("  " + sensitivity["note"].replace("\n", "\n  "))

    for state_mode, block in (surrogate.get("by_state_mode") or {}).items():
        print()
        print("-" * 78)
        print(
            "SURROGATE SWEEP (critic-only, a_beta = a0 + clip(grad)/beta)   "
            "state = {0}".format(state_mode)
        )
        print("-" * 78)
        rec = block["recorded_action"]
        print(
            "recorded chunk a0:  Q mean {0}   raw ||grad|| median {1}   clip rate {2}".format(
                fmt(rec["q"]["mean"]),
                fmt(rec["grad_norm_raw"]["median"], 9, 6),
                fmt(rec["clip_rate"], 9, 3),
            )
        )
        print(
            "recorded chunk a0:  outside p01-p99 {0}   outside min-max {1}".format(
                fmt(rec["out_of_p01_p99_fraction"]["mean"]),
                fmt(rec["out_of_min_max_fraction"]["mean"]),
            )
        )
        print()
        header = (
            "  beta | shift L2  shift max | mean dQ  frac dQ>0 | OOD band   OOD hard"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for beta in report["betas"]:
            row = block["per_beta"][str(beta)]
            dq = row["delta_q"] or {}
            print(
                "  {0:>4} | {1} {2} | {3} {4} | {5} {6}".format(
                    beta,
                    fmt((row["shift_l2"] or {}).get("mean"), 9, 5),
                    fmt((row["shift_max_abs"] or {}).get("mean"), 10, 5),
                    fmt(dq.get("mean"), 8, 5),
                    fmt(row.get("delta_q_positive_fraction"), 10, 3),
                    fmt((row["out_of_p01_p99_fraction"] or {}).get("mean")),
                    fmt((row["out_of_min_max_fraction"] or {}).get("mean")),
                )
            )

    if len(report.get("state_modes", [])) > 1:
        print()
        print("!" * 78)
        print("STATE-CONVENTION WARNING")
        print("!" * 78)
        print(
            "The critic was trained on the RAW state cached in the feature files, but the\n"
            "installed QGF processor hands it batch['observation.state'], which the SmolVLA\n"
            "preprocessor has already MEAN_STD-normalized. The two tables above are the same\n"
            "sweep under the two conventions. If they disagree, the 'deployed' table is what\n"
            "the robot would actually do today, and the disagreement should be fixed before\n"
            "any robot time is spent on this beta."
        )


def _gate_rows(report, args, state_mode):
    """Score every beta against the declared gates for one state convention."""
    exact = report.get("exact") or {}
    block = (exact.get("by_state_mode") or {}).get(state_mode)
    source = "exact"
    if block is None:
        block = ((report.get("surrogate") or {}).get("by_state_mode") or {}).get(state_mode)
        source = "surrogate"
    if block is None:
        return source, []
    unguided_hard = None
    if source == "exact":
        unguided_hard = (block["unguided"]["out_of_min_max_fraction"] or {}).get("mean")
    else:
        unguided_hard = (block["recorded_action"]["out_of_min_max_fraction"] or {}).get("mean")

    rows = []
    for beta in report["betas"]:
        row = block["per_beta"][str(beta)]
        ratio = row.get("ratio_action_dims") if source == "exact" else None
        dq = row.get("delta_q") or {}
        hard = (row.get("out_of_min_max_fraction") or {}).get("mean")
        clip_rate = row.get("clip_rate")
        if clip_rate is None and source == "surrogate":
            clip_rate = (block["recorded_action"] or {}).get("clip_rate")
        checks = {}
        if ratio is not None:
            checks["A_strength_floor"] = (
                ratio.get("median") is not None and ratio["median"] >= args.gate_ratio_floor
            )
            checks["B_strength_ceiling"] = (
                ratio.get("p90") is not None and ratio["p90"] <= args.gate_ratio_ceiling
            )
        else:
            checks["A_strength_floor"] = None
            checks["B_strength_ceiling"] = None
        checks["C_critic_agrees"] = bool(
            row.get("delta_q_positive_fraction") is not None
            and row["delta_q_positive_fraction"] >= args.gate_dq_positive_frac
            and (dq.get("mean") or 0.0) > 0.0
        )
        checks["D_in_distribution"] = bool(
            hard is not None
            and unguided_hard is not None
            and hard <= unguided_hard + args.gate_ood_slack
        )
        checks["E_clip_linearity"] = bool(
            clip_rate is not None and clip_rate <= args.gate_clip_rate
        )
        decided = [v for v in checks.values() if v is not None]
        rows.append(
            {
                "beta": beta,
                "source": source,
                "checks": checks,
                "passes_all_decidable": all(decided) and bool(decided),
                "undecidable_gates": [k for k, v in checks.items() if v is None],
                "delta_q_mean": dq.get("mean"),
                "delta_q_positive_fraction": row.get("delta_q_positive_fraction"),
                "ratio_median": (ratio or {}).get("median"),
                "ratio_p90": (ratio or {}).get("p90"),
                "out_of_min_max_fraction": hard,
                "unguided_out_of_min_max_fraction": unguided_hard,
                "clip_rate": clip_rate,
            }
        )
    return source, rows


def recommend(report, args):
    out = {
        "gate_definitions": {
            "A_strength_floor": "median ratio >= {0} (below this the guidance is a no-op and "
            "the robot run cannot distinguish QGF from baseline)".format(args.gate_ratio_floor),
            "B_strength_ceiling": "p90 ratio <= {0} (above this the guidance overwhelms the "
            "learned flow)".format(args.gate_ratio_ceiling),
            "C_critic_agrees": "delta-Q > 0 on at least {0:.0%} of paired samples AND mean "
            "delta-Q > 0".format(args.gate_dq_positive_frac),
            "D_in_distribution": "guided out-of-min-max element fraction <= unguided + {0}".format(
                args.gate_ood_slack
            ),
            "E_clip_linearity": "gradient clip rate <= {0} (otherwise 1/beta scaling is not "
            "linear at this beta)".format(args.gate_clip_rate),
        },
        "by_state_mode": {},
    }
    for state_mode in report.get("state_modes", []):
        source, rows = _gate_rows(report, args, state_mode)
        passing = [r for r in rows if r["passes_all_decidable"]]
        undecidable = sorted({g for r in rows for g in r["undecidable_gates"]})
        choice = None
        if passing and not undecidable:
            # mean delta-Q is monotone in 1/beta, so among the gate-passing betas
            # this is exactly "the strongest guidance that is still admissible".
            choice = max(passing, key=lambda r: ((r["delta_q_mean"] or 0.0), r["beta"]))
        if undecidable:
            reason = (
                "REFUSING to name a beta: gates {0} could not be decided because "
                "||velocity_t|| was never measured. Mean delta-Q rises monotonically as "
                "beta falls, so with no strength ceiling 'largest delta-Q' would always "
                "name the smallest beta in the sweep, which is not a selection. Re-run "
                "with the SmolVLA denoiser available.".format(undecidable)
            )
        elif choice:
            reason = (
                "the strongest guidance (smallest beta) that still passes every gate; "
                "among gate-passing betas mean delta-Q is monotone in 1/beta, so the "
                "largest delta-Q is the strongest admissible guidance"
            )
        else:
            reason = "no candidate beta passed every decidable gate"
        exact_block = ((report.get("exact") or {}).get("by_state_mode") or {}).get(state_mode)
        implied = (exact_block or {}).get("implied_beta_for_median_ratio") or {}
        out["by_state_mode"][state_mode] = {
            "evidence_source": source,
            "implied_beta_for_median_ratio": implied,
            "undecidable_gates": undecidable,
            "rows": rows,
            "betas_passing_all_gates": [r["beta"] for r in passing],
            "strongest_beta_still_in_support": (
                min(
                    (r["beta"] for r in rows if r["checks"].get("D_in_distribution")),
                    default=None,
                )
            ),
            "suggested_beta": (choice or {}).get("beta"),
            "suggested_beta_reason": reason,
        }
    return out


def print_recommendation(recommendation, report, args):
    print()
    print("=" * 78)
    print("RECOMMENDATION TO A HUMAN  (this script decides nothing)")
    print("=" * 78)
    print("gates used (all overridable on the command line):")
    for name, text in recommendation["gate_definitions"].items():
        print("  {0}: {1}".format(name, text))
    print()
    for state_mode, block in recommendation["by_state_mode"].items():
        print("-" * 78)
        print(
            "state convention '{0}'   evidence: {1} block".format(
                state_mode, block["evidence_source"]
            )
        )
        for row in block["rows"]:
            marks = " ".join(
                "{0}={1}".format(
                    key.split("_")[0], "?" if value is None else ("ok" if value else "NO")
                )
                for key, value in row["checks"].items()
            )
            print(
                "  beta {0:>5} : {1}   (mean dQ {2}, frac dQ>0 {3})".format(
                    row["beta"],
                    marks,
                    fmt(row["delta_q_mean"], 8, 5),
                    fmt(row["delta_q_positive_fraction"], 6, 2),
                )
            )
        if block.get("undecidable_gates"):
            print("  -> {0}".format(block["suggested_beta_reason"]))
            print(
                "     The only thing this block can bound is strength from above: the "
                "smallest\n     beta whose surrogate action still sits inside the training "
                "min-max is {0}.\n     Treat that as a ceiling on guidance strength, not as a "
                "recommendation.".format(block.get("strongest_beta_still_in_support"))
            )
        elif block["suggested_beta"] is None:
            failed_a = [
                r["beta"] for r in block["rows"] if r["checks"].get("A_strength_floor") is False
            ]
            failed_c = [
                r["beta"] for r in block["rows"] if r["checks"].get("C_critic_agrees") is False
            ]
            print("  -> NO beta in this sweep passed every gate.")
            if len(failed_a) == len(block["rows"]) and block["rows"]:
                print(
                    "     Every beta failed gate A: the guidance is a no-op at this scale, so a\n"
                    "     robot run would compare QGF against a policy that is doing almost\n"
                    "     exactly the same thing. Either extend the sweep to smaller beta or\n"
                    "     conclude that this critic's gradient is too small to matter."
                )
                for target in sorted(block.get("implied_beta_for_median_ratio") or {}, key=float):
                    print(
                        "       to reach a median ratio of {0}, beta would have to be about "
                        "{1:.4g}".format(
                            target, block["implied_beta_for_median_ratio"][target]
                        )
                    )
            if failed_c:
                print(
                    "     Gate C failed at beta {0}: the guidance did not reliably raise even "
                    "the\n     critic's own score there.".format(failed_c)
                )
            print(
                "     Spending robot time before this is understood risks a headline number "
                "that\n     measures nothing."
            )
        else:
            print(
                "  -> suggested beta = {0}   ({1})".format(
                    block["suggested_beta"], block["suggested_beta_reason"]
                )
            )
        if block["evidence_source"] == "surrogate":
            print(
                "  -> NOTE: this came from the SURROGATE block. The relative-perturbation "
                "gates\n     A and B could not be decided because ||velocity_t|| was never "
                "measured."
            )
    print()
    print("-" * 78)
    print("CAVEATS - these are not optional reading")
    print("-" * 78)
    print(
        "1. Every score above is the critic's own opinion of its own guidance. A\n"
        "   miscalibrated critic can be maximised to arbitrary delta-Q while the real\n"
        "   robot gets no better, or gets worse. Passing gate C means the guidance is\n"
        "   doing what the critic asks, not that the critic asks for the right thing.\n"
        "2. 'Held out' here means held out from CRITIC TRAINING. These five episodes are\n"
        "   not a substitute for a robot comparison, and beta chosen here must not be\n"
        "   re-tuned on the robot episodes that the headline number is reported on.\n"
        "3. The in-distribution check compares against the 45 TRAINING episodes' action\n"
        "   support only. An action inside that support can still be wrong for the scene\n"
        "   in front of the arm.\n"
        "4. delta-Q is a paired statistic over {0} samples drawn from only {1} episodes;\n"
        "   samples inside one episode are strongly correlated, so the effective sample\n"
        "   size is closer to the episode count than to the sample count. Read the sign\n"
        "   and the magnitude, not a p-value.".format(
            report.get("sample_count"), len(report.get("val_episodes", []))
        )
    )


if __name__ == "__main__":
    sys.exit(main())
