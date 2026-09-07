#!/usr/bin/env python3
"""Offline verification of the guidance_mode switch, on the Orin, with no robot.

Pure tensors against the DEPLOYED critic. No ROS import, no robot module, no
network, no arm, no camera, no policy server. It reads the same critic file the
rollout would load and drives the guidance function directly.

What it must show:
  critic              unchanged behaviour, direction cosine exactly 1.0
  random_matched_norm same |grad| before and after clipping, same applied
                      magnitude, direction cosine near 0, reproducible for a
                      seed, and the global torch RNG untouched
  zero                the critic forward AND backward still ran (raw and clipped
                      norms identical to critic mode), but the applied guidance
                      is exactly 0 and the guided velocity equals the input

If any of that fails, the ablation arms are not safe to collect with.
"""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO = os.environ.get("QGF_ORIN_REPO", "/home/nvidia/work/telop/SmolVLA-with-QGF")
CRITIC = os.environ.get(
    "SMOKE_CRITIC",
    "/home/nvidia/work/telop/models/qgf/stapler_into_box_single_q_45_5_20260830/critic_member_00.pt")
sys.path.insert(0, os.path.join(REPO, "qgf/src"))

import torch  # noqa: E402
from guided_action_flow.critics.checkpoint import load_action_chunk_critic  # noqa: E402
from guided_action_flow.guidance.qgf import (  # noqa: E402
    GUIDANCE_MODES,
    QGuidanceConfig,
    q_guided_velocity_smolvla_reverse_time,
    reset_random_generators,
)
from guided_action_flow.policies.smolvla_qgf import SmolVLAVisualCriticAdapter  # noqa: E402

FAIL = []


def check(ok, what):
    print("  %-4s %s" % ("ok" if ok else "FAIL", what))
    if not ok:
        FAIL.append(what)


print("modes available:", GUIDANCE_MODES)
print("critic         :", CRITIC)
critic, meta = load_action_chunk_critic(CRITIC, device="cpu")
cfg = meta["critic_config"]
print("arch           :", meta.get("critic_arch"), "| action_dim", cfg["action_dim"],
      "| horizon", cfg["action_horizon"], "| visual", cfg["visual_tokens"], "x", cfg["visual_token_dim"])

B = 2
g = torch.Generator().manual_seed(0)
state = torch.randn(B, cfg["state_dim"], generator=g)
visual = torch.randn(B, cfg["visual_tokens"], cfg["visual_token_dim"], generator=g)
action_t = torch.randn(B, cfg["action_horizon"], cfg["action_dim"], generator=g)
velocity_t = torch.randn(B, cfg["action_horizon"], cfg["action_dim"], generator=g)

adapter = SmolVLAVisualCriticAdapter(critic)
adapter.set_visual_tokens(visual)


def run(mode, seed=None, beta=0.5, clip=1.0):
    return q_guided_velocity_smolvla_reverse_time(
        critic=adapter, obs_features=state, action_t=action_t, velocity_t=velocity_t,
        time_t=0.3, critic_action_dim=cfg["action_dim"],
        config=QGuidanceConfig(beta=beta, grad_clip_norm=clip, uncertainty_scale=0.0,
                               min_gate=0.0, guidance_mode=mode, random_seed=seed),
    )


print("\n=== critic mode is unchanged and is the default ===")
v_def, d_def = q_guided_velocity_smolvla_reverse_time(
    critic=adapter, obs_features=state, action_t=action_t, velocity_t=velocity_t,
    time_t=0.3, critic_action_dim=cfg["action_dim"],
    config=QGuidanceConfig(beta=0.5, grad_clip_norm=1.0, uncertainty_scale=0.0, min_gate=0.0))
v_cri, d_cri = run("critic")
check(torch.equal(v_def, v_cri), "the default config still means critic mode")
check(abs(float(d_cri["q_direction_cos_mean"]) - 1.0) < 1e-6, "direction cosine is 1.0")
check(float(d_cri["q_guidance_norm_mean"]) > 0, "guidance is actually applied")
print("       raw|grad| %.6g  clipped %.6g  applied %.6g  Q %.6g" % (
    float(d_cri["q_grad_norm_raw_mean"]), float(d_cri["q_grad_norm_mean"]),
    float(d_cri["q_guidance_norm_mean"]), float(d_cri["q_value_mean"])))

print("\n=== zero mode: critic still runs, nothing is applied (B0-LM) ===")
v_zero, d_zero = run("zero")
check(float(d_zero["q_grad_norm_raw_mean"]) > 0, "the backward ran (raw norm is non-zero)")
check(torch.allclose(d_zero["q_grad_norm_raw_mean"], d_cri["q_grad_norm_raw_mean"]),
      "raw gradient norm identical to critic mode")
check(torch.allclose(d_zero["q_grad_norm_mean"], d_cri["q_grad_norm_mean"]),
      "clipped gradient norm identical to critic mode")
check(torch.allclose(d_zero["q_value_mean"], d_cri["q_value_mean"]),
      "Q value identical to critic mode")
check(float(d_zero["q_guidance_norm_mean"]) == 0.0, "applied guidance is exactly zero")
check(torch.equal(v_zero, velocity_t), "guided velocity equals the input velocity")

print("\n=== random_matched_norm: same magnitude, random direction ===")
reset_random_generators()
v_rnd, d_rnd = run("random_matched_norm", seed=20260903)
check(torch.allclose(d_rnd["q_grad_norm_raw_mean"], d_cri["q_grad_norm_raw_mean"]),
      "raw norm matches the real gradient's")
check(torch.allclose(d_rnd["q_grad_norm_mean"], d_cri["q_grad_norm_mean"], rtol=1e-4),
      "post-clip norm matches")
check(torch.allclose(d_rnd["q_guidance_norm_mean"], d_cri["q_guidance_norm_mean"], rtol=1e-4),
      "applied magnitude matches")
check(abs(float(d_rnd["q_direction_cos_mean"])) < 0.3, "direction is random (|cos| small)")
check(not torch.equal(v_rnd, v_cri), "the guided velocity differs from critic mode")
print("       cos %.4f  applied %.6g vs critic %.6g" % (
    float(d_rnd["q_direction_cos_mean"]), float(d_rnd["q_guidance_norm_mean"]),
    float(d_cri["q_guidance_norm_mean"])))

print("\n=== reproducibility and RNG isolation ===")
reset_random_generators(); v1, _ = run("random_matched_norm", seed=42)
reset_random_generators(); v2, _ = run("random_matched_norm", seed=42)
reset_random_generators(); v3, _ = run("random_matched_norm", seed=43)
check(torch.equal(v1, v2), "same seed gives the same direction")
check(not torch.equal(v1, v3), "a different seed gives a different direction")
reset_random_generators()
torch.manual_seed(999)
before = torch.get_rng_state().clone()
run("random_matched_norm", seed=7)
check(torch.equal(before, torch.get_rng_state()),
      "the global torch RNG is untouched (SmolVLA's own noise is unaffected)")

print("\n=== refusals ===")
try:
    run("random_matched_norm", seed=None); check(False, "a seedless random arm is refused")
except ValueError as exc:
    check("random_seed" in str(exc), "a seedless random arm is refused")
try:
    run("best_of_n"); check(False, "an unknown mode is refused")
except ValueError as exc:
    check("guidance_mode" in str(exc), "an unknown mode is refused")

print("\n=== safety declaration ===")
bad = [m for m in sys.modules if m.split(".")[0] in
       ("rclpy", "rospy", "std_msgs", "sensor_msgs", "trajectory_msgs",
        "lerobot_robot_armstrong_ros2")]
check(not bad, "no ROS or robot module was imported (%s)" % (bad or "none"))
print("  this run powered nothing on, enabled nothing, entered no servo mode,")
print("  sent no gripper command and moved no joint. Tensors and files only.")

print()
if FAIL:
    print("SMOKE FAILED (%d):" % len(FAIL))
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("GUIDANCE_MODE SMOKE OK - the three arms behave as the ablation requires")
