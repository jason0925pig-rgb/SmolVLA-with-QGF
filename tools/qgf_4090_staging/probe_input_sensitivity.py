"""Does this Q critic actually use its visual input?

Not required by the handoff.  Run because the LIBERO Q line (E8) found the
critic was effectively scene-blind: swapping the entire visual scene moved Q by
under 1%, which made QGF structurally unable to react to scene perturbations.
If the same holds here, that is decision-relevant BEFORE any on-robot QGF run.

Method: hold two of (state, visual, action) fixed and replace the third with the
corresponding tensor from a different held-out episode, then measure how far Q
moves relative to its own spread across real samples.

Read-only.  Runs on the 4090.  No robot involved.
"""
import io
import json
import sys
from pathlib import Path

import torch

RUN = Path("/opt/qgf_real_robot/runs/mug_purple_box_single_q_45_5_20260829")
CK = RUN / "outputs/single_qcritic/critic_member_00.pt"
sys.path.insert(0, "/opt/qgf_real_robot/repos/SmolVLA-with-QGF/qgf/src")
from guided_action_flow.critics.checkpoint import load_action_chunk_critic  # noqa: E402

split = json.load(io.open(RUN / "manifest/episode_split_45_5.json", encoding="utf-8"))
val = [int(x) for x in split["val_episode_indices"]]
smap = json.load(io.open(
    "/opt/qgf_real_robot/datasets/mug_purple_box_baseline50_20260829/source_episode_map.json",
    encoding="utf-8"))
outcome = {e["dest_episode_index"]: e["outcome"] for e in smap["episodes"]}
print("validation episodes:", [(i, outcome[i]) for i in val])

critic, _ck = load_action_chunk_critic(str(CK), device="cpu")
net = critic            # callable wrapper: net(state, visual_tokens, action_chunk)
mod = critic.module     # the nn.Module, already in eval mode after loading


def load(ep):
    d = torch.load(RUN / f"features/episode_{ep:06d}.pt", map_location="cpu", weights_only=False)
    return (d["state"].float(), d["visual_features"].float(), d["action_chunk"].float())


def q(s, z, a):
    with torch.no_grad():
        return net(s, z, a).flatten()


A, B = val[0], val[1]
sA, zA, aA = load(A)
sB, zB, aB = load(B)
N = min(len(sA), len(sB), 64)
sA, zA, aA = sA[:N], zA[:N], aA[:N]
sB, zB, aB = sB[:N], zB[:N], aB[:N]
print(f"\ncomparing episode {A} ({outcome[A]}) against episode {B} ({outcome[B]}), N={N}")

base = q(sA, zA, aA)
spread = float(base.max() - base.min())
print(f"\nbaseline Q over {N} real samples of ep{A}:")
print(f"  mean {base.mean():.6f}  std {base.std():.6f}  min {base.min():.6f}  max {base.max():.6f}")
print(f"  spread (max-min) = {spread:.6f}   <- the natural scale to compare against")

print("\n=== swap ONE input at a time, keep the other two ===")
rows = [
    ("visual  z: ep%d -> ep%d" % (A, B), q(sA, zB, aA)),
    ("action  a: ep%d -> ep%d" % (A, B), q(sA, zA, aB)),
    ("state   s: ep%d -> ep%d" % (A, B), q(sB, zA, aA)),
    ("visual  z -> zeros", q(sA, torch.zeros_like(zA), aA)),
    ("visual  z -> gaussian noise", q(sA, torch.randn_like(zA) * zA.std(), aA)),
]
print(f"{'perturbation':<34}{'mean |dQ|':>12}{'as % of Q':>12}{'as % of spread':>17}")
for name, qq in rows:
    d = (qq - base).abs()
    pct_q = float((d / base.abs().clamp_min(1e-9)).mean()) * 100
    pct_sp = float(d.mean()) / spread * 100 if spread > 0 else float("nan")
    print(f"{name:<34}{float(d.mean()):>12.6f}{pct_q:>11.2f}%{pct_sp:>16.1f}%")

print("\n=== gradient magnitudes at the real inputs ===")
s = sA.clone().requires_grad_(True)
z = zA.clone().requires_grad_(True)
a = aA.clone().requires_grad_(True)
out = mod(s, z, a).sum()
out.backward()
for nm, g, t in (("dQ/ds", s.grad, sA), ("dQ/dz", z.grad, zA), ("dQ/da", a.grad, aA)):
    gn = float(g.norm())
    # scale-free: relative sensitivity = |grad| * |input| / |Q|
    rel = gn * float(t.norm()) / float(base.abs().sum().clamp_min(1e-9))
    print(f"  {nm}: grad_norm={gn:.6e}  input_norm={float(t.norm()):.4e}  relative_sensitivity={rel:.4f}")

print("\n=== does Q separate success from failure episodes at all? ===")
for ep in val:
    s_, z_, a_ = load(ep)
    qv = q(s_, z_, a_)
    print(f"  ep{ep:<3d} {outcome[ep]:<8s} n={len(qv):4d}  Q mean {qv.mean():.4f}  "
          f"std {qv.std():.4f}  last-chunk Q {qv[-1]:.4f}")

print("""
READING THIS:
  'as % of spread' is the honest number.  If replacing the entire visual scene
  moves Q by only a few percent of the range Q already covers across ordinary
  samples, the critic is not really conditioning on vision, and QGF guidance
  will not react to visual perturbations on the robot.  Compare the visual row
  against the action row: QGF differentiates Q with respect to the action, so
  the action row is the channel that actually drives guidance.
""")
