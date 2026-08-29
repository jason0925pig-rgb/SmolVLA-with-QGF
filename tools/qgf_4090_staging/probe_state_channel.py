"""Follow-up: is the state input actually wired into this critic, or is the
exact 0.000000 change under a state swap a bug in my probe?

Three independent ways to tell them apart:
  1. confirm the two state tensors really differ
  2. push the state to absurd values and see if Q moves at all
  3. look at the norm of the state-embedding weights inside the module
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
critic, _ = load_action_chunk_critic(str(CK), device="cpu")


def load(ep):
    d = torch.load(RUN / f"features/episode_{ep:06d}.pt", map_location="cpu", weights_only=False)
    return d["state"].float(), d["visual_features"].float(), d["action_chunk"].float()


def q(s, z, a):
    with torch.no_grad():
        return critic(s, z, a).flatten()


A, B = val[0], val[1]
sA, zA, aA = load(A)
sB, _, _ = load(B)
N = min(len(sA), len(sB), 64)
sA, zA, aA, sB = sA[:N], zA[:N], aA[:N], sB[:N]

print("=== 1. do the two state tensors actually differ? ===")
print(f"  sA[0] = {[round(float(v), 4) for v in sA[0]]}")
print(f"  sB[0] = {[round(float(v), 4) for v in sB[0]]}")
print(f"  mean |sA - sB| = {float((sA - sB).abs().mean()):.6f}")
print(f"  max  |sA - sB| = {float((sA - sB).abs().max()):.6f}")
print(f"  identical? {bool(torch.equal(sA, sB))}")

print()
print("=== 2. absurd state values ===")
base = q(sA, zA, aA)
for label, s in (
    ("state -> zeros", torch.zeros_like(sA)),
    ("state -> +100", torch.full_like(sA, 100.0)),
    ("state -> -100", torch.full_like(sA, -100.0)),
    ("state -> 1e6", torch.full_like(sA, 1e6)),
    ("state -> random N(0,10)", torch.randn_like(sA) * 10),
):
    d = (q(s, zA, aA) - base).abs()
    print(f"  {label:<26} mean|dQ| = {float(d.mean()):.8f}   max|dQ| = {float(d.max()):.8f}")

print()
print("=== 3. does the module have a state pathway at all? ===")
mod = critic.module
found = False
for name, p in mod.named_parameters():
    if "state" in name.lower():
        found = True
        print(f"  {name:<40} shape {tuple(p.shape)}  norm {float(p.norm()):.6e}  "
              f"max|w| {float(p.abs().max()):.6e}")
if not found:
    print("  no parameter name contains 'state'; listing all top-level params:")
    for name, p in mod.named_parameters():
        print(f"    {name:<44} {tuple(p.shape)}  norm {float(p.norm()):.4e}")

print()
print("=== 4. for contrast: same absurd-value test on the action input ===")
for label, a in (
    ("action -> zeros", torch.zeros_like(aA)),
    ("action -> +10", torch.full_like(aA, 10.0)),
    ("action -> random N(0,1)", torch.randn_like(aA)),
):
    d = (q(sA, zA, a) - base).abs()
    print(f"  {label:<26} mean|dQ| = {float(d.mean()):.8f}   max|dQ| = {float(d.max()):.8f}")
