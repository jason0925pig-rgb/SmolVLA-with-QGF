"""Handoff section 13 offline smoke test, run ON THE ORIN.

Loads the deployed mug Q critic, checks dimensions and the 50x8 action shape,
runs one forward pass on synthetic tensors, and confirms nothing reached the
network.  It does NOT touch any robot service, does not power on, enable,
servo, or move anything.
"""
import json
import os
import socket
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Hard-block outbound sockets so a hidden download would raise instead of hang.
_real_socket = socket.socket


class _NoNet(_real_socket):
    def connect(self, *a, **k):
        raise RuntimeError(f"network access attempted: {a}")

    def connect_ex(self, *a, **k):
        raise RuntimeError(f"network access attempted: {a}")


socket.socket = _NoNet

import torch  # noqa: E402

sys.path.insert(0, "/home/nvidia/work/telop/SmolVLA-with-QGF/qgf/src")
from guided_action_flow.critics.checkpoint import load_action_chunk_critic  # noqa: E402

D = "/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829"
P = f"{D}/critic_member_00.pt"

print("=== checkpoint identity ===")
ck = torch.load(P, map_location="cpu", weights_only=False)
cfg = ck["critic_config"]
print(f"  path             {P}")
print(f"  critic_arch      {ck['critic_arch']}")
print(f"  selected_epoch   {ck['selected_epoch']}")
print(f"  selected_val_td  {ck['selected_val_td_loss']:.6f}")
print(f"  ensemble_member  {ck['ensemble_member_index']}  (single critic)")
print(f"  config           {cfg}")

fail = []
for k, want in (("state_dim", 8), ("action_dim", 8), ("action_horizon", 50),
                ("visual_tokens", 128), ("visual_token_dim", 960)):
    ok = cfg.get(k) == want
    print(f"  {'PASS' if ok else 'FAIL'}  {k} == {want} (got {cfg.get(k)})")
    if not ok:
        fail.append(k)

print()
print("=== offline load + one forward pass on synthetic tensors ===")
model = load_action_chunk_critic(P, device="cpu")
net = model[0] if isinstance(model, (tuple, list)) else model
if hasattr(net, "eval"):
    net.eval()

B = 4
s = torch.zeros(B, cfg["state_dim"])
a = torch.zeros(B, cfg["action_horizon"], cfg["action_dim"])
z = torch.zeros(B, cfg["visual_tokens"], cfg["visual_token_dim"])
with torch.no_grad():
    q = net(s, z, a)
print(f"  input  state {tuple(s.shape)}  visual {tuple(z.shape)}  action {tuple(a.shape)}")
print(f"  output Q {tuple(q.shape)}  finite={bool(torch.isfinite(q).all())}")
print(f"  Q values: {[round(float(v), 6) for v in q.flatten()[:4]]}")
if q.shape[0] != B:
    fail.append("batch dim")
if not torch.isfinite(q).all():
    fail.append("non-finite Q")

print()
print("=== 50x8 action shape is enforced ===")
try:
    with torch.no_grad():
        net(s, z, torch.zeros(B, 25, cfg["action_dim"]))
    print("  WARN  a 25-step chunk was accepted; horizon is not enforced by shape")
except Exception as exc:
    print(f"  PASS  a 25-step chunk is rejected: {type(exc).__name__}")

print()
print("=== deployment runtime settings the operator must export ===")
prov = json.load(open(f"{D}/training_provenance.json", encoding="utf-8"))
for k, v in prov["deployment"]["runtime_env"].items():
    print(f"  {k} = {v}")

print()
print("=== no network was used ===")
print("  outbound sockets were blocked for the whole run; nothing raised")

print()
print("=== safety ===")
print("  no robot service was called: no power-on, no enable, no servo,")
print("  no gripper command, no arm motion.  Pure file + tensor operations.")

print()
if fail:
    print(f"FAILED: {fail}")
    sys.exit(1)
print("ORIN OFFLINE SMOKE OK")
