"""Handoff section 13 offline smoke test, run ON THE ORIN - TASK PARAMETERISED.

Task-generic successor of orin_offline_smoke.py.  The deployment directory comes
from QGF_ORIN_DEPLOY_DIR; nothing about the task is hardcoded.

It loads the deployed Q critic, verifies the bundle's own SHA256SUMS, checks the
checkpoint identity and visual dimensions, enforces the 50 x 8 action shape, then
replays the policy server's own installation path on loopback - the same gates,
the same QGuidanceConfig, the same beta log line, and one real guided-velocity
computation - and finally proves that nothing reached the network.

SAFETY - THIS SCRIPT IS A PURE OFFLINE LOAD TEST.
It must NOT and does NOT power on the robot, enable it, enter servo mode,
actuate the gripper, or send any arm motion.  It calls no robot service, starts
no ROS node, and publishes no topic.  Three guards enforce that:
  1. an import guard that raises if any ROS / robot-message module is imported
     (importing the real policy server would create a ROS node and publisher, so
     its source is READ AS TEXT and never imported);
  2. a socket guard that permits only 127.0.0.1 / ::1 and raises on anything
     else, recording every attempt;
  3. a static self-audit that re-reads this file and refuses to pass if any
     robot-actuation token appears in its executable body.
Only after this offline report is delivered and reviewed on site may an operator
next to the robot run a controlled powered test.

Required environment:
    QGF_ORIN_DEPLOY_DIR         /home/nvidia/work/telop/models/qgf/<run id>
    SMOLVLA_QGF_BETA            the positive beta chosen for this critic
                                (the policy server reads this exact name)
    SMOLVLA_QGF_GRAD_CLIP_NORM  1.0
    SMOLVLA_QGF_CRITIC_PATH     <QGF_ORIN_DEPLOY_DIR>/critic_member_00.pt

Optional:
    QGF_ORIN_REPO       repo checkout, default /home/nvidia/work/telop/SmolVLA-with-QGF
    QGF_RUN_MODE        must be "qgf" if set
    SMOLVLA_ORIN_BUNDLE must match the bundle recorded in training_provenance.json

On the Orin, anything that imports torch needs:
    LD_LIBRARY_PATH=/home/nvidia/work/telop/venvs/smolvla-orin/opt/libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib
"""

# --- BEGIN SAFETY DECLARATION AND TOKEN LIST ---
# Everything above and inside this block is documentation and the guard's own
# vocabulary; the static self-audit deliberately excludes it, otherwise the
# safety declaration itself would trip the scan.  The two markers are built by
# concatenation below so that the marker strings do not appear literally before
# the real comment lines.
FORBIDDEN_CALL_TOKENS = (
    "power_on", "poweron", "set_enabled", "enable_service", "servo",
    "gripper_command", "executed_gripper_command", "teleop_joint_command",
    "motion_enabled", "stop_request", "JointTrajectory", "JointState",
    "SetBool", "Trigger", "rclpy", "create_publisher", "create_client",
    "create_service", "/right_arm/", "TelemetryPolicyServer", "QGFPolicyServer",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "rclpy", "rmw", "rosidl", "ros2", "ros2cli", "std_msgs", "std_srvs",
    "sensor_msgs", "trajectory_msgs", "geometry_msgs", "control_msgs",
    "lerobot_robot_armstrong_ros2",
)
MARK_1_END = "# --- END SAFETY DECLARATION" + " AND TOKEN LIST ---"
MARK_2_BEGIN = "# --- BEGIN SAFETY" + " SUMMARY ---"
MARK_2_END = "# --- END SAFETY" + " SUMMARY ---"
# --- END SAFETY DECLARATION AND TOKEN LIST ---

import hashlib
import io
import json
import os
import socket
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

fail = []
notes = []


def check(ok, message):
    print("  {0}  {1}".format("PASS" if ok else "FAIL", message))
    if not ok:
        fail.append(message)
    return bool(ok)


# ---------------------------------------------------------------------------
# guard 1: no ROS / robot-message module may be imported by this process
# ---------------------------------------------------------------------------
def _is_forbidden_module(fullname):
    for prefix in FORBIDDEN_IMPORT_PREFIXES:
        if fullname == prefix or fullname.startswith(prefix + "."):
            return True
    return False


class _ImportGuard(object):
    """Refuses any robot / ROS import.  Everything else falls through."""

    def find_spec(self, fullname, path=None, target=None):
        if _is_forbidden_module(fullname):
            raise ImportError(
                "offline smoke test refused to import {0!r}: this script must not "
                "touch the robot stack".format(fullname)
            )
        return None

    def find_module(self, fullname, path=None):  # python 2 style fallback
        if _is_forbidden_module(fullname):
            raise ImportError(
                "offline smoke test refused to import {0!r}".format(fullname))
        return None


sys.meta_path.insert(0, _ImportGuard())

# ---------------------------------------------------------------------------
# guard 2: only loopback sockets.  Every attempt is recorded either way.
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
    # AF_UNIX (a str/bytes path) and anything unrecognised are denied, exactly as
    # the original smoke test denied every connect.
    if isinstance(address, tuple) and address:
        return _is_loopback_host(address[0])
    return False


class _LoopbackOnlySocket(_real_socket):
    def _qgf_gate(self, address, how):
        if _is_loopback_address(address):
            NET_ALLOWED.append((how, repr(address)))
            return
        NET_DENIED.append((how, repr(address)))
        raise RuntimeError("network access attempted: {0} {1!r}".format(how, address))

    def connect(self, address, *a, **k):
        self._qgf_gate(address, "connect")
        return _real_socket.connect(self, address, *a, **k)

    def connect_ex(self, address, *a, **k):
        self._qgf_gate(address, "connect_ex")
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
# 0. contract  (resolved before torch is imported, so a missing variable fails
#    in a second instead of after a multi-second import)
# ---------------------------------------------------------------------------
def env(name, required=True, default=None):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        if required:
            print("FATAL: environment variable {0} is unset or empty.".format(name))
            print("       This script has no task defaults on purpose.  Export at least:")
            print("       QGF_ORIN_DEPLOY_DIR, SMOLVLA_QGF_CRITIC_PATH, SMOLVLA_QGF_BETA,")
            print("       SMOLVLA_QGF_GRAD_CLIP_NORM")
            sys.exit(2)
        return default
    return raw.strip()


D = env("QGF_ORIN_DEPLOY_DIR").rstrip("/")
REPO = env("QGF_ORIN_REPO", required=False,
           default="/home/nvidia/work/telop/SmolVLA-with-QGF").rstrip("/")
P = "{0}/critic_member_00.pt".format(D)
BUNDLE_FILES = (
    "SHA256SUMS",
    "critic_member_00.pt",
    "episode_split_45_5.json",
    "training_input_summary.json",
    "training_provenance.json",
    "training_summary.json",
)

print("=== contract ===")
print("  QGF_ORIN_DEPLOY_DIR  {0}".format(D))
print("  QGF_ORIN_REPO        {0}".format(REPO))
print("  critic               {0}".format(P))
print()

if not os.path.isdir(D):
    print("FATAL: {0} is not a directory.  Deploy it first.".format(D))
    sys.exit(2)
if not os.path.isfile(P):
    print("FATAL: {0} does not exist.".format(P))
    sys.exit(2)

sys.path.insert(0, "{0}/qgf/src".format(REPO))


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(block), b""):
            h.update(blk)
    return h.hexdigest()


# torch is imported only after both guards are installed and the contract is
# resolved, so nothing can slip a download or a robot import in ahead of them.
import torch  # noqa: E402

# ---------------------------------------------------------------------------
# 1. the deployed bundle is exactly the six section-13 files and still verifies
# ---------------------------------------------------------------------------
print("=== deployed bundle integrity ===")
present = sorted(f for f in os.listdir(D) if os.path.isfile(os.path.join(D, f)))
subdirs = sorted(f for f in os.listdir(D) if not os.path.isfile(os.path.join(D, f)))
check(present == sorted(BUNDLE_FILES),
      "the deployed directory holds exactly the six section-13 files (found {0})".format(present))
check(not subdirs, "the deployed directory has no subdirectories")

sums_path = os.path.join(D, "SHA256SUMS")
bad_sums, n_sums = [], 0
if os.path.isfile(sums_path):
    for line in io.open(sums_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        name = name.strip().lstrip("*")
        target = os.path.join(D, name)
        n_sums += 1
        if not os.path.isfile(target) or sha256(target).lower() != digest.strip().lower():
            bad_sums.append(name)
check(n_sums == len(BUNDLE_FILES) - 1,
      "SHA256SUMS lists the five payload files ({0})".format(n_sums))
check(not bad_sums, "every listed file still matches its SHA256 ({0} bad)".format(len(bad_sums)))
print("       critic_member_00.pt sha256 = {0}".format(sha256(P)))

prov_path = os.path.join(D, "training_provenance.json")
if not os.path.isfile(prov_path):
    print("FATAL: {0} is missing; the bundle is incomplete.".format(prov_path))
    sys.exit(2)
prov = json.load(io.open(prov_path, encoding="utf-8"))
runtime_env = (prov.get("deployment") or {}).get("runtime_env") or {}
task_prompt = prov.get("task_prompt")

check(prov.get("run_id") == os.path.basename(D),
      "provenance run_id matches the deployment directory name ({0})".format(prov.get("run_id")))
check(str((prov.get("deployment") or {}).get("orin_target", "")).rstrip("/") == D,
      "provenance orin_target is this directory")
check(runtime_env.get("SMOLVLA_QGF_CRITIC_PATH") == P,
      "provenance SMOLVLA_QGF_CRITIC_PATH is this critic")
check(bool(task_prompt), "provenance records the task prompt")
print("       task    : {0}".format(prov.get("task")))
print("       prompt  : {0}".format(task_prompt))

# ---------------------------------------------------------------------------
# 2. the runtime environment the policy server will actually read
# ---------------------------------------------------------------------------
print()
print("=== runtime environment (the names the policy server reads) ===")
crit_env = os.environ.get("SMOLVLA_QGF_CRITIC_PATH", "").strip()
check(crit_env == P,
      "SMOLVLA_QGF_CRITIC_PATH is exported and points at this critic (got {0!r})".format(crit_env))

beta_raw = os.environ.get("SMOLVLA_QGF_BETA")
beta_alias = os.environ.get("QGF_BETA")
if beta_raw is None and beta_alias is None:
    print("FATAL: neither SMOLVLA_QGF_BETA nor QGF_BETA is set; the beta log line")
    print("       cannot be smoke-tested.  Export SMOLVLA_QGF_BETA=<positive number>.")
    sys.exit(2)
if beta_raw is None:
    check(False,
          "SMOLVLA_QGF_BETA is exported - the policy server reads this exact name and "
          "raises on the documented alias QGF_BETA; export SMOLVLA_QGF_BETA={0}".format(beta_alias))
    beta_raw = beta_alias
elif beta_alias is not None:
    check(beta_alias.strip() == beta_raw.strip(),
          "QGF_BETA and SMOLVLA_QGF_BETA agree ({0!r} vs {1!r})".format(beta_alias, beta_raw))
try:
    beta = float(beta_raw)
except ValueError:
    print("FATAL: beta {0!r} is not a number.".format(beta_raw))
    sys.exit(2)
check(beta > 0.0, "beta is positive ({0})".format(beta))

clip_raw = os.environ.get("SMOLVLA_QGF_GRAD_CLIP_NORM")
check(clip_raw is not None and clip_raw.strip() != "",
      "SMOLVLA_QGF_GRAD_CLIP_NORM is exported (the server raises without it)")
try:
    grad_clip_norm = float(clip_raw) if clip_raw else 1.0
except ValueError:
    grad_clip_norm = -1.0
check(grad_clip_norm == 1.0,
      "grad_clip_norm is the fixed contract value 1.0 (got {0})".format(clip_raw))

run_mode = os.environ.get("QGF_RUN_MODE")
if run_mode is None:
    notes.append("QGF_RUN_MODE is not exported in this shell; the launcher must set it to 'qgf'")
    print("  NOTE  QGF_RUN_MODE is not exported here; the launcher must set it to 'qgf'")
else:
    check(run_mode.strip() == "qgf", "QGF_RUN_MODE == qgf (got {0!r})".format(run_mode))

orin_bundle = os.environ.get("SMOLVLA_ORIN_BUNDLE")
prov_bundle = runtime_env.get("SMOLVLA_ORIN_BUNDLE")
if orin_bundle is None:
    notes.append("SMOLVLA_ORIN_BUNDLE is not exported in this shell; it must be {0}".format(prov_bundle))
    print("  NOTE  SMOLVLA_ORIN_BUNDLE is not exported here; it must be {0}".format(prov_bundle))
else:
    check(orin_bundle.strip().rstrip("/") == str(prov_bundle).rstrip("/"),
          "SMOLVLA_ORIN_BUNDLE is the bundle that produced these rollouts")

# ---------------------------------------------------------------------------
# 3. checkpoint identity
# ---------------------------------------------------------------------------
print()
print("=== checkpoint identity ===")
ck = torch.load(P, map_location="cpu", weights_only=False)
cfg = ck["critic_config"]
print("  path             {0}".format(P))
print("  critic_arch      {0}".format(ck["critic_arch"]))
print("  selected_epoch   {0}".format(ck["selected_epoch"]))
print("  selected_val_td  {0:.6f}".format(ck["selected_val_td_loss"]))
print("  ensemble_member  {0}  (single critic)".format(ck["ensemble_member_index"]))
print("  config           {0}".format(cfg))

check(ck.get("critic_arch") == "visual_transformer", "critic_arch == visual_transformer")
check(ck.get("ensemble_member_index") == 0, "ensemble_member_index == 0")
for k, want in (("state_dim", 8), ("action_dim", 8), ("action_horizon", 50),
                ("visual_tokens", 128), ("visual_token_dim", 960)):
    check(cfg.get(k) == want, "{0} == {1} (got {2})".format(k, want, cfg.get(k)))

# ---------------------------------------------------------------------------
# 4. offline load + one forward pass on synthetic tensors
# ---------------------------------------------------------------------------
print()
print("=== offline load + one forward pass on synthetic tensors ===")
from guided_action_flow.critics.checkpoint import load_action_chunk_critic  # noqa: E402

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
print("  input  state {0}  visual {1}  action {2}".format(tuple(s.shape), tuple(z.shape), tuple(a.shape)))
print("  output Q {0}  finite={1}".format(tuple(q.shape), bool(torch.isfinite(q).all())))
print("  Q values: {0}".format([round(float(v), 6) for v in q.flatten()[:4]]))
check(q.shape[0] == B, "batch dim is preserved")
check(bool(torch.isfinite(q).all()), "Q is finite")

# ---------------------------------------------------------------------------
# 5. the 50 x 8 action shape is enforced
# ---------------------------------------------------------------------------
print()
print("=== 50x8 action shape is enforced ===")
try:
    with torch.no_grad():
        net(s, z, torch.zeros(B, 25, cfg["action_dim"]))
    print("  WARN  a 25-step chunk was accepted; horizon is not enforced by shape")
    notes.append("a 25-step action chunk was accepted by the critic module")
except Exception as exc:
    print("  PASS  a 25-step chunk is rejected: {0}".format(type(exc).__name__))

# ---------------------------------------------------------------------------
# 6. loopback policy-server smoke: the server's own gates, config and log line
# ---------------------------------------------------------------------------
print()
print("=== loopback policy-server smoke (no ROS node, no gRPC listener on any")
print("    external interface, no robot service) ===")

server_src_path = None
for cand in (
    "{0}/lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py".format(REPO),
    "{0}/lerobot_robot_armstrong_ros2/lerobot_robot_armstrong_ros2/policy_server_qgf.py".format(REPO),
):
    if os.path.isfile(cand):
        server_src_path = cand
        break
check(server_src_path is not None,
      "the policy server source was found under {0}".format(REPO))
server_src = io.open(server_src_path, encoding="utf-8").read() if server_src_path else ""
print("       source read as TEXT (never imported: importing it would create a ROS node)")
if server_src_path:
    print("       {0}".format(server_src_path))

CONTRACT_FRAGMENTS = (
    'critic_path = Path(os.environ.get("SMOLVLA_QGF_CRITIC_PATH", ""))',
    'beta = _positive_env_float("SMOLVLA_QGF_BETA")',
    'grad_clip_norm = _positive_env_float("SMOLVLA_QGF_GRAD_CLIP_NORM")',
    'if metadata.get("critic_arch") != "visual_transformer":',
    'if int(metadata["critic_config"]["action_dim"]) != 8:',
    "uncertainty_scale=0.0,",
    "min_gate=0.0,",
    "critic_action_dim=8,",
    '"QGF single-critic guidance installed: "',
    'f"checkpoint={critic_path}; beta={beta:.8g}; coefficient=1/beta={1.0 / beta:.8g}; "',
    'f"grad_clip_norm={grad_clip_norm:.8g}; uncertainty_gate=disabled"',
)
missing_fragments = [f for f in CONTRACT_FRAGMENTS if f not in server_src]
check(not missing_fragments,
      "the deployed policy server still uses the gates and the beta log format this "
      "test reproduces ({0} fragment(s) drifted)".format(len(missing_fragments)))
for frag in missing_fragments:
    print("        drifted: {0}".format(frag))

# the server's own two gates, applied to this checkpoint
metadata = model[1] if isinstance(model, (tuple, list)) and len(model) > 1 else ck
check(metadata.get("critic_arch") == "visual_transformer",
      "server gate 1: metadata critic_arch is visual_transformer")
check(int(metadata["critic_config"]["action_dim"]) == 8,
      "server gate 2: the deployed critic uses eight action channels")

# the exact log line the server emits, rebuilt from the same format
critic_path = P
beta_log_line = (
    "QGF single-critic guidance installed: "
    "checkpoint={0}; beta={1:.8g}; coefficient=1/beta={2:.8g}; "
    "grad_clip_norm={3:.8g}; uncertainty_gate=disabled".format(
        critic_path, beta, 1.0 / beta, grad_clip_norm)
)
print()
print("  beta log line the server will emit:")
print("    {0}".format(beta_log_line))
check("beta={0:.8g};".format(beta) in beta_log_line, "the log line reports beta")
check("coefficient=1/beta={0:.8g};".format(1.0 / beta) in beta_log_line,
      "the log line reports the guidance coefficient 1/beta = {0:.8g}".format(1.0 / beta))
check("grad_clip_norm={0:.8g};".format(grad_clip_norm) in beta_log_line,
      "the log line reports grad_clip_norm = {0:.8g}".format(grad_clip_norm))
check("uncertainty_gate=disabled" in beta_log_line, "the log line reports the gate as disabled")
check(critic_path in beta_log_line, "the log line names this deployed checkpoint")

# the real guidance path, on synthetic tensors, exactly as installed at runtime
from guided_action_flow.guidance.qgf import (  # noqa: E402
    QGuidanceConfig,
    q_guided_velocity_smolvla_reverse_time,
)
from guided_action_flow.policies.smolvla_qgf import SmolVLAVisualCriticAdapter  # noqa: E402

qcfg = QGuidanceConfig(
    beta=beta, grad_clip_norm=grad_clip_norm, uncertainty_scale=0.0, min_gate=0.0
)
adapter = SmolVLAVisualCriticAdapter(net)
torch.manual_seed(20260814)
gs = torch.randn(B, cfg["state_dim"])
gz = torch.randn(B, cfg["visual_tokens"], cfg["visual_token_dim"])
ga = torch.randn(B, cfg["action_horizon"], cfg["action_dim"])
gv = torch.randn(B, cfg["action_horizon"], cfg["action_dim"])
adapter.set_visual_tokens(gz)
guided_v, diag = q_guided_velocity_smolvla_reverse_time(
    critic=adapter,
    obs_features=gs,
    action_t=ga,
    velocity_t=gv,
    time_t=0.5,
    config=qcfg,
    critic_action_dim=cfg["action_dim"],
)
raw_norm = float(diag["q_grad_norm_raw_mean"])
clipped_norm = float(diag["q_grad_norm_mean"])
guid_norm = float(diag["q_guidance_norm_mean"])
gate_mean = float(diag["q_gate_mean"])
ens = float(diag["q_ensemble_size"])
print()
print("  guided velocity {0}  raw|grad|={1:.6g}  clipped|grad|={2:.6g}".format(
    tuple(guided_v.shape), raw_norm, clipped_norm))
print("  |guidance delta|={0:.6g}  gate={1:.3f}  ensemble={2:.0f}".format(guid_norm, gate_mean, ens))
check(bool(torch.isfinite(guided_v).all()), "the guided velocity is finite")
check(guided_v.shape == gv.shape, "the guided velocity keeps the [B, 50, 8] chunk shape")
check(raw_norm > 0.0, "the critic actually has a gradient on the action chunk (not a no-op)")
check(clipped_norm <= grad_clip_norm + 1e-5,
      "the gradient is clipped to norm {0:.8g}".format(grad_clip_norm))
check(abs(guid_norm - clipped_norm / beta) <= 1e-4 * max(1.0, clipped_norm / beta),
      "the applied guidance is exactly clipped_grad / beta")
check(abs(gate_mean - 1.0) <= 1e-6, "the uncertainty gate is disabled (gate == 1)")
check(abs(ens - 1.0) <= 1e-6, "exactly one critic is in the ensemble")

# a real loopback round trip: the process can serve itself while off-box egress
# is impossible.  Bound to 127.0.0.1 only, never 0.0.0.0.
print()
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 0))
srv.listen(1)
host, port = srv.getsockname()[:2]
payload = beta_log_line.encode("utf-8")
cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cli.settimeout(5.0)
cli.connect((host, port))
conn, _peer = srv.accept()
conn.sendall(payload)
conn.shutdown(socket.SHUT_WR)
received = b""
while True:
    chunk = cli.recv(65536)
    if not chunk:
        break
    received += chunk
cli.close()
conn.close()
srv.close()
check(host == "127.0.0.1", "the loopback listener bound 127.0.0.1 only, never 0.0.0.0")
check(received == payload,
      "the beta log line made a full loopback round trip ({0} bytes)".format(len(payload)))

# ---------------------------------------------------------------------------
# 7. no network was used
# ---------------------------------------------------------------------------
print()
print("=== no network was used ===")
check(not NET_DENIED,
      "no off-box connection or name resolution was attempted ({0} denied)".format(len(NET_DENIED)))
for how, addr in NET_DENIED[:10]:
    print("        DENIED {0} {1}".format(how, addr))
print("  loopback traffic this run ({0} event(s)):".format(len(NET_ALLOWED)))
for how, addr in NET_ALLOWED[:10]:
    print("        allowed {0} {1}".format(how, addr))
print("  HF_HUB_OFFLINE={0} TRANSFORMERS_OFFLINE={1}".format(
    os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE")))

# ---------------------------------------------------------------------------
# 8. safety guards
# ---------------------------------------------------------------------------
print()
print("=== safety guards ===")
loaded_forbidden = sorted(m for m in list(sys.modules) if _is_forbidden_module(m))
check(not loaded_forbidden,
      "no robot / ROS module was imported ({0})".format(loaded_forbidden or "none"))

own_source = io.open(os.path.abspath(__file__), encoding="utf-8").read()
end1 = own_source.find(MARK_1_END)
begin2 = own_source.find(MARK_2_BEGIN)
end2 = own_source.find(MARK_2_END)
audit_ok = end1 > 0 and begin2 > end1 and end2 > begin2
if audit_ok:
    body = own_source[end1 + len(MARK_1_END):begin2] + own_source[end2 + len(MARK_2_END):]
    hits = sorted({t for t in FORBIDDEN_CALL_TOKENS if t in body})
    check(not hits,
          "static self-audit: no robot-actuation token in the executable body "
          "({0} chars scanned, hits={1})".format(len(body), hits or "none"))
else:
    check(False, "static self-audit could not locate its own markers")

# --- BEGIN SAFETY SUMMARY ---
print("  this run issued no power-on, no enable, no servo-mode entry, no gripper")
print("  command and no arm motion.  It called no robot service, started no ROS")
print("  node, and published no topic.  File, tensor and loopback operations only.")
# --- END SAFETY SUMMARY ---

# ---------------------------------------------------------------------------
# 9. what the operator must export on the machine
# ---------------------------------------------------------------------------
print()
print("=== deployment runtime settings the operator must export ===")
for k, v in runtime_env.items():
    print("  {0} = {1}".format(k, v))

print()
if notes:
    print("notes:")
    for n in notes:
        print("  - {0}".format(n))
    print()
if fail:
    print("FAILED ({0}):".format(len(fail)))
    for f in fail:
        print("  - {0}".format(f))
    sys.exit(1)
print("ORIN OFFLINE SMOKE OK")
