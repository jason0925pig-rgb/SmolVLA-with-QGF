#!/usr/bin/env python3
"""Install the guidance_mode ablation switch on the Orin, in place.

WHY ANCHORED EDITS RATHER THAN COPYING FILES IN
  The Orin's checkout is not the repo: it sits at an old commit with local
  uncommitted edits to the control files. Overwriting qgf.py or
  policy_server_qgf.py with a copy generated against origin/main would silently
  revert whatever those local edits are. This script edits the Orin's OWN copies
  by matching exact anchors, so it fails loudly if the file is not what it expects
  rather than clobbering it.

WHAT IT ADDS
  QGuidanceConfig.guidance_mode in {critic, random_matched_norm, zero} + random_seed
    critic              the deployed behaviour, unchanged and still the default
    random_matched_norm same per-sample |grad|, clip, gate and 1/beta; random direction
    zero                B0-LM: the critic forward AND backward still run and are
                        logged at their true values; only the applied guidance is
                        zeroed, immediately before velocity - grad/beta
  Two diagnostics: q_direction_cos_mean (1.0 for critic, ~0 for random, 0 for zero)
  and q_guidance_mode.
  The policy server reads SMOLVLA_QGF_GUIDANCE_MODE / SMOLVLA_QGF_RANDOM_SEED and
  logs a per-chunk inference time as "QGF_CHUNK idx=.. infer_ms=.. mode=.. beta=..".

SAFETY
  Refuses while any session holds the arm or the cameras. Backs both files up
  first. Touches nothing else. --check reports without writing; --rollback
  restores the backups.

  python3 deploy_guidance_mode.py [--check|--rollback]
"""
import argparse
import ast
import os
import shutil
import subprocess
import sys
import time

REPO = os.environ.get("QGF_ORIN_REPO", "/home/nvidia/work/telop/SmolVLA-with-QGF")
QGF = os.path.join(REPO, "qgf/src/guided_action_flow/guidance/qgf.py")
SRV = os.path.join(REPO, "lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py")
STAMP = time.strftime("%Y%m%d_%H%M%S")

BUSY = ("run_qgf_collection_sessio[n]|policy_server_qg[f]|policy_server_telemetr[y]|"
        "async_clien[t]|safe_one_arm_serv[o]|safe_gripper_controlle[r]|"
        "udp_leader_bridg[e]|ros2_episode_recorde[r]|dataset_cameras.launch.p[y]")


def busy():
    if not shutil.which("pgrep"):
        sys.exit("FATAL: pgrep is unavailable, so the busy check cannot run. Refusing.")
    return subprocess.call(["pgrep", "-f", BUSY], stdout=subprocess.DEVNULL) == 0


def edit(text, marker, old, new, label):
    # The marker exists only once the edit is in place. Testing the anchor is not
    # enough: five of these replacements append to their anchor, so the anchor
    # survives and a second run would insert the block twice.
    if marker in text:
        return text, "already"
    if old not in text:
        sys.exit("FATAL: anchor not found in the deployed file: %s\n"
                 "       The Orin's copy is not what this patch expects. Nothing written." % label)
    return text.replace(old, new, 1), "applied"


# ------------------------------------------------------------------ qgf.py
QGF_EDITS = [
    ("config dataclass",
     "GUIDANCE_MODES = (",
     '''@dataclass(frozen=True)
class QGuidanceConfig:
    beta: float = 10.0
    grad_clip_norm: float | None = 1.0
    uncertainty_scale: float = 0.0
    min_gate: float = 0.0
''',
     '''GUIDANCE_MODES = ("critic", "random_matched_norm", "zero")


@dataclass(frozen=True)
class QGuidanceConfig:
    beta: float = 10.0
    grad_clip_norm: float | None = 1.0
    uncertainty_scale: float = 0.0
    min_gate: float = 0.0
    # Ablation switch. Everything else - the critic forward, the backward, the
    # clip, the gate, the 1/beta scaling, the RTC hook and therefore the timing
    # and the RNG consumption - is identical across the three modes.
    guidance_mode: str = "critic"
    # Seed for the random direction, drawn from a dedicated generator so that
    # SmolVLA's own flow-noise sampling from the global RNG is untouched.
    random_seed: int | None = None


_RANDOM_GENERATORS: dict = {}


def _random_generator(device, seed):
    import torch

    key = (str(device), int(seed))
    gen = _RANDOM_GENERATORS.get(key)
    if gen is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        _RANDOM_GENERATORS[key] = gen
    return gen


def reset_random_generators() -> None:
    _RANDOM_GENERATORS.clear()
'''),

    ("validation",
     "QGuidanceConfig.guidance_mode must be one of",
     '''    if config.min_gate < 0 or config.min_gate > 1:
        raise ValueError("QGuidanceConfig.min_gate must be in [0, 1].")
''',
     '''    if config.min_gate < 0 or config.min_gate > 1:
        raise ValueError("QGuidanceConfig.min_gate must be in [0, 1].")
    if config.guidance_mode not in GUIDANCE_MODES:
        raise ValueError(
            f"QGuidanceConfig.guidance_mode must be one of {GUIDANCE_MODES}, "
            f"got {config.guidance_mode!r}."
        )
    if config.guidance_mode == "random_matched_norm" and config.random_seed is None:
        raise ValueError(
            "random_matched_norm requires QGuidanceConfig.random_seed (no global RNG)."
        )
'''),

    ("direction swap",
     "real_grad = grad",
     '''        raw_grad_norm = grad.reshape(grad.shape[0], -1).norm(dim=-1)
        if config.grad_clip_norm is not None:''',
     '''        raw_grad_norm = grad.reshape(grad.shape[0], -1).norm(dim=-1)
        real_grad = grad
        if config.guidance_mode == "random_matched_norm":
            # Keep the real gradient's per-sample norm; replace only its direction.
            # Done BEFORE clipping, so the clip and the gate see exactly what they
            # would have seen for a real gradient of the same magnitude.
            gen = _random_generator(grad.device, config.random_seed)
            rand = torch.randn(
                grad.shape, generator=gen, device=grad.device, dtype=grad.dtype
            )
            rand_norm = rand.reshape(rand.shape[0], -1).norm(dim=-1)
            view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
            grad = rand * (raw_grad_norm / (rand_norm + 1.0e-6)).reshape(view_shape)
        if config.grad_clip_norm is not None:'''),

    ("zero + cosine",
     'if config.guidance_mode == "zero":',
     '''        grad = grad * gate.reshape(view_shape)
        guided_velocity = velocity_t - grad / config.beta
        guidance_delta = velocity_t - guided_velocity
''',
     '''        grad = grad * gate.reshape(view_shape)
        if config.guidance_mode == "zero":
            # B0-LM. The critic forward, the backward, the clip and the gate have
            # all run above and are logged at their true values; only the APPLIED
            # guidance is zeroed here, so timing and RNG consumption match "critic".
            grad = torch.zeros_like(grad)
        guided_velocity = velocity_t - grad / config.beta
        guidance_delta = velocity_t - guided_velocity
        flat_real = real_grad.detach().reshape(real_grad.shape[0], -1)
        flat_used = grad.detach().reshape(grad.shape[0], -1)
        direction_cos = torch.nn.functional.cosine_similarity(
            flat_real, flat_used, dim=-1, eps=1.0e-12
        )
'''),

    ("diagnostics",
     "q_direction_cos_mean",
     '''        "q_guidance_norm_mean": guidance_delta.detach()
        .reshape(guidance_delta.shape[0], -1)
        .norm(dim=-1)
        .mean(),
    }''',
     '''        "q_guidance_norm_mean": guidance_delta.detach()
        .reshape(guidance_delta.shape[0], -1)
        .norm(dim=-1)
        .mean(),
        # 1.0 in critic mode, ~0 in random_matched_norm, 0 in zero.
        "q_direction_cos_mean": direction_cos.mean(),
        "q_guidance_mode": q_value.detach().new_tensor(
            float(GUIDANCE_MODES.index(config.guidance_mode))
        ),
    }'''),
]

# --------------------------------------------------- policy_server_qgf.py
SRV_EDITS = [
    ("import", "import os\nimport torch", "import os\n", "import os\nimport torch\n"),

    ("GUIDANCE_MODES import",
     "import GUIDANCE_MODES, QGuidanceConfig",
     "from guided_action_flow.guidance.qgf import QGuidanceConfig",
     "from guided_action_flow.guidance.qgf import GUIDANCE_MODES, QGuidanceConfig"),

    ("per-chunk latency",
     "_qgf_chunk_counter",
     '''class QGFPolicyServer(TelemetryPolicyServer):
    """Install exactly one visual Q critic after the policy is loaded."""
''',
     '''class QGFPolicyServer(TelemetryPolicyServer):
    """Install exactly one visual Q critic after the policy is loaded."""

    _qgf_chunk_counter = 0

    def _get_action_chunk(self, observation):
        # Per-chunk wall clock, so every arm carries a MEASURED latency rather
        # than an assumed one. B0-LM's whole point is the timing comparison, and
        # no QGF-on latency has ever been recorded on this robot.
        import time as _time

        t0 = _time.perf_counter()
        chunk = super()._get_action_chunk(observation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        QGFPolicyServer._qgf_chunk_counter += 1
        self.logger.info(
            "QGF_CHUNK idx=%d infer_ms=%.1f mode=%s beta=%s",
            QGFPolicyServer._qgf_chunk_counter,
            elapsed_ms,
            os.environ.get("SMOLVLA_QGF_GUIDANCE_MODE", "critic") or "critic",
            os.environ.get("SMOLVLA_QGF_BETA", "?"),
        )
        return chunk
'''),

    ("env reads",
     # NOT SMOLVLA_QGF_GUIDANCE_MODE: the per-chunk latency edit above also
     # inserts that name, which made this edit look already-applied and skipped.
     "seed_text = os.environ.get",
     '''        grad_clip_norm = _positive_env_float("SMOLVLA_QGF_GRAD_CLIP_NORM")
''',
     '''        grad_clip_norm = _positive_env_float("SMOLVLA_QGF_GRAD_CLIP_NORM")
        guidance_mode = (
            os.environ.get("SMOLVLA_QGF_GUIDANCE_MODE", "critic").strip() or "critic"
        )
        if guidance_mode not in GUIDANCE_MODES:
            raise RuntimeError(
                "SMOLVLA_QGF_GUIDANCE_MODE must be one of "
                f"{GUIDANCE_MODES}; got {guidance_mode!r}."
            )
        seed_text = os.environ.get("SMOLVLA_QGF_RANDOM_SEED", "").strip()
        random_seed = int(seed_text) if seed_text else None
        if guidance_mode == "random_matched_norm" and random_seed is None:
            raise RuntimeError(
                "SMOLVLA_QGF_RANDOM_SEED is required when "
                "SMOLVLA_QGF_GUIDANCE_MODE=random_matched_norm."
            )
'''),

    ("config kwargs",
     "guidance_mode=guidance_mode",
     '''                uncertainty_scale=0.0,
                min_gate=0.0,
            ),''',
     '''                uncertainty_scale=0.0,
                min_gate=0.0,
                guidance_mode=guidance_mode,
                random_seed=random_seed,
            ),'''),

    ("log line",
     "guidance_mode={guidance_mode}",
     '''            f"grad_clip_norm={grad_clip_norm:.8g}; uncertainty_gate=disabled"
        )''',
     '''            f"grad_clip_norm={grad_clip_norm:.8g}; uncertainty_gate=disabled; "
            f"guidance_mode={guidance_mode}; random_seed={random_seed}"
        )'''),
]


# Every string here must exist once the file is patched, whatever the markers say.
REQUIRED = {
    "qgf.py": (
        'GUIDANCE_MODES = ("critic", "random_matched_norm", "zero")',
        "guidance_mode: str = \"critic\"",
        "random_seed: int | None = None",
        "def _random_generator(",
        "def reset_random_generators(",
        "QGuidanceConfig.guidance_mode must be one of",
        "random_matched_norm requires QGuidanceConfig.random_seed",
        "real_grad = grad",
        'if config.guidance_mode == "random_matched_norm":',
        'if config.guidance_mode == "zero":',
        "grad = torch.zeros_like(grad)",
        "direction_cos = torch.nn.functional.cosine_similarity(",
        '"q_direction_cos_mean"',
        '"q_guidance_mode"',
    ),
    "policy_server_qgf.py": (
        "import torch",
        "from guided_action_flow.guidance.qgf import GUIDANCE_MODES, QGuidanceConfig",
        "_qgf_chunk_counter",
        "QGF_CHUNK idx=%d infer_ms=%.1f",
        'os.environ.get("SMOLVLA_QGF_GUIDANCE_MODE", "critic")',
        "SMOLVLA_QGF_RANDOM_SEED",
        "seed_text = os.environ.get",
        "guidance_mode=guidance_mode",
        "random_seed=random_seed",
        "guidance_mode={guidance_mode}",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--rollback", action="store_true", help="restore the newest backups")
    args = ap.parse_args()

    for p in (QGF, SRV):
        if not os.path.isfile(p):
            sys.exit("FATAL: not found: %s" % p)

    if args.rollback:
        for p in (QGF, SRV):
            backups = sorted(g for g in os.listdir(os.path.dirname(p))
                             if g.startswith(os.path.basename(p) + ".bak."))
            if not backups:
                print("  no backup for %s" % p)
                continue
            src = os.path.join(os.path.dirname(p), backups[-1])
            shutil.copy2(src, p)
            print("  restored %s from %s" % (p, backups[-1]))
        return

    # The busy check gates WRITING, not reporting: --check touches nothing, and
    # being able to review the plan while a session is running is the point.
    if not args.check and busy():
        sys.exit("FATAL: a session is holding the arm or the cameras. Refusing to patch\n"
                 "       the inference path underneath a running rollout or teleop session.")

    plans = []
    for path, edits in ((QGF, QGF_EDITS), (SRV, SRV_EDITS)):
        text = open(path, encoding="utf-8").read()
        original = text
        states = []
        for label, marker, old, new in edits:
            text, state = edit(text, marker, old, new, "%s: %s" % (os.path.basename(path), label))
            states.append((label, state))
        try:
            ast.parse(text)
        except SyntaxError as exc:
            sys.exit("FATAL: the patched %s would not parse: %s" % (path, exc))
        # Post-conditions, independent of the applied-markers. A marker that
        # another edit can introduce would silently skip an edit; these catch it.
        for needed in REQUIRED.get(os.path.basename(path), ()):
            if needed not in text:
                sys.exit("FATAL: after patching, %s still lacks %r.\n"
                         "       An edit was skipped. Nothing written."
                         % (os.path.basename(path), needed))
        plans.append((path, original, text, states))

    print("=== plan ===")
    for path, original, text, states in plans:
        print("  %s" % path)
        for label, state in states:
            print("      %-22s %s" % (label, state))
        print("      %d -> %d lines" % (len(original.splitlines()), len(text.splitlines())))

    if args.check:
        print("\n--check: nothing written")
        return

    for path, original, text, states in plans:
        bak = "%s.bak.%s" % (path, STAMP)
        shutil.copy2(path, bak)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("  wrote %s   (backup %s)" % (path, os.path.basename(bak)))

    print("\nDEPLOY OK")
    print("verify next, with no robot involved:")
    print("  LD_LIBRARY_PATH=/home/nvidia/work/telop/venvs/smolvla-orin/opt/"
          "libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib \\")
    print("    /home/nvidia/work/telop/.venvs/onearm-lerobot/bin/python "
          "/home/nvidia/work/telop/tmp/smoke_guidance_mode.py")


if __name__ == "__main__":
    main()
