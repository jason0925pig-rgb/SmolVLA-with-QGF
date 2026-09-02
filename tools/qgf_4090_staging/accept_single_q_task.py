"""Handoff section 12 acceptance for a single-Q critic, TASK-PARAMETERIZED.

Same seven acceptance groups as the mug-specific accept_single_q.py; the task
identity now comes from the environment instead of being baked in, and the
45/5 split, the 80 epochs and every shape/hyperparameter stay fixed because
handoff sections 8 and 11 fix them.

Required environment (no defaults - an unset variable is a hard failure):
  QGF_SSD_ROOT       /opt/qgf_real_robot
  QGF_TASK_KEY       red_parcel
  QGF_RUN_ID         red_parcel_single_q_45_5_20260902
  QGF_EPISODE_FIRST  0
  QGF_EPISODE_LAST   49
  CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=1, so that "cuda" here
  really is physical GPU 1 (handoff iron rule 1).

Optional: argv[1] overrides the run directory that would be derived from
QGF_SSD_ROOT / QGF_RUN_ID.  The environment is still required, because an
acceptance report with no task label is exactly what ends up pasted into the
wrong task's result document.

Key names verified against train_real_robot_visual_iql.py, not guessed:
  training_input_summary.json = metadata
  training_summary.json       = metadata + {"members": [{member_index, path,
                                selected_epoch, selected_val_td_loss}]}
  critic_member_00.pt         = {format, critic_arch, critic_config,
                                model_state_dict, value_model_state_dict,
                                selected_epoch, selected_val_td_loss, history,
                                **metadata, ensemble_member_index, member_seed}
  history entry               = {epoch, train_q_loss, train_v_loss,
                                val_td_loss, val_q_mean, val_q_success_mean,
                                val_q_failure_mean, val_q_success_failure_gap,
                                val_positive_reward_samples, val_samples}

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python accept_single_q_task.py [run_dir]
"""
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- fixed values
# Handoff sections 8, 11 and 12.  These are not task parameters.
EXPECTED_TRAIN_EPISODES = 45
EXPECTED_VAL_EPISODES = 5
EXPECTED_EPOCHS = 80
PHYSICAL_GPU = 1
SPLIT_NAME = "episode_split_%d_%d.json" % (EXPECTED_TRAIN_EPISODES, EXPECTED_VAL_EPISODES)
EXPECTED_SHAPES = (
    ("state_dim", 8),
    ("action_dim", 8),
    ("action_horizon", 50),
    ("visual_tokens", 128),
    ("visual_token_dim", 960),
)
EXPECTED_NET = (
    ("d_model", 256),
    ("num_layers", 3),
    ("num_heads", 4),
    ("dropout", 0.1),
)
EXPECTED_TRAINING_ARGS = (
    ("ensemble_size", 1),
    ("epochs", EXPECTED_EPOCHS),
    ("batch_size", 16),
    ("lr", 3e-4),
    ("weight_decay", 1e-4),
    ("gamma", 0.99),
    ("expectile", 0.7),
    ("polyak", 0.005),
    ("d_model", 256),
    ("layers", 3),
    ("heads", 4),
    ("dropout", 0.1),
    ("seed", 20260814),
    ("device", "cuda"),
    ("expected_train_episodes", EXPECTED_TRAIN_EPISODES),
    ("expected_val_episodes", EXPECTED_VAL_EPISODES),
)

fail, warn = [], []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


def note(cond, msg):
    print(("  ok    " if cond else "  WARN  ") + msg)
    if not cond:
        warn.append(msg)


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def same_number(got, want):
    if isinstance(want, str):
        return got == want
    if isinstance(got, bool) or got is None:
        return False
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- environment
REQUIRED = ("QGF_SSD_ROOT", "QGF_TASK_KEY", "QGF_RUN_ID", "QGF_EPISODE_FIRST", "QGF_EPISODE_LAST")
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    print("FATAL: required environment variable(s) unset or empty: " + ", ".join(missing))
    print("       Refusing to guess a task.")
    sys.exit(2)

SSD_ROOT = Path(os.environ["QGF_SSD_ROOT"])
TASK_KEY = os.environ["QGF_TASK_KEY"]
RUN_ID = os.environ["QGF_RUN_ID"]
try:
    EP_FIRST = int(os.environ["QGF_EPISODE_FIRST"])
    EP_LAST = int(os.environ["QGF_EPISODE_LAST"])
except ValueError:
    print("FATAL: QGF_EPISODE_FIRST / QGF_EPISODE_LAST must be integers.")
    sys.exit(2)
if EP_LAST < EP_FIRST:
    print("FATAL: QGF_EPISODE_LAST < QGF_EPISODE_FIRST.")
    sys.exit(2)
EXPECTED_EPISODES = sorted(range(EP_FIRST, EP_LAST + 1))

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else SSD_ROOT / "runs" / RUN_ID
OUT = RUN / "outputs/single_qcritic"
FEATURES = RUN / "features"
SPLIT_FILE = RUN / "manifest" / SPLIT_NAME

print("task     : %s" % TASK_KEY)
print("run id   : %s" % RUN_ID)
print("run dir  : %s" % RUN)
print("episodes : %d..%d (%d)" % (EP_FIRST, EP_LAST, len(EXPECTED_EPISODES)))
print("split    : %d train / %d val" % (EXPECTED_TRAIN_EPISODES, EXPECTED_VAL_EPISODES))
print()

# ---------------------------------------------------------------- 0
print("=== 0. task wiring and GPU policy ===")
check(RUN.is_dir(), "run directory exists: %s" % RUN)
check(OUT.is_dir(), "output directory exists: %s" % OUT)
note(TASK_KEY in RUN_ID,
     "QGF_TASK_KEY %r appears in QGF_RUN_ID %r (task/run mismatch would mis-label this report)"
     % (TASK_KEY, RUN_ID))
check(len(EXPECTED_EPISODES) == EXPECTED_TRAIN_EPISODES + EXPECTED_VAL_EPISODES,
      "episode range covers %d episodes, the fixed split needs %d (got %d)"
      % (EXPECTED_TRAIN_EPISODES + EXPECTED_VAL_EPISODES,
         EXPECTED_TRAIN_EPISODES + EXPECTED_VAL_EPISODES, len(EXPECTED_EPISODES)))

cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
cdo = os.environ.get("CUDA_DEVICE_ORDER")
check(cvd == str(PHYSICAL_GPU),
      "CUDA_VISIBLE_DEVICES == %s so that 'cuda' is physical GPU %d (got %r)"
      % (PHYSICAL_GPU, PHYSICAL_GPU, cvd))
check(cdo == "PCI_BUS_ID",
      "CUDA_DEVICE_ORDER == PCI_BUS_ID, otherwise the physical index is not pinned (got %r)" % cdo)


def physical_gpu_uuid(index):
    """UUID of a physical GPU, asked with the CUDA masking variables stripped."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("CUDA_DEVICE_ORDER", None)
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "nvidia-smi failed")
    table = {}
    for line in out.stdout.splitlines():
        if "," not in line:
            continue
        idx, uuid = line.split(",", 1)
        table[idx.strip()] = uuid.strip()
    return table.get(str(index)), table


SMI_UUID, SMI_TABLE = None, {}
try:
    SMI_UUID, SMI_TABLE = physical_gpu_uuid(PHYSICAL_GPU)
    print("        nvidia-smi physical GPUs: %s" % SMI_TABLE)
    check(SMI_UUID is not None, "nvidia-smi reports a physical GPU %d" % PHYSICAL_GPU)
except Exception as exc:
    note(False, "nvidia-smi GPU table unavailable (%s: %s)" % (type(exc).__name__, exc))

# ---------------------------------------------------------------- 1
print()
print("=== 1. training_input_summary.json records %d/%d and the sample counts ==="
      % (EXPECTED_TRAIN_EPISODES, EXPECTED_VAL_EPISODES))
tis_p = OUT / "training_input_summary.json"
check(tis_p.is_file(), "exists: %s" % tis_p)
tis = load(tis_p) if tis_p.is_file() else {}
tr, va = [], []
if tis:
    tr = [int(x) for x in tis.get("train_episode_indices", [])]
    va = [int(x) for x in tis.get("val_episode_indices", [])]
    check(len(tr) == EXPECTED_TRAIN_EPISODES,
          "train_episode_indices == %d (got %d)" % (EXPECTED_TRAIN_EPISODES, len(tr)))
    check(len(va) == EXPECTED_VAL_EPISODES,
          "val_episode_indices == %d (got %d)" % (EXPECTED_VAL_EPISODES, len(va)))
    shared = sorted(set(tr) & set(va))
    check(not shared, "disjoint (shared: %s)" % (shared or "none"))
    check(sorted(set(tr) | set(va)) == EXPECTED_EPISODES,
          "train + val cover exactly episodes %d..%d (missing: %s, extra: %s)"
          % (EP_FIRST, EP_LAST,
             sorted(set(EXPECTED_EPISODES) - set(tr) - set(va)) or "none",
             sorted((set(tr) | set(va)) - set(EXPECTED_EPISODES)) or "none"))
    print("        train_samples=%s  val_samples=%s"
          % (tis.get("train_samples"), tis.get("val_samples")))
    print("        train_positive_rewards=%s  val_positive_rewards=%s"
          % (tis.get("train_positive_rewards"), tis.get("val_positive_rewards")))
    print("        val episodes: %s" % sorted(va))
    check(isinstance(tis.get("train_samples"), int) and tis["train_samples"] > 0,
          "train_samples > 0 (got %s)" % tis.get("train_samples"))
    check(isinstance(tis.get("val_samples"), int) and tis["val_samples"] > 0,
          "val_samples > 0 (got %s)" % tis.get("val_samples"))

    ta = tis.get("training_args", {})
    print("        --- fixed configuration recorded by the trainer ---")
    for key, want in EXPECTED_TRAINING_ARGS:
        got = ta.get(key, "<absent>")
        print("        arg %s = %r" % (key, got))
        check(same_number(got, want), "training_args.%s == %r (got %r)" % (key, want, got))
    # the trainer must have been pointed at THIS run, not at another task's run
    for key, want_path in (("data_dir", FEATURES), ("split_file", SPLIT_FILE),
                           ("output_dir", OUT)):
        got = ta.get(key)
        ok = False
        if isinstance(got, str):
            try:
                ok = Path(got).resolve() == want_path.resolve()
            except OSError:
                ok = os.path.normpath(got) == os.path.normpath(str(want_path))
        check(ok, "training_args.%s points at this run (%s), got %r" % (key, want_path, got))

# ---------------------------------------------------------------- 2
print()
print("=== 2. split disjoint, both outcomes on each side, validation has positive reward ===")
check(SPLIT_FILE.is_file(), "exists: %s" % SPLIT_FILE)
split = load(SPLIT_FILE) if SPLIT_FILE.is_file() else {}
if split:
    s_tr = [int(x) for x in split.get("train_episode_indices", [])]
    s_va = [int(x) for x in split.get("val_episode_indices", [])]
    print("        strategy: %s   seed: %s" % (split.get("strategy"), split.get("seed")))
    print("        train outcomes: %s" % split.get("train_outcome_counts"))
    print("        val   outcomes: %s" % split.get("val_outcome_counts"))
    print("        whole cohort  : %s" % split.get("outcome_counts"))
    check(sorted(s_tr) == sorted(tr) and sorted(s_va) == sorted(va),
          "the trainer used the audited split file (indices identical)")
    check(not (set(s_tr) & set(s_va)),
          "split train/val disjoint (shared: %s)" % (sorted(set(s_tr) & set(s_va)) or "none"))
    tc = split.get("train_outcome_counts", {}) or {}
    vc = split.get("val_outcome_counts", {}) or {}
    for side, counts in (("train", tc), ("val", vc)):
        for outcome in ("success", "failure"):
            n = int(counts.get(outcome, 0) or 0)
            check(n > 0,
                  "%s side has at least one %s episode (got %d) - a single-outcome side "
                  "cannot support Q checkpoint selection" % (side, outcome, n))

check(int(tis.get("val_positive_rewards", 0) or 0) > 0,
      "val_positive_rewards > 0 (got %s)" % tis.get("val_positive_rewards"))
check(int(tis.get("train_positive_rewards", 0) or 0) > 0,
      "train_positive_rewards > 0 (got %s)" % tis.get("train_positive_rewards"))

# ---------------------------------------------------------------- 3
print()
print("=== 3. exactly one critic member ===")
members = sorted(OUT.glob("critic_member_*.pt"))
check(len(members) == 1,
      "exactly one critic_member_*.pt (got %d: %s)" % (len(members), [m.name for m in members]))
if members:
    check(members[0].name == "critic_member_00.pt",
          "named critic_member_00.pt (got %s)" % members[0].name)
    print("        size %.1f MiB" % (members[0].stat().st_size / 2 ** 20))

# ---------------------------------------------------------------- 4
print()
print("=== 4. loads on CPU and physical GPU %d; arch and shapes correct ===" % PHYSICAL_GPU)
ckpt = None
if members:
    import torch
    from guided_action_flow.critics.checkpoint import load_action_chunk_critic

    check(torch.cuda.is_available(), "torch sees a CUDA device")
    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        check(n_visible == 1,
              "exactly one visible CUDA device, i.e. GPU 0 is masked away (got %d)" % n_visible)
        print("        visible device name: %s" % torch.cuda.get_device_name(0))
        torch_uuid = None
        try:
            torch_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        except Exception as exc:
            note(False, "torch could not report the device UUID (%s: %s)" % (type(exc).__name__, exc))
        if torch_uuid and SMI_UUID:
            check(torch_uuid.lower() == SMI_UUID.lower(),
                  "the visible CUDA device IS physical GPU %d (torch %s vs nvidia-smi %s)"
                  % (PHYSICAL_GPU, torch_uuid, SMI_UUID))
        elif torch_uuid:
            note(False, "no nvidia-smi UUID to cross-check torch device %s against" % torch_uuid)

    for dev in ("cpu", "cuda"):
        try:
            obj = load_action_chunk_critic(str(members[0]), device=dev)
            print("        load_action_chunk_critic(%s) -> %s" % (dev, type(obj).__name__))
            check(True, "load_action_chunk_critic on %s" % dev)
        except Exception as exc:
            check(False, "load_action_chunk_critic on %s: %s: %s" % (dev, type(exc).__name__, exc))

    ckpt = torch.load(str(members[0]), map_location="cpu", weights_only=False)
    check(ckpt.get("critic_arch") == "visual_transformer",
          "critic_arch == visual_transformer (got %r)" % ckpt.get("critic_arch"))
    cfg = ckpt.get("critic_config", {})
    for key, want in EXPECTED_SHAPES:
        check(cfg.get(key) == want, "critic_config.%s == %s (got %s)" % (key, want, cfg.get(key)))
    for key, want in EXPECTED_NET:
        check(same_number(cfg.get(key), want),
              "critic_config.%s == %s (got %s)" % (key, want, cfg.get(key)))
    check(ckpt.get("ensemble_member_index") == 0,
          "ensemble_member_index == 0 (got %s)" % ckpt.get("ensemble_member_index"))

# ---------------------------------------------------------------- 5
print()
print("=== 5. selected epoch == argmin(validation TD loss) ===")
ts_p = OUT / "training_summary.json"
check(ts_p.is_file(), "exists: %s" % ts_p)
ts = load(ts_p) if ts_p.is_file() else {}
if ts:
    mem = ts.get("members", [])
    check(len(mem) == 1, "training_summary lists exactly one member (got %d)" % len(mem))
if ckpt:
    hist = ckpt.get("history", [])
    sel = ckpt.get("selected_epoch")
    sel_loss = ckpt.get("selected_val_td_loss")
    check(len(hist) == EXPECTED_EPOCHS,
          "full %d-epoch history in the checkpoint (got %d)" % (EXPECTED_EPOCHS, len(hist)))
    if sel_loss is not None:
        print("        selected_epoch=%s  selected_val_td_loss=%.6f" % (sel, sel_loss))
    else:
        print("        selected_epoch=%s" % sel)
    if hist:
        lo = min(hist, key=lambda h: h["val_td_loss"])
        print("        argmin val_td_loss: epoch %s = %.6f" % (lo["epoch"], lo["val_td_loss"]))
        check(lo["epoch"] == sel, "selected_epoch %s == argmin epoch %s" % (sel, lo["epoch"]))
        check(sel != hist[-1]["epoch"] or lo["epoch"] == hist[-1]["epoch"],
              "selection is by validation TD loss, not by taking the last epoch")
        if sel_loss is not None:
            check(same_number(sel_loss, lo["val_td_loss"]),
                  "selected_val_td_loss %s == the argmin value %s" % (sel_loss, lo["val_td_loss"]))
        last = hist[-1]
        print("        last epoch %s: val_td_loss=%.6f" % (last["epoch"], last["val_td_loss"]))
        sm = lo.get("val_q_success_mean")
        fm = lo.get("val_q_failure_mean")
        gp = lo.get("val_q_success_failure_gap")
        print("        at selected epoch: val_q_success_mean=%s  val_q_failure_mean=%s  gap=%s"
              % (sm, fm, gp))
        check(sm is not None and fm is not None and gp is not None,
              "val_q_success_mean / val_q_failure_mean / gap are all reported at the selected epoch")
        print("        NOTE: this gap is a Q-value separation on %d held-out episodes."
              % EXPECTED_VAL_EPISODES)
        print("              It is NOT a success rate and must not be reported as one.")

# ---------------------------------------------------------------- 6
print()
print("=== 6. no NaN / Inf in history or weights ===")
if ckpt:
    import torch

    bad_h = ["epoch %s.%s" % (h["epoch"], k) for h in ckpt.get("history", [])
             for k, v in h.items()
             if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
    check(not bad_h, "history finite (bad: %s)" % (bad_h[:5] or "none"))
    for sk in ("model_state_dict", "value_model_state_dict"):
        sd = ckpt.get(sk) or {}
        nb = [k for k, v in sd.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
        n_tensors = sum(1 for v in sd.values() if torch.is_tensor(v))
        check(n_tensors > 0, "%s holds tensors (got %d)" % (sk, n_tensors))
        check(not nb, "%s all finite (%d tensors, bad: %s)" % (sk, n_tensors, nb[:3] or "none"))
    for key in ("selected_val_td_loss",):
        v = ckpt.get(key)
        check(isinstance(v, float) and math.isfinite(v), "%s is finite (got %r)" % (key, v))

# ---------------------------------------------------------------- 7
print()
print("=== 7. reload determinism: same fixed validation batch, same output ===")
if members:
    import torch
    from guided_action_flow.critics.checkpoint import load_action_chunk_critic

    try:
        vidx = sorted(int(x) for x in load(SPLIT_FILE)["val_episode_indices"])
        feat = FEATURES / ("episode_%06d.pt" % vidx[0])
        blob = torch.load(feat, map_location="cpu", weights_only=False)
        s = blob["state"][:8].float()
        a = blob["action_chunk"][:8].float()
        z = blob["visual_features"][:8].float()
        print("        fixed batch from %s: state%s action%s visual%s"
              % (feat.name, list(s.shape), list(a.shape), list(z.shape)))

        outs = []
        for _ in range(2):
            m = load_action_chunk_critic(str(members[0]), device="cpu")
            net = m[0] if isinstance(m, (tuple, list)) else m
            if hasattr(net, "eval"):
                net.eval()
            with torch.no_grad():
                outs.append(net(s, z, a).flatten())
        same = torch.allclose(outs[0], outs[1], atol=0, rtol=0)
        check(same, "two independent reloads give bit-identical Q on the same batch (max diff %.3e)"
              % (outs[0] - outs[1]).abs().max().item())
        check(bool(torch.isfinite(outs[0]).all()), "Q on the fixed validation batch is finite")
        print("        Q on 8 val samples: %s" % [round(float(v), 4) for v in outs[0][:8]])
    except Exception as exc:
        check(False, "reload determinism check could not run: %s: %s" % (type(exc).__name__, exc))

# ---------------------------------------------------------------- summary
print()
if warn:
    print("%d warning(s):" % len(warn))
    for w in warn:
        print("  -", w)
if fail:
    print("\nFAILED %d check(s):" % len(fail))
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("\nALL HARD CHECKS PASSED  (task=%s run=%s)" % (TASK_KEY, RUN_ID))
