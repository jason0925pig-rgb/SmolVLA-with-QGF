"""Handoff section 12 acceptance for the mug single-Q critic.

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

Usage: accept_single_q.py <run_dir>
"""
import io
import json
import math
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
OUT = RUN / "outputs/single_qcritic"
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


# ---------------------------------------------------------------- 1
print("=== 1. training_input_summary.json records 45/5 and the sample counts ===")
tis_p = OUT / "training_input_summary.json"
check(tis_p.is_file(), f"exists: {tis_p}")
tis = load(tis_p) if tis_p.is_file() else {}
if tis:
    tr = tis.get("train_episode_indices", [])
    va = tis.get("val_episode_indices", [])
    check(len(tr) == 45, f"train_episode_indices == 45 (got {len(tr)})")
    check(len(va) == 5, f"val_episode_indices == 5 (got {len(va)})")
    check(not (set(tr) & set(va)), f"disjoint (shared: {sorted(set(tr) & set(va)) or 'none'})")
    print(f"        train_samples={tis.get('train_samples')}  val_samples={tis.get('val_samples')}")
    print(f"        train_positive_rewards={tis.get('train_positive_rewards')}"
          f"  val_positive_rewards={tis.get('val_positive_rewards')}")
    print(f"        val episodes: {sorted(va)}")
    ta = tis.get("training_args", {})
    for k in ("ensemble_size", "epochs", "batch_size", "lr", "weight_decay", "gamma",
              "expectile", "polyak", "d_model", "layers", "heads", "dropout", "seed",
              "expected_train_episodes", "expected_val_episodes"):
        if k in ta:
            print(f"        arg {k} = {ta[k]}")

# ---------------------------------------------------------------- 2
print()
print("=== 2. validation contains positive-reward chunks ===")
check(tis.get("val_positive_rewards", 0) > 0,
      f"val_positive_rewards > 0 (got {tis.get('val_positive_rewards')})")
check(tis.get("train_positive_rewards", 0) > 0,
      f"train_positive_rewards > 0 (got {tis.get('train_positive_rewards')})")

# ---------------------------------------------------------------- 3
print()
print("=== 3. exactly one critic member ===")
members = sorted(OUT.glob("critic_member_*.pt"))
check(len(members) == 1, f"exactly one critic_member_*.pt (got {len(members)}: {[m.name for m in members]})")
if members:
    check(members[0].name == "critic_member_00.pt", f"named critic_member_00.pt (got {members[0].name})")
    print(f"        size {members[0].stat().st_size / 2**20:.1f} MiB")

# ---------------------------------------------------------------- 4
print()
print("=== 4. loads on CPU and GPU 0; arch and shapes correct ===")
ckpt = None
if members:
    import torch
    from guided_action_flow.critics.checkpoint import load_action_chunk_critic

    for dev in ("cpu", "cuda"):
        try:
            obj = load_action_chunk_critic(str(members[0]), device=dev)
            print(f"        load_action_chunk_critic({dev}) -> {type(obj).__name__}")
        except Exception as exc:
            check(False, f"load_action_chunk_critic on {dev}: {type(exc).__name__}: {exc}")

    ckpt = torch.load(str(members[0]), map_location="cpu", weights_only=False)
    check(ckpt.get("critic_arch") == "visual_transformer",
          f"critic_arch == visual_transformer (got {ckpt.get('critic_arch')!r})")
    cfg = ckpt.get("critic_config", {})
    for key, want in (("state_dim", 8), ("action_dim", 8), ("action_horizon", 50),
                      ("visual_tokens", 128), ("visual_token_dim", 960)):
        check(cfg.get(key) == want, f"critic_config.{key} == {want} (got {cfg.get(key)})")
    for key, want in (("d_model", 256), ("num_layers", 3), ("num_heads", 4), ("dropout", 0.1)):
        check(cfg.get(key) == want, f"critic_config.{key} == {want} (got {cfg.get(key)})")
    check(ckpt.get("ensemble_member_index") == 0,
          f"ensemble_member_index == 0 (got {ckpt.get('ensemble_member_index')})")

# ---------------------------------------------------------------- 5
print()
print("=== 5. selected epoch == argmin(validation TD loss) ===")
ts_p = OUT / "training_summary.json"
check(ts_p.is_file(), f"exists: {ts_p}")
ts = load(ts_p) if ts_p.is_file() else {}
if ts:
    mem = ts.get("members", [])
    check(len(mem) == 1, f"training_summary lists exactly one member (got {len(mem)})")
if ckpt:
    hist = ckpt.get("history", [])
    sel = ckpt.get("selected_epoch")
    sel_loss = ckpt.get("selected_val_td_loss")
    check(len(hist) == 80, f"full 80-epoch history in the checkpoint (got {len(hist)})")
    print(f"        selected_epoch={sel}  selected_val_td_loss={sel_loss:.6f}"
          if sel_loss is not None else f"        selected_epoch={sel}")
    if hist:
        lo = min(hist, key=lambda h: h["val_td_loss"])
        print(f"        argmin val_td_loss: epoch {lo['epoch']} = {lo['val_td_loss']:.6f}")
        check(lo["epoch"] == sel, f"selected_epoch {sel} == argmin epoch {lo['epoch']}")
        check(sel != hist[-1]["epoch"] or lo["epoch"] == hist[-1]["epoch"],
              "selection is by validation TD loss, not by taking the last epoch")
        last = hist[-1]
        print(f"        last epoch {last['epoch']}: val_td_loss={last['val_td_loss']:.6f}")
        sm = lo.get("val_q_success_mean")
        fm = lo.get("val_q_failure_mean")
        gp = lo.get("val_q_success_failure_gap")
        print(f"        at selected epoch: val_q_success_mean={sm}  val_q_failure_mean={fm}  gap={gp}")
        print("        NOTE: this gap is a Q-value separation on 5 held-out episodes.")
        print("              It is NOT a success rate and must not be reported as one.")

# ---------------------------------------------------------------- 6
print()
print("=== 6. no NaN / Inf in history or weights ===")
if ckpt:
    import torch

    bad_h = [f"epoch {h['epoch']}.{k}" for h in ckpt.get("history", [])
             for k, v in h.items()
             if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
    check(not bad_h, f"history finite (bad: {bad_h[:5] or 'none'})")
    for sk in ("model_state_dict", "value_model_state_dict"):
        sd = ckpt.get(sk) or {}
        nb = [k for k, v in sd.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
        check(not nb, f"{sk} all finite ({sum(1 for v in sd.values() if torch.is_tensor(v))} tensors, bad: {nb[:3] or 'none'})")

# ---------------------------------------------------------------- 7
print()
print("=== 7. reload determinism: same fixed validation batch, same output ===")
if members:
    import torch
    from guided_action_flow.critics.checkpoint import load_action_chunk_critic

    try:
        split = load(RUN / "manifest/episode_split_45_5.json")
        vidx = [int(x) for x in split["val_episode_indices"]]
        blob = torch.load(RUN / f"features/episode_{vidx[0]:06d}.pt", map_location="cpu", weights_only=False)
        s = blob["state"][:8].float()
        a = blob["action_chunk"][:8].float()
        z = blob["visual_features"][:8].float()

        outs = []
        for _ in range(2):
            m = load_action_chunk_critic(str(members[0]), device="cpu")
            net = m[0] if isinstance(m, (tuple, list)) else m
            if hasattr(net, "eval"):
                net.eval()
            with torch.no_grad():
                outs.append(net(s, z, a).flatten())
        same = torch.allclose(outs[0], outs[1], atol=0, rtol=0)
        check(same, f"two independent reloads give bit-identical Q on the same batch (max diff {(outs[0] - outs[1]).abs().max().item():.3e})")
        print(f"        Q on 8 val samples: {[round(float(v), 4) for v in outs[0][:8]]}")
    except Exception as exc:
        note(False, f"reload determinism check could not run: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- summary
print()
if warn:
    print(f"{len(warn)} warning(s):")
    for w in warn:
        print("  -", w)
if fail:
    print(f"\nFAILED {len(fail)} check(s):")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("\nALL HARD CHECKS PASSED")
