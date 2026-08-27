#!/usr/bin/env python3
"""Offline validation for the 2026-08-27 handoff, run on the 5 held-out
validation episodes only.

Modes:
  light  - validation flow loss + 7D joint MAE + raw gripper MAE (fast)
  replay - 15 Hz closed-loop-style replay: at every 15 Hz tick, query the
           checkpoint with the real observation, take the first action of the
           chunk, pass its gripper through the deployment-style filter
           (open<=0.40 / close>=0.85, 5-tick confirmation; contact-hold not
           simulatable offline and reported as such), and compare gripper
           EVENTS against ground truth processed by the same rule.

Usage:
  offline_eval.py --run-dir RUN --ds-name DSDIR --repo-id ID \
      --ckpt STEP [--ckpt STEP ...] --mode light|replay
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

OPEN_T, CLOSE_T, CONFIRM = 0.40, 0.85, 5
VAL_EPS = [9, 24, 29, 40, 49]


def load_policy(ckpt_dir, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(ckpt_dir)
    policy.to(device).eval()
    return policy


def make_processors(ckpt_dir, ds_meta, policy):
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=ckpt_dir,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(next(policy.parameters()).device)}},
    )
    return pre, post


def filtered_events(g, open_t=OPEN_T, close_t=CLOSE_T, confirm=CONFIRM):
    """Return (state_seq, t_close_idx, t_open_idx, flips). state 0=open 1=closed.
    Confirmation completes at the index of the confirm-th consecutive tick."""
    state, states = 0, []
    run_c = run_o = 0
    t_close = t_open = None
    flips = 0
    for i, v in enumerate(g):
        run_c = run_c + 1 if v >= close_t else 0
        run_o = run_o + 1 if v <= open_t else 0
        if state == 0 and run_c >= confirm:
            state = 1
            flips += 1
            if t_close is None:
                t_close = i
        elif state == 1 and run_o >= confirm:
            state = 0
            flips += 1
            if t_close is not None and t_open is None:
                t_open = i
        states.append(state)
    return np.array(states), t_close, t_open, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ds-name", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--train-tag", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--mode", choices=["light", "replay"], required=True)
    ap.add_argument("--light-samples", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run_dir).expanduser()
    root = run / args.ds_name
    out_dir = run / "validation_reports"
    out_dir.mkdir(exist_ok=True)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # first checkpoint decides the delta_timestamps (same config for all steps)
    from lerobot.policies import SmolVLAConfig  # noqa: F401 - registers 'smolvla'
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    ck0 = run / f"outputs/train_{args.train_tag}/checkpoints/{args.ckpt[0]}/pretrained_model"
    pcfg = PreTrainedConfig.from_pretrained(str(ck0))
    meta = LeRobotDatasetMetadata(args.repo_id, root=str(root))
    dts = resolve_delta_timestamps(pcfg, meta)
    ds = LeRobotDataset(args.repo_id, root=str(root), episodes=VAL_EPS,
                        delta_timestamps=dts)
    print("val frames:", ds.num_frames, "episodes:", ds.num_episodes,
          "| delta keys:", list(dts.keys()) if dts else None)

    for step in args.ckpt:
        ck = run / f"outputs/train_{args.train_tag}/checkpoints/{step}/pretrained_model"
        if not ck.is_dir():
            print(f"[{step}] MISSING {ck}")
            continue
        policy = load_policy(str(ck), device)
        pre, post = make_processors(str(ck), ds.meta, policy)
        report = {"checkpoint": step, "mode": args.mode, "val_episodes": VAL_EPS}

        if args.mode == "light":
            rng = np.random.default_rng(1000)
            idxs = rng.choice(len(ds), size=min(args.light_samples, len(ds)), replace=False)
            losses, jmae, gmae = [], [], []
            for i in idxs:
                item = ds[int(i)]
                batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                         for k, v in item.items()}
                batch["task"] = [item["task"]]
                proc = pre(batch)
                with torch.no_grad():
                    loss, _ = policy.forward(proc)
                    losses.append(float(loss.item()))
                    policy.reset()
                    chunk = policy.predict_action_chunk(proc)
                    act = post(chunk[:, 0]).cpu().numpy()[0]
                gt = item["action"].numpy()
                if gt.ndim == 2:
                    gt = gt[0]
                jmae.append(np.abs(act[:7] - gt[:7]).mean())
                gmae.append(abs(act[7] - gt[7]))
            report.update(val_flow_loss=float(np.mean(losses)),
                          joint_mae_rad=float(np.mean(jmae)),
                          gripper_raw_mae=float(np.mean(gmae)),
                          n_samples=len(idxs))
        else:  # replay at 15 Hz
            per_ep, agg = [], {"close_err": [], "open_err": [], "early": 0,
                               "missed": 0, "flips": [], "f1_tp": 0, "f1_fp": 0,
                               "f1_fn": 0, "joint_chunk_mae": []}
            from collections import defaultdict
            ep_frames = defaultdict(list)
            for i in range(len(ds)):
                ep_frames[int(ds[i]["episode_index"])].append(i)
            for ep, frame_ids in sorted(ep_frames.items()):
                frame_ids = frame_ids[::2]  # 30 fps -> 15 Hz
                pred_g, gt_g, cmae = [], [], []
                for fi in frame_ids:
                    item = ds[fi]
                    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                             for k, v in item.items()}
                    batch["task"] = [item["task"]]
                    proc = pre(batch)
                    with torch.no_grad():
                        policy.reset()
                        chunk = policy.predict_action_chunk(proc)
                        acts = post(chunk).cpu().numpy()[0]
                    pred_g.append(float(acts[0, 7]))
                    _g = item["action"].numpy()
                    gt_g.append(float((_g[0] if _g.ndim == 2 else _g)[7]))
                    horizon = min(len(acts), (len(frame_ids) - len(pred_g) + 1))
                    _a = item["action"].numpy()
                    _a = _a[0] if _a.ndim == 2 else _a
                    cmae.append(float(np.abs(acts[0, :7] - _a[:7]).mean()))
                ps, ptc, pto, pfl = filtered_events(np.array(pred_g))
                gs, gtc, gto, _ = filtered_events(np.array(gt_g))
                dt = 1 / 15.0
                ce = (ptc - gtc) * dt if (ptc is not None and gtc is not None) else None
                oe = (pto - gto) * dt if (pto is not None and gto is not None) else None
                early = int(pto is not None and gto is not None and (pto - gto) * dt < -1.0)
                missed = int(gto is not None and pto is None)
                tp = int(((ps == 1) & (gs == 1)).sum()); fp = int(((ps == 1) & (gs == 0)).sum())
                fn = int(((ps == 0) & (gs == 1)).sum())
                per_ep.append({"ep": ep, "close_err_s": ce, "open_err_s": oe,
                               "early_open": early, "missed_open": missed,
                               "flips_pred": pfl, "chunk_first_mae": float(np.mean(cmae))})
                if ce is not None: agg["close_err"].append(abs(ce))
                if oe is not None: agg["open_err"].append(abs(oe))
                agg["early"] += early; agg["missed"] += missed; agg["flips"].append(pfl)
                agg["f1_tp"] += tp; agg["f1_fp"] += fp; agg["f1_fn"] += fn
                agg["joint_chunk_mae"].append(float(np.mean(cmae)))
                print(f"  [{step}] ep{ep}: close_err={ce} open_err={oe} flips={pfl}", flush=True)
            prec = agg["f1_tp"] / max(1, agg["f1_tp"] + agg["f1_fp"])
            rec = agg["f1_tp"] / max(1, agg["f1_tp"] + agg["f1_fn"])
            report.update(per_episode=per_ep,
                          close_err_mean_s=float(np.mean(agg["close_err"])) if agg["close_err"] else None,
                          open_err_mean_s=float(np.mean(agg["open_err"])) if agg["open_err"] else None,
                          early_open_rate=agg["early"] / 5.0, missed_open_rate=agg["missed"] / 5.0,
                          closed_f1=float(2 * prec * rec / max(1e-9, prec + rec)),
                          flips_mean=float(np.mean(agg["flips"])),
                          joint_chunk_mae=float(np.mean(agg["joint_chunk_mae"])),
                          contact_hold="not simulatable offline (no contact signal)")

        out = out_dir / f"{args.mode}_{step}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"[{step}] {args.mode} done ->", out, flush=True)
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
