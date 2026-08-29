#!/usr/bin/env python3
"""Sequence evaluation per handoff section 4.3 (2026-08-28 revision).

Replays each held-out validation episode on the 15 Hz action axis the way the
Orin actually runs: re-plan every REPLAN ticks and execute the chunk's
successive actions in between (open-loop within a chunk), instead of querying
the policy with fresh ground truth at every tick. Reports both the raw policy
gripper stream and the filtered command stream that the robot would receive.

Metrics (all on the 5 validation episodes):
  - 7-joint action MAE and max error
  - first full action chunk: per-joint min/max, max |adjacent diff|,
    2*pi jump detection, out-of-training-range detection
  - filtered gripper open/close F1
  - first-close and first-open timing error
  - early-open / missed-open rate, state flips

Usage:
  seq_eval.py --run-dir RUN --ds-name DS --repo-id ID --train-tag TAG
              --ckpt STEP [--ckpt STEP ...] [--replan 25]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OPEN_T, CLOSE_T, CONFIRM = 0.15, 0.85, 5
TWO_PI = 2.0 * np.pi


def filtered_stream(g):
    """Deployment gripper state machine. Returns (states, t_close_i, t_open_i, flips)."""
    state, states = 0, []
    run_c = run_o = 0
    tc = to = None
    flips = 0
    for i, v in enumerate(g):
        run_c = run_c + 1 if v >= CLOSE_T else 0
        run_o = run_o + 1 if v <= OPEN_T else 0
        if state == 0 and run_c >= CONFIRM:
            state = 1; flips += 1
            if tc is None:
                tc = i
        elif state == 1 and run_o >= CONFIRM:
            state = 0; flips += 1
            if tc is not None and to is None:
                to = i
        states.append(state)
    return np.array(states), tc, to, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ds-name", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--train-tag", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--replan", type=int, default=25)
    args = ap.parse_args()

    run = Path(args.run_dir).expanduser()
    root = run / args.ds_name
    out = run / "validation_reports"
    out.mkdir(exist_ok=True)
    val = json.loads((run / "split_manifest.json").read_text())["validation_new_indices"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from lerobot.policies import SmolVLAConfig  # noqa: F401 registers 'smolvla'
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ck0 = run / f"outputs/train_{args.train_tag}/checkpoints/{args.ckpt[0]}/pretrained_model"
    pcfg = PreTrainedConfig.from_pretrained(str(ck0))
    meta = LeRobotDatasetMetadata(args.repo_id, root=str(root))
    dts = resolve_delta_timestamps(pcfg, meta)
    ds = LeRobotDataset(args.repo_id, root=str(root), episodes=val, delta_timestamps=dts)
    print("val episodes:", val, "frames:", ds.num_frames, flush=True)

    # training-data joint range (from the dataset stats) for the sanity check
    st = json.loads((root / "meta/stats.json").read_text())
    lo = np.asarray(st["action"]["min"], dtype=np.float64)[:7]
    hi = np.asarray(st["action"]["max"], dtype=np.float64)[:7]

    ep_frames = defaultdict(list)
    for i in range(len(ds)):
        ep_frames[int(ds[i]["episode_index"])].append(i)

    for step in args.ckpt:
        ck = run / f"outputs/train_{args.train_tag}/checkpoints/{step}/pretrained_model"
        if not ck.is_dir():
            print(f"[{step}] MISSING"); continue
        policy = SmolVLAPolicy.from_pretrained(str(ck)).to(device).eval()
        pre, post = make_pre_post_processors(
            policy.config, pretrained_path=str(ck), dataset_stats=ds.meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}})

        per_ep, first_chunks = [], []
        agg = {"mae": [], "maxerr": [], "ce": [], "oe": [], "early": 0,
               "missed": 0, "flips": [], "tp": 0, "fp": 0, "fn": 0}

        for ep, fids in sorted(ep_frames.items()):
            fids = fids[::2]                      # 30 fps -> 15 Hz axis
            chunk = None; chunk_at = -10**9
            pred_g, gt_g, errs = [], [], []
            for t, fi in enumerate(fids):
                item = ds[fi]
                if chunk is None or (t - chunk_at) >= args.replan or (t - chunk_at) >= chunk.shape[0]:
                    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                             for k, v in item.items()}
                    batch["task"] = [item["task"]]
                    with torch.no_grad():
                        policy.reset()
                        raw = policy.predict_action_chunk(pre(batch))
                        chunk = post(raw).cpu().numpy()[0]
                    chunk_at = t
                    if t == 0:
                        first_chunks.append((ep, chunk.copy()))
                act = chunk[t - chunk_at]
                gt = item["action"].numpy()
                gt = gt[0] if gt.ndim == 2 else gt
                errs.append(np.abs(act[:7] - gt[:7]))
                pred_g.append(float(act[7]))
                gt_g.append(float(gt[7]))

            errs = np.stack(errs)
            ps, ptc, pto, pfl = filtered_stream(np.array(pred_g))
            gs, gtc, gto, _ = filtered_stream(np.array(gt_g))
            dt = 1 / 15.0
            ce = (ptc - gtc) * dt if (ptc is not None and gtc is not None) else None
            oe = (pto - gto) * dt if (pto is not None and gto is not None) else None
            early = int(pto is not None and gto is not None and (pto - gto) * dt < -1.0)
            missed = int(gto is not None and pto is None)
            agg["tp"] += int(((ps == 1) & (gs == 1)).sum())
            agg["fp"] += int(((ps == 1) & (gs == 0)).sum())
            agg["fn"] += int(((ps == 0) & (gs == 1)).sum())
            agg["mae"].append(float(errs.mean())); agg["maxerr"].append(float(errs.max()))
            if ce is not None: agg["ce"].append(abs(ce))
            if oe is not None: agg["oe"].append(abs(oe))
            agg["early"] += early; agg["missed"] += missed; agg["flips"].append(pfl)
            per_ep.append({"ep": ep, "joint_mae": float(errs.mean()),
                           "joint_max_err": float(errs.max()),
                           "close_err_s": ce, "open_err_s": oe,
                           "early_open": early, "missed_open": missed, "flips": pfl})
            print(f"  [{step}] ep{ep}: mae={errs.mean():.4f} max={errs.max():.4f} "
                  f"close={ce} open={oe} flips={pfl}", flush=True)

        # first-chunk diagnostics (2*pi jumps, range, adjacent diffs)
        fc = []
        for ep, c in first_chunks:
            j = c[:, :7]
            diff = np.abs(np.diff(j, axis=0))
            below = (j < lo - 0.5).any(0); above = (j > hi + 0.5).any(0)
            twopi = bool((diff > TWO_PI - 1.0).any())
            fc.append({"ep": ep,
                       "min": j.min(0).tolist(), "max": j.max(0).tolist(),
                       "max_adjacent_diff": float(diff.max()),
                       "max_adjacent_diff_per_joint": diff.max(0).tolist(),
                       "twopi_jump": twopi,
                       "joints_below_train_range": np.where(below)[0].tolist(),
                       "joints_above_train_range": np.where(above)[0].tolist()})
        prec = agg["tp"] / max(1, agg["tp"] + agg["fp"])
        rec = agg["tp"] / max(1, agg["tp"] + agg["fn"])
        n = len(per_ep)
        report = {
            "checkpoint": step, "replan_ticks": args.replan, "val_episodes": val,
            "joint_mae": float(np.mean(agg["mae"])),
            "joint_max_err": float(np.max(agg["maxerr"])),
            "close_err_mean_s": float(np.mean(agg["ce"])) if agg["ce"] else None,
            "open_err_mean_s": float(np.mean(agg["oe"])) if agg["oe"] else None,
            "early_open_rate": agg["early"] / n, "missed_open_rate": agg["missed"] / n,
            "gripper_f1": float(2 * prec * rec / max(1e-9, prec + rec)),
            "flips_mean": float(np.mean(agg["flips"])),
            "first_chunk": fc,
            "offline_pass": bool(not any(f["twopi_jump"] for f in fc)
                                 and not any(f["joints_below_train_range"] or
                                             f["joints_above_train_range"] for f in fc)),
            "contact_hold": "not simulatable offline (no contact signal)",
            "per_episode": per_ep,
        }
        (out / f"seq_{step}.json").write_text(json.dumps(report, indent=2))
        print(f"[{step}] joint_mae={report['joint_mae']:.4f} F1={report['gripper_f1']:.3f} "
              f"close={report['close_err_mean_s']} open={report['open_err_mean_s']} "
              f"pass={report['offline_pass']}", flush=True)
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
