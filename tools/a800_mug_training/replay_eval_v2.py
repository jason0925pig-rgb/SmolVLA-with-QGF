#!/usr/bin/env python3
"""15 Hz gripper-timing replay validation (handoff section 5), v2 - auditable.

For each candidate checkpoint, replays the 5 validation episodes at 15 Hz
mimicking async deployment: re-plan every REPLAN ticks, execute the chunk's
corresponding action between plans. The executed gripper stream goes through
the deployment filter (close>=0.85 / open<=0.40, 5 consecutive 15 Hz ticks);
contact-hold/dwell cannot be simulated offline and is explicitly noted.
GT events use the same filter on the demo gripper action channel.

Selection (lexicographic, both tasks identical):
  1. missed_open_rate + early_open_rate  (lower)
  2. filtered open/close F1              (higher)
  3. |close_err| + |open_err| mean       (lower; None = tie, skipped)
  4. executed-action 7-joint MAE         (lower)
  5. validation flow loss (from light report) (lower)

Usage: replay_eval_v2.py <run_dir> <dataset_subdir> <task_text> <gpu> <ck1,ck2,...>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RUN = Path(sys.argv[1])
DS_ROOT = RUN / sys.argv[2]
TASK = sys.argv[3]
CKPTS = sys.argv[5].split(",")
REPLAN = 25
CLOSE_T, OPEN_T, CONFIRM = 0.85, 0.40, 5

split = json.loads((RUN / "split_manifest.json").read_text())
VAL = split["validation_new_indices"]

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

ds = LeRobotDataset("local/replay", root=str(DS_ROOT))
meta_eps = ds.meta.episodes


def episode_range(ep):
    row = meta_eps[ep]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def filter_events(g15):
    """Deployment-style hysteresis on a 15Hz gripper stream.
    Returns (state_seq, t_close_idx, t_open_idx, flips)."""
    state = 0  # 0=open, 1=closed (start open)
    run_c = run_o = 0
    states, t_close, t_open, flips = [], None, None, 0
    for i, g in enumerate(g15):
        if g >= CLOSE_T:
            run_c += 1
        else:
            run_c = 0
        if g <= OPEN_T:
            run_o += 1
        else:
            run_o = 0
        if state == 0 and run_c >= CONFIRM:
            state = 1
            flips += 1
            if t_close is None:
                t_close = i
        elif state == 1 and run_o >= CONFIRM:
            state = 0
            flips += 1
            if t_close is not None and t_open is None:
                t_open = i
        states.append(state)
    return np.array(states), t_close, t_open, flips


def load_policy(ck_dir):
    cfg = PreTrainedConfig.from_pretrained(ck_dir)
    cfg.pretrained_path = ck_dir
    policy = get_policy_class(cfg.type).from_pretrained(ck_dir, config=cfg)
    policy.eval().cuda()
    pre, post = make_pre_post_processors(cfg, pretrained_path=ck_dir,
                                         dataset_stats=None)
    return policy, pre, post


out_dir = RUN / "validation_reports_v2"
out_dir.mkdir(exist_ok=True)

for ck in CKPTS:
    ck_dir = None
    for cand in (RUN / "outputs").glob(f"*/checkpoints/{ck}/pretrained_model"):
        ck_dir = cand
    if ck_dir is None:
        print(f"{ck}: checkpoint dir missing, skip", flush=True)
        continue
    policy, pre, post = load_policy(str(ck_dir))
    rows = []
    for ep in VAL:
        lo, hi = episode_range(ep)
        # GT gripper on the 15Hz timeline from the data parquet action channel
        dfile = DS_ROOT / f"data/chunk-000/file-{ep:03d}.parquet"
        d = pd.read_parquet(dfile, columns=["action"])
        gt_act = np.stack(d["action"].to_numpy())            # 30 Hz
        gt15 = gt_act[::2]                                    # 15 Hz
        n15 = len(gt15)
        gt_states, gt_c, gt_o, _ = filter_events(gt15[:, 7])

        exec_act = np.zeros((n15, 8), dtype=np.float64)
        chunk = None
        chunk_at = -10**9
        with torch.no_grad():
            for t in range(n15):
                if chunk is None or (t - chunk_at) >= REPLAN or (t - chunk_at) >= chunk.shape[0]:
                    gidx = lo + t * 2
                    item = ds[gidx]
                    obs = {
                        "observation.state": item["observation.state"].unsqueeze(0).cuda(),
                        "observation.images.chest": item["observation.images.chest"].unsqueeze(0).cuda(),
                        "observation.images.wrist_right": item["observation.images.wrist_right"].unsqueeze(0).cuda(),
                        "task": TASK,
                    }
                    batch = pre(obs)
                    acts = policy.predict_action_chunk(batch)
                    acts = post(acts)
                    chunk = acts[0].detach().float().cpu().numpy()
                    chunk_at = t
                exec_act[t] = chunk[t - chunk_at]
        pr_states, pr_c, pr_o, flips = filter_events(exec_act[:, 7])

        # timing errors (seconds on the 15 Hz axis)
        close_err = abs(pr_c - gt_c) / 15.0 if (gt_c is not None and pr_c is not None) else None
        open_err = abs(pr_o - gt_o) / 15.0 if (gt_o is not None and pr_o is not None) else None
        missed_open = 1 if (gt_o is not None and pr_o is None) else 0
        early_open = 0
        if gt_o is not None and pr_o is not None and pr_o < gt_c:
            early_open = 1     # released before GT even closed = clearly early
        # filtered state F1 (closed = positive class)
        tp = int(((gt_states == 1) & (pr_states == 1)).sum())
        fp = int(((gt_states == 0) & (pr_states == 1)).sum())
        fn = int(((gt_states == 1) & (pr_states == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
        jmae = float(np.abs(exec_act[:, :7] - gt15[:, :7]).mean())
        rows.append({"ep": ep, "gt_close": gt_c, "gt_open": gt_o,
                     "pr_close": pr_c, "pr_open": pr_o,
                     "close_err_s": close_err, "open_err_s": open_err,
                     "missed_open": missed_open, "early_open": early_open,
                     "flips": flips, "f1": round(f1, 4), "jmae": round(jmae, 5),
                     "gt_open_simulable": gt_o is not None})

    def mean_of(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    rep = {"checkpoint": ck, "task": TASK, "replan_ticks": REPLAN,
           "note": "contact-hold/dwell not simulable offline; async replay re-plans every 25 ticks",
           "val_episodes": VAL, "per_episode": rows,
           "missed_open_rate": mean_of("missed_open"),
           "early_open_rate": mean_of("early_open"),
           "close_err_mean_s": mean_of("close_err_s"),
           "open_err_mean_s": mean_of("open_err_s"),
           "f1_mean": mean_of("f1"), "jmae_mean": mean_of("jmae"),
           "flips_mean": mean_of("flips"),
           "gt_open_simulable_all": all(r["gt_open_simulable"] for r in rows)}
    (out_dir / f"replay_{ck}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"{ck}: missed+early={rep['missed_open_rate']}+{rep['early_open_rate']} "
          f"F1={rep['f1_mean']} closeerr={rep['close_err_mean_s']} openerr={rep['open_err_mean_s']} "
          f"jmae={rep['jmae_mean']} flips={rep['flips_mean']}", flush=True)

print("REPLAY_V2_DONE")
