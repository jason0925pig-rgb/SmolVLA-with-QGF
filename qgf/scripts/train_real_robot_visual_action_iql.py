#!/usr/bin/env python3
"""Train the no-state Q(visual-tokens, action-chunk) real-robot IQL ablation.

The input feature cache and episode-level 90/10 split are identical to the
full visual Q critic.  Proprioceptive state and next-state tensors are loaded
only to verify cache alignment; they are never passed into either model.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ensemble-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--polyak", type=float, default=0.005)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expected-train-episodes", type=int, default=90)
    p.add_argument("--expected-val-episodes", type=int, default=10)
    return p.parse_args()


def clone_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def expectile_loss(advantage, expectile):
    import torch
    return (torch.where(advantage > 0, expectile, 1.0 - expectile) * advantage.square()).mean()


def load_episode(data_dir, index):
    import torch
    path = data_dir / f"episode_{index:06d}.pt"
    item = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state", "action_chunk", "next_state", "reward", "success", "done", "terminated", "truncated", "visual_features", "next_visual_features", "next_visual_valid"}
    missing = required - set(item)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    n = int(item["action_chunk"].shape[0])
    if tuple(item["action_chunk"].shape[1:]) != (50, 8) or tuple(item["visual_features"].shape[1:]) != (128, 960):
        raise ValueError(f"{path}: incompatible action/visual shape")
    if any(int(item[k].shape[0]) != n for k in required if hasattr(item[k], "shape")):
        raise ValueError(f"{path}: inconsistent row count")
    return item


def join(data_dir, ids):
    import torch
    fields = ("action_chunk", "reward", "success", "done", "terminated", "truncated", "visual_features", "next_visual_features")
    cols = {k: [] for k in fields}
    for index in ids:
        item = load_episode(data_dir, index)
        for key in fields:
            cols[key].append(item[key])
    out = {k: torch.cat(v, dim=0) for k, v in cols.items()}
    out["done"] = out["done"].bool() | out["terminated"].bool() | out["truncated"].bool() | out["success"].bool()
    out["reward"] = torch.maximum(out["reward"].float(), out["success"].float())
    return out


class Rows:
    def __init__(self, data): self.data = data
    def __len__(self): return int(self.data["action_chunk"].shape[0])
    def __getitem__(self, i):
        return tuple(self.data[k][i] for k in ("visual_features", "action_chunk", "next_visual_features", "reward", "done"))


def evaluate(q, v, data, a, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    q.eval(); v.eval(); losses=[]; scores=[]; positive=[]
    with torch.inference_mode():
        for visual, action, next_visual, reward, done in DataLoader(Rows(data), batch_size=a.batch_size):
            visual, action, next_visual = visual.to(device), action.to(device), next_visual.to(device)
            reward, done = reward.to(device), done.to(device).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = q(visual, action)
                target = reward + a.gamma * (1.0 - done) * v.forward_value(next_visual)
                losses.append(float(F.mse_loss(pred.float(), target.float()).cpu()))
            scores.append(pred.float().cpu()); positive.append((reward > 0).cpu())
    scores, positive = torch.cat(scores), torch.cat(positive)
    good, bad = scores[positive], scores[~positive]
    return {"td_loss": float(sum(losses)/len(losses)), "q_mean": float(scores.mean()), "q_success_mean": float(good.mean()) if good.numel() else None, "q_failure_mean": float(bad.mean()) if bad.numel() else None, "q_success_failure_gap": float(good.mean()-bad.mean()) if good.numel() and bad.numel() else None, "positive_reward_samples": int(positive.sum()), "samples": int(scores.numel())}


def train_member(member, a, train_data, val_data, cfg, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from guided_action_flow.critics.visual_action_transformer_critic import VisualActionTransformerCritic
    seed=a.seed+member*1009; torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    q=VisualActionTransformerCritic(cfg).module.to(device); target=VisualActionTransformerCritic(cfg).module.to(device); target.load_state_dict(copy.deepcopy(q.state_dict()))
    for p in target.parameters(): p.requires_grad_(False)
    v=VisualActionTransformerCritic(cfg).module.to(device)
    oq=torch.optim.AdamW(q.parameters(),lr=a.lr,weight_decay=a.weight_decay); ov=torch.optim.AdamW(v.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    loader=DataLoader(Rows(train_data),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best={"val_td_loss":float("inf"),"epoch":None,"q_state_dict":None,"value_state_dict":None}; history=[]
    for epoch in range(1,a.epochs+1):
        q.train(); v.train(); qlosses=[]; vlosses=[]
        for visual, action, next_visual, reward, done in loader:
            visual, action, next_visual=visual.to(device),action.to(device),next_visual.to(device); reward,done=reward.to(device),done.to(device).float()
            with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): target_q=target(visual,action)
            value=v.forward_value(visual); vl=expectile_loss(target_q.float()-value.float(),a.expectile); ov.zero_grad(set_to_none=True); vl.backward(); torch.nn.utils.clip_grad_norm_(v.parameters(),10.0); ov.step()
            with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): td=reward+a.gamma*(1.0-done)*v.forward_value(next_visual)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): pred=q(visual,action); ql=F.mse_loss(pred.float(),td.float())
            oq.zero_grad(set_to_none=True); ql.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(),10.0); oq.step()
            with torch.no_grad():
                for tp,p in zip(target.parameters(),q.parameters()): tp.mul_(1.0-a.polyak).add_(p,alpha=a.polyak)
            qlosses.append(float(ql.detach().cpu())); vlosses.append(float(vl.detach().cpu()))
        metrics=evaluate(q,v,val_data,a,device); row={"epoch":epoch,"train_q_loss":float(sum(qlosses)/len(qlosses)),"train_v_loss":float(sum(vlosses)/len(vlosses)),**{f"val_{k}":x for k,x in metrics.items()}}; history.append(row); print(json.dumps({"member":member,**row}),flush=True)
        if metrics["td_loss"]<best["val_td_loss"]: best={"val_td_loss":metrics["td_loss"],"epoch":epoch,"q_state_dict":clone_cpu(q.state_dict()),"value_state_dict":clone_cpu(v.state_dict())}
    return best,history


def main():
    a=args(); import torch
    from guided_action_flow.critics.visual_action_transformer_critic import VisualActionTransformerCriticConfig
    split=json.loads(a.split_file.read_text(encoding="utf-8")); train_ids=[int(x) for x in split["train_episode_indices"]]; val_ids=[int(x) for x in split["val_episode_indices"]]
    if len(train_ids)!=a.expected_train_episodes or len(val_ids)!=a.expected_val_episodes or set(train_ids)&set(val_ids): raise ValueError("Episode split does not match the declared fixed train/validation sizes")
    device=torch.device(a.device if torch.cuda.is_available() or a.device=="cpu" else "cpu"); a.output_dir.mkdir(parents=True,exist_ok=True); train_data=join(a.data_dir,train_ids); val_data=join(a.data_dir,val_ids)
    cfg=VisualActionTransformerCriticConfig(action_dim=8,action_horizon=50,visual_tokens=int(train_data["visual_features"].shape[-2]),visual_token_dim=int(train_data["visual_features"].shape[-1]),d_model=a.d_model,num_layers=a.layers,num_heads=a.heads,dropout=a.dropout)
    meta={"format":"armstrong-qgf-visual-action-iql-v1","ablation":"no_state_Q_z_a","device":str(device),"critic_arch":"visual_action_transformer","critic_config":asdict(cfg),"train_episode_indices":train_ids,"val_episode_indices":val_ids,"train_samples":int(train_data["action_chunk"].shape[0]),"val_samples":int(val_data["action_chunk"].shape[0]),"train_positive_rewards":int((train_data["reward"]>0).sum()),"val_positive_rewards":int((val_data["reward"]>0).sum()),"training_args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},"provenance":"Uses the full visual-Q frozen feature cache and exact episode split; state and next-state inputs are intentionally not passed to the critic."}
    (a.output_dir/"training_input_summary.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8"); summaries=[]
    for m in range(a.ensemble_size):
        best,history=train_member(m,a,train_data,val_data,cfg,device); ckpt={**meta,"model_state_dict":best["q_state_dict"],"value_model_state_dict":best["value_state_dict"],"selected_epoch":best["epoch"],"selected_val_td_loss":best["val_td_loss"],"history":history,"ensemble_member_index":m,"member_seed":a.seed+m*1009}; out=a.output_dir/f"critic_member_{m:02d}.pt"; torch.save(ckpt,out); summaries.append({"member_index":m,"path":out.name,"selected_epoch":best["epoch"],"selected_val_td_loss":best["val_td_loss"]})
    (a.output_dir/"training_summary.json").write_text(json.dumps({**meta,"members":summaries},indent=2)+"\n",encoding="utf-8"); print(json.dumps({"TRAINING_COMPLETE":True,"members":summaries},indent=2))


if __name__ == "__main__": main()
