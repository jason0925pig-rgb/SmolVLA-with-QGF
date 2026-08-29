"""Why is Q flat on failure episodes, and why is the state channel dead?

Tests four competing explanations against the actual trained critic and data,
instead of guessing:

  H1  "no early signal": Q cannot separate success from failure at episode
      start, only near the goal.  -> measure Q vs normalised episode progress.
  H2  "almost nothing is grounded": with terminal-only reward, the fraction of
      training targets carrying real reward or a done flag is tiny, so Q is
      almost entirely bootstrapped from V.  -> count them.
  H3  "LayerNorm(8) destroys proprioception": normalising ACROSS the 8 joint
      dims (not per-joint across the dataset) removes exactly the information
      that distinguishes arm poses.  -> measure what survives.
  H4  "expectile optimism lifts the failure floor": V is fit at the 0.7
      expectile, so failure states inherit value from lookalike success states.
      -> compare V on failure vs success states.

Read-only.  No robot involved.
"""
import io
import json
import sys
from pathlib import Path

import torch

RUN = Path("/opt/qgf_real_robot/runs/mug_purple_box_single_q_45_5_20260829")
CK = RUN / "outputs/single_qcritic/critic_member_00.pt"
DS = Path("/opt/qgf_real_robot/datasets/mug_purple_box_baseline50_20260829")
sys.path.insert(0, "/opt/qgf_real_robot/repos/SmolVLA-with-QGF/qgf/src")
from guided_action_flow.critics.checkpoint import load_action_chunk_critic  # noqa: E402

split = json.load(io.open(RUN / "manifest/episode_split_45_5.json", encoding="utf-8"))
train_eps = [int(x) for x in split["train_episode_indices"]]
val_eps = [int(x) for x in split["val_episode_indices"]]
smap = json.load(io.open(DS / "source_episode_map.json", encoding="utf-8"))
outcome = {e["dest_episode_index"]: e["outcome"] for e in smap["episodes"]}

critic, ckpt = load_action_chunk_critic(str(CK), device="cpu")
mod = critic.module

# the value head, if the checkpoint carries one
vsd = ckpt.get("value_model_state_dict")


def load(ep):
    return torch.load(RUN / f"features/episode_{ep:06d}.pt", map_location="cpu", weights_only=False)


probe = load(val_eps[0])
print("feature cache keys:", sorted(probe.keys()))
print()

# ---------------------------------------------------------------- H2
print("=" * 72)
print("H2: how much of the training signal is actually grounded?")
print("=" * 72)
tot = pos = done = 0
for ep in train_eps:
    d = load(ep)
    r = d["reward"].float().flatten()
    tot += len(r)
    pos += int((r > 0).sum())
    if "done" in d:
        done += int(d["done"].float().flatten().sum())
    elif "terminated" in d:
        done += int(d["terminated"].float().flatten().sum())
print(f"  training samples                  {tot}")
print(f"  samples with reward > 0           {pos}   ({pos / tot * 100:.2f}%)")
print(f"  samples with done/terminal flag   {done}  ({done / tot * 100:.2f}%)")
print(f"  purely bootstrapped (r=0, d=0)    {tot - pos - max(done - pos, 0)}  "
      f"({(tot - pos - max(done - pos, 0)) / tot * 100:.2f}%)")
print()
print("  Reading: every non-grounded target is r + gamma*(1-d)*V(s'), i.e. it says")
print("  nothing except 'be consistent with V'.  The only statements about the")
print("  world are the grounded ones.")

# ---------------------------------------------------------------- H1
print()
print("=" * 72)
print("H1: can Q separate success from failure EARLY in an episode?")
print("=" * 72)


def q_of(d):
    with torch.no_grad():
        return critic(d["state"].float(), d["visual_features"].float(),
                      d["action_chunk"].float()).flatten()


buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
agg = {("success", b): [] for b in buckets}
agg.update({("failure", b): [] for b in buckets})
sample_eps = val_eps + train_eps[:20]
for ep in sample_eps:
    d = load(ep)
    q = q_of(d)
    n = len(q)
    t = torch.arange(n).float() / max(n - 1, 1)
    for b in buckets:
        m = (t >= b[0]) & (t < b[1] if b[1] < 1.0 else t <= 1.0)
        if m.any():
            agg[(outcome[ep], b)].append(q[m])

print(f"  over {len(sample_eps)} episodes ({sum(1 for e in sample_eps if outcome[e]=='success')} success"
      f" / {sum(1 for e in sample_eps if outcome[e]=='failure')} failure)")
print()
print(f"  {'progress':<12}{'success Q':>22}{'failure Q':>22}{'separation':>13}")
for b in buckets:
    s = torch.cat(agg[("success", b)]) if agg[("success", b)] else torch.tensor([float("nan")])
    f = torch.cat(agg[("failure", b)]) if agg[("failure", b)] else torch.tensor([float("nan")])
    sep = float(s.mean() - f.mean())
    print(f"  {b[0]:.0%}-{b[1]:.0%}      "
          f"{float(s.mean()):>10.4f} +/- {float(s.std()):<8.4f}"
          f"{float(f.mean()):>10.4f} +/- {float(f.std()):<8.4f}"
          f"{sep:>12.4f}")
print()
print("  Reading: if separation is ~0 in the first buckets, the critic has no")
print("  opinion until the task is nearly done - which is exactly the window")
print("  where QGF guidance would need to matter.")

# ---------------------------------------------------------------- H3
print()
print("=" * 72)
print("H3: what does LayerNorm(8) do to the proprioceptive state?")
print("=" * 72)
S = torch.cat([load(ep)["state"].float() for ep in val_eps])
ln = torch.nn.LayerNorm(8, elementwise_affine=False)
Sn = ln(S)
print(f"  {len(S)} state vectors from the 5 validation episodes")
print()
print(f"  {'dim':<6}{'raw std':>12}{'raw range':>22}{'after LN std':>16}")
names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "grip"]
for i in range(8):
    print(f"  {names[i]:<6}{float(S[:, i].std()):>12.4f}"
          f"   [{float(S[:, i].min()):>7.3f},{float(S[:, i].max()):>7.3f}]"
          f"{float(Sn[:, i].std()):>16.4f}")
print()
# how much of the between-sample variance survives?
raw_var = float(S.var(dim=0).sum())
ln_var = float(Sn.var(dim=0).sum())
print(f"  total between-sample variance   raw {raw_var:.4f}   after LN {ln_var:.4f}")
# per-sample mean/scale are exactly what LN throws away
mu = S.mean(dim=1)
sd = S.std(dim=1)
print(f"  the two statistics LN discards:")
print(f"    per-sample mean across joints   std {float(mu.std()):.4f}  range [{float(mu.min()):.3f}, {float(mu.max()):.3f}]")
print(f"    per-sample std  across joints   std {float(sd.std()):.4f}  range [{float(sd.min()):.3f}, {float(sd.max()):.3f}]")
print()
print("  Reading: LayerNorm here normalises ACROSS the 8 joints of one sample,")
print("  not per-joint across the dataset. Two different arm poses with the same")
print("  'shape' of joint spread collapse onto the same vector, and the 0/1")
print("  gripper bit is rescaled by whatever the joints happen to be doing.")

# how many distinct poses collapse?
d_raw = torch.cdist(S[:200], S[:200])
d_ln = torch.cdist(Sn[:200], Sn[:200])
iu = torch.triu_indices(200, 200, offset=1)
print()
print(f"  pairwise distance between 200 states:")
print(f"    raw   mean {float(d_raw[iu[0], iu[1]].mean()):.4f}")
print(f"    afterLN mean {float(d_ln[iu[0], iu[1]].mean()):.4f}")
corr = torch.corrcoef(torch.stack([d_raw[iu[0], iu[1]], d_ln[iu[0], iu[1]]]))[0, 1]
print(f"    correlation between the two distance matrices: {float(corr):.4f}")
print("    (a low correlation means LN reshuffles which poses look alike)")

# ---------------------------------------------------------------- H4
print()
print("=" * 72)
print("H4: does the value head lift failure states toward success values?")
print("=" * 72)
if vsd is None:
    print("  no value_model_state_dict in the checkpoint; skipping")
else:
    from guided_action_flow.critics.visual_transformer_critic import (
        VisualTransformerCritic,
        VisualTransformerCriticConfig,
    )

    vcfg = VisualTransformerCriticConfig(**ckpt["critic_config"])
    vcritic = VisualTransformerCritic(vcfg)
    vcritic.load_state_dict(vsd)
    vmod = vcritic.module
    vmod.eval()
    print(f"  {'episode':<10}{'outcome':<10}{'V mean':>10}{'V std':>10}{'V first':>10}{'V last':>10}")
    for ep in val_eps:
        d = load(ep)
        with torch.no_grad():
            v = vmod.forward_value(d["state"].float(), d["visual_features"].float()).flatten()
        print(f"  ep{ep:<8d}{outcome[ep]:<10}{float(v.mean()):>10.4f}{float(v.std()):>10.4f}"
              f"{float(v[0]):>10.4f}{float(v[-1]):>10.4f}")
    print()
    print("  Reading: V is fit with expectile 0.7, i.e. deliberately optimistic.")
    print("  If V on failure states sits well above 0, that optimism is what holds")
    print("  the failure-episode Q floor up at ~0.2 instead of decaying to 0.")
