# QGF ablation arms — Random Guidance and B0-LM

Branch contents, what they answer, and what is already verified. Two tasks:
**water bottle into carton** and **red parcel out of box**.

---

## 1. The two arms

The experiment table has a `Random Guidance` column and every cell is empty. This
branch is what fills it, plus a second arm the same switch gives us for free.

| arm | what changes | what it answers |
| --- | --- | --- |
| `qgf` | the deployed behaviour: `v ← v − clip(∇Q)/β` | the method |
| **`qgf_random`** | same per-sample \|∇Q\|, same clip, same gate, same 1/β — **only the direction is randomised** | is the gain from the Q *direction*, or would any perturbation of the same size do? |
| **`qgf_zero`** (B0-LM) | the critic forward **and backward still run**; only the applied guidance is zeroed | is the gain from Q, or from the critic merely being *present* — the extra latency and the different code path? |

`qgf_zero` is not a formality. See §3.

---

## 2. Where the switch lives

`QGuidanceConfig.guidance_mode ∈ {critic, random_matched_norm, zero}` plus
`random_seed`, in `qgf/src/guided_action_flow/guidance/qgf.py`.

Three details that were not obvious and are easy to get wrong:

**The random direction is applied BEFORE clipping**, right after the real
gradient's norm is recorded. The clip and the uncertainty gate then see exactly
what they would have seen for a real gradient of that magnitude. Matching the
norm after clipping would be a different, weaker control.

**The zero is applied AFTER the gate multiply and immediately before
`v − grad/β`.** Zeroing right after `autograd.grad` instead would make
`raw_grad_norm` log as 0, the clip a no-op, and the diagnostics a lie — so the
arm would silently stop being the control it is meant to be. The whole point of
B0-LM is that the critic forward, the backward, the clip and the gate all still
happen and are all still logged at their true values.

**The random draw uses a dedicated `torch.Generator`, never the global RNG.**
SmolVLA draws its own initial flow noise from the global stream
(`smolvla_qgf.py` passes `noise=None`), so consuming from it here would change
the policy's own sampling and the arm would no longer differ from `critic` in
only one respect.

Two new diagnostics come out per denoise step: `q_direction_cos_mean`
(1.0 in critic mode, ≈0 in random, 0 in zero) and `q_guidance_mode`.

The policy server reads `SMOLVLA_QGF_GUIDANCE_MODE` and `SMOLVLA_QGF_RANDOM_SEED`,
and logs one line per chunk:

```
QGF_CHUNK idx=… infer_ms=… mode=… beta=…
```

That line exists because **no QGF-on inference latency has ever been measured on
this robot** — the 1.5–1.8 s figures in the tree are hand-written comments about
baseline SmolVLA, and they disagree with each other.

---

## 3. Why B0-LM matters more than it looks

An offline probe on 2026-09-03 measured, on the five held-out episodes of each
critic, the ratio `‖∇Q/β‖ / ‖v_t‖` over the eight action dimensions:

| critic | ratio at β=0.5 | vs the 0.02 strength floor | probe's suggested β |
| --- | ---: | ---: | :---: |
| mug | 0.00005 | 400× under | none passes, at any β tried |
| stapler | 0.00119 | 17× under | 0.005 |
| red parcel | 0.00237 | 8× under | 0.01 |

The gradient clip never fires (clip rate 0 at every β), so the 1/β scaling is
linear and these extrapolate.

At β=0.5 the guided velocity field differs from the unguided one by about one
part in 20 000 per step for mug, and the largest of the three is still 8× below
the floor. **Every QGF cohort in the table was collected at that setting.** That
does not make the measured effects false, but it does mean they cannot be
explained by the Q gradient — so something else in the QGF code path is a live
candidate, and `qgf_zero` is the arm that separates the two.

Read the per-task magnitude, not the cross-task ordering: comparing effect sizes
across mug/stapler/red-parcel confounds gradient scale with task difficulty,
baseline rate, day and lighting.

Reports: `runs/<task>/outputs/beta_probe/beta_probe_report.json` on the 4090.

---

## 4. What is verified

**Unit tests** — `qgf/tests/test_qgf_guidance_mode.py`, 9 tests, and the 7
existing `test_qgf.py` tests still pass unchanged. They pin: the default config
still means `critic`; zero mode keeps the raw and clipped norms and the Q value
identical to critic mode while applying exactly nothing; random mode matches the
applied magnitude per sample; the direction is random; the same seed reproduces;
the global RNG is untouched; a seedless random arm and an unknown mode are both
refused; and clipping is still exercised in all three modes.

**On the Orin, against the deployed stapler critic**, `smoke_guidance_mode.py`
(no ROS import, no policy server, no arm — it says so and checks its own imports):

```
critic    raw|grad| 0.104055   clipped 0.104055   applied 0.20811   cos 1.0
zero      raw|grad| 0.104055   clipped 0.104055   applied 0        v_guided == v
random    raw|grad| 0.104055   clipped 0.104055   applied 0.20811   cos -0.0035
```

The same numbers came out on the 4090 against the same checkpoint.

---

## 5. Deploying it

The Orin's checkout is not this repo — it sits at an older commit with local
uncommitted edits to the control files. Copying files in would silently revert
those, so `tools/qgf_ablation/deploy_guidance_mode.py` edits the Orin's **own**
copies by matching exact anchors and fails loudly if they are not what it expects.

```bash
python3 deploy_guidance_mode.py --check      # print the plan, write nothing
python3 deploy_guidance_mode.py              # apply, with backups
python3 deploy_guidance_mode.py --rollback   # restore the newest backups
```

It refuses while any session holds the arm or the cameras, and asserts 24
post-conditions on the patched text before writing anything.

`check_markers.py` guards the deployer itself. Its applied-marker scheme failed
twice: first it used the replacement's first line, so five edits that append to
their anchor would have double-applied; then `env reads` used
`SMOLVLA_QGF_GUIDANCE_MODE` as its marker — which the per-chunk latency edit,
running earlier, also inserts — so that edit was judged already-applied and
skipped, and `SMOLVLA_QGF_RANDOM_SEED` never reached the file.

**Already deployed on the Orin as of 2026-09-05**, smoke passed, backups
alongside the two files as `*.bak.20260905_225231`.

---

## 6. Collecting

```
.\tools\my_run_ablation_arm.cmd -Task red_parcel -Arm qgf_random -Beta 0.5 -EpisodeCount 20
.\tools\my_run_ablation_arm.cmd -Task red_parcel -Arm qgf_zero   -Beta 0.5 -EpisodeCount 20
```

`-DryRun` verifies every precondition, prints the remote command and exits 3 if
anything would block, without starting anything. **Without it, running this IS
starting a rollout on the real robot.**

β must match the QGF cell of the same table row, because these arms are controls
for that cell: **0.5** for red parcel, **2** for the bottle rows.

The launcher refuses to start when a rollout **or teleop** session is running,
when the switch is not on the Orin, when the bundle or critic is missing, or when
the task prompt does not match its recorded sha256. Episodes land in a separate
cohort tag `<task>_ablation_normal` so nothing mixes with the existing cohorts,
and every episode's notes carry `arm=`, `guidance_mode=`, `qgf_beta=`,
`qgf_random_seed=`, `qgf_critic=` and `policy_bundle=`.

**One trap worth knowing about**, since both busy gates in the tree now avoid it:
`pgrep -f` matches the whole command line of every process, including the ssh
that carries the check. The bracket trick (`safe_one_arm_serv[o]`) stops the
pattern matching its own text, but not other text on the same command line — a
sibling check that greps `.../policy_server_qgf.py` made the gate fire on an idle
robot. The gate is now its own ssh call, containing only the pattern.

---

## 7. Known gaps

* **The bottle's β=2 rests on one operator's recollection.** It is not in any
  episode's metadata. If it was not 2, the bottle rows' controls are invalid.
* **`arm.log` is not archived at the end of a session** (`/tmp/one_arm_smolvla_*`,
  cleared on reboot; 0 archived so far). B0-LM's other product is the latency
  distribution, and without the log there is no P95/P99.
* **The bottle has no rollout root yet**; the launcher writes to a new
  `bottle_real_rollouts`, deliberately not the emptied `qgf_real_rollouts`.
* **Raw episodes keep disappearing.** bottle ep0-241 (08-28), mug ep50-149
  (09-02, index still lists 150), red parcel ep50-89 (by 09-07, index still lists
  90). Those last 40 are the raw data behind the 92.5% QGF cell. The table's
  numbers survive; the episodes behind them do not.
