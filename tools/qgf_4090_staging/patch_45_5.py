"""Apply the minimal, backward-compatible 45/5 change required by handoff section 8.

Three edits, nothing else:
  1. manifest builder: split filename derived from the real counts, not "90_10".
  2. manifest builder: the "Cannot make a 90/10 split" message becomes generic.
  3. trainer: --expected-train-episodes / --expected-val-episodes, defaulting to
     90 / 10 so every existing invocation behaves exactly as before.

Run on the 4090 inside the repo snapshot.  Refuses to run twice.
"""
import io
import sys
from pathlib import Path

REPO = Path("/opt/qgf_real_robot/repos/SmolVLA-with-QGF")
BUILD = REPO / "qgf/scripts/build_real_robot_visual_iql_manifest.py"
TRAIN = REPO / "qgf/scripts/train_real_robot_visual_iql.py"


def sub_once(text, old, new, label):
    n = text.count(old)
    if n == 0:
        raise SystemExit(f"FAIL [{label}]: anchor not found:\n{old}")
    if n > 1:
        raise SystemExit(f"FAIL [{label}]: anchor occurs {n} times, refusing")
    print(f"  ok  {label}")
    return text.replace(old, new)


# ---------- 1 + 2: manifest builder ----------
b = io.open(BUILD, encoding="utf-8").read()

b = sub_once(
    b,
    '        raise RuntimeError("Cannot make a 90/10 split: one or more requested episodes are incomplete.")',
    '        raise RuntimeError(\n'
    '            "Cannot make the episode split: one or more requested episodes are incomplete."\n'
    '        )',
    "builder: generic incomplete-episode message",
)

b = sub_once(
    b,
    '    split = _stratified_episode_split(outcomes, val_count=args.val_count, seed=args.split_seed)\n'
    '    split_path = args.output_dir / "episode_split_90_10.json"',
    '    split = _stratified_episode_split(outcomes, val_count=args.val_count, seed=args.split_seed)\n'
    '    n_train = len(split["train_episode_indices"])\n'
    '    n_val = len(split["val_episode_indices"])\n'
    '    split_path = args.output_dir / f"episode_split_{n_train}_{n_val}.json"',
    "builder: split filename from real counts",
)

# ---------- 3: trainer ----------
t = io.open(TRAIN, encoding="utf-8").read()

t = sub_once(
    t,
    '    parser.add_argument("--device", default="cuda")',
    '    parser.add_argument("--device", default="cuda")\n'
    '    parser.add_argument(\n'
    '        "--expected-train-episodes",\n'
    '        type=int,\n'
    '        default=90,\n'
    '        help="Number of training episodes the split file must contain.",\n'
    '    )\n'
    '    parser.add_argument(\n'
    '        "--expected-val-episodes",\n'
    '        type=int,\n'
    '        default=10,\n'
    '        help="Number of validation episodes the split file must contain.",\n'
    '    )',
    "trainer: add --expected-*-episodes",
)

t = sub_once(
    t,
    '    if len(train_episodes) != 90 or len(val_episodes) != 10 or set(train_episodes) & set(val_episodes):\n'
    '        raise ValueError("Expected a disjoint fixed 90/10 episode split.")',
    '    if (\n'
    '        len(train_episodes) != args.expected_train_episodes\n'
    '        or len(val_episodes) != args.expected_val_episodes\n'
    '        or set(train_episodes) & set(val_episodes)\n'
    '    ):\n'
    '        raise ValueError(\n'
    '            "Expected a disjoint fixed "\n'
    '            f"{args.expected_train_episodes}/{args.expected_val_episodes} episode split, got "\n'
    '            f"{len(train_episodes)}/{len(val_episodes)} with "\n'
    '            f"{len(set(train_episodes) & set(val_episodes))} shared episodes."\n'
    '        )',
    "trainer: honour --expected-*-episodes",
)

io.open(BUILD, "w", encoding="utf-8", newline="\n").write(b)
io.open(TRAIN, "w", encoding="utf-8", newline="\n").write(t)

import ast

for p in (BUILD, TRAIN):
    ast.parse(io.open(p, encoding="utf-8").read())
    print(f"  syntax OK  {p.name}")

if "90_10" in b or "90/10" in b:
    print("  WARN: builder still mentions 90_10 somewhere")
print("patched")
