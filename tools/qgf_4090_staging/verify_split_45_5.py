"""Handoff section 8 acceptance test for the 45/5 episode split.

Checks: disjoint, totals 50, both outcomes present on BOTH sides, and that the
validation side is not single-outcome (which would make Q checkpoint selection
meaningless).  Exits non-zero on any failure so it can gate the training run.

Usage: python3 verify_split_45_5.py <split_file.json> <source_episode_map.json>
"""
import io
import json
import sys
from collections import Counter

split_path, map_path = sys.argv[1], sys.argv[2]
split = json.load(io.open(split_path, encoding="utf-8"))
smap = json.load(io.open(map_path, encoding="utf-8"))

train = [int(x) for x in split["train_episode_indices"]]
val = [int(x) for x in split["val_episode_indices"]]
outcome = {e["dest_episode_index"]: e["outcome"] for e in smap["episodes"]}

fail = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


print(f"split file: {split_path}")
print(f"strategy  : {split.get('strategy')}   seed: {split.get('seed')}")
print()
check(len(train) == 45, f"train episode count == 45 (got {len(train)})")
check(len(val) == 5, f"val episode count == 5 (got {len(val)})")
overlap = set(train) & set(val)
check(not overlap, f"train/val disjoint (shared: {sorted(overlap) or 'none'})")
check(len(train) + len(val) == 50, f"train+val == 50 (got {len(train) + len(val)})")
check(
    len(set(train) | set(val)) == 50,
    f"union covers 50 distinct episodes (got {len(set(train) | set(val))})",
)
missing = [i for i in train + val if i not in outcome]
check(not missing, f"every split episode is in source_episode_map (missing: {missing or 'none'})")

tc = Counter(outcome.get(i) for i in train)
vc = Counter(outcome.get(i) for i in val)
print()
print(f"  train outcomes: {dict(tc)}")
print(f"  val   outcomes: {dict(vc)}")
print()
check(tc.get("success", 0) > 0, f"train has successes ({tc.get('success', 0)})")
check(tc.get("failure", 0) > 0, f"train has failures ({tc.get('failure', 0)})")
check(
    vc.get("success", 0) > 0,
    f"val has at least one success ({vc.get('success', 0)}) - needed for a positive-reward chunk",
)
check(
    vc.get("failure", 0) > 0,
    f"val has at least one failure ({vc.get('failure', 0)}) - single-outcome val cannot select a Q checkpoint",
)

whole = Counter(outcome.values())
print()
print(f"  whole cohort: {dict(whole)}")
tr_rate = tc.get("success", 0) / max(len(train), 1)
va_rate = vc.get("success", 0) / max(len(val), 1)
print(f"  success rate  train {tr_rate:.1%}   val {va_rate:.1%}   overall {whole.get('success', 0) / 50:.1%}")

print()
if fail:
    print(f"FAILED {len(fail)} check(s):")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
