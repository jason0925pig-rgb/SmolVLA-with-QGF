"""Verify the deploy script's edit tables are internally sound.

Three properties, each of which has already been violated once:
  1. no marker may be a substring of any OTHER edit's replacement text - that is
     the collision that silently skipped the env-reads edit;
  2. every marker must actually appear in its own replacement, or the edit can
     never be seen as applied and would double-apply;
  3. no marker may already exist in the pristine source files, or the edit would
     be skipped on a clean deployment.
"""
import importlib.util
import io
import os
import sys

SP = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dep", os.path.join(SP, "deploy_guidance_mode.py"))
dep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dep)

# The pristine files this patch expects, relative to the repo root.
REPO = os.environ.get("QGF_PRISTINE") or os.path.abspath(os.path.join(SP, "..", ".."))
SRC = {
    "qgf.py": os.path.join(REPO, "qgf/src/guided_action_flow/guidance/qgf.py"),
    "policy_server_qgf.py": os.path.join(
        REPO, "lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py"),
}

# On a branch that already carries the change, every marker is present by
# definition. That is not a defect, and reporting it as 34 failures is worse than
# useless - say so and stop, rather than making the reader work it out.
PATCHED = [f for f in SRC
           if all(n in io.open(SRC[f], encoding="utf-8").read()
                  for n in dep.REQUIRED[f])]
if PATCHED:
    print("These files already carry the guidance_mode change:")
    for f in PATCHED:
        print("   ", SRC[f])
    print()
    print("This script checks the deployer's marker scheme against PRISTINE sources,")
    print("so it has nothing to say here - every marker is present by construction.")
    print("Run it against a checkout without the change, e.g.")
    print("    git worktree add /tmp/pristine origin/main")
    print("    QGF_PRISTINE=/tmp/pristine python3 check_markers.py")
    print()
    print("SKIPPED (sources already patched)")
    sys.exit(0)

fail = 0
for fname, edits in (("qgf.py", dep.QGF_EDITS), ("policy_server_qgf.py", dep.SRV_EDITS)):
    print("=== %s: %d edits ===" % (fname, len(edits)))
    pristine = io.open(SRC[fname], encoding="utf-8").read()
    for i, (label, marker, old, new) in enumerate(edits):
        problems = []
        # 2) the marker must be in its own replacement
        if marker not in new:
            problems.append("marker is not in its own replacement")
        # 3) the marker must not already exist in the untouched file
        if marker in pristine:
            problems.append("marker already exists in the pristine file")
        # 1) the marker must not be introduced by any other edit
        for j, (other_label, _, _, other_new) in enumerate(edits):
            if i != j and marker in other_new:
                problems.append("marker is introduced by edit %r" % other_label)
        # also check across files, since both are patched in one run
        for other_fname, other_edits in (("qgf.py", dep.QGF_EDITS),
                                         ("policy_server_qgf.py", dep.SRV_EDITS)):
            if other_fname == fname:
                continue
            for other_label, _, _, other_new in other_edits:
                if marker in other_new:
                    problems.append("marker also introduced by %s:%r" % (other_fname, other_label))
        status = "ok" if not problems else "FAIL"
        print("  %-4s %-22s marker=%r" % (status, label, marker[:44]))
        for p in problems:
            print("         - %s" % p)
            fail += 1

print()
print("=== REQUIRED post-conditions are absent from the pristine files ===")
for fname, needs in dep.REQUIRED.items():
    pristine = io.open(SRC[fname], encoding="utf-8").read()
    already = [n for n in needs if n in pristine]
    print("  %-24s %d post-conditions, %d already present%s"
          % (fname, len(needs), len(already), (" -> " + repr(already[:3])) if already else ""))
    # "import torch" legitimately may already exist; flag only, do not fail on it
    hard = [n for n in already if n not in ("import torch",)]
    if hard:
        print("         FAIL: these cannot serve as post-conditions:", hard)
        fail += len(hard)

print()
print("PASSED" if fail == 0 else "FAILED (%d problems)" % fail)
sys.exit(0 if fail == 0 else 1)
