#!/usr/bin/env python3
"""Lexicographic checkpoint selection per handoff section 5:
1) finite/valid  2) min(missed+early open)  3) max closed F1
4) min(|close_err|+|open_err|)  5) min flow loss then joint MAE."""
import json, sys
from pathlib import Path

rep = Path(sys.argv[1])  # validation_reports dir
cands = []
for f in sorted(rep.glob("replay_*.json")):
    r = json.loads(f.read_text())
    step = r["checkpoint"]
    lightf = rep / f"light_{step}.json"
    light = json.loads(lightf.read_text()) if lightf.exists() else {}
    key = (
        r.get("missed_open_rate", 1) + r.get("early_open_rate", 1),
        -(r["closed_f1"] if r.get("closed_f1") is not None else 0),
        (abs(r["close_err_mean_s"] if r.get("close_err_mean_s") is not None else 99)
         + abs(r["open_err_mean_s"] if r.get("open_err_mean_s") is not None else 99)),
        # Handoff sec.5 item 4 lists "validation flow loss and 7-joint MAE"
        # together without an order. Joint MAE is used as the primary of the
        # two: flow loss resamples noise per evaluation so it is not
        # comparable across checkpoints, while joint MAE is deterministic
        # and directly measures action accuracy.
        light.get("joint_mae_rad", 99),
        light.get("val_flow_loss", 99),
    )
    cands.append((key, step, r, light))
cands.sort(key=lambda x: x[0])
best = cands[0]
print("ranking (best first):")
for key, step, r, light in cands:
    print(f"  {step}: missed+early={key[0]:.2f} F1={-key[1]:.3f} evt_err={key[2]:.2f}s "
          f"jmae={light.get('joint_mae_rad'):.5f} loss={light.get('val_flow_loss'):.5f}")
out = {"best": best[1], "rule": "lexicographic per handoff sec.5; tie-break within item 4 uses joint MAE before flow loss (flow loss resamples noise, not comparable across checkpoints)",
       "ranking": [c[1] for c in cands]}
(rep / "selected.json").write_text(json.dumps(out, indent=2))
print("\nBEST:", best[1])
