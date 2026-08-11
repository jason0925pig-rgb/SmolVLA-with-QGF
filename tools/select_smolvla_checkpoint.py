#!/usr/bin/env python3
"""Select a SmolVLA checkpoint from comparable offline validation reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for path in args.report:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise ValueError(f"validation did not succeed: {path}")
        for key in (
            "joint_action_mae_rad",
            "joint_action_max_abs_error_rad",
            "gripper_binary_accuracy",
            "maximum_demonstrated_envelope_overshoot",
        ):
            if key not in payload:
                raise KeyError(f"{path} is missing {key}")
        reports.append(payload)

    # Gripper correctness is safety-relevant, then prefer lower continuous
    # joint error, maximum error, demonstrated-envelope overshoot and loss.
    # The chosen 20K checkpoint dominates the earlier checkpoints on every
    # listed metric, so the result does not depend on a subjective weighting.
    selected = min(
        reports,
        key=lambda item: (
            -float(item["gripper_binary_accuracy"]),
            float(item["joint_action_mae_rad"]),
            float(item["joint_action_max_abs_error_rad"]),
            float(item["maximum_demonstrated_envelope_overshoot"]),
            float(item["loss"]["mean"]),
        ),
    )
    summary = {
        "format": "onearm-smolvla-checkpoint-selection-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "highest gripper binary accuracy, then lowest joint MAE, maximum "
            "joint error, demonstrated-envelope overshoot and mean validation loss"
        ),
        "important_limit": (
            "All reports use deterministic samples from the training dataset; "
            "this checks numerical and pipeline quality but is not an unseen-task evaluation."
        ),
        "selected_checkpoint": selected["checkpoint"],
        "selected_metrics": {
            "loss_mean": selected["loss"]["mean"],
            "joint_action_mae_rad": selected["joint_action_mae_rad"],
            "joint_action_max_abs_error_rad": selected["joint_action_max_abs_error_rad"],
            "gripper_binary_accuracy": selected["gripper_binary_accuracy"],
            "maximum_demonstrated_envelope_overshoot": selected[
                "maximum_demonstrated_envelope_overshoot"
            ],
        },
        "candidates": [
            {
                "checkpoint": item["checkpoint"],
                "loss_mean": item["loss"]["mean"],
                "joint_action_mae_rad": item["joint_action_mae_rad"],
                "joint_action_max_abs_error_rad": item["joint_action_max_abs_error_rad"],
                "gripper_binary_accuracy": item["gripper_binary_accuracy"],
                "maximum_demonstrated_envelope_overshoot": item[
                    "maximum_demonstrated_envelope_overshoot"
                ],
            }
            for item in reports
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SELECTED_CHECKPOINT={summary['selected_checkpoint']}")
    print(f"SELECTION_REPORT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
