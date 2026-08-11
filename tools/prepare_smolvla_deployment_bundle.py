#!/usr/bin/env python3
"""Create a relocatable SmolVLA checkpoint + VLM deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--lerobot-source", type=Path)
    parser.add_argument("--armstrong-plugin", type=Path)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination, symlinks=False)


def patch_json(path: Path, transform) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    vlm = args.vlm.resolve()
    output = args.output.resolve()
    for required in (
        checkpoint / "config.json",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "model.safetensors",
        vlm / "config.json",
        vlm / "model.safetensors",
        args.validation_report,
        args.training_log,
        args.dataset_info,
        args.dataset_stats,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    checkpoint_out = output / "checkpoint"
    copy_tree(checkpoint, checkpoint_out)
    vlm_out = output / "vlm"
    vlm_out.mkdir()
    for source in sorted(vlm.iterdir()):
        if source.is_file() and source.name not in {".gitattributes", "README.md"}:
            shutil.copy2(source, vlm_out / source.name)

    patch_json(
        checkpoint_out / "config.json",
        lambda payload: payload.__setitem__("vlm_model_name", "vlm"),
    )

    def patch_preprocessor(payload: dict) -> None:
        for step in payload.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                step.setdefault("config", {})["tokenizer_name"] = "vlm"

    patch_json(checkpoint_out / "policy_preprocessor.json", patch_preprocessor)

    shutil.copy2(args.validation_report, output / "offline_validation.json")
    shutil.copy2(args.training_log, output / "train.log")
    shutil.copy2(args.dataset_info, output / "dataset_info.json")
    shutil.copy2(args.dataset_stats, output / "dataset_stats.json")
    if args.lerobot_source is not None:
        source = args.lerobot_source.resolve()
        if not (source / "pyproject.toml").is_file():
            raise FileNotFoundError(source / "pyproject.toml")
        shutil.copytree(
            source,
            output / "software" / "lerobot-v0.4.4",
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
        )
    if args.armstrong_plugin is not None:
        source = args.armstrong_plugin.resolve()
        if not (source / "pyproject.toml").is_file():
            raise FileNotFoundError(source / "pyproject.toml")
        shutil.copytree(
            source,
            output / "software" / "lerobot_robot_armstrong_ros2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    if args.project_root is not None:
        project_root = args.project_root.resolve()
        deployment_files = (
            "docs/SMOLVLA_DEPLOYMENT.md",
            "docs/SMOLVLA_DEPLOYMENT_TEST_RESULTS_20260805.json",
            "docs/SMOLVLA_TRAINING_RESULT_20260805.md",
            "servo_controller/config/smolvla_first_rollout.yaml",
            "tools/setup_smolvla_client_ubuntu.sh",
            "tools/select_smolvla_checkpoint.py",
            "tools/smoke_smolvla_policy_server.py",
            "tools/start_smolvla_policy_server.sh",
            "tools/start_smolvla_ros_client.sh",
            "tools/test_smolvla_bundle.sh",
            "tools/ubuntu_smolvla_stack.sh",
            "tools/validate_smolvla_checkpoint.py",
        )
        for relative in deployment_files:
            source = project_root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = output / "deployment" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if destination.suffix == ".sh":
                destination.chmod(0o755)
        policy_launcher = output / "start_policy_server.sh"
        policy_launcher.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "BUNDLE_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
            "cd \"${BUNDLE_ROOT}\"\n"
            "export SMOLVLA_PHYSICAL_GPU=\"${SMOLVLA_PHYSICAL_GPU:-4}\"\n"
            "exec \"${BUNDLE_ROOT}/deployment/tools/start_smolvla_policy_server.sh\"\n",
            encoding="utf-8",
        )
        policy_launcher.chmod(0o755)

    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "format": "onearm-smolvla-deployment-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_source": str(checkpoint),
        "vlm_source": str(vlm),
        "camera_features": [
            "observation.images.chest",
            "observation.images.wrist_right",
        ],
        "state_dim": 8,
        "action_dim": 8,
        "fps": 30,
        "gripper_convention": "0=open, 1=closed",
        "relative_vlm_path": "vlm",
        "launch_working_directory": ".",
        "robot_action_gate_default": "disabled",
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"DEPLOYMENT_BUNDLE_READY={output}")
    print(f"FILES={len(files) + 1}")
    print(f"BYTES={sum(item['bytes'] for item in files) + manifest_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
