#!/usr/bin/env python3
"""Smoke-test a large SmolVLA backbone and record GPU usage.

This intentionally does not assume a fine-tuned LeRobot checkpoint exists for the
large backbone. It instantiates SmolVLA from config with the requested VLM
backbone, moves it to CUDA through the normal policy constructor, then attempts a
single fake-observation select_action call.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def query_gpu() -> list[str] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return None
    line = result.stdout.strip().splitlines()[0]
    return [part.strip() for part in line.split(",")]


def monitor_gpu(csv_path: Path, stop_event: threading.Event, interval_s: float) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "wall_time_s",
                "timestamp",
                "name",
                "utilization_gpu_pct",
                "memory_used_mib",
                "memory_total_mib",
                "power_draw_w",
                "temperature_gpu_c",
            ]
        )
        start = time.time()
        while not stop_event.is_set():
            row = query_gpu()
            if row is not None:
                writer.writerow([f"{time.time() - start:.3f}", *row])
                f.flush()
            time.sleep(interval_s)
        row = query_gpu()
        if row is not None:
            writer.writerow([f"{time.time() - start:.3f}", *row])


def read_peak_gpu(csv_path: Path) -> dict[str, float | int | None]:
    peak_mem = 0
    peak_util = 0
    peak_power = 0.0
    samples = 0
    if not csv_path.exists():
        return {"samples": 0, "peak_memory_mib": None, "peak_utilization_pct": None, "peak_power_w": None}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples += 1
            try:
                peak_mem = max(peak_mem, int(row["memory_used_mib"]))
                peak_util = max(peak_util, int(row["utilization_gpu_pct"]))
                peak_power = max(peak_power, float(row["power_draw_w"]))
            except ValueError:
                continue
    return {
        "samples": samples,
        "peak_memory_mib": peak_mem,
        "peak_utilization_pct": peak_util,
        "peak_power_w": peak_power,
    }


def build_config(args: argparse.Namespace) -> SmolVLAConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        f"{OBS_IMAGES}.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        f"{OBS_IMAGES}.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        f"{OBS_IMAGES}.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        use_amp=False,
        vlm_model_name=args.model_id,
        load_vlm_weights=True,
        num_vlm_layers=args.num_vlm_layers,
        num_expert_layers=args.num_expert_layers,
        expert_width_multiplier=args.expert_width_multiplier,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        num_steps=args.num_steps,
        pad_language_to="max_length",
    )


def make_fake_observation(image_size: int) -> dict:
    image = torch.zeros(3, image_size, image_size, dtype=torch.float32)
    return {
        OBS_STATE: torch.zeros(6, dtype=torch.float32),
        f"{OBS_IMAGES}.camera1": image.clone(),
        f"{OBS_IMAGES}.camera2": image.clone(),
        f"{OBS_IMAGES}.camera3": image.clone(),
        "task": "pick up the black bowl and place it on the plate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--output-dir", default="runs/smolvla_2p25b_smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-vlm-layers", type=int, default=16)
    parser.add_argument("--num-expert-layers", type=int, default=-1)
    parser.add_argument("--expert-width-multiplier", type=float, default=0.75)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--monitor-interval-s", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_csv = output_dir / "gpu_usage.csv"
    summary_path = output_dir / "summary.json"

    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(gpu_csv, stop_event, args.monitor_interval_s))
    monitor.start()

    summary: dict = {
        "model_id": args.model_id,
        "device": args.device,
        "config": {
            "num_vlm_layers": args.num_vlm_layers,
            "num_expert_layers": args.num_expert_layers,
            "expert_width_multiplier": args.expert_width_multiplier,
            "chunk_size": args.chunk_size,
            "n_action_steps": args.n_action_steps,
            "num_steps": args.num_steps,
        },
        "events": [],
        "success": False,
    }

    t0 = time.time()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        summary["events"].append({"name": "build_config_start", "t_s": time.time() - t0})
        config = build_config(args)

        summary["events"].append({"name": "policy_init_start", "t_s": time.time() - t0})
        policy = SmolVLAPolicy(config)
        summary["events"].append({"name": "policy_init_done", "t_s": time.time() - t0})

        param_count = sum(p.numel() for p in policy.parameters())
        summary["param_count"] = param_count
        summary["param_count_b"] = param_count / 1e9

        summary["events"].append({"name": "policy_to_device_start", "t_s": time.time() - t0})
        policy.to(config.device)
        policy.eval()
        summary["events"].append({"name": "policy_to_device_done", "t_s": time.time() - t0})

        if not args.skip_forward:
            preprocessor, _ = make_smolvla_pre_post_processors(config, dataset_stats=None)
            obs = make_fake_observation(args.image_size)
            summary["events"].append({"name": "preprocess_start", "t_s": time.time() - t0})
            batch = preprocessor(obs)
            summary["events"].append({"name": "preprocess_done", "t_s": time.time() - t0})
            summary["events"].append({"name": "select_action_start", "t_s": time.time() - t0})
            with torch.inference_mode():
                action = policy.select_action(batch)
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()
            summary["events"].append({"name": "select_action_done", "t_s": time.time() - t0})
            summary["action_shape"] = list(action.shape)
            summary["action_dtype"] = str(action.dtype)

        summary["success"] = True
    except BaseException as exc:  # noqa: BLE001
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
    finally:
        summary["elapsed_s"] = time.time() - t0
        if torch.cuda.is_available():
            try:
                summary["torch_cuda_max_memory_allocated_mib"] = (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                )
                summary["torch_cuda_max_memory_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024 / 1024
            except Exception:
                pass
        stop_event.set()
        monitor.join()
        summary["gpu_monitor"] = read_peak_gpu(gpu_csv)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
