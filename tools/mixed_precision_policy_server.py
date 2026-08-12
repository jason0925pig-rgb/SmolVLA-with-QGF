#!/usr/bin/env python3
"""Stock LeRobot policy server with an explicit AMP inference mode.

This is deliberately a standalone script: it imports no ROS package and has
no path to robot publishers, services, or SDK calls.
"""

from __future__ import annotations

import os

import torch

from lerobot.async_inference.policy_server import PolicyServer
from lerobot_robot_armstrong_ros2.policy_server_telemetry import serve


_STOCK_GET_ACTION_CHUNK = PolicyServer._get_action_chunk


def _amp_dtype() -> torch.dtype | None:
    name = os.environ.get("SMOLVLA_INFERENCE_DTYPE", "fp32").strip().lower()
    choices = {
        "fp32": None,
        "float32": None,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in choices:
        raise ValueError("SMOLVLA_INFERENCE_DTYPE must be fp32, fp16, or bf16")
    return choices[name]


def _get_action_chunk_with_amp(self: PolicyServer, observation: dict[str, torch.Tensor]) -> torch.Tensor:
    dtype = _amp_dtype()
    if dtype is None:
        return _STOCK_GET_ACTION_CHUNK(self, observation)
    if not torch.cuda.is_available():
        raise RuntimeError("mixed-precision inference requires CUDA")
    # Keep parameters in their checkpoint dtype while Tensor-Core compatible
    # operations run in FP16/BF16. Numerically sensitive operations remain FP32.
    with torch.autocast(device_type="cuda", dtype=dtype):
        return _STOCK_GET_ACTION_CHUNK(self, observation)


# TelemetryPolicyServer deliberately inherits the stock PolicyServer, so
# patching this base method preserves its ROS-only normalized-chunk publisher
# while adding AMP around the actual model inference.  Importing stock `serve`
# here would silently bypass that telemetry subclass and make a QGF recorder
# wait forever for /smolvla/normalized_action_chunk.
PolicyServer._get_action_chunk = _get_action_chunk_with_amp


if __name__ == "__main__":
    serve()
