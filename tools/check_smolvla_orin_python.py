#!/usr/bin/env python3
"""Import and CUDA smoke check for the Orin SmolVLA runtime."""

import cv2
import grpc
import lerobot
import lerobot.async_inference.policy_server
import lerobot_robot_armstrong_ros2
import lerobot_robot_armstrong_ros2.async_client
import rclpy
import torch
import torchvision
import transformers
from torchvision.transforms import v2


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    image = torch.rand(1, 3, 64, 64, device="cuda")
    resized = v2.Resize((32, 32))(image)
    if tuple(resized.shape) != (1, 3, 32, 32):
        raise RuntimeError(f"unexpected resize result: {tuple(resized.shape)}")
    print(
        "SMOLVLA_ORIN_RUNTIME_OK",
        f"torch={torch.__version__}",
        f"torchvision={torchvision.__version__}",
        f"transformers={transformers.__version__}",
        f"grpc={grpc.__version__}",
        f"opencv={cv2.__version__}",
        f"device={torch.cuda.get_device_name(0)}",
    )


if __name__ == "__main__":
    main()
