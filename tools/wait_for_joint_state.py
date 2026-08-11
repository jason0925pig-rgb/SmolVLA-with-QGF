#!/usr/bin/env python3
"""Wait for several valid JointState samples without failing on one bad read."""

from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


EXPECTED_NAMES = tuple(f"right_joint{index}" for index in range(1, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/right_arm/joint_states")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--minimum-messages", type=int, default=3)
    return parser.parse_args()


def valid_positions(message: JointState) -> tuple[float, ...] | None:
    if len(message.position) != 7:
        return None
    if message.name:
        lookup = dict(zip(message.name, message.position, strict=False))
        if any(name not in lookup for name in EXPECTED_NAMES):
            return None
        positions = tuple(float(lookup[name]) for name in EXPECTED_NAMES)
    else:
        positions = tuple(float(value) for value in message.position)
    if not all(math.isfinite(value) for value in positions):
        return None
    return positions


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.minimum_messages <= 0:
        raise SystemExit("timeout and minimum-messages must be positive")

    rclpy.init(args=None)
    node = Node("wait_for_valid_right_arm_joint_state")
    valid_count = 0
    invalid_count = 0
    spin_errors = 0
    latest: tuple[float, ...] | None = None

    def callback(message: JointState) -> None:
        nonlocal valid_count, invalid_count, latest
        positions = valid_positions(message)
        if positions is None:
            invalid_count += 1
            return
        latest = positions
        valid_count += 1

    node.create_subscription(JointState, args.topic, callback, 10)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and valid_count < args.minimum_messages:
            try:
                rclpy.spin_once(node, timeout_sec=0.20)
            except Exception as exc:  # Keep waiting through a transient RMW/DDS read failure.
                spin_errors += 1
                print(
                    f"WARNING: transient joint-state receive error ignored: {exc}",
                    file=sys.stderr,
                )
                time.sleep(0.05)
        if valid_count < args.minimum_messages:
            print(
                "ERROR: valid seven-joint feedback did not become stable within "
                f"{args.timeout:.1f}s; valid={valid_count} invalid={invalid_count} "
                f"receive_errors={spin_errors}",
                file=sys.stderr,
            )
            return 1
        assert latest is not None
        formatted = ",".join(f"{value:.6f}" for value in latest)
        print(
            f"JOINT_STREAM_READY valid={valid_count} invalid={invalid_count} "
            f"receive_errors={spin_errors} latest_rad=[{formatted}]"
        )
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
