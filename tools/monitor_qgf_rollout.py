#!/usr/bin/env python3
"""Monitor one active rollout and classify its stop source."""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def fields(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for field in value.split(";"):
        key, separator, item = field.partition("=")
        if separator:
            parsed[key.strip()] = item.strip()
    return parsed


class Monitor(Node):
    def __init__(self):
        super().__init__("qgf_rollout_monitor")
        self.smolvla: dict[str, str] = {}
        self.safety: dict[str, str] = {}
        self.seen_enabled = False
        self.create_subscription(String, "/smolvla/status", self._smolvla_cb, 20)
        self.create_subscription(String, "/right_arm/safety_status", self._safety_cb, 20)

    def _smolvla_cb(self, message: String) -> None:
        self.smolvla = fields(message.data)
        self.seen_enabled = self.seen_enabled or self.smolvla.get("action_enabled") == "1"

    def _safety_cb(self, message: String) -> None:
        self.safety = fields(message.data)

    def result(self) -> tuple[int, str] | None:
        if self.safety.get("robot_emergency_stop") == "1":
            return 20, "physical_emergency_stop"
        if self.safety.get("robot_protective_stop") == "1":
            return 21, "protective_stop"
        error = self.safety.get("robot_error_code")
        if error not in (None, "0"):
            return 22, f"robot_error_{error}"
        if self.smolvla.get("task_completed") == "1":
            return 0, "task_completed_returned_home"
        if self.seen_enabled and self.smolvla.get("action_enabled") == "0":
            return 23, "policy_gate_closed_without_completion"
        if self.seen_enabled and self.safety.get("robot_enabled") == "0":
            return 24, "robot_enable_lost"
        if self.seen_enabled and self.safety.get("robot_powered_on") == "0":
            return 25, "robot_power_lost"
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    rclpy.init(args=None)
    node = Monitor()
    start = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            result = node.result()
            if result is not None:
                code, source = result
                print(f"QGF_ROLLOUT_MONITOR source={source}", flush=True)
                return code
            elapsed = time.monotonic() - start
            if not node.seen_enabled and elapsed > args.startup_timeout:
                print("QGF_ROLLOUT_MONITOR source=action_gate_never_enabled", flush=True)
                return 26
            if elapsed > args.timeout:
                print("QGF_ROLLOUT_MONITOR source=episode_timeout", flush=True)
                return 27
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 28


if __name__ == "__main__":
    raise SystemExit(main())
