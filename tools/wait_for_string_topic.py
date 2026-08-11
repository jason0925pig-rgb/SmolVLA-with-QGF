#!/usr/bin/env python3
"""Wait for a std_msgs/String topic without relying on the ROS CLI daemon."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--contains", action="append", default=[])
    parser.add_argument("--regex", action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    patterns = [re.compile(value) for value in args.regex]
    rclpy.init(args=None)
    node = Node(f"onearm_string_topic_wait_{os.getpid()}")
    last_data: str | None = None
    message_count = 0

    def on_message(message: String) -> None:
        nonlocal last_data, message_count
        last_data = message.data
        message_count += 1

    # Keep a strong reference for the lifetime of the wait. Some rclpy
    # versions otherwise allow the subscription object to be collected.
    subscription = node.create_subscription(String, args.topic, on_message, 10)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.10)
            if last_data is None:
                continue
            if not all(value in last_data for value in args.contains):
                continue
            if not all(pattern.search(last_data) for pattern in patterns):
                continue
            print(f"data: {last_data}")
            return 0

        print(
            f"ERROR: topic condition timed out: topic={args.topic} "
            f"messages={message_count} contains={args.contains!r} "
            f"regex={args.regex!r}",
            file=sys.stderr,
        )
        if last_data is not None:
            print(f"last_data: {last_data}", file=sys.stderr)
        return 1
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
