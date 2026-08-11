#!/usr/bin/env python3
"""Passively verify the two dataset camera topics and their header cadence."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


DEFAULT_TOPICS = (
    "/camera_head/color/image_raw/compressed",
    "/camera_wrist/color/image_raw/compressed",
)


@dataclass
class Samples:
    header_ns: list[int] = field(default_factory=list)
    received_monotonic: list[float] = field(default_factory=list)


class CameraRateProbe(Node):
    def __init__(self, topics: tuple[str, ...]) -> None:
        super().__init__("onearm_camera_rate_probe")
        self.samples = {topic: Samples() for topic in topics}
        self.camera_subscriptions = [
            self.create_subscription(
                CompressedImage,
                topic,
                lambda message, selected=topic: self.on_image(
                    selected, message
                ),
                qos_profile_sensor_data,
            )
            for topic in topics
        ]

    def on_image(self, topic: str, message: CompressedImage) -> None:
        stamp = message.header.stamp
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if timestamp_ns <= 0:
            timestamp_ns = self.get_clock().now().nanoseconds
        self.samples[topic].header_ns.append(timestamp_ns)
        self.samples[topic].received_monotonic.append(time.monotonic())


def summarize(samples: Samples) -> dict[str, float | int]:
    if len(samples.header_ns) < 2:
        return {
            "frames": len(samples.header_ns),
            "header_fps": 0.0,
            "receive_fps": 0.0,
            "median_period_ms": float("inf"),
            "maximum_gap_ms": float("inf"),
        }
    # Adjacent-pair iteration intentionally compares N samples with N-1
    # successors. strict=True would reject the correct shape here.
    header_deltas = [
        (current - previous) / 1_000_000_000
        for previous, current in zip(
            samples.header_ns, samples.header_ns[1:]
        )
        if current > previous
    ]
    receive_span = (
        samples.received_monotonic[-1] - samples.received_monotonic[0]
    )
    if not header_deltas or receive_span <= 0:
        return {
            "frames": len(samples.header_ns),
            "header_fps": 0.0,
            "receive_fps": 0.0,
            "median_period_ms": float("inf"),
            "maximum_gap_ms": float("inf"),
        }
    return {
        "frames": len(samples.header_ns),
        "header_fps": 1.0 / statistics.mean(header_deltas),
        "receive_fps": (len(samples.received_monotonic) - 1) / receive_span,
        "median_period_ms": statistics.median(header_deltas) * 1000.0,
        "maximum_gap_ms": max(header_deltas) * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe to the head and right-wrist compressed RGB streams. "
            "This program never publishes a robot command."
        )
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--first-frame-timeout", type=float, default=20.0)
    parser.add_argument("--minimum-fps", type=float, default=27.0)
    parser.add_argument("--maximum-fps", type=float, default=33.0)
    parser.add_argument("--maximum-gap-ms", type=float, default=100.0)
    parser.add_argument("--topic", action="append", dest="topics")
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.first_frame_timeout <= 0:
        raise SystemExit(
            "duration and first-frame-timeout must be positive; "
            "warmup must be nonnegative"
        )
    topics = tuple(args.topics or DEFAULT_TOPICS)
    if len(topics) != 2 or len(set(topics)) != 2:
        raise SystemExit("exactly two different camera topics are required")

    rclpy.init()
    node = CameraRateProbe(topics)
    try:
        # Orbbec devices can need several seconds to load presets and start
        # their USB streams.  Do not spend the fixed measurement window while
        # one camera is still initializing.
        first_frame_deadline = time.monotonic() + args.first_frame_timeout
        while (
            not all(samples.header_ns for samples in node.samples.values())
            and time.monotonic() < first_frame_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        missing_topics = [
            topic
            for topic, samples in node.samples.items()
            if not samples.header_ns
        ]
        if missing_topics:
            print(
                "CAMERA_FIRST_FRAME_TIMEOUT missing="
                + ",".join(missing_topics)
            )
            return 4

        warmup_deadline = time.monotonic() + args.warmup
        while time.monotonic() < warmup_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        for samples in node.samples.values():
            samples.header_ns.clear()
            samples.received_monotonic.clear()

        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    failed = False
    for topic, samples in node.samples.items():
        result = summarize(samples)
        print(
            f"{topic}: frames={result['frames']} "
            f"header_fps={result['header_fps']:.3f} "
            f"receive_fps={result['receive_fps']:.3f} "
            f"median_period_ms={result['median_period_ms']:.3f} "
            f"maximum_gap_ms={result['maximum_gap_ms']:.3f}"
        )
        if not (
            args.minimum_fps
            <= float(result["header_fps"])
            <= args.maximum_fps
        ):
            failed = True
        if float(result["receive_fps"]) < args.minimum_fps:
            failed = True
        if float(result["maximum_gap_ms"]) > args.maximum_gap_ms:
            failed = True
    if failed:
        print("CAMERA_FPS_CHECK_FAILED")
        return 3
    print("CAMERA_FPS_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
