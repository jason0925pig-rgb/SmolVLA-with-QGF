#!/usr/bin/env python3
"""Passively record one real Armstrong rollout for QGF critic training.

The recorder never creates publishers or services.  It samples the latest
robot/policy telemetry at a fixed rate and writes both camera streams while
the SmolVLA action gate is enabled.  A separate finalizer assigns the human
success/failure label and constructs transition rows.
"""

from __future__ import annotations

import argparse
import json
import signal
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory


JOINT_NAMES = tuple(f"right_joint{i}" for i in range(1, 8))
GRIPPER_NAME = "right_gripper_closed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_status(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in value.split(";"):
        key, separator, item = field.partition("=")
        if separator:
            result[key.strip()] = item.strip()
    return result


def ordered_positions(message: JointState, expected: tuple[str, ...]) -> list[float] | None:
    if message.name:
        lookup = dict(zip(message.name, message.position, strict=False))
        if any(name not in lookup for name in expected):
            return None
        return [float(lookup[name]) for name in expected]
    if len(message.position) != len(expected):
        return None
    return [float(value) for value in message.position]


def message_timestamp_ns(message: JointState) -> int:
    stamp = message.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else time.time_ns()


class VideoSink:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.latest_timestamp_ns = 0
        self.latest_frame_index = -1
        # Preserve the exact JPEG byte stream received from ROS in addition
        # to the convenient MP4.  The length-prefixed archive is future-proof
        # input for a frozen SmolVLA/VLM visual encoder and avoids an extra
        # lossy encode/decode when visual_features are computed later.
        self.archive_path = path.with_suffix(".mjpeg")
        self.archive = self.archive_path.open("wb")

    def write(self, message: CompressedImage) -> None:
        encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            return
        stamp = message.header.stamp
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        payload = bytes(message.data)
        self.archive.write(struct.pack("<QI", timestamp_ns, len(payload)))
        self.archive.write(payload)
        height, width = frame.shape[:2]
        if self.writer is None:
            self.width, self.height = width, height
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot open video writer: {self.path}")
        elif (width, height) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.writer.write(frame)
        self.latest_frame_index = self.frame_count
        self.frame_count += 1
        self.latest_timestamp_ns = timestamp_ns

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.archive.close()


class QgfEpisodeRecorder(Node):
    def __init__(self, output_dir: Path, sample_hz: float, camera_fps: float, task: str):
        super().__init__("qgf_episode_recorder")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.raw_path = output_dir / "samples.jsonl"
        self.raw_file = self.raw_path.open("w", encoding="utf-8", buffering=1)
        self.chunk_path = output_dir / "normalized_policy_chunks.jsonl"
        self.chunk_file = self.chunk_path.open("w", encoding="utf-8", buffering=1)
        self.policy_observation_path = output_dir / "policy_observations.jsonl"
        self.policy_observation_file = self.policy_observation_path.open(
            "w", encoding="utf-8", buffering=1
        )
        self.chunk_count = 0
        self.policy_observation_count = 0
        self.chest = VideoSink(output_dir / "chest.mp4", camera_fps)
        self.wrist = VideoSink(output_dir / "wrist_right.mp4", camera_fps)
        self.action_enabled = False
        self.ever_enabled = False
        self.state: list[float] | None = None
        self.state_timestamp_ns = 0
        self.gripper_closed: float | None = None
        self.raw_action: list[float] | None = None
        self.raw_action_timestamp_ns = 0
        self.policy_action_timestep = -1
        self.guarded_action: list[float] | None = None
        self.guarded_action_timestamp_ns = 0
        self.executed_joints: list[float] | None = None
        self.executed_action_timestamp_ns = 0
        self.executed_gripper_closed: float | None = None
        self.gripper_contact = False
        self.sample_count = 0
        self.started_at = utc_now()

        self.create_subscription(JointState, "/right_arm/joint_states", self._state_cb, 20)
        self.create_subscription(
            JointState, "/right_arm/executed_joint_command", self._executed_joint_cb, 20
        )
        self.create_subscription(
            JointState, "/smolvla/raw_policy_action", self._raw_action_cb, 20
        )
        self.create_subscription(
            JointState, "/smolvla/guarded_policy_action", self._guarded_action_cb, 20
        )
        self.create_subscription(Bool, "/right_arm/gripper_state", self._gripper_state_cb, 20)
        self.create_subscription(
            Bool, "/right_arm/executed_gripper_command", self._executed_gripper_cb, 20
        )
        self.create_subscription(Bool, "/right_arm/gripper_contact", self._contact_cb, 20)
        self.create_subscription(String, "/smolvla/status", self._status_cb, 20)
        chunk_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            JointTrajectory,
            "/smolvla/normalized_action_chunk",
            self._normalized_chunk_cb,
            chunk_qos,
        )
        self.create_subscription(
            JointState,
            "/smolvla/policy_observation_state",
            self._policy_observation_cb,
            chunk_qos,
        )
        self.create_subscription(
            CompressedImage,
            "/camera_chest/color/image_raw/compressed",
            self._chest_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CompressedImage,
            "/camera_wrist/color/image_raw/compressed",
            self._wrist_cb,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / sample_hz, self._sample)

    def _state_cb(self, message: JointState) -> None:
        value = ordered_positions(message, JOINT_NAMES)
        if value is not None:
            self.state = value
            self.state_timestamp_ns = message_timestamp_ns(message)

    def _executed_joint_cb(self, message: JointState) -> None:
        value = ordered_positions(message, JOINT_NAMES)
        if value is not None:
            self.executed_joints = value
            self.executed_action_timestamp_ns = message_timestamp_ns(message)

    def _raw_action_cb(self, message: JointState) -> None:
        self.raw_action = ordered_positions(message, (*JOINT_NAMES, GRIPPER_NAME))
        if self.raw_action is not None:
            self.raw_action_timestamp_ns = message_timestamp_ns(message)
            try:
                self.policy_action_timestep = int(message.header.frame_id)
            except ValueError:
                self.policy_action_timestep = -1
            # send_action only publishes this topic after the action gate has
            # opened.  This starts capture immediately instead of waiting up
            # to 0.5 s for the periodic status topic.
            self.action_enabled = True
            self.ever_enabled = True

    def _guarded_action_cb(self, message: JointState) -> None:
        self.guarded_action = ordered_positions(message, (*JOINT_NAMES, GRIPPER_NAME))
        if self.guarded_action is not None:
            self.guarded_action_timestamp_ns = message_timestamp_ns(message)

    def _gripper_state_cb(self, message: Bool) -> None:
        # ROS Bool means requested_open; QGF/LeRobot uses 1=closed.
        self.gripper_closed = float(not message.data)

    def _executed_gripper_cb(self, message: Bool) -> None:
        self.executed_gripper_closed = float(not message.data)

    def _contact_cb(self, message: Bool) -> None:
        self.gripper_contact = bool(message.data)

    def _status_cb(self, message: String) -> None:
        enabled = parse_status(message.data).get("action_enabled") == "1"
        self.action_enabled = enabled
        self.ever_enabled = self.ever_enabled or enabled

    def _normalized_chunk_cb(self, message: JointTrajectory) -> None:
        if tuple(message.joint_names) != (*JOINT_NAMES, GRIPPER_NAME):
            return
        try:
            observation_timestep = int(message.header.frame_id)
        except ValueError:
            return
        chunk = [[float(value) for value in point.positions] for point in message.points]
        if not chunk or any(len(row) != 8 for row in chunk):
            return
        stamp = message.header.stamp
        row = {
            "chunk_index": self.chunk_count,
            "observation_timestep": observation_timestep,
            "timestamp_ns": int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
            "action_chunk_normalized": chunk,
        }
        self.chunk_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.chunk_count += 1

    def _policy_observation_cb(self, message: JointState) -> None:
        state = ordered_positions(message, (*JOINT_NAMES, GRIPPER_NAME))
        if state is None:
            return
        try:
            observation_timestep = int(message.header.frame_id)
        except ValueError:
            return
        row = {
            "observation_timestep": observation_timestep,
            "timestamp_ns": message_timestamp_ns(message),
            "state": state,
            "chest_frame_index": self.chest.latest_frame_index,
            "chest_timestamp_ns": self.chest.latest_timestamp_ns,
            "wrist_frame_index": self.wrist.latest_frame_index,
            "wrist_timestamp_ns": self.wrist.latest_timestamp_ns,
        }
        self.policy_observation_file.write(
            json.dumps(row, separators=(",", ":")) + "\n"
        )
        self.policy_observation_count += 1

    def _chest_cb(self, message: CompressedImage) -> None:
        if self.action_enabled:
            self.chest.write(message)

    def _wrist_cb(self, message: CompressedImage) -> None:
        if self.action_enabled:
            self.wrist.write(message)

    def _sample(self) -> None:
        if not self.action_enabled:
            return
        required = (
            self.state,
            self.gripper_closed,
            self.raw_action,
            self.guarded_action,
            self.executed_joints,
        )
        if any(item is None for item in required):
            return
        if self.chest.latest_frame_index < 0 or self.wrist.latest_frame_index < 0:
            return
        executed_gripper = (
            self.gripper_closed
            if self.executed_gripper_closed is None
            else self.executed_gripper_closed
        )
        row: dict[str, Any] = {
            "sample_index": self.sample_count,
            "timestamp_ns": time.time_ns(),
            "state": [*self.state, self.gripper_closed],
            "state_timestamp_ns": self.state_timestamp_ns,
            "action_policy": self.raw_action,
            "policy_action_timestep": self.policy_action_timestep,
            "action_policy_timestamp_ns": self.raw_action_timestamp_ns,
            "action_guarded": self.guarded_action,
            "action_guarded_timestamp_ns": self.guarded_action_timestamp_ns,
            "action_executed": [*self.executed_joints, executed_gripper],
            "action_executed_timestamp_ns": self.executed_action_timestamp_ns,
            "gripper_contact": self.gripper_contact,
            "chest_frame_index": self.chest.latest_frame_index,
            "chest_timestamp_ns": self.chest.latest_timestamp_ns,
            "wrist_frame_index": self.wrist.latest_frame_index,
            "wrist_timestamp_ns": self.wrist.latest_timestamp_ns,
        }
        self.raw_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.sample_count += 1

    def close(self) -> None:
        self.chest.close()
        self.wrist.close()
        self.raw_file.close()
        self.chunk_file.close()
        self.policy_observation_file.close()
        summary = {
            "schema_version": "qgf-real-rollout-1.0",
            "task": self.task,
            "started_at_utc": self.started_at,
            "finished_at_utc": utc_now(),
            "sample_count": self.sample_count,
            "normalized_policy_chunk_count": self.chunk_count,
            "policy_observation_count": self.policy_observation_count,
            "action_gate_was_enabled": self.ever_enabled,
            "videos": {
                "chest": {
                    "path": "chest.mp4",
                    "exact_jpeg_archive": "chest.mjpeg",
                    "frames": self.chest.frame_count,
                    "width": self.chest.width,
                    "height": self.chest.height,
                    "fps": self.chest.fps,
                },
                "wrist_right": {
                    "path": "wrist_right.mp4",
                    "exact_jpeg_archive": "wrist_right.mjpeg",
                    "frames": self.wrist.frame_count,
                    "width": self.wrist.width,
                    "height": self.wrist.height,
                    "fps": self.wrist.fps,
                },
            },
        }
        (self.output_dir / "capture_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sample-hz", type=float, default=15.0)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.sample_hz <= 0 or args.camera_fps <= 0:
        parser.error("rates must be positive")

    rclpy.init(args=None)
    recorder = QgfEpisodeRecorder(args.output_dir, args.sample_hz, args.camera_fps, args.task)
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while rclpy.ok() and not stopping:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    finally:
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
