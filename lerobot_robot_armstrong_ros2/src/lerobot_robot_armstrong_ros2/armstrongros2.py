from __future__ import annotations

import math
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from lerobot.robots.robot import Robot

from lerobot_robot_armstrong_ros2.smolvla_guard import (
    GripperTemporalConfig,
    GripperTemporalFilter,
    PolicySafetyConfig,
    PolicySafetyError,
    TaskCompletionConfig,
    TaskCompletionDetector,
    guard_policy_action,
    validate_initial_pose,
)

from .configuration_armstrong_ros2 import ArmstrongRos2Config


JOINT_NAMES = tuple(f"right_joint{index}" for index in range(1, 8))
GRIPPER_NAME = "right_gripper_closed"


def ros_requested_open_to_lerobot_closed(requested_open: bool) -> bool:
    return not bool(requested_open)


def lerobot_closed_to_ros_requested_open(closed: bool) -> bool:
    return not bool(closed)


class ArmstrongRos2(Robot):
    """LeRobot hardware facade over the existing safe ROS 2 controllers.

    Connecting this adapter is observation-only.  ``send_action`` publishes
    nothing until ``/smolvla/set_enabled`` succeeds.  The underlying arm node
    retains its independent servo-mode gate, limits, slew limiter and watchdog.
    """

    config_class = ArmstrongRos2Config
    name = "armstrong_ros2"

    def __init__(self, config: ArmstrongRos2Config):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._action_enabled = False
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._joint_state: tuple[float, ...] | None = None
        self._joint_time = 0.0
        self._arm_motion_enabled = False
        self._chest: np.ndarray | None = None
        self._chest_time = 0.0
        self._wrist: np.ndarray | None = None
        self._wrist_time = 0.0
        self._gripper_closed = False
        self._gripper_contact = False
        self._last_gripper_command: bool | None = None
        self._model_gripper_command_count = 0
        self._model_gripper_open_count = 0
        self._model_gripper_close_count = 0
        self._last_safe_joint_command: tuple[float, ...] | None = None
        self._policy_queue_size = 0
        self._policy_expected_chunk_size = 0
        self._policy_chunk_ready = False
        self._completion_result = None
        self._completion_stop_published = False
        self._completion_initial_joints: tuple[float, ...] | None = None
        self._completion_home_joints = tuple(config.completion_home_joints)
        self._rollout_end_logged = False
        self._episode_reset_callback = None
        self._latest_policy_observation_timestep = -1
        self._current_policy_action_timestep = -1
        self._owns_rclpy_context = False
        self._guard_config = PolicySafetyConfig(
            task_lower=tuple(config.task_lower),
            task_upper=tuple(config.task_upper),
            initial_lower=tuple(config.initial_lower),
            initial_upper=tuple(config.initial_upper),
            max_target_error_rad=config.max_target_error_rad,
            small_envelope_overshoot_rad=config.small_envelope_overshoot_rad,
            gripper_open_threshold=config.gripper_open_threshold,
            gripper_close_threshold=config.gripper_close_threshold,
        )
        self._gripper_filter = GripperTemporalFilter(
            GripperTemporalConfig(
                open_threshold=config.gripper_open_threshold,
                close_threshold=config.gripper_close_threshold,
                confirmation_frames=config.gripper_confirmation_frames,
                min_state_dwell_seconds=config.gripper_min_state_dwell_seconds,
                contact_hold_seconds=config.gripper_contact_hold_seconds,
            ),
            initial_closed=False,
            now=time.monotonic(),
        )
        self._completion_detector = TaskCompletionDetector(
            TaskCompletionConfig(
                departure_threshold_rad=config.completion_departure_threshold_rad,
                return_tolerance_rad=config.completion_return_tolerance_rad,
                stable_duration_seconds=config.completion_stable_duration_seconds,
                minimum_episode_seconds=config.completion_minimum_episode_seconds,
                maximum_stable_speed_rad_s=config.completion_maximum_stable_speed_rad_s,
            )
        )

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {
            **{name: float for name in JOINT_NAMES},
            GRIPPER_NAME: float,
            "chest": (self.config.image_height, self.config.image_width, 3),
            "wrist_right": (self.config.image_height, self.config.image_width, 3),
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {
            **{name: float for name in JOINT_NAMES},
            GRIPPER_NAME: float,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_enabled(self) -> bool:
        return self._action_enabled

    @property
    def task_completed(self) -> bool:
        with self._lock:
            return bool(
                self._completion_result is not None
                and self._completion_result.completed
            )

    def update_policy_queue_state(
        self,
        queue_size: int,
        expected_chunk_size: int,
        chunk_ready: bool,
    ) -> None:
        """Expose async-policy preload state through ``/smolvla/status``."""
        with self._lock:
            self._policy_queue_size = max(0, int(queue_size))
            self._policy_expected_chunk_size = max(0, int(expected_chunk_size))
            self._policy_chunk_ready = bool(chunk_ready)

    def calibrate(self) -> None:
        return None

    def set_episode_reset_callback(self, callback: Any) -> None:
        self._episode_reset_callback = callback

    def set_policy_action_timestep(self, timestep: int) -> None:
        self._current_policy_action_timestep = int(timestep)

    def _reset_episode_service(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self._action_enabled:
            response.success = False
            response.message = "disable the policy action gate before resetting an episode"
            return response
        if self._episode_reset_callback is None:
            response.success = False
            response.message = "async policy queue reset callback is not registered"
            return response
        try:
            self._episode_reset_callback()
        except Exception as exc:
            response.success = False
            response.message = f"episode queue reset failed: {exc}"
            return response
        response.success = True
        response.message = "policy action queue cleared for the next episode"
        return response

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected:
            return
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self._node = Node("armstrong_lerobot_client")
        self._joint_pub = self._node.create_publisher(
            JointState, self.config.joint_command_topic, 10
        )
        self._gripper_pub = self._node.create_publisher(
            Bool, self.config.gripper_command_topic, 10
        )
        self._stop_pub = self._node.create_publisher(Bool, self.config.stop_topic, 10)
        self._status_pub = self._node.create_publisher(String, "/smolvla/status", 10)
        # Dataset-only telemetry.  These topics are observations of what the
        # policy requested and what passed the safety guard; they are never
        # consumed by the robot controllers and therefore cannot command
        # motion.  A QGF rollout recorder uses them to reconstruct
        # (state, action, next_state, reward) transitions.
        self._raw_policy_action_pub = self._node.create_publisher(
            JointState, "/smolvla/raw_policy_action", 10
        )
        self._guarded_policy_action_pub = self._node.create_publisher(
            JointState, "/smolvla/guarded_policy_action", 10
        )
        telemetry_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._policy_observation_pub = self._node.create_publisher(
            JointState, "/smolvla/policy_observation_state", telemetry_qos
        )
        self._node.create_subscription(
            JointState, self.config.joint_state_topic, self._joint_callback, 10
        )
        self._node.create_subscription(
            Bool,
            self.config.motion_enabled_topic,
            self._motion_enabled_callback,
            10,
        )
        self._node.create_subscription(
            Bool, self.config.gripper_state_topic, self._gripper_callback, 10
        )
        self._node.create_subscription(
            Bool, self.config.gripper_contact_topic, self._gripper_contact_callback, 10
        )
        self._node.create_subscription(
            CompressedImage,
            self.config.chest_topic,
            self._chest_callback,
            qos_profile_sensor_data,
        )
        self._node.create_subscription(
            CompressedImage,
            self.config.wrist_topic,
            self._wrist_callback,
            qos_profile_sensor_data,
        )
        self._node.create_service(SetBool, self.config.enable_service, self._set_enabled)
        self._node.create_service(
            Trigger, "/smolvla/reset_episode", self._reset_episode_service
        )
        self._node.create_timer(0.5, self._publish_status)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        self._connected = True
        if self.config.start_action_enabled:
            raise RuntimeError(
                "start_action_enabled=true is intentionally unsupported; arm through the ROS service"
            )
        self._node.get_logger().warn(
            "Armstrong LeRobot adapter connected in observation-only mode; no action is published"
        )

    def _decode_image(self, message: CompressedImage) -> np.ndarray:
        encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG image could not be decoded")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.config.image_height, self.config.image_width):
            rgb = cv2.resize(
                rgb,
                (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _joint_callback(self, message: JointState) -> None:
        if len(message.position) != 7:
            return
        if message.name:
            lookup = dict(zip(message.name, message.position, strict=False))
            if any(name not in lookup for name in JOINT_NAMES):
                return
            ordered = tuple(float(lookup[name]) for name in JOINT_NAMES)
        else:
            ordered = tuple(float(value) for value in message.position)
        with self._lock:
            self._joint_state = ordered
            self._joint_time = time.monotonic()

    def _gripper_callback(self, message: Bool) -> None:
        # The safe gripper controller Bool means ``requested_open``. LeRobot
        # uses the opposite convention for this feature: 0=open, 1=closed.
        with self._lock:
            self._gripper_closed = ros_requested_open_to_lerobot_closed(
                message.data
            )

    def _gripper_contact_callback(self, message: Bool) -> None:
        with self._lock:
            self._gripper_contact = bool(message.data)

    def _motion_enabled_callback(self, message: Bool) -> None:
        publish_stop = False
        endpoint = None
        with self._lock:
            self._arm_motion_enabled = bool(message.data)
            if self._action_enabled and not self._arm_motion_enabled:
                self._action_enabled = False
                self._last_safe_joint_command = None
                self._last_gripper_command = None
                self._gripper_filter.reset(
                    initial_closed=self._gripper_closed,
                    now=time.monotonic(),
                )
                if not self._rollout_end_logged and self._joint_state is not None:
                    endpoint = (self._completion_initial_joints, self._joint_state)
                    self._rollout_end_logged = True
                publish_stop = True
        if publish_stop:
            if self._node is not None:
                if endpoint is not None:
                    self._node.get_logger().error(
                        "TELEMETRY_EVENT event=rollout_end "
                        "source=arm_motion_gate_closed "
                        + self._pose_comparison_fields(*endpoint)
                        + " "
                        + self._home_comparison_fields(endpoint[1])
                    )
                self._node.get_logger().error(
                    "TELEMETRY_EVENT event=policy_stop "
                    "source=arm_motion_gate_closed; policy and gripper "
                    "stopped together"
                )
            self._publish_stop()

    def _chest_callback(self, message: CompressedImage) -> None:
        try:
            decoded = self._decode_image(message)
        except Exception as exc:
            if self._node is not None:
                self._node.get_logger().error(f"chest image decode failed: {exc}")
            return
        with self._lock:
            self._chest = decoded
            self._chest_time = time.monotonic()

    def _wrist_callback(self, message: CompressedImage) -> None:
        try:
            decoded = self._decode_image(message)
        except Exception as exc:
            if self._node is not None:
                self._node.get_logger().error(f"wrist image decode failed: {exc}")
            return
        with self._lock:
            self._wrist = decoded
            self._wrist_time = time.monotonic()

    def _snapshot(self) -> tuple[tuple[float, ...], bool, np.ndarray, np.ndarray]:
        now = time.monotonic()
        with self._lock:
            joints = self._joint_state
            gripper_closed = self._gripper_closed
            chest = None if self._chest is None else self._chest.copy()
            wrist = None if self._wrist is None else self._wrist.copy()
            joint_age = now - self._joint_time
            chest_age = now - self._chest_time
            wrist_age = now - self._wrist_time
        if joints is None or joint_age > self.config.state_timeout_seconds:
            raise RuntimeError(f"joint state is missing or stale ({joint_age:.3f}s)")
        if chest is None or chest_age > self.config.camera_timeout_seconds:
            raise RuntimeError(f"chest image is missing or stale ({chest_age:.3f}s)")
        if wrist is None or wrist_age > self.config.camera_timeout_seconds:
            raise RuntimeError(f"wrist image is missing or stale ({wrist_age:.3f}s)")
        return joints, gripper_closed, chest, wrist

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("robot adapter is not connected")
        joints, gripper_closed, chest, wrist = self._snapshot()
        self._update_task_completion(joints, gripper_closed)
        return {
            **dict(zip(JOINT_NAMES, joints, strict=True)),
            GRIPPER_NAME: float(gripper_closed),
            "chest": chest,
            "wrist_right": wrist,
        }

    def _set_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        if not request.data:
            endpoint = None
            with self._lock:
                if (
                    self._action_enabled
                    and not self._rollout_end_logged
                    and self._joint_state is not None
                ):
                    endpoint = (self._completion_initial_joints, self._joint_state)
                    self._rollout_end_logged = True
                self._action_enabled = False
                self._last_safe_joint_command = None
                self._last_gripper_command = None
                self._gripper_filter.reset(
                    initial_closed=self._gripper_closed,
                    now=time.monotonic(),
                )
                self._completion_detector.deactivate()
                self._completion_result = None
                self._completion_stop_published = False
            self._publish_stop()
            if self._node is not None:
                if endpoint is not None:
                    self._node.get_logger().error(
                        "TELEMETRY_EVENT event=rollout_end "
                        "source=set_enabled_false "
                        + self._pose_comparison_fields(*endpoint)
                        + " "
                        + self._home_comparison_fields(endpoint[1])
                    )
                self._node.get_logger().error(
                    "TELEMETRY_EVENT event=policy_stop "
                    "source=set_enabled_false"
                )
            response.success = True
            response.message = "SmolVLA action gate disabled and STOP published"
            return response
        try:
            joints, gripper_closed, _, _ = self._snapshot()
            with self._lock:
                arm_motion_enabled = self._arm_motion_enabled
            if not arm_motion_enabled:
                raise PolicySafetyError("arm servo motion gate is not enabled")
            state = (*joints, float(gripper_closed))
            validate_initial_pose(
                state,
                self._guard_config,
                require_open_gripper=self.config.require_open_gripper_at_start,
            )
            if self._joint_pub.get_subscription_count() != 1:
                raise PolicySafetyError(
                    "expected exactly one safe arm-command subscriber"
                )
            if self._gripper_pub.get_subscription_count() != 1:
                raise PolicySafetyError(
                    "expected exactly one safe gripper-command subscriber"
                )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            return response
        with self._lock:
            self._gripper_filter.reset(
                initial_closed=gripper_closed,
                now=time.monotonic(),
            )
            # Initialization has already opened the follower gripper.  Do not
            # re-publish a binary command until the model earns a confirmed
            # transition through the temporal filter.
            self._last_gripper_command = gripper_closed
            self._completion_detector.reset(
                self._completion_home_joints,
                initial_gripper_closed=gripper_closed,
                now=time.monotonic(),
            )
            self._completion_result = None
            self._completion_stop_published = False
            self._completion_initial_joints = tuple(joints)
            self._rollout_end_logged = False
            self._action_enabled = True
        if self._node is not None:
            self._node.get_logger().warn(
                "TELEMETRY_EVENT event=rollout_start source=policy_gate_enabled "
                f"initial_joints_rad={self._format_joint_values(joints)} "
                f"initial_joints_deg={self._format_joint_values(self._to_degrees(joints))} "
                f"home_joints_rad={self._format_joint_values(self._completion_home_joints)} "
                f"home_joints_deg={self._format_joint_values(self._to_degrees(self._completion_home_joints))} "
                f"departure_threshold_rad={self.config.completion_departure_threshold_rad:.6f} "
                f"return_tolerance_rad={self.config.completion_return_tolerance_rad:.6f} "
                f"return_tolerance_deg={self.config.completion_return_tolerance_rad * 180.0 / math.pi:.6f}"
            )
        response.success = True
        response.message = (
            "SmolVLA action gate enabled; gripper temporal hysteresis active"
        )
        return response

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("robot adapter is not connected")
        if not self._action_enabled:
            return action
        joints, gripper_closed, _, _ = self._snapshot()
        predicted = tuple(float(action[name]) for name in (*JOINT_NAMES, GRIPPER_NAME))
        self._publish_policy_action(self._raw_policy_action_pub, predicted)
        try:
            guarded_joints, _ = guard_policy_action(
                predicted,
                (*joints, float(gripper_closed)),
                gripper_closed,
                self._guard_config,
            )
        except PolicySafetyError as exc:
            self._action_enabled = False
            self._last_safe_joint_command = None
            self._last_gripper_command = None
            self._publish_stop()
            if self._node is not None:
                self._node.get_logger().error(
                    "TELEMETRY_EVENT event=model_safety_guard "
                    f'reason="{exc}" raw_gripper={predicted[-1]:.6f} '
                    f"actual_joints={joints} predicted_joints={predicted[:7]}"
                )
            raise

        now = time.monotonic()
        with self._lock:
            temporal_gripper = self._gripper_filter.update(
                predicted[-1],
                now=now,
                contact_active=self._gripper_contact,
            )
        guarded_gripper = temporal_gripper.command_closed
        self._publish_policy_action(
            self._guarded_policy_action_pub,
            (*guarded_joints, float(guarded_gripper)),
        )

        self._last_safe_joint_command = tuple(guarded_joints)
        self._publish_joint_command(self._last_safe_joint_command)
        if temporal_gripper.transitioned:
            gripper_message = Bool()
            # ROS command=True requests OPEN; guarded_gripper=True means
            # CLOSED in the LeRobot action space, so the value must invert.
            gripper_message.data = lerobot_closed_to_ros_requested_open(guarded_gripper)
            self._gripper_pub.publish(gripper_message)
            self._last_gripper_command = guarded_gripper
            self._model_gripper_command_count += 1
            if guarded_gripper:
                self._model_gripper_close_count += 1
            else:
                self._model_gripper_open_count += 1
            if self._node is not None:
                self._node.get_logger().warn(
                    "TELEMETRY_EVENT event=model_gripper_transition "
                    f"sequence={self._model_gripper_command_count} "
                    f"action={'close' if guarded_gripper else 'open'} "
                    f"raw_model={predicted[-1]:.6f} "
                    f"confirmed_frames={self.config.gripper_confirmation_frames} "
                    f"contact={int(self._gripper_contact)} "
                    f"opens={self._model_gripper_open_count} "
                    f"closes={self._model_gripper_close_count} "
                    f"actual_joints={joints} guarded_joints={guarded_joints}"
                )
        return {
            **dict(zip(JOINT_NAMES, guarded_joints, strict=True)),
            GRIPPER_NAME: float(guarded_gripper),
        }

    def _publish_policy_action(
        self,
        publisher: Any,
        values: tuple[float, ...],
    ) -> None:
        """Publish one eight-dimensional policy action for passive recording."""
        if self._node is None:
            return
        message = JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = str(self._current_policy_action_timestep)
        message.name = [*JOINT_NAMES, GRIPPER_NAME]
        message.position = [float(value) for value in values]
        publisher.publish(message)

    def publish_policy_observation(
        self, timestep: int, observation: dict[str, Any]
    ) -> None:
        """Publish the exact proprioceptive state paired with a policy request."""
        if self._node is None:
            return
        try:
            values = [
                float(observation[name]) for name in (*JOINT_NAMES, GRIPPER_NAME)
            ]
        except (KeyError, TypeError, ValueError):
            return
        message = JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = str(int(timestep))
        message.name = [*JOINT_NAMES, GRIPPER_NAME]
        message.position = values
        self._policy_observation_pub.publish(message)
        self._latest_policy_observation_timestep = int(timestep)

    def _update_task_completion(
        self,
        joints: tuple[float, ...],
        gripper_closed: bool,
    ) -> None:
        should_stop = False
        result = None
        with self._lock:
            if self._action_enabled and self._completion_detector.active:
                result = self._completion_detector.update(
                    joints,
                    gripper_closed=gripper_closed,
                    now=time.monotonic(),
                )
                self._completion_result = result
                if result.completed and not self._completion_stop_published:
                    self._completion_stop_published = True
                    self._action_enabled = False
                    self._last_safe_joint_command = None
                    self._last_gripper_command = None
                    self._rollout_end_logged = True
                    should_stop = True
        if should_stop:
            if self._node is not None:
                self._node.get_logger().warn(
                    "TELEMETRY_EVENT event=task_complete "
                    "source=returned_home "
                    f"max_home_error_rad={result.maximum_start_error_rad:.6f} "
                    f"max_speed_rad_s={result.maximum_speed_rad_s:.6f} "
                    f"stable_seconds={result.stable_seconds:.3f} "
                    + self._pose_comparison_fields(
                        self._completion_initial_joints, joints
                    )
                    + " "
                    + self._home_comparison_fields(joints)
                )
            self._publish_stop()

    @staticmethod
    def _format_joint_values(values: tuple[float, ...]) -> str:
        return "[" + ",".join(f"{value:.6f}" for value in values) + "]"

    @staticmethod
    def _to_degrees(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(value * 180.0 / math.pi for value in values)

    @classmethod
    def _pose_comparison_fields(
        cls,
        initial_joints: tuple[float, ...] | None,
        actual_joints: tuple[float, ...],
    ) -> str:
        if initial_joints is None:
            return (
                "initial_joints_rad=unavailable "
                f"actual_joints_rad={cls._format_joint_values(actual_joints)}"
            )
        deltas = tuple(
            actual - initial
            for actual, initial in zip(actual_joints, initial_joints, strict=True)
        )
        return (
            f"initial_joints_rad={cls._format_joint_values(initial_joints)} "
            f"actual_joints_rad={cls._format_joint_values(actual_joints)} "
            f"delta_joints_rad={cls._format_joint_values(deltas)} "
            f"delta_joints_deg={cls._format_joint_values(cls._to_degrees(deltas))}"
        )

    def _home_comparison_fields(self, actual_joints: tuple[float, ...]) -> str:
        deltas = tuple(
            actual - home
            for actual, home in zip(
                actual_joints, self._completion_home_joints, strict=True
            )
        )
        return (
            f"home_joints_rad={self._format_joint_values(self._completion_home_joints)} "
            f"home_delta_joints_rad={self._format_joint_values(deltas)} "
            f"home_delta_joints_deg={self._format_joint_values(self._to_degrees(deltas))}"
        )

    def _publish_joint_command(self, joints: tuple[float, ...]) -> None:
        message = JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = list(joints)
        self._joint_pub.publish(message)

    def resend_last_joint_command(self) -> bool:
        """Keep the last guarded target alive while the next chunk is inferred."""
        if not self._connected or not self._action_enabled:
            return False
        command = self._last_safe_joint_command
        if command is None:
            return False
        self._publish_joint_command(command)
        return True

    def _publish_stop(self) -> None:
        if self._node is None:
            return
        message = Bool()
        message.data = True
        for _ in range(3):
            self._stop_pub.publish(message)

    def _publish_status(self) -> None:
        if self._node is None:
            return
        now = time.monotonic()
        with self._lock:
            joint_present = self._joint_state is not None
            chest_present = self._chest is not None
            wrist_present = self._wrist is not None
            joint_age = now - self._joint_time if joint_present else -1.0
            chest_age = now - self._chest_time if chest_present else -1.0
            wrist_age = now - self._wrist_time if wrist_present else -1.0
            policy_queue_size = self._policy_queue_size
            policy_expected_chunk_size = self._policy_expected_chunk_size
            policy_chunk_ready = self._policy_chunk_ready
            arm_motion_enabled = self._arm_motion_enabled
            gripper_contact = self._gripper_contact
            gripper_candidate = self._gripper_filter.candidate_closed
            gripper_candidate_count = self._gripper_filter.candidate_count
            gripper_command_closed = self._gripper_filter.command_closed
            completion_result = self._completion_result
            completion_active = self._completion_detector.active
        task_completed = bool(completion_result and completion_result.completed)
        completion_departed = bool(completion_result and completion_result.departed)
        completion_saw_close = bool(completion_result and completion_result.saw_close)
        completion_saw_release = bool(completion_result and completion_result.saw_release)
        completion_home = bool(completion_result and completion_result.inside_return_envelope)
        completion_stable_seconds = (
            0.0 if completion_result is None else completion_result.stable_seconds
        )
        observation_ready = (
            joint_present
            and chest_present
            and wrist_present
            and joint_age <= self.config.state_timeout_seconds
            and chest_age <= self.config.camera_timeout_seconds
            and wrist_age <= self.config.camera_timeout_seconds
        )
        message = String()
        message.data = (
            f"connected={int(self._connected)};"
            f"action_enabled={int(self._action_enabled)};"
            f"arm_motion_enabled={int(arm_motion_enabled)};"
            f"observation_ready={int(observation_ready)};"
            f"policy_chunk_ready={int(policy_chunk_ready)};"
            f"action_queue_size={policy_queue_size};"
            f"expected_action_chunk_size={policy_expected_chunk_size};"
            f"gripper_command_closed={int(gripper_command_closed)};"
            f"gripper_contact={int(gripper_contact)};"
            f"gripper_candidate={'none' if gripper_candidate is None else int(gripper_candidate)};"
            f"gripper_candidate_count={gripper_candidate_count};"
            f"completion_active={int(completion_active)};"
            f"completion_departed={int(completion_departed)};"
            f"completion_saw_close={int(completion_saw_close)};"
            f"completion_saw_release={int(completion_saw_release)};"
            f"completion_home={int(completion_home)};"
            f"completion_stable_s={completion_stable_seconds:.3f};"
            f"task_completed={int(task_completed)};"
            f"joint_age_s={joint_age:.3f};"
            f"chest_age_s={chest_age:.3f};"
            f"wrist_age_s={wrist_age:.3f}"
        )
        self._status_pub.publish(message)

    def disconnect(self) -> None:
        if not self._connected:
            return
        with self._lock:
            self._action_enabled = False
            self._last_safe_joint_command = None
            self._last_gripper_command = None
            self._gripper_filter.reset(
                initial_closed=self._gripper_closed,
                now=time.monotonic(),
            )
            self._completion_detector.deactivate()
        self._publish_stop()
        self._connected = False
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()
