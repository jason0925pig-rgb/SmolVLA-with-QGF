from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("armstrong_ros2")
@dataclass
class ArmstrongRos2Config(RobotConfig):
    joint_state_topic: str = "/right_arm/joint_states"
    joint_command_topic: str = "/right_arm/teleop_joint_command"
    motion_enabled_topic: str = "/right_arm/motion_enabled"
    gripper_state_topic: str = "/right_arm/executed_gripper_command"
    gripper_contact_topic: str = "/right_arm/gripper_contact"
    gripper_command_topic: str = "/right_arm/gripper_command"
    chest_topic: str = "/camera_chest/color/image_raw/compressed"
    wrist_topic: str = "/camera_wrist/color/image_raw/compressed"
    stop_topic: str = "/teleop/stop_request"
    enable_service: str = "/smolvla/set_enabled"
    image_height: int = 720
    image_width: int = 1280
    state_timeout_seconds: float = 0.30
    camera_timeout_seconds: float = 0.30
    start_action_enabled: bool = False
    require_open_gripper_at_start: bool = True

    # Demonstrated task envelope plus 0.05 rad.  This is deliberately narrower
    # than the robot's broad controller limits and is specific to this dataset.
    task_lower: tuple[float, ...] = (
        -2.834686,
        -1.316365,
        -0.721879,
        -2.381663,
        4.118480,
        -0.769072,
        3.900149,
    )
    task_upper: tuple[float, ...] = (
        -1.356595,
        0.583672,
        0.911651,
        -0.737084,
        5.245894,
        0.146565,
        4.888320,
    )

    # All 50 demonstrated first frames plus 15 degrees, clamped to the task
    # envelope above.  The gripper starts open.
    initial_lower: tuple[float, ...] = (
        -2.834686,
        -0.305685,
        -0.713077,
        -2.381663,
        4.118480,
        -0.769072,
        3.900149,
    )
    initial_upper: tuple[float, ...] = (
        -1.553967,
        0.583672,
        0.533122,
        -1.449920,
        5.177725,
        0.146565,
        4.888320,
    )
    max_target_error_rad: float = 0.50
    small_envelope_overshoot_rad: float = 0.03
    gripper_open_threshold: float = 0.15
    gripper_close_threshold: float = 0.85
    gripper_confirmation_frames: int = 10
    gripper_min_state_dwell_seconds: float = 2.0
    gripper_contact_hold_seconds: float = 3.0

    # Fixed return-pose completion detector. This is the original attended
    # 2026-08-07 screenshot endpoint, using its unrounded robot feedback
    # rather than the two-decimal values shown in the screenshot.
    completion_home_joints: tuple[float, ...] = (
        -2.370767,
        0.275709,
        0.038841,
        -2.055510,
        4.440476,
        -0.314679,
        4.712930,
    )
    completion_departure_threshold_rad: float = 0.40
    completion_return_tolerance_rad: float = 0.2617993877991494  # 15 degrees
    # Zero disables the low-speed dwell: completion fires immediately after
    # the completed grasp/release cycle returns inside the home envelope.
    completion_stable_duration_seconds: float = 0.0
    completion_minimum_episode_seconds: float = 15.0
    completion_maximum_stable_speed_rad_s: float = 0.05
