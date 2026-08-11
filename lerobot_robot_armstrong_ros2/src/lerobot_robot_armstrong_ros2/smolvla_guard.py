"""Pure safety checks for one-arm SmolVLA actions.

The functions in this module are ROS-independent so they can be unit tested
without robot hardware.  Passing these checks is necessary but never replaces
the controller's joint limits, slew limiter, watchdog, or physical E-stop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


JOINT_COUNT = 7
ACTION_DIM = 8


class PolicySafetyError(ValueError):
    """Raised when an observation or predicted action is unsafe to execute."""


@dataclass(frozen=True)
class PolicySafetyConfig:
    task_lower: tuple[float, ...]
    task_upper: tuple[float, ...]
    initial_lower: tuple[float, ...]
    initial_upper: tuple[float, ...]
    max_target_error_rad: float = 0.25
    small_envelope_overshoot_rad: float = 0.03
    gripper_open_threshold: float = 0.15
    gripper_close_threshold: float = 0.85

    def __post_init__(self) -> None:
        for name in ("task_lower", "task_upper", "initial_lower", "initial_upper"):
            values = getattr(self, name)
            if len(values) != JOINT_COUNT or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain seven finite values")
        for lower, upper in zip(self.task_lower, self.task_upper, strict=True):
            if lower >= upper:
                raise ValueError("each task lower limit must be below its upper limit")
        for lower, upper in zip(self.initial_lower, self.initial_upper, strict=True):
            if lower >= upper:
                raise ValueError("each initial lower limit must be below its upper limit")
        if not math.isfinite(self.max_target_error_rad) or self.max_target_error_rad <= 0:
            raise ValueError("max_target_error_rad must be positive")
        if self.small_envelope_overshoot_rad < 0:
            raise ValueError("small_envelope_overshoot_rad cannot be negative")
        if not 0 <= self.gripper_open_threshold < self.gripper_close_threshold <= 1:
            raise ValueError("gripper thresholds must satisfy 0 <= open < close <= 1")


@dataclass(frozen=True)
class GripperTemporalConfig:
    """Temporal hysteresis for model gripper commands.

    Thresholds classify only decisive model outputs.  A state change then has
    to remain decisive for ``confirmation_frames`` consecutive actions.  The
    dwell and contact holds prevent adjacent policy chunks from immediately
    undoing a close command.
    """

    open_threshold: float = 0.15
    close_threshold: float = 0.85
    confirmation_frames: int = 10
    min_state_dwell_seconds: float = 2.0
    contact_hold_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not 0 <= self.open_threshold < self.close_threshold <= 1:
            raise ValueError("gripper thresholds must satisfy 0 <= open < close <= 1")
        if self.confirmation_frames <= 0:
            raise ValueError("confirmation_frames must be positive")
        if not math.isfinite(self.min_state_dwell_seconds) or self.min_state_dwell_seconds < 0:
            raise ValueError("min_state_dwell_seconds cannot be negative")
        if not math.isfinite(self.contact_hold_seconds) or self.contact_hold_seconds < 0:
            raise ValueError("contact_hold_seconds cannot be negative")


@dataclass(frozen=True)
class GripperTemporalResult:
    command_closed: bool
    transitioned: bool
    candidate_closed: bool | None
    candidate_count: int
    blocked_reason: str | None


class GripperTemporalFilter:
    """Stateful, time-aware filter for binary gripper commands."""

    def __init__(
        self,
        config: GripperTemporalConfig,
        *,
        initial_closed: bool = False,
        now: float = 0.0,
    ) -> None:
        self.config = config
        self.reset(initial_closed=initial_closed, now=now)

    def reset(self, *, initial_closed: bool, now: float) -> None:
        self.command_closed = bool(initial_closed)
        self.candidate_closed: bool | None = None
        self.candidate_count = 0
        self.last_transition_time = float(now)
        self.contact_started_time: float | None = None
        self.contact_active = False

    def note_contact(self, active: bool, *, now: float) -> None:
        active = bool(active)
        if active and not self.contact_active:
            self.contact_started_time = float(now)
        elif not active:
            self.contact_started_time = None
        self.contact_active = active

    def update(
        self,
        raw_value: float,
        *,
        now: float,
        contact_active: bool,
    ) -> GripperTemporalResult:
        raw_value = float(raw_value)
        now = float(now)
        if not math.isfinite(raw_value) or not math.isfinite(now):
            raise PolicySafetyError("gripper command/time contains NaN or infinity")
        self.note_contact(contact_active, now=now)

        candidate: bool | None
        if raw_value <= self.config.open_threshold:
            candidate = False
        elif raw_value >= self.config.close_threshold:
            candidate = True
        else:
            candidate = None

        if candidate is None or candidate == self.command_closed:
            self.candidate_closed = None
            self.candidate_count = 0
            return GripperTemporalResult(
                self.command_closed, False, None, 0, None
            )

        if candidate == self.candidate_closed:
            self.candidate_count += 1
        else:
            self.candidate_closed = candidate
            self.candidate_count = 1

        if self.candidate_count < self.config.confirmation_frames:
            return GripperTemporalResult(
                self.command_closed,
                False,
                self.candidate_closed,
                self.candidate_count,
                "confirmation",
            )

        elapsed = max(0.0, now - self.last_transition_time)
        if elapsed < self.config.min_state_dwell_seconds:
            return GripperTemporalResult(
                self.command_closed,
                False,
                self.candidate_closed,
                self.candidate_count,
                "minimum_dwell",
            )

        if (
            candidate is False
            and self.contact_started_time is not None
            and now - self.contact_started_time < self.config.contact_hold_seconds
        ):
            return GripperTemporalResult(
                self.command_closed,
                False,
                self.candidate_closed,
                self.candidate_count,
                "contact_hold",
            )

        self.command_closed = candidate
        self.last_transition_time = now
        self.candidate_closed = None
        self.candidate_count = 0
        return GripperTemporalResult(
            self.command_closed, True, None, 0, None
        )


@dataclass(frozen=True)
class TaskCompletionConfig:
    """Conditions for declaring a learned rollout complete.

    The policy remains responsible for returning the arm.  This detector only
    closes the action gate after a complete grasp/release cycle and a return
    near the pose captured when model control was enabled. A nonzero stable
    duration can optionally require a low-speed dwell.
    """

    departure_threshold_rad: float = 0.40
    return_tolerance_rad: float = 0.30
    stable_duration_seconds: float = 0.0
    minimum_episode_seconds: float = 15.0
    maximum_stable_speed_rad_s: float = 0.05

    def __post_init__(self) -> None:
        finite_positive = (
            "departure_threshold_rad",
            "return_tolerance_rad",
            "minimum_episode_seconds",
            "maximum_stable_speed_rad_s",
        )
        for name in finite_positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.stable_duration_seconds)
            or self.stable_duration_seconds < 0
        ):
            raise ValueError("stable_duration_seconds must be finite and nonnegative")
        if self.departure_threshold_rad <= self.return_tolerance_rad:
            raise ValueError(
                "departure_threshold_rad must exceed return_tolerance_rad "
                "to provide completion hysteresis"
            )


@dataclass(frozen=True)
class TaskCompletionResult:
    completed: bool
    departed: bool
    saw_close: bool
    saw_release: bool
    inside_return_envelope: bool
    stable_seconds: float
    maximum_start_error_rad: float
    maximum_speed_rad_s: float


class TaskCompletionDetector:
    """Recognize leave, grasp, release, and stable-return in that order."""

    def __init__(self, config: TaskCompletionConfig) -> None:
        self.config = config
        self.active = False
        self.completed = False

    def reset(
        self,
        initial_joints: Sequence[float],
        *,
        initial_gripper_closed: bool,
        now: float,
    ) -> None:
        self.initial_joints = _finite_tuple(initial_joints, JOINT_COUNT, "initial joints")
        self.previous_joints = self.initial_joints
        self.previous_time = float(now)
        self.started_time = float(now)
        self.return_stable_since: float | None = None
        self.departed = False
        # A closed initial gripper must not count as the task's grasp event.
        self.saw_close = False
        self.saw_release = False
        self.active = True
        self.completed = False

    def deactivate(self) -> None:
        self.active = False
        self.return_stable_since = None

    def update(
        self,
        joints: Sequence[float],
        *,
        gripper_closed: bool,
        now: float,
    ) -> TaskCompletionResult:
        if not self.active:
            raise RuntimeError("task completion detector is not active")
        checked = _finite_tuple(joints, JOINT_COUNT, "completion joints")
        now = float(now)
        if not math.isfinite(now) or now < self.previous_time:
            raise ValueError("completion time must be finite and monotonic")

        start_errors = tuple(
            abs(value - initial)
            for value, initial in zip(checked, self.initial_joints, strict=True)
        )
        maximum_start_error = max(start_errors)
        if maximum_start_error >= self.config.departure_threshold_rad:
            self.departed = True

        if self.departed and gripper_closed:
            self.saw_close = True
        if self.saw_close and not gripper_closed:
            self.saw_release = True

        dt = now - self.previous_time
        maximum_speed = 0.0
        if dt > 0:
            maximum_speed = max(
                abs(value - previous) / dt
                for value, previous in zip(checked, self.previous_joints, strict=True)
            )

        inside_return_envelope = all(
            error <= self.config.return_tolerance_rad for error in start_errors
        )
        stability_disabled = self.config.stable_duration_seconds == 0
        eligible = (
            self.departed
            and self.saw_close
            and self.saw_release
            and not gripper_closed
            and now - self.started_time >= self.config.minimum_episode_seconds
            and inside_return_envelope
            and (
                stability_disabled
                or maximum_speed <= self.config.maximum_stable_speed_rad_s
            )
        )
        if eligible:
            if self.return_stable_since is None:
                self.return_stable_since = now
        else:
            self.return_stable_since = None

        stable_seconds = (
            0.0
            if self.return_stable_since is None
            else max(0.0, now - self.return_stable_since)
        )
        if eligible and stable_seconds >= self.config.stable_duration_seconds:
            self.completed = True
            self.active = False

        self.previous_joints = checked
        self.previous_time = now
        return TaskCompletionResult(
            completed=self.completed,
            departed=self.departed,
            saw_close=self.saw_close,
            saw_release=self.saw_release,
            inside_return_envelope=inside_return_envelope,
            stable_seconds=stable_seconds,
            maximum_start_error_rad=maximum_start_error,
            maximum_speed_rad_s=maximum_speed,
        )


def _finite_tuple(values: Sequence[float], expected: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected:
        raise PolicySafetyError(f"{name} must contain exactly {expected} values")
    if not all(math.isfinite(value) for value in result):
        raise PolicySafetyError(f"{name} contains NaN or infinity")
    return result


def validate_initial_pose(
    state: Sequence[float],
    config: PolicySafetyConfig,
    *,
    require_open_gripper: bool = True,
) -> tuple[float, ...]:
    """Validate the live 7+1 state before an autonomous rollout is armed."""

    checked = _finite_tuple(state, ACTION_DIM, "state")
    joints = checked[:JOINT_COUNT]
    outside = [
        index + 1
        for index, (value, lower, upper) in enumerate(
            zip(joints, config.initial_lower, config.initial_upper, strict=True)
        )
        if value < lower or value > upper
    ]
    if outside:
        raise PolicySafetyError(
            "initial pose is outside the demonstrated start envelope on joints "
            + ", ".join(map(str, outside))
        )
    if require_open_gripper and checked[-1] >= config.gripper_close_threshold:
        raise PolicySafetyError("initial gripper state is closed; the demonstrations start open")
    return checked


def guard_policy_action(
    predicted_action: Sequence[float],
    current_state: Sequence[float],
    previous_gripper_closed: bool,
    config: PolicySafetyConfig,
) -> tuple[tuple[float, ...], bool]:
    """Reject implausible targets and clamp only tiny task-envelope overshoot.

    The returned joint target remains an absolute target in radians.  A target
    that is far from the current measured pose is rejected rather than slowly
    chasing a potentially nonsensical model output.
    """

    action = _finite_tuple(predicted_action, ACTION_DIM, "predicted action")
    state = _finite_tuple(current_state, ACTION_DIM, "current state")
    guarded: list[float] = []
    for index, (target, current, lower, upper) in enumerate(
        zip(
            action[:JOINT_COUNT],
            state[:JOINT_COUNT],
            config.task_lower,
            config.task_upper,
            strict=True,
        )
    ):
        if target < lower - config.small_envelope_overshoot_rad or target > upper + config.small_envelope_overshoot_rad:
            raise PolicySafetyError(
                f"joint {index + 1} target {target:.4f} is outside the demonstrated task envelope"
            )
        target = min(max(target, lower), upper)
        if abs(target - current) > config.max_target_error_rad:
            raise PolicySafetyError(
                f"joint {index + 1} target error {target - current:+.4f} rad exceeds "
                f"{config.max_target_error_rad:.4f} rad"
            )
        guarded.append(target)

    gripper_value = action[-1]
    if gripper_value <= config.gripper_open_threshold:
        gripper_closed = False
    elif gripper_value >= config.gripper_close_threshold:
        gripper_closed = True
    else:
        gripper_closed = bool(previous_gripper_closed)
    return tuple(guarded), gripper_closed
