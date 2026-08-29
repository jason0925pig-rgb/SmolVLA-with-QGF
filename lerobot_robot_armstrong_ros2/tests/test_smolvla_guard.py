import math
from types import SimpleNamespace
import unittest

from lerobot_robot_armstrong_ros2.armstrongros2 import ArmstrongRos2, JOINT_NAMES
from lerobot_robot_armstrong_ros2.smolvla_guard import (
    GripperTemporalConfig,
    GripperTemporalFilter,
    PolicySafetyConfig,
    PolicySafetyError,
    TaskCompletionConfig,
    TaskCompletionDetector,
    canonicalize_joints_for_envelope,
    controller_targets_near_measured_joints,
    guard_policy_action,
    validate_initial_pose,
)


def config() -> PolicySafetyConfig:
    return PolicySafetyConfig(
        task_lower=(-1.0,) * 7,
        task_upper=(1.0,) * 7,
        initial_lower=(-0.2,) * 7,
        initial_upper=(0.2,) * 7,
        max_target_error_rad=0.25,
        joint2_max_target_error_rad=0.25,
        small_envelope_overshoot_rad=0.03,
    )


class SmolVLAGuardTests(unittest.TestCase):
    def test_initial_pose_and_open_gripper(self):
        self.assertEqual(validate_initial_pose((0.0,) * 8, config()), (0.0,) * 8)
        with self.assertRaises(PolicySafetyError):
            validate_initial_pose((0.3,) + (0.0,) * 7, config())
        with self.assertRaises(PolicySafetyError):
            validate_initial_pose((0.0,) * 7 + (1.0,), config())

    def test_initial_pose_explicit_twenty_degree_tolerance(self):
        tolerant = PolicySafetyConfig(
            task_lower=(-1.0,) * 7,
            task_upper=(1.0,) * 7,
            initial_lower=(-0.2,) * 7,
            initial_upper=(0.2,) * 7,
            initial_envelope_overshoot_rad=math.radians(20.0),
        )
        # 0.50 rad is outside the base 0.20-rad start envelope but inside the
        # requested additional 20-degree tolerance (effective upper 0.5491).
        accepted = validate_initial_pose((0.50,) + (0.0,) * 7, tolerant)
        self.assertAlmostEqual(accepted[0], 0.50)
        with self.assertRaises(PolicySafetyError):
            validate_initial_pose((0.56,) + (0.0,) * 7, tolerant)

    def test_action_is_absolute_and_hysteretic(self):
        joints, gripper = guard_policy_action(
            (0.1,) * 7 + (0.5,),
            (0.0,) * 8,
            True,
            config(),
        )
        self.assertEqual(joints, (0.1,) * 7)
        self.assertTrue(gripper)

    def test_small_envelope_overshoot_is_clamped(self):
        joints, _ = guard_policy_action(
            (1.02,) + (0.0,) * 6 + (0.0,),
            (0.9,) + (0.0,) * 7,
            False,
            config(),
        )
        self.assertEqual(joints[0], 1.0)

    def test_large_overshoot_target_jump_and_nonfinite_are_rejected(self):
        with self.assertRaises(PolicySafetyError):
            guard_policy_action((1.04,) + (0.0,) * 7, (0.9,) + (0.0,) * 7, False, config())
        with self.assertRaises(PolicySafetyError):
            guard_policy_action((0.3,) + (0.0,) * 7, (0.0,) * 8, False, config())
        with self.assertRaises(PolicySafetyError):
            guard_policy_action((math.nan,) + (0.0,) * 7, (0.0,) * 8, False, config())

    def test_joint2_target_error_override_does_not_weaken_other_joints(self):
        limits = PolicySafetyConfig(
            task_lower=(-1.0,) * 7,
            task_upper=(1.0,) * 7,
            initial_lower=(-0.2,) * 7,
            initial_upper=(0.2,) * 7,
            max_target_error_rad=0.50,
            joint2_max_target_error_rad=0.75,
        )
        accepted, _ = guard_policy_action(
            (0.0, 0.70) + (0.0,) * 6,
            (0.0,) * 8,
            False,
            limits,
        )
        self.assertAlmostEqual(accepted[1], 0.70)
        with self.assertRaisesRegex(PolicySafetyError, r"joint 1 target error.*0\.5000"):
            guard_policy_action(
                (0.70,) + (0.0,) * 7,
                (0.0,) * 8,
                False,
                limits,
            )

    def test_raw_policy_input_and_canonical_safety_coordinates_are_separate(self):
        lower = [-1.0] * 7
        upper = [1.0] * 7
        lower[4] = 4.118480
        upper[4] = 5.177725
        wrapped = PolicySafetyConfig(
            task_lower=tuple(lower),
            task_upper=tuple(upper),
            initial_lower=tuple(lower),
            initial_upper=tuple(upper),
            max_target_error_rad=0.50,
            joint2_max_target_error_rad=0.50,
            wraparound_joint_indices=(4,),
        )
        raw_joints = [0.0] * 7
        raw_joints[4] = -1.978191
        canonical = canonicalize_joints_for_envelope(
            raw_joints,
            wrapped.task_lower,
            wrapped.task_upper,
            wrapped.wraparound_joint_indices,
        )
        self.assertAlmostEqual(raw_joints[4], -1.978191, places=6)
        self.assertAlmostEqual(canonical[4], raw_joints[4] + 2.0 * math.pi, places=6)

        # A raw-coordinate checkpoint can emit a negative J5.  The guard maps
        # it into the positive safety envelope, then the controller mapping
        # returns the equivalent target close to the measured raw position.
        predicted = [0.0] * 8
        predicted[4] = -1.93
        guarded, _ = guard_policy_action(
            tuple(predicted), tuple(raw_joints) + (0.0,), False, wrapped
        )
        self.assertAlmostEqual(guarded[4], -1.93 + 2.0 * math.pi, places=6)
        controller = controller_targets_near_measured_joints(
            guarded, raw_joints, wrapped.wraparound_joint_indices
        )
        self.assertAlmostEqual(controller[4], -1.93, places=6)

    def test_only_configured_axes_receive_periodic_conversion(self):
        raw = (-2.43, 0.23, -0.24, -2.11, -1.94, -0.53, 4.62)
        lower = (-2.86, -0.68, -1.26, -2.34, 3.97, -1.15, 4.26)
        upper = (-1.16, 0.65, 0.55, -0.98, 5.09, 0.08, 5.36)
        canonical = canonicalize_joints_for_envelope(
            raw, lower, upper, (0, 2, 4, 6)
        )
        changed = [
            index for index, (before, after) in enumerate(zip(raw, canonical))
            if not math.isclose(before, after, abs_tol=1e-9)
        ]
        self.assertEqual(changed, [4])

    def test_raw_policy_observation_keeps_completion_in_safety_coordinates(self):
        lower = [-3.0] * 7
        upper = [3.0] * 7
        lower[4] = 4.0
        upper[4] = 5.2
        robot = ArmstrongRos2.__new__(ArmstrongRos2)
        robot._connected = True
        robot.config = SimpleNamespace(canonicalize_policy_observation=False)
        robot._guard_config = PolicySafetyConfig(
            task_lower=tuple(lower),
            task_upper=tuple(upper),
            initial_lower=tuple(lower),
            initial_upper=tuple(upper),
            wraparound_joint_indices=(4,),
        )
        raw = (0.0, 0.1, 0.2, 0.3, -1.94, 0.5, 4.6)
        robot._snapshot = lambda: (raw, False, "chest", "wrist")
        completion_inputs = []
        robot._update_task_completion = (
            lambda joints, gripper: completion_inputs.append((joints, gripper))
        )

        observation = ArmstrongRos2.get_observation(robot)
        self.assertAlmostEqual(observation[JOINT_NAMES[4]], raw[4], places=6)
        self.assertAlmostEqual(
            completion_inputs[0][0][4], raw[4] + 2.0 * math.pi, places=6
        )

    def test_gripper_requires_consecutive_decisive_frames(self):
        filter_ = GripperTemporalFilter(
            GripperTemporalConfig(
                confirmation_frames=3,
                min_state_dwell_seconds=0.0,
                contact_hold_seconds=0.0,
            )
        )
        self.assertFalse(filter_.update(0.90, now=0.0, contact_active=False).transitioned)
        # An ambiguous frame clears the candidate sequence.
        self.assertFalse(filter_.update(0.50, now=0.1, contact_active=False).transitioned)
        self.assertFalse(filter_.update(0.90, now=0.2, contact_active=False).transitioned)
        self.assertFalse(filter_.update(0.90, now=0.3, contact_active=False).transitioned)
        result = filter_.update(0.90, now=0.4, contact_active=False)
        self.assertTrue(result.transitioned)
        self.assertTrue(result.command_closed)

    def test_gripper_close_dwell_and_contact_hold_block_reopen(self):
        filter_ = GripperTemporalFilter(
            GripperTemporalConfig(
                confirmation_frames=2,
                min_state_dwell_seconds=1.0,
                contact_hold_seconds=3.0,
            ),
            now=-2.0,
        )
        filter_.update(1.0, now=0.0, contact_active=False)
        self.assertTrue(filter_.update(1.0, now=0.1, contact_active=False).transitioned)
        filter_.note_contact(True, now=0.5)

        filter_.update(0.0, now=1.2, contact_active=True)
        held = filter_.update(0.0, now=1.3, contact_active=True)
        self.assertFalse(held.transitioned)
        self.assertEqual(held.blocked_reason, "contact_hold")

        opened = filter_.update(0.0, now=3.6, contact_active=True)
        self.assertTrue(opened.transitioned)
        self.assertFalse(opened.command_closed)

    def test_completion_requires_depart_grasp_release_and_stable_return(self):
        detector = TaskCompletionDetector(
            TaskCompletionConfig(
                departure_threshold_rad=0.40,
                return_tolerance_rad=0.30,
                stable_duration_seconds=2.0,
                minimum_episode_seconds=3.0,
                maximum_stable_speed_rad_s=0.05,
            )
        )
        detector.reset((0.0,) * 7, initial_gripper_closed=False, now=0.0)
        self.assertFalse(
            detector.update((0.0,) * 7, gripper_closed=False, now=1.0).completed
        )
        departed = detector.update(
            (0.41,) + (0.0,) * 6, gripper_closed=False, now=2.0
        )
        self.assertTrue(departed.departed)
        detector.update((0.41,) + (0.0,) * 6, gripper_closed=True, now=3.0)
        detector.update((0.20,) + (0.0,) * 6, gripper_closed=False, now=4.0)
        self.assertFalse(
            detector.update((0.20,) + (0.0,) * 6, gripper_closed=False, now=5.0).completed
        )
        self.assertFalse(
            detector.update((0.20,) + (0.0,) * 6, gripper_closed=False, now=6.0).completed
        )
        completed = detector.update(
            (0.20,) + (0.0,) * 6, gripper_closed=False, now=7.0
        )
        self.assertTrue(completed.completed)

    def test_completion_does_not_trigger_without_a_grasp_cycle(self):
        detector = TaskCompletionDetector(
            TaskCompletionConfig(
                departure_threshold_rad=0.40,
                return_tolerance_rad=0.30,
                stable_duration_seconds=1.0,
                minimum_episode_seconds=1.0,
                maximum_stable_speed_rad_s=0.05,
            )
        )
        detector.reset((0.0,) * 7, initial_gripper_closed=False, now=0.0)
        detector.update((0.50,) + (0.0,) * 6, gripper_closed=False, now=1.0)
        detector.update((0.20,) + (0.0,) * 6, gripper_closed=False, now=2.0)
        result = detector.update((0.20,) + (0.0,) * 6, gripper_closed=False, now=4.0)
        self.assertFalse(result.completed)
        self.assertFalse(result.saw_close)

    def test_completion_can_fire_immediately_without_low_speed_dwell(self):
        detector = TaskCompletionDetector(
            TaskCompletionConfig(
                departure_threshold_rad=0.40,
                return_tolerance_rad=0.30,
                stable_duration_seconds=0.0,
                minimum_episode_seconds=1.0,
                maximum_stable_speed_rad_s=0.05,
            )
        )
        detector.reset((0.0,) * 7, initial_gripper_closed=False, now=0.0)
        detector.update((0.50,) + (0.0,) * 6, gripper_closed=False, now=1.0)
        detector.update((0.50,) + (0.0,) * 6, gripper_closed=True, now=2.0)
        # The 0.25 rad movement in 0.1 seconds is deliberately much faster
        # than the configured speed threshold. With dwell disabled, entering
        # the return envelope after release completes immediately.
        result = detector.update(
            (0.25,) + (0.0,) * 6, gripper_closed=False, now=2.1
        )
        self.assertTrue(result.completed)


if __name__ == "__main__":
    unittest.main()
