from guided_action_flow.rewards.libero import LiberoSparseSuccessReward
from guided_action_flow.types import Transition


def test_libero_sparse_success_reads_success_key():
    reward_fn = LiberoSparseSuccessReward()
    transition = Transition({}, None, 0.0, {}, True, {"success": True})

    output = reward_fn(transition)

    assert output.reward == 1.0
    assert output.diagnostics["success"] == 1.0


def test_libero_sparse_success_defaults_to_failure():
    reward_fn = LiberoSparseSuccessReward()
    transition = Transition({}, None, 0.0, {}, True, {})

    output = reward_fn(transition)

    assert output.reward == 0.0
    assert output.diagnostics["success"] == 0.0

