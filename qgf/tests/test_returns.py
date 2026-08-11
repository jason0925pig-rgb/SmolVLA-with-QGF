from guided_action_flow.training.returns import discounted_returns


def test_discounted_returns_reset_at_done():
    returns = discounted_returns([1.0, 1.0, 1.0], [False, True, False], gamma=0.9)

    assert returns == [1.9, 1.0, 1.0]

