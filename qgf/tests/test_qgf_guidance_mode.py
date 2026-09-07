"""Tests for the guidance_mode ablation switch in q_guided_velocity_smolvla_reverse_time.

What must hold, per the ablation design:
  critic             : byte-identical to the pre-patch behaviour.
  random_matched_norm: same per-sample |grad| as critic mode BEFORE clipping, same
                       post-clip norm, direction ~ orthogonal (|cos| small), drawn from a
                       dedicated generator (global torch RNG state untouched),
                       reproducible for the same seed.
  zero               : critic forward AND backward ran (q_grad_norm_raw_mean > 0, and
                       q_grad_norm_mean equals critic mode's), applied guidance is exactly
                       zero (guided_velocity == velocity_t, q_guidance_norm_mean == 0).
"""
import torch

from guided_action_flow.guidance.qgf import (
    GUIDANCE_MODES,
    QGuidanceConfig,
    q_guided_velocity_smolvla_reverse_time,
    reset_random_generators,
)


class QuadraticCritic:
    """Q(a) = -||a - target||^2 summed; gradient is non-trivial and state-dependent."""

    def __init__(self, target):
        self.target = target

    def __call__(self, *, obs_features, action_chunk, proprio=None, task_features=None):
        del proprio, task_features
        shift = obs_features[:, :1, None] if obs_features.ndim == 2 else 0.0
        diff = action_chunk - (self.target + shift)
        return -(diff.reshape(diff.shape[0], -1) ** 2).sum(dim=-1)


def _inputs(seed=0, batch=3, horizon=50, dim=8):
    g = torch.Generator().manual_seed(seed)
    action_t = torch.randn(batch, horizon, dim, generator=g)
    velocity_t = torch.randn(batch, horizon, dim, generator=g)
    obs = torch.randn(batch, 4, generator=g)
    target = torch.randn(1, horizon, dim, generator=g)
    return action_t, velocity_t, obs, target


def _run(mode, seed=None, clip=1.0, beta=0.5, inputs=None):
    action_t, velocity_t, obs, target = inputs or _inputs()
    cfg = QGuidanceConfig(beta=beta, grad_clip_norm=clip, guidance_mode=mode, random_seed=seed)
    return q_guided_velocity_smolvla_reverse_time(
        critic=QuadraticCritic(target),
        obs_features=obs,
        action_t=action_t,
        velocity_t=velocity_t,
        time_t=0.3,
        config=cfg,
    )


def test_default_mode_is_critic_and_matches_explicit_critic():
    inputs = _inputs()
    v_default, d_default = q_guided_velocity_smolvla_reverse_time(
        critic=QuadraticCritic(inputs[3]),
        obs_features=inputs[2],
        action_t=inputs[0],
        velocity_t=inputs[1],
        time_t=0.3,
        config=QGuidanceConfig(beta=0.5, grad_clip_norm=1.0),
    )
    v_critic, d_critic = _run("critic", inputs=inputs)
    assert torch.equal(v_default, v_critic)
    assert float(d_critic["q_direction_cos_mean"]) == 1.0
    assert float(d_critic["q_guidance_mode"]) == float(GUIDANCE_MODES.index("critic"))
    assert float(d_default["q_guidance_norm_mean"]) > 0.0


def test_zero_mode_runs_critic_but_applies_no_guidance():
    inputs = _inputs()
    v_critic, d_critic = _run("critic", inputs=inputs)
    v_zero, d_zero = _run("zero", inputs=inputs)
    # the backward ran and is logged at its true, non-zero value
    assert float(d_zero["q_grad_norm_raw_mean"]) > 0.0
    assert torch.allclose(d_zero["q_grad_norm_raw_mean"], d_critic["q_grad_norm_raw_mean"])
    assert torch.allclose(d_zero["q_grad_norm_mean"], d_critic["q_grad_norm_mean"])
    assert torch.allclose(d_zero["q_value_mean"], d_critic["q_value_mean"])
    # but nothing was applied
    assert torch.equal(v_zero, inputs[1])
    assert float(d_zero["q_guidance_norm_mean"]) == 0.0
    assert float(d_zero["q_direction_cos_mean"]) == 0.0
    assert not torch.equal(v_zero, v_critic)


def test_random_matched_norm_keeps_magnitude_and_randomises_direction():
    reset_random_generators()
    inputs = _inputs()
    v_critic, d_critic = _run("critic", inputs=inputs)
    v_rand, d_rand = _run("random_matched_norm", seed=123, inputs=inputs)
    # raw norm is the REAL gradient's norm (logged before the swap)
    assert torch.allclose(d_rand["q_grad_norm_raw_mean"], d_critic["q_grad_norm_raw_mean"])
    # post-clip norm equals what the real gradient would have had
    assert torch.allclose(d_rand["q_grad_norm_mean"], d_critic["q_grad_norm_mean"], rtol=1e-4)
    # and so does the applied guidance magnitude
    assert torch.allclose(d_rand["q_guidance_norm_mean"], d_critic["q_guidance_norm_mean"], rtol=1e-4)
    # direction is random: |cos| small in a 400-dim space
    assert abs(float(d_rand["q_direction_cos_mean"])) < 0.25
    assert not torch.allclose(v_rand, v_critic)
    assert float(d_rand["q_guidance_mode"]) == float(GUIDANCE_MODES.index("random_matched_norm"))


def test_random_matched_norm_per_sample_norms_match_exactly():
    reset_random_generators()
    inputs = _inputs(batch=5)
    action_t, velocity_t, obs, target = inputs
    # no clipping so the applied guidance norm equals |grad|/beta exactly per sample
    _, d_c = _run("critic", clip=None, inputs=inputs)
    _, d_r = _run("random_matched_norm", seed=7, clip=None, inputs=inputs)
    assert torch.allclose(d_r["q_guidance_norm_mean"], d_c["q_guidance_norm_mean"], rtol=1e-5)


def test_random_matched_norm_is_reproducible_and_seed_dependent():
    inputs = _inputs()
    reset_random_generators()
    v1, _ = _run("random_matched_norm", seed=42, inputs=inputs)
    reset_random_generators()
    v2, _ = _run("random_matched_norm", seed=42, inputs=inputs)
    reset_random_generators()
    v3, _ = _run("random_matched_norm", seed=43, inputs=inputs)
    assert torch.equal(v1, v2)
    assert not torch.equal(v1, v3)


def test_random_matched_norm_does_not_touch_global_rng():
    reset_random_generators()
    inputs = _inputs()
    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    _run("random_matched_norm", seed=5, inputs=inputs)
    after = torch.get_rng_state()
    assert torch.equal(before, after), "guidance consumed the global RNG stream"
    # and the policy's own next draw is unchanged by having run the arm
    torch.manual_seed(999)
    a = torch.randn(4)
    torch.manual_seed(999)
    _run("random_matched_norm", seed=5, inputs=inputs)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_random_matched_norm_requires_seed():
    inputs = _inputs()
    try:
        _run("random_matched_norm", seed=None, inputs=inputs)
    except ValueError as exc:
        assert "random_seed" in str(exc)
    else:
        raise AssertionError("random_matched_norm without a seed must be refused")


def test_unknown_mode_is_refused():
    inputs = _inputs()
    try:
        _run("best_of_n", inputs=inputs)
    except ValueError as exc:
        assert "guidance_mode" in str(exc)
    else:
        raise AssertionError("unknown guidance_mode must be refused")


def test_clipping_still_exercised_in_all_modes():
    # a critic with a huge gradient so the clip at 1.0 is active
    inputs = _inputs()
    action_t, velocity_t, obs, target = inputs
    big_target = target * 1000.0
    outs = {}
    for mode, seed in (("critic", None), ("random_matched_norm", 3), ("zero", None)):
        cfg = QGuidanceConfig(beta=0.5, grad_clip_norm=1.0, guidance_mode=mode, random_seed=seed)
        _, d = q_guided_velocity_smolvla_reverse_time(
            critic=QuadraticCritic(big_target), obs_features=obs, action_t=action_t,
            velocity_t=velocity_t, time_t=0.3, config=cfg,
        )
        outs[mode] = d
    for mode in outs:
        assert float(outs[mode]["q_grad_norm_raw_mean"]) > 1.0          # clip was needed
        assert abs(float(outs[mode]["q_grad_norm_mean"]) - 1.0) < 1e-4  # and applied
