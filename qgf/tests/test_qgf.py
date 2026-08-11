import torch

from guided_action_flow.guidance.qgf import (
    QGuidanceConfig,
    estimate_clean_action_smolvla_reverse_time,
    q_guided_velocity_smolvla_reverse_time,
)


class LinearActionCritic:
    def __init__(self, target_direction):
        self.target_direction = target_direction

    def __call__(self, *, obs_features, action_chunk, proprio=None):
        del obs_features, proprio
        return (action_chunk * self.target_direction).flatten(1).sum(dim=-1)


class BiasLinearCritic:
    def __init__(self, bias: float):
        self.bias = bias

    def __call__(self, *, obs_features, action_chunk, proprio=None):
        del obs_features, proprio
        return action_chunk.flatten(1).sum(dim=-1) + self.bias


class TaskFeatureCritic:
    def __call__(self, *, obs_features, action_chunk, proprio=None, task_features=None):
        del obs_features, proprio
        if task_features is None:
            raise AssertionError("task_features were not passed to the critic")
        scale = task_features[:, :1].reshape(action_chunk.shape[0], 1, 1)
        return (action_chunk * scale).flatten(1).sum(dim=-1)


def test_smolvla_reverse_time_clean_action_estimate_recovers_action():
    action = torch.tensor([[[0.2, -0.4], [0.5, 0.1]]])
    noise = torch.tensor([[[1.0, 0.5], [-0.3, 0.7]]])
    time = torch.tensor([0.7])

    x_t = time[:, None, None] * noise + (1.0 - time[:, None, None]) * action
    velocity = noise - action

    clean_action = estimate_clean_action_smolvla_reverse_time(x_t, velocity, time)

    torch.testing.assert_close(clean_action, action)


def test_smolvla_qgf_velocity_sign_increases_clean_action_value():
    action_t = torch.zeros(1, 2, 2)
    velocity_t = torch.zeros_like(action_t)
    time = torch.tensor([0.5])
    target_direction = torch.ones_like(action_t)
    critic = LinearActionCritic(target_direction)
    obs_features = torch.zeros(1, 3)

    guided_velocity, diagnostics = q_guided_velocity_smolvla_reverse_time(
        critic=critic,
        obs_features=obs_features,
        action_t=action_t,
        velocity_t=velocity_t,
        time_t=time,
        config=QGuidanceConfig(beta=2.0, grad_clip_norm=None),
    )

    base_clean = estimate_clean_action_smolvla_reverse_time(action_t, velocity_t, time)
    guided_clean = estimate_clean_action_smolvla_reverse_time(action_t, guided_velocity, time)
    base_value = critic(obs_features=obs_features, action_chunk=base_clean)
    guided_value = critic(obs_features=obs_features, action_chunk=guided_clean)

    torch.testing.assert_close(guided_velocity, torch.full_like(action_t, -0.5))
    assert guided_value.item() > base_value.item()
    assert diagnostics["q_value_mean"].item() == base_value.item()


def test_smolvla_qgf_clips_action_gradient_before_scaling():
    action_t = torch.zeros(1, 2, 2)
    velocity_t = torch.zeros_like(action_t)
    time = torch.tensor([0.5])
    target_direction = torch.full_like(action_t, 3.0)
    critic = LinearActionCritic(target_direction)
    obs_features = torch.zeros(1, 3)

    guided_velocity, diagnostics = q_guided_velocity_smolvla_reverse_time(
        critic=critic,
        obs_features=obs_features,
        action_t=action_t,
        velocity_t=velocity_t,
        time_t=time,
        config=QGuidanceConfig(beta=1.0, grad_clip_norm=0.5),
    )

    correction = velocity_t - guided_velocity
    correction_norm = correction.flatten(1).norm(dim=-1)

    torch.testing.assert_close(correction_norm, torch.tensor([0.5]), atol=1.0e-6, rtol=1.0e-6)
    assert diagnostics["q_grad_norm_raw_mean"].item() > diagnostics["q_grad_norm_mean"].item()


def test_smolvla_qgf_can_limit_critic_to_unpadded_action_dims():
    action_t = torch.zeros(1, 2, 4)
    velocity_t = torch.zeros_like(action_t)
    time = torch.tensor([0.5])
    target_direction = torch.ones(1, 2, 2)
    critic = LinearActionCritic(target_direction)
    obs_features = torch.zeros(1, 3)

    guided_velocity, _ = q_guided_velocity_smolvla_reverse_time(
        critic=critic,
        obs_features=obs_features,
        action_t=action_t,
        velocity_t=velocity_t,
        time_t=time,
        config=QGuidanceConfig(beta=2.0, grad_clip_norm=None),
        critic_action_dim=2,
    )

    torch.testing.assert_close(guided_velocity[..., :2], torch.full((1, 2, 2), -0.5))
    torch.testing.assert_close(guided_velocity[..., 2:], torch.zeros(1, 2, 2))


def test_smolvla_qgf_averages_ensemble_gradients():
    action_t = torch.zeros(1, 2, 2)
    velocity_t = torch.zeros_like(action_t)
    time = torch.tensor([0.5])
    obs_features = torch.zeros(1, 3)
    critics = [
        LinearActionCritic(torch.ones_like(action_t)),
        LinearActionCritic(torch.full_like(action_t, 3.0)),
    ]

    guided_velocity, diagnostics = q_guided_velocity_smolvla_reverse_time(
        critic=critics,
        obs_features=obs_features,
        action_t=action_t,
        velocity_t=velocity_t,
        time_t=time,
        config=QGuidanceConfig(beta=2.0, grad_clip_norm=None),
    )

    torch.testing.assert_close(guided_velocity, torch.full_like(action_t, -1.0))
    assert diagnostics["q_ensemble_size"].item() == 2.0


def test_smolvla_qgf_adaptive_gate_reduces_high_disagreement_guidance():
    action_t = torch.zeros(1, 2, 2)
    velocity_t = torch.zeros_like(action_t)
    time = torch.tensor([0.5])
    obs_features = torch.zeros(1, 3)
    critics = [BiasLinearCritic(0.0), BiasLinearCritic(2.0)]

    guided_velocity, diagnostics = q_guided_velocity_smolvla_reverse_time(
        critic=critics,
        obs_features=obs_features,
        action_t=action_t,
        velocity_t=velocity_t,
        time_t=time,
        config=QGuidanceConfig(
            beta=1.0,
            grad_clip_norm=None,
            uncertainty_scale=1.0,
        ),
    )

    expected_gate = torch.exp(torch.tensor(-1.0))
    torch.testing.assert_close(guided_velocity, torch.full_like(action_t, -expected_gate))
    torch.testing.assert_close(diagnostics["q_gate_mean"], expected_gate)
    torch.testing.assert_close(diagnostics["q_value_std_mean"], torch.tensor(1.0))


def test_smolvla_qgf_passes_task_features_to_critic():
    action_t = torch.zeros(1, 2, 2)
    velocity_t = torch.zeros_like(action_t)
    time = torch.tensor([0.5])
    obs_features = torch.zeros(1, 3)
    task_features = torch.tensor([[2.0, 0.0, 0.0]])

    guided_velocity, diagnostics = q_guided_velocity_smolvla_reverse_time(
        critic=TaskFeatureCritic(),
        obs_features=obs_features,
        task_features=task_features,
        action_t=action_t,
        velocity_t=velocity_t,
        time_t=time,
        config=QGuidanceConfig(beta=2.0, grad_clip_norm=None),
    )

    torch.testing.assert_close(guided_velocity, torch.full_like(action_t, -1.0))
    assert diagnostics["q_value_mean"].item() == 0.0
