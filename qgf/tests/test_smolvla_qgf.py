from types import SimpleNamespace
import math

import torch

from guided_action_flow.guidance.qgf import QGuidanceConfig
from guided_action_flow.policies.smolvla_qgf import SmolVLAQGFProcessor, install_smolvla_qgf


class LinearActionCritic:
    def __call__(self, *, obs_features, action_chunk, proprio=None, task_features=None):
        del proprio, task_features
        return action_chunk.flatten(1).sum(dim=-1) + obs_features.flatten(1).sum(dim=-1) * 0.0


class TaskConditionedLinearActionCritic:
    def __call__(self, *, obs_features, action_chunk, proprio=None, task_features=None):
        del obs_features, proprio
        if task_features is None:
            raise AssertionError("task_features were not provided")
        scale = task_features[:, :1].reshape(action_chunk.shape[0], 1, 1)
        return (action_chunk * scale).flatten(1).sum(dim=-1)


def test_qgf_processor_guides_only_unpadded_action_dims():
    processor = SmolVLAQGFProcessor(
        critic=LinearActionCritic(),
        config=QGuidanceConfig(beta=2.0, grad_clip_norm=None),
        critic_action_dim=2,
    )
    processor.set_obs_features(torch.zeros(1, 3))
    x_t = torch.zeros(1, 2, 4)

    guided = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=None,
        inference_delay=None,
        time=torch.tensor([0.5]),
        original_denoise_step_partial=lambda input_x_t: torch.zeros_like(input_x_t),
        execution_horizon=None,
    )

    torch.testing.assert_close(guided[..., :2], torch.full((1, 2, 2), -0.5))
    torch.testing.assert_close(guided[..., 2:], torch.zeros(1, 2, 2))


def test_install_smolvla_qgf_uses_model_rtc_hook_without_policy_rtc_assertion():
    calls = {}

    class FakePolicy:
        def __init__(self):
            self.config = SimpleNamespace(rtc_config=None)
            self.model = SimpleNamespace(config=self.config, rtc_processor=None)

        def _get_action_chunk(self, batch, noise=None, **kwargs):
            del noise, kwargs
            calls["obs_features"] = self.model.rtc_processor.obs_features
            return batch["out"]

        def _rtc_enabled(self):
            return True

    policy = FakePolicy()
    processor = install_smolvla_qgf(
        policy,
        critic=LinearActionCritic(),
        config=QGuidanceConfig(),
        critic_action_dim=7,
    )

    output = policy._get_action_chunk(
        {
            "observation.state": torch.ones(1, 8),
            "out": torch.zeros(1, 50, 7),
        }
    )

    assert policy.model.rtc_processor is processor
    assert policy.model.config.rtc_config.enabled is True
    assert policy._rtc_enabled() is False
    torch.testing.assert_close(calls["obs_features"], torch.ones(1, 8))
    torch.testing.assert_close(output, torch.zeros(1, 50, 7))


def test_qgf_processor_builds_task_features_from_language_tokens():
    processor = SmolVLAQGFProcessor(
        critic=TaskConditionedLinearActionCritic(),
        config=QGuidanceConfig(beta=2.0, grad_clip_norm=None),
        critic_action_dim=2,
        task_feature_dim=16,
    )
    processor.set_batch_features(
        {
            "observation.state": torch.zeros(1, 3),
            "observation.language.tokens": torch.tensor([[5, 7, 0]]),
            "observation.language.attention_mask": torch.tensor([[True, True, False]]),
        }
    )
    x_t = torch.zeros(1, 2, 2)

    guided = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=None,
        inference_delay=None,
        time=torch.tensor([0.5]),
        original_denoise_step_partial=lambda input_x_t: torch.zeros_like(input_x_t),
        execution_horizon=None,
    )

    assert processor.task_features is not None
    assert processor.task_features.shape == (1, 16)
    assert not torch.allclose(guided, torch.zeros_like(guided))


def test_qgf_processor_builds_task_features_from_vlm_hidden_states():
    class FakeVLMWithExpert:
        def embed_language_tokens(self, tokens):
            return tokens.float().unsqueeze(-1).repeat(1, 1, 2)

        def forward(
            self,
            *,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            fill_kv_cache,
        ):
            del attention_mask, position_ids, past_key_values, use_cache, fill_kv_cache
            return [inputs_embeds[0] + 2.0, None], None

    class FakePolicyModel:
        vlm_with_expert = FakeVLMWithExpert()

    processor = SmolVLAQGFProcessor(
        critic=TaskConditionedLinearActionCritic(),
        config=QGuidanceConfig(beta=2.0, grad_clip_norm=None),
        critic_action_dim=2,
        task_feature_dim=2,
        task_feature_source="vlm_hidden",
        policy_model=FakePolicyModel(),
    )
    processor.set_batch_features(
        {
            "observation.state": torch.zeros(1, 3),
            "observation.language.tokens": torch.tensor([[3, 5, 0]]),
            "observation.language.attention_mask": torch.tensor([[True, True, False]]),
        }
    )

    assert processor.task_features is not None
    expected = 4.0 * math.sqrt(2.0) + 2.0
    torch.testing.assert_close(processor.task_features, torch.full((1, 2), expected))
