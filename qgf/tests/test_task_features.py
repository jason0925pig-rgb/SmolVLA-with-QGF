import math

import torch

from guided_action_flow.training.task_features import (
    hash_token_features,
    masked_mean_pool,
    smolvla_text_hidden_features,
)


def test_hash_token_features_is_masked_and_deterministic():
    tokens_a = torch.tensor([[5, 7, 0], [5, 7, 99]])
    mask = torch.tensor([[True, True, False], [True, True, False]])

    features = hash_token_features(tokens_a, mask, feature_dim=16)

    assert features.shape == (2, 16)
    torch.testing.assert_close(features[0], features[1])
    torch.testing.assert_close(features.norm(dim=-1), torch.ones(2))


def test_hash_token_features_changes_when_unmasked_tokens_change():
    first = hash_token_features(
        torch.tensor([[5, 7, 0]]),
        torch.tensor([[True, True, False]]),
        feature_dim=16,
    )
    second = hash_token_features(
        torch.tensor([[5, 9, 0]]),
        torch.tensor([[True, True, False]]),
        feature_dim=16,
    )

    assert not torch.allclose(first, second)


def test_masked_mean_pool_ignores_padding_tokens():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[5.0, 7.0], [11.0, 13.0], [17.0, 19.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, True, True]])

    pooled = masked_mean_pool(hidden, mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0], [14.0, 16.0]]))


def test_smolvla_text_hidden_features_pool_frozen_vlm_outputs():
    class FakeVLMWithExpert:
        def embed_language_tokens(self, tokens):
            return tokens.float().unsqueeze(-1).repeat(1, 1, 3)

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
            del past_key_values, use_cache
            assert attention_mask.shape == (1, 3, 3)
            assert fill_kv_cache is True
            torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 1]]))
            return [inputs_embeds[0] + 10.0, None], None

    class FakeModel:
        vlm_with_expert = FakeVLMWithExpert()

    features = smolvla_text_hidden_features(
        FakeModel(),
        torch.tensor([[2, 4, 0]]),
        torch.tensor([[True, True, False]]),
    )

    expected = 3.0 * math.sqrt(3.0) + 10.0
    torch.testing.assert_close(features, torch.full((1, 3), expected))
