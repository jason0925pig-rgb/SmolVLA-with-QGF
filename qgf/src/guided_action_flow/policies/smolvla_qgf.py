from __future__ import annotations

from types import SimpleNamespace

from guided_action_flow.guidance.qgf import (
    QGuidanceConfig,
    q_guided_velocity_smolvla_reverse_time,
)
from guided_action_flow.training.task_features import (
    hash_token_features,
    smolvla_text_hidden_features,
)

OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


class SmolVLAVisualCriticAdapter:
    """Adapt the visual IQL critic to the generic QGF critic call signature.

    The real-robot critic is trained on frozen image tokens from the deployed
    SmolVLA image encoder.  At runtime those tokens belong to the current
    policy observation, while QGF supplies the differentiable action chunk.
    This adapter keeps the visual tokens fixed and preserves gradients only
    with respect to the proposed action chunk.
    """

    def __init__(self, critic):
        self.critic = critic
        self._visual_tokens = None

    def set_visual_tokens(self, visual_tokens) -> None:
        self._visual_tokens = visual_tokens.detach()

    def __call__(self, *, obs_features, action_chunk, proprio=None, task_features=None):
        del proprio, task_features
        if self._visual_tokens is None:
            raise RuntimeError("QGF visual tokens were not prepared for this observation.")
        if self._visual_tokens.shape[0] != obs_features.shape[0]:
            raise RuntimeError(
                "QGF visual-token batch size does not match proprioception: "
                f"{self._visual_tokens.shape[0]} != {obs_features.shape[0]}."
            )
        return self.critic.module(obs_features, self._visual_tokens, action_chunk)


def encode_smolvla_visual_tokens(policy, batch):
    """Encode the two deployed camera views exactly as offline IQL did."""

    import torch

    with torch.no_grad():
        images, _ = policy.prepare_images(batch)
        if len(images) != 2:
            raise RuntimeError(
                "Real-robot QGF requires exactly two image views (chest and wrist); "
                f"SmolVLA received {len(images)}."
            )
        chest_tokens = policy.model.vlm_with_expert.embed_image(images[0])
        wrist_tokens = policy.model.vlm_with_expert.embed_image(images[1])
        tokens = torch.cat((chest_tokens, wrist_tokens), dim=1).detach()
    adapter = getattr(policy, "_gaf_visual_critic", None)
    expected = getattr(adapter, "critic", adapter)
    config = getattr(expected, "config", None)
    if config is not None and tuple(tokens.shape[1:]) != (config.visual_tokens, config.visual_token_dim):
        raise RuntimeError(
            "Runtime SmolVLA visual-token shape differs from the trained critic: "
            f"{tuple(tokens.shape[1:])} != {(config.visual_tokens, config.visual_token_dim)}."
        )
    return tokens


class SmolVLAQGFProcessor:
    """RTC-compatible processor that applies QGF inside SmolVLA denoising."""

    def __init__(
        self,
        *,
        critic,
        config: QGuidanceConfig,
        critic_action_dim: int,
        task_feature_dim: int = 0,
        task_feature_source: str = "tokens",
        policy_model=None,
    ):
        self.critic = critic
        self.config = config
        self.critic_action_dim = critic_action_dim
        self.task_feature_dim = task_feature_dim
        self.task_feature_source = task_feature_source
        self.policy_model = policy_model
        self.obs_features = None
        self.task_features = None
        self.diagnostics = []

    def set_obs_features(self, obs_features) -> None:
        self.obs_features = obs_features
        self.task_features = None

    def set_batch_features(self, batch) -> None:
        self.obs_features = _latest_state(batch)
        self.task_features = _task_features_from_batch(
            batch,
            self.task_feature_dim,
            task_feature_source=self.task_feature_source,
            policy_model=self.policy_model,
        )
        set_visual_tokens = getattr(self.critic, "set_visual_tokens", None)
        if set_visual_tokens is not None:
            if self.policy_model is None:
                raise RuntimeError("Visual QGF critic requires the active SmolVLA policy model.")
            # The adapter receives the policy object through install below;
            # this attribute intentionally contains the public policy wrapper,
            # not only its underlying model.
            visual_policy = getattr(self, "visual_policy", None)
            if visual_policy is None:
                raise RuntimeError("Visual QGF critic adapter was installed without a policy wrapper.")
            set_visual_tokens(encode_smolvla_visual_tokens(visual_policy, batch))

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ):
        del prev_chunk_left_over, inference_delay, execution_horizon
        if self.obs_features is None:
            raise RuntimeError("QGF obs_features were not set before denoise_step().")

        v_t = original_denoise_step_partial(x_t)
        guided_velocity, diagnostics = q_guided_velocity_smolvla_reverse_time(
            critic=self.critic,
            obs_features=self.obs_features,
            action_t=x_t,
            velocity_t=v_t,
            time_t=time,
            config=self.config,
            task_features=self.task_features,
            critic_action_dim=self.critic_action_dim,
        )
        self.diagnostics.append({key: value.detach().cpu() for key, value in diagnostics.items()})
        return guided_velocity

    def is_debug_enabled(self) -> bool:
        return False

    def track(self, *args, **kwargs) -> None:
        del args, kwargs


def _latest_state(batch):
    state = batch[OBS_STATE]
    if getattr(state, "ndim", 0) > 2:
        return state[:, -1, :]
    return state


def _task_features_from_batch(
    batch,
    task_feature_dim: int,
    *,
    task_feature_source: str,
    policy_model=None,
):
    if task_feature_dim <= 0 or task_feature_source == "none":
        return None
    if OBS_LANGUAGE_TOKENS not in batch:
        raise RuntimeError("QGF task-conditioned critic requires observation.language.tokens.")
    tokens = batch[OBS_LANGUAGE_TOKENS]
    mask = batch.get(OBS_LANGUAGE_ATTENTION_MASK)

    if task_feature_source == "tokens":
        features = hash_token_features(tokens, mask, feature_dim=task_feature_dim).to(
            device=tokens.device
        )
    elif task_feature_source == "vlm_hidden":
        if policy_model is None:
            raise RuntimeError("QGF VLM-hidden task features require a SmolVLA policy model.")
        features = smolvla_text_hidden_features(policy_model, tokens, mask).to(device=tokens.device)
    else:
        raise RuntimeError(
            f"Unsupported runtime task_feature_source={task_feature_source!r}."
        )

    if features.shape[-1] != task_feature_dim:
        raise RuntimeError(
            "Runtime task feature dimension does not match critic config: "
            f"{features.shape[-1]} != {task_feature_dim}."
        )
    return features


def install_smolvla_qgf(
    policy,
    *,
    critic,
    config: QGuidanceConfig,
    critic_action_dim: int,
    task_feature_dim: int = 0,
    task_feature_source: str = "tokens",
) -> SmolVLAQGFProcessor:
    """Install QGF through SmolVLA's existing RTC denoise hook."""

    processor = SmolVLAQGFProcessor(
        critic=critic,
        config=config,
        critic_action_dim=critic_action_dim,
        task_feature_dim=task_feature_dim,
        task_feature_source=task_feature_source,
        policy_model=policy.model,
    )
    processor.visual_policy = policy
    # Kept on the policy for the shape assertion in encode_smolvla_visual_tokens.
    policy._gaf_visual_critic = critic

    if not hasattr(policy, "_gaf_original_get_action_chunk"):
        policy._gaf_original_get_action_chunk = policy._get_action_chunk
    if not hasattr(policy, "_gaf_original_rtc_enabled"):
        policy._gaf_original_rtc_enabled = policy._rtc_enabled

    original_get_action_chunk = policy._gaf_original_get_action_chunk

    def _get_action_chunk_with_qgf(batch, noise=None, **kwargs):
        processor.set_batch_features(batch)
        return original_get_action_chunk(batch, noise=noise, **kwargs)

    rtc_config = SimpleNamespace(enabled=True)
    policy._get_action_chunk = _get_action_chunk_with_qgf
    policy._rtc_enabled = lambda: False
    policy.rtc_processor = processor
    policy.config.rtc_config = rtc_config
    policy.model.config.rtc_config = rtc_config
    policy.model.rtc_processor = processor
    policy._gaf_qgf_processor = processor
    return processor
