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
