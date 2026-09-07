from __future__ import annotations

from dataclasses import dataclass


GUIDANCE_MODES = ("critic", "random_matched_norm", "zero")


@dataclass(frozen=True)
class QGuidanceConfig:
    beta: float = 10.0
    grad_clip_norm: float | None = 1.0
    uncertainty_scale: float = 0.0
    min_gate: float = 0.0
    # Ablation switch. Everything else - the critic forward, the backward, the
    # clip, the gate, the 1/beta scaling, the RTC hook and therefore the timing
    # and the RNG consumption - is identical across the three modes.
    guidance_mode: str = "critic"
    # Seed for the random direction, drawn from a dedicated generator so that
    # SmolVLA's own flow-noise sampling from the global RNG is untouched.
    random_seed: int | None = None


_RANDOM_GENERATORS: dict = {}


def _random_generator(device, seed):
    import torch

    key = (str(device), int(seed))
    gen = _RANDOM_GENERATORS.get(key)
    if gen is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        _RANDOM_GENERATORS[key] = gen
    return gen


def reset_random_generators() -> None:
    _RANDOM_GENERATORS.clear()


def _as_critic_list(critic):
    if isinstance(critic, (list, tuple)):
        if not critic:
            raise ValueError("At least one critic is required.")
        return list(critic)
    return [critic]


def _time_like(time_t, reference):
    import torch

    if torch.is_tensor(time_t):
        time = time_t.to(device=reference.device, dtype=reference.dtype)
    else:
        time = torch.tensor(time_t, device=reference.device, dtype=reference.dtype)

    if time.ndim == 0:
        return time
    if time.ndim == 1:
        return time.reshape((time.shape[0],) + (1,) * (reference.ndim - 1))
    return time


def _call_critic(critic, *, obs_features, action_chunk, proprio=None, task_features=None):
    kwargs = {
        "obs_features": obs_features,
        "action_chunk": action_chunk,
        "proprio": proprio,
    }
    if task_features is not None:
        kwargs["task_features"] = task_features
    return critic(**kwargs)


def estimate_clean_action_smolvla_reverse_time(action_t, velocity_t, time_t):
    """Estimate the final action for SmolVLA's `t=1 noise -> t=0 action` sampler.

    SmolVLA trains with `x_t = t * noise + (1 - t) * action` and
    `v_t = noise - action`, so the clean action estimate is `x_t - t * v_t`.
    """

    time = _time_like(time_t, action_t)
    return action_t - time * velocity_t


def q_guided_velocity_smolvla_reverse_time(
    *,
    critic,
    obs_features,
    action_t,
    velocity_t,
    time_t,
    config: QGuidanceConfig,
    proprio=None,
    task_features=None,
    critic_action_dim: int | None = None,
):
    """Return QGF-guided velocity for SmolVLA reverse-time flow sampling.

    The critic is queried only on the estimated clean action chunk. The clean
    action is detached before taking `grad_a Q`, so this implements test-time
    guidance without backpropagating through the denoiser.
    """

    import torch

    if config.beta <= 0:
        raise ValueError("QGuidanceConfig.beta must be positive.")
    if config.uncertainty_scale < 0:
        raise ValueError("QGuidanceConfig.uncertainty_scale must be non-negative.")
    if config.min_gate < 0 or config.min_gate > 1:
        raise ValueError("QGuidanceConfig.min_gate must be in [0, 1].")
    if config.guidance_mode not in GUIDANCE_MODES:
        raise ValueError(
            f"QGuidanceConfig.guidance_mode must be one of {GUIDANCE_MODES}, "
            f"got {config.guidance_mode!r}."
        )
    if config.guidance_mode == "random_matched_norm" and config.random_seed is None:
        raise ValueError(
            "random_matched_norm requires QGuidanceConfig.random_seed (no global RNG)."
        )

    with torch.inference_mode(False), torch.enable_grad():
        action_t = action_t.detach().clone()
        velocity_t = velocity_t.detach().clone()

        clean_action = estimate_clean_action_smolvla_reverse_time(action_t, velocity_t, time_t)
        clean_action = clean_action.detach().clone().requires_grad_(True)
        if critic_action_dim is not None:
            if critic_action_dim < 1 or critic_action_dim > clean_action.shape[-1]:
                raise ValueError("critic_action_dim must be within the action dimension.")
            critic_action = clean_action[..., :critic_action_dim]
        else:
            critic_action = clean_action
        critics = _as_critic_list(critic)
        critic_obs_features = obs_features.detach().clone()
        critic_proprio = proprio.detach().clone() if proprio is not None else None
        critic_task_features = (
            task_features.detach().clone() if task_features is not None else None
        )
        q_values = [
            _call_critic(
                critic_item,
                obs_features=critic_obs_features,
                action_chunk=critic_action,
                proprio=critic_proprio,
                task_features=critic_task_features,
            )
            for critic_item in critics
        ]
        q_value_stack = torch.stack(q_values, dim=0)
        q_value = q_value_stack.mean(dim=0)
        if len(critics) > 1:
            q_value_std = q_value_stack.std(dim=0, unbiased=False)
        else:
            q_value_std = torch.zeros_like(q_value)

        grad = torch.autograd.grad(q_value.sum(), clean_action, create_graph=False)[0]

        raw_grad_norm = grad.reshape(grad.shape[0], -1).norm(dim=-1)
        real_grad = grad
        if config.guidance_mode == "random_matched_norm":
            # Keep the real gradient's per-sample norm; replace only its direction.
            # Done BEFORE clipping, so the clip and the gate see exactly what they
            # would have seen for a real gradient of the same magnitude.
            gen = _random_generator(grad.device, config.random_seed)
            rand = torch.randn(
                grad.shape, generator=gen, device=grad.device, dtype=grad.dtype
            )
            rand_norm = rand.reshape(rand.shape[0], -1).norm(dim=-1)
            view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
            grad = rand * (raw_grad_norm / (rand_norm + 1.0e-6)).reshape(view_shape)
        if config.grad_clip_norm is not None:
            max_norm = float(config.grad_clip_norm)
            scale = (max_norm / (raw_grad_norm + 1.0e-6)).clamp(max=1.0)
            view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
            grad = grad * scale.reshape(view_shape)

        grad_norm = grad.reshape(grad.shape[0], -1).norm(dim=-1)
        if config.uncertainty_scale > 0:
            gate = torch.exp(-float(config.uncertainty_scale) * q_value_std)
            gate = gate.clamp(min=float(config.min_gate), max=1.0)
        else:
            gate = torch.ones_like(q_value)
        view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
        grad = grad * gate.reshape(view_shape)
        if config.guidance_mode == "zero":
            # B0-LM. The critic forward, the backward, the clip and the gate have
            # all run above and are logged at their true values; only the APPLIED
            # guidance is zeroed here, so timing and RNG consumption match "critic".
            grad = torch.zeros_like(grad)
        guided_velocity = velocity_t - grad / config.beta
        guidance_delta = velocity_t - guided_velocity
        flat_real = real_grad.detach().reshape(real_grad.shape[0], -1)
        flat_used = grad.detach().reshape(grad.shape[0], -1)
        direction_cos = torch.nn.functional.cosine_similarity(
            flat_real, flat_used, dim=-1, eps=1.0e-12
        )

    diagnostics = {
        "q_value_mean": q_value.detach().mean(),
        "q_value_std_mean": q_value_std.detach().mean(),
        "q_ensemble_size": q_value.detach().new_tensor(float(len(critics))),
        "q_gate_mean": gate.detach().mean(),
        "q_grad_norm_raw_mean": raw_grad_norm.detach().mean(),
        "q_grad_norm_mean": grad_norm.detach().mean(),
        "q_guidance_norm_mean": guidance_delta.detach()
        .reshape(guidance_delta.shape[0], -1)
        .norm(dim=-1)
        .mean(),
        # 1.0 in critic mode, ~0 in random_matched_norm, 0 in zero.
        "q_direction_cos_mean": direction_cos.mean(),
        "q_guidance_mode": q_value.detach().new_tensor(
            float(GUIDANCE_MODES.index(config.guidance_mode))
        ),
    }
    return guided_velocity.detach(), diagnostics


q_guided_velocity = q_guided_velocity_smolvla_reverse_time
