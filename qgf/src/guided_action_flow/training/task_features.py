from __future__ import annotations

import math


def _as_2d_long_tensor(value, *, name: str):
    import torch

    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def masked_mean_pool(hidden_states, attention_mask):
    import torch

    hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must be a 3D tensor, "
            f"got shape {tuple(hidden_states.shape)}."
        )

    mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=hidden_states.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "attention_mask must match hidden_states first two dimensions, "
            f"got {tuple(mask.shape)} and {tuple(hidden_states.shape[:2])}."
        )

    weights = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    summed = (hidden_states * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return summed / count


def _model_device(module, fallback):
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return fallback


def smolvla_text_hidden_features(model, token_ids, attention_mask=None):
    """Compute frozen SmolVLA/VLM text hidden-state task features.

    The feature is language-only and mean-pooled over non-padding tokens. It uses
    the pinned SmolVLA VLM text stack, but does not backpropagate through it.
    """

    import torch

    if not hasattr(model, "vlm_with_expert"):
        raise ValueError("model must expose SmolVLA's vlm_with_expert module.")

    vlm_with_expert = model.vlm_with_expert
    tokens = _as_2d_long_tensor(token_ids, name="token_ids")
    device = _model_device(vlm_with_expert, tokens.device)
    tokens = tokens.to(device=device)

    if attention_mask is None:
        mask = torch.ones_like(tokens, dtype=torch.bool, device=device)
    else:
        mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != tokens.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids, "
                f"got {tuple(mask.shape)} and {tuple(tokens.shape)}."
            )

    with torch.inference_mode():
        lang_emb = vlm_with_expert.embed_language_tokens(tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        att_2d_mask = mask[:, None, :] & mask[:, :, None]
        position_ids = torch.cumsum(mask.to(dtype=torch.long), dim=1) - 1
        position_ids = position_ids.clamp_min(0)
        outputs_embeds, _ = vlm_with_expert.forward(
            attention_mask=att_2d_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[lang_emb, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        hidden_states = outputs_embeds[0].to(dtype=torch.float32)
        return masked_mean_pool(hidden_states, mask).detach()


def hash_token_features(
    token_ids,
    attention_mask=None,
    *,
    feature_dim: int,
    normalize: bool = True,
):
    """Hash tokenizer ids into a fixed-size bag-of-token feature vector.

    This is intentionally lightweight: it avoids a second text encoder while
    still conditioning the critic on SmolVLA's tokenized task description.
    """

    import torch

    if feature_dim < 1:
        raise ValueError("feature_dim must be >= 1.")

    tokens = _as_2d_long_tensor(token_ids, name="token_ids")
    if attention_mask is None:
        mask = torch.ones_like(tokens, dtype=torch.bool)
    else:
        mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=tokens.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != tokens.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids, "
                f"got {tuple(mask.shape)} and {tuple(tokens.shape)}."
            )

    device = tokens.device
    features = torch.zeros(tokens.shape[0], feature_dim, dtype=torch.float32, device=device)

    bucket = torch.remainder(tokens * 1_000_003 + 97_531, feature_dim)
    sign = torch.where(
        torch.remainder(tokens * 91_739 + 19_993, 2) == 0,
        torch.ones_like(tokens, dtype=torch.float32),
        -torch.ones_like(tokens, dtype=torch.float32),
    )
    values = sign * mask.to(dtype=torch.float32)
    features.scatter_add_(dim=1, index=bucket, src=values)

    if normalize:
        norm = features.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        features = torch.where(norm > 0, features / norm, features)
    return features
