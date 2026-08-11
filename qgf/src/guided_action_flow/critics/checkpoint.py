from __future__ import annotations


def load_action_chunk_critic(checkpoint_path, *, device="cpu"):
    import torch

    from guided_action_flow.critics.action_chunk_critic import (
        ActionChunkCritic,
        ActionChunkCriticConfig,
    )
    from guided_action_flow.critics.transformer_action_chunk_critic import (
        TransformerActionChunkCritic,
        TransformerActionChunkCriticConfig,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    critic_arch = checkpoint.get("critic_arch", "mlp")
    if critic_arch == "mlp":
        config = ActionChunkCriticConfig(**checkpoint["critic_config"])
        critic = ActionChunkCritic(config)
    elif critic_arch == "transformer":
        config = TransformerActionChunkCriticConfig(**checkpoint["critic_config"])
        critic = TransformerActionChunkCritic(config)
    else:
        raise ValueError(f"Unsupported critic_arch={critic_arch!r}.")
    critic.load_state_dict(checkpoint["model_state_dict"])
    critic.module.to(device)
    critic.module.eval()
    return critic, checkpoint

