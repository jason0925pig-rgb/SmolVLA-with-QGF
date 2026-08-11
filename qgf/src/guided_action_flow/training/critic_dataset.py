from __future__ import annotations

from pathlib import Path


def success_to_go_targets(successes, gamma: float):
    import torch

    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1].")

    successes = torch.as_tensor(successes, dtype=torch.bool)
    targets = torch.zeros(successes.shape[0], dtype=torch.float32)
    success_indices = torch.nonzero(successes, as_tuple=False).flatten()
    if success_indices.numel() == 0:
        return targets

    first_success = int(success_indices[0].item())
    for step in range(successes.shape[0]):
        future_successes = success_indices[success_indices >= step]
        if future_successes.numel() == 0:
            continue
        distance = int(future_successes[0].item()) - step
        targets[step] = float(gamma**distance)
    targets[first_success:] = 1.0
    return targets


def _episode_task_key(episode):
    return (
        episode.get("task_group"),
        episode.get("task_id"),
        episode.get("task"),
    )


def retrieval_progress_targets(
    episodes,
    *,
    gamma: float,
    obs_key: str = "state",
    k: int = 5,
):
    """Estimate dense progress by retrieving similar states from successful rollouts."""

    import torch

    if k < 1:
        raise ValueError("k must be >= 1.")

    episode_states = [_as_2d_tensor(episode[obs_key], name=obs_key) for episode in episodes]
    episode_successes = [
        torch.as_tensor(episode["success"], dtype=torch.bool).flatten()
        for episode in episodes
    ]
    episode_returns = [
        success_to_go_targets(successes, gamma=gamma) for successes in episode_successes
    ]

    banks_by_task = {}
    global_states = []
    global_values = []
    for episode, states, successes, returns in zip(
        episodes,
        episode_states,
        episode_successes,
        episode_returns,
    ):
        if not bool(successes.any()):
            continue
        task_key = _episode_task_key(episode)
        banks_by_task.setdefault(task_key, {"states": [], "values": []})
        banks_by_task[task_key]["states"].append(states)
        banks_by_task[task_key]["values"].append(returns)
        global_states.append(states)
        global_values.append(returns)

    if not global_states:
        return [torch.zeros(states.shape[0], dtype=torch.float32) for states in episode_states]

    global_bank = {
        "states": torch.cat(global_states, dim=0),
        "values": torch.cat(global_values, dim=0),
    }
    task_banks = {}
    for task_key, bank in banks_by_task.items():
        task_banks[task_key] = {
            "states": torch.cat(bank["states"], dim=0),
            "values": torch.cat(bank["values"], dim=0),
        }

    progress_targets = []
    for episode, states in zip(episodes, episode_states):
        bank = task_banks.get(_episode_task_key(episode), global_bank)
        bank_states = bank["states"]
        bank_values = bank["values"]
        mean = bank_states.mean(dim=0, keepdim=True)
        std = bank_states.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        normalized_states = (states - mean) / std
        normalized_bank_states = (bank_states - mean) / std
        distances = torch.cdist(normalized_states, normalized_bank_states)
        nearest_count = min(k, bank_values.shape[0])
        nearest = distances.topk(nearest_count, largest=False, dim=1).indices
        progress = bank_values[nearest].mean(dim=1).clamp(0.0, 1.0)
        progress_targets.append(progress)

    return progress_targets


def _as_2d_tensor(value, *, name: str):
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def build_action_chunk_dataset(
    episodes,
    *,
    action_horizon: int,
    stride: int = 1,
    gamma: float = 0.99,
    obs_key: str = "state",
    action_key: str = "action_policy",
    target_mode: str = "success_to_go",
    progress_key: str = "progress_score",
    success_weight: float = 0.7,
    progress_weight: float = 0.3,
    retrieval_k: int = 5,
    task_feature_source: str = "none",
    task_feature_dim: int = 0,
    task_feature_key: str = "task_features",
):
    import torch

    from guided_action_flow.training.task_features import hash_token_features

    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if task_feature_source not in {"none", "tokens", "episode", "vlm_hidden"}:
        raise ValueError(
            "task_feature_source must be one of "
            "{'none', 'tokens', 'episode', 'vlm_hidden'}."
        )
    valid_target_modes = {"success_to_go", "progress_blend", "retrieval_progress_blend"}
    if target_mode not in valid_target_modes:
        raise ValueError(f"target_mode must be one of {sorted(valid_target_modes)}.")
    if success_weight < 0 or progress_weight < 0:
        raise ValueError("success_weight and progress_weight must be non-negative.")
    use_task_features = task_feature_source != "none"
    if task_feature_source == "tokens" and task_feature_dim < 1:
        raise ValueError("task_feature_dim must be >= 1 when task features are enabled.")

    obs_features = []
    action_chunks = []
    targets = []
    episode_indices = []
    frame_indices = []
    task_features = []
    retrieval_progress = None
    if target_mode == "retrieval_progress_blend":
        retrieval_progress = retrieval_progress_targets(
            episodes,
            gamma=gamma,
            obs_key=obs_key,
            k=retrieval_k,
        )

    for episode_index, episode in enumerate(episodes):
        states = _as_2d_tensor(episode[obs_key], name=obs_key)
        actions = _as_2d_tensor(episode[action_key], name=action_key)
        successes = torch.as_tensor(episode["success"], dtype=torch.bool).flatten()
        episode_task_features = None
        if task_feature_source == "tokens":
            if "task_tokens" not in episode or "task_attention_mask" not in episode:
                raise ValueError(
                    "Episodes must contain task_tokens and task_attention_mask when "
                    "task_feature_source='tokens'."
                )
            episode_task_features = hash_token_features(
                episode["task_tokens"],
                episode["task_attention_mask"],
                feature_dim=task_feature_dim,
            ).squeeze(0)
        elif task_feature_source in {"episode", "vlm_hidden"}:
            if task_feature_key not in episode:
                raise ValueError(
                    f"Episodes must contain {task_feature_key!r} when "
                    f"task_feature_source={task_feature_source!r}."
                )
            episode_task_features = torch.as_tensor(
                episode[task_feature_key],
                dtype=torch.float32,
            ).flatten()

        if states.shape[0] != actions.shape[0] or states.shape[0] != successes.shape[0]:
            raise ValueError(
                "state, action, and success tensors must have the same first dimension."
            )
        if actions.shape[0] < action_horizon:
            continue

        returns = success_to_go_targets(successes, gamma=gamma)
        if target_mode == "progress_blend" and progress_key in episode:
            progress = torch.as_tensor(episode[progress_key], dtype=torch.float32).flatten()
            if progress.shape[0] != returns.shape[0]:
                raise ValueError(
                    f"{progress_key!r} must have the same length as success, "
                    f"got {progress.shape[0]} and {returns.shape[0]}."
                )
            total_weight = max(success_weight + progress_weight, 1.0e-6)
            returns = (
                success_weight * returns + progress_weight * progress.clamp(0.0, 1.0)
            ) / total_weight
            returns = returns.clamp(0.0, 1.0)
        elif target_mode == "retrieval_progress_blend":
            progress = retrieval_progress[episode_index]
            total_weight = max(success_weight + progress_weight, 1.0e-6)
            returns = (
                success_weight * returns + progress_weight * progress.clamp(0.0, 1.0)
            ) / total_weight
            returns = returns.clamp(0.0, 1.0)
        for start in range(0, actions.shape[0] - action_horizon + 1, stride):
            obs_features.append(states[start])
            action_chunks.append(actions[start : start + action_horizon])
            targets.append(returns[start])
            episode_indices.append(episode_index)
            frame_indices.append(start)
            if episode_task_features is not None:
                task_features.append(episode_task_features)

    if not action_chunks:
        raise ValueError(
            "No action chunks were produced. Check horizon, stride, and rollout length."
        )

    dataset = {
        "obs_features": torch.stack(obs_features),
        "action_chunks": torch.stack(action_chunks),
        "targets": torch.stack(targets).float(),
        "episode_indices": torch.tensor(episode_indices, dtype=torch.long),
        "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
    }
    if use_task_features:
        dataset["task_features"] = torch.stack(task_features)
    return dataset


def split_indices_by_episode(episode_indices, *, val_fraction: float, generator=None):
    import torch

    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    episode_indices = torch.as_tensor(episode_indices, dtype=torch.long).flatten()
    if episode_indices.numel() == 0:
        raise ValueError("episode_indices must not be empty.")

    all_indices = torch.arange(episode_indices.numel(), dtype=torch.long)
    unique_episodes = torch.unique(episode_indices, sorted=True)
    if val_fraction == 0.0 or unique_episodes.numel() < 2:
        return all_indices, all_indices[:0]

    val_episode_count = int(unique_episodes.numel() * val_fraction)
    val_episode_count = max(1, min(val_episode_count, unique_episodes.numel() - 1))
    order = torch.randperm(unique_episodes.numel(), generator=generator)
    val_episodes = unique_episodes[order[:val_episode_count]]
    val_mask = (episode_indices[:, None] == val_episodes[None, :]).any(dim=1)

    return all_indices[~val_mask], all_indices[val_mask]


def load_episode_files(paths):
    import torch

    episodes = []
    for path in paths:
        path = Path(path)
        if path.suffix == ".parquet":
            episodes.append(load_real_robot_parquet(path))
        else:
            episodes.append(torch.load(path, map_location="cpu", weights_only=False))
    return episodes


def load_real_robot_parquet(path: str | Path):
    """Load the standardized real-robot transition table as one episode.

    Videos remain external MP4 files.  Frame indices/timestamps are retained
    in the returned mapping so a future vision encoder can retrieve aligned
    frames without changing the control/action interface.
    """

    import torch

    path = Path(path)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read real-robot QGF data") from exc
    table = pq.read_table(path)
    columns = table.to_pydict()
    required = {"state", "action_policy", "next_state", "reward", "success", "done"}
    missing = sorted(required - columns.keys())
    if missing:
        raise ValueError(f"{path} is missing required QGF columns: {missing}")
    episode = {
        "state": torch.tensor(columns["state"], dtype=torch.float32),
        "action_policy": torch.tensor(columns["action_policy"], dtype=torch.float32),
        "next_state": torch.tensor(columns["next_state"], dtype=torch.float32),
        "reward": torch.tensor(columns["reward"], dtype=torch.float32),
        "success": torch.tensor(columns["success"], dtype=torch.bool),
        "done": torch.tensor(columns["done"], dtype=torch.bool),
        "source_path": str(path),
        "source_kind": "qgf-real-rollout-1.0",
    }
    for key in (
        "action_guarded",
        "action_executed",
        "terminated",
        "truncated",
        "episode_success",
        "gripper_contact",
        "chest_frame_index",
        "chest_timestamp_ns",
        "wrist_frame_index",
        "wrist_timestamp_ns",
    ):
        if key in columns:
            value = columns[key]
            if key.startswith("action_"):
                episode[key] = torch.tensor(value, dtype=torch.float32)
            elif key.endswith("timestamp_ns") or key.endswith("frame_index"):
                episode[key] = torch.tensor(value, dtype=torch.long)
            else:
                episode[key] = torch.tensor(value, dtype=torch.bool)
    tasks = columns.get("task", [])
    episode["task"] = tasks[0] if tasks else ""
    episode["camera_videos"] = {
        "chest": str(path.parent / "chest.mp4"),
        "wrist_right": str(path.parent / "wrist_right.mp4"),
    }
    return episode


def discover_episode_files(data_dir: str | Path):
    data_path = Path(data_dir)
    return sorted(
        [*data_path.rglob("episode_*.pt"), *data_path.rglob("transitions.parquet")],
        key=lambda path: str(path),
    )
