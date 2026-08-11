import torch

from guided_action_flow.training.critic_dataset import (
    build_action_chunk_dataset,
    discover_episode_files,
    load_real_robot_parquet,
    split_indices_by_episode,
    success_to_go_targets,
)


def test_real_robot_parquet_is_discovered_and_loaded(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    episode_dir = tmp_path / "episodes" / "episode_000000"
    episode_dir.mkdir(parents=True)
    path = episode_dir / "transitions.parquet"
    rows = [
        {
            "state": [0.0, 0.5],
            "action_policy": [0.1, 1.0],
            "next_state": [0.1, 1.0],
            "reward": 0.0,
            "success": False,
            "done": False,
            "task": "put bottle in box",
        },
        {
            "state": [0.1, 1.0],
            "action_policy": [0.2, 0.0],
            "next_state": [0.2, 0.0],
            "reward": 1.0,
            "success": True,
            "done": True,
            "task": "put bottle in box",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)

    assert discover_episode_files(tmp_path) == [path]
    episode = load_real_robot_parquet(path)
    assert episode["state"].shape == (2, 2)
    assert episode["action_policy"].shape == (2, 2)
    assert episode["success"].tolist() == [False, True]
    assert episode["task"] == "put bottle in box"


def test_success_to_go_targets_use_first_future_success():
    successes = torch.tensor([False, False, False, True, True])

    targets = success_to_go_targets(successes, gamma=0.9)

    torch.testing.assert_close(targets, torch.tensor([0.9**3, 0.9**2, 0.9, 1.0, 1.0]))


def test_build_action_chunk_dataset_uses_policy_actions_and_state_at_chunk_start():
    episode = {
        "state": torch.tensor(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
                [4.0, 14.0],
            ]
        ),
        "action_policy": torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        "success": torch.tensor([False, False, False, True, True]),
    }

    dataset = build_action_chunk_dataset([episode], action_horizon=3, stride=2, gamma=0.9)

    torch.testing.assert_close(dataset["obs_features"], torch.tensor([[0.0, 10.0], [2.0, 12.0]]))
    torch.testing.assert_close(
        dataset["action_chunks"],
        torch.tensor([[[0.0], [1.0], [2.0]], [[2.0], [3.0], [4.0]]]),
    )
    torch.testing.assert_close(dataset["targets"], torch.tensor([0.9**3, 0.9]))


def test_build_action_chunk_dataset_can_attach_task_token_features():
    episode = {
        "state": torch.tensor(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
            ]
        ),
        "action_policy": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "success": torch.tensor([False, False, True, True]),
        "task_tokens": torch.tensor([5, 7, 0]),
        "task_attention_mask": torch.tensor([True, True, False]),
    }

    dataset = build_action_chunk_dataset(
        [episode],
        action_horizon=2,
        stride=1,
        gamma=0.9,
        task_feature_source="tokens",
        task_feature_dim=16,
    )

    assert dataset["task_features"].shape == (3, 16)
    torch.testing.assert_close(dataset["task_features"][0], dataset["task_features"][1])
    torch.testing.assert_close(dataset["task_features"].norm(dim=-1), torch.ones(3))


def test_build_action_chunk_dataset_can_attach_episode_task_features():
    episode = {
        "state": torch.tensor([[0.0], [1.0], [2.0]]),
        "action_policy": torch.tensor([[0.0], [1.0], [2.0]]),
        "success": torch.tensor([False, True, True]),
        "task_vlm_hidden": torch.tensor([0.25, 0.5, 0.75]),
    }

    dataset = build_action_chunk_dataset(
        [episode],
        action_horizon=2,
        stride=1,
        gamma=0.9,
        task_feature_source="vlm_hidden",
        task_feature_key="task_vlm_hidden",
    )

    assert dataset["task_features"].shape == (2, 3)
    torch.testing.assert_close(
        dataset["task_features"],
        torch.tensor([[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]]),
    )


def test_retrieval_progress_blend_gives_failed_rollouts_dense_targets():
    successful_episode = {
        "state": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "action_policy": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "success": torch.tensor([False, False, False, True]),
        "task_group": "libero_spatial",
        "task_id": 0,
        "task": "move object",
    }
    failed_episode = {
        "state": torch.tensor([[1.8], [2.0], [2.2], [2.4]]),
        "action_policy": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "success": torch.tensor([False, False, False, False]),
        "task_group": "libero_spatial",
        "task_id": 0,
        "task": "move object",
    }

    dataset = build_action_chunk_dataset(
        [successful_episode, failed_episode],
        action_horizon=2,
        gamma=0.9,
        target_mode="retrieval_progress_blend",
        success_weight=0.0,
        progress_weight=1.0,
        retrieval_k=1,
    )

    failed_targets = dataset["targets"][dataset["episode_indices"] == 1]
    assert torch.all(failed_targets > 0.0)


def test_split_indices_by_episode_keeps_rollouts_disjoint():
    episode_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    generator = torch.Generator().manual_seed(0)

    train_indices, val_indices = split_indices_by_episode(
        episode_indices,
        val_fraction=0.34,
        generator=generator,
    )

    train_episodes = set(episode_indices[train_indices].tolist())
    val_episodes = set(episode_indices[val_indices].tolist())

    assert train_episodes
    assert val_episodes
    assert train_episodes.isdisjoint(val_episodes)
    assert sorted([*train_indices.tolist(), *val_indices.tolist()]) == list(
        range(episode_indices.numel())
    )
