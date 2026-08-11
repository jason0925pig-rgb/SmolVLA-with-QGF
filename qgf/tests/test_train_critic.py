import importlib.util
from pathlib import Path


def _load_train_critic_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_critic.py"
    spec = importlib.util.spec_from_file_location("train_critic_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_checkpoint_metadata_prefers_best_validation_loss():
    train_critic = _load_train_critic_module()
    history = [
        {"epoch": 0, "train_loss": 0.3, "val_loss": 0.2},
        {"epoch": 1, "train_loss": 0.2, "val_loss": 0.1},
        {"epoch": 2, "train_loss": 0.1, "val_loss": 0.15},
    ]

    selected = train_critic._select_checkpoint_metadata(history)

    assert selected["selected_epoch"] == 1
    assert selected["selected_metric"] == "val_loss"
    assert selected["selected_val_loss"] == 0.1


def test_select_checkpoint_metadata_falls_back_to_final_epoch_without_validation():
    train_critic = _load_train_critic_module()
    history = [
        {"epoch": 0, "train_loss": 0.3},
        {"epoch": 1, "train_loss": 0.2},
    ]

    selected = train_critic._select_checkpoint_metadata(history)

    assert selected["selected_epoch"] == 1
    assert selected["selected_metric"] == "final"
    assert selected["selected_val_loss"] is None


def test_train_critic_parser_accepts_task_feature_options():
    train_critic = _load_train_critic_module()

    args = train_critic._build_arg_parser().parse_args(
        [
            "--data-dir",
            "runs/data",
            "--output-dir",
            "runs/critic",
            "--task-feature-source",
            "tokens",
            "--task-feature-dim",
            "64",
            "--task-feature-key",
            "task_vlm_hidden",
        ]
    )

    assert args.task_feature_source == "tokens"
    assert args.task_feature_dim == 64
    assert args.task_feature_key == "task_vlm_hidden"


def test_train_critic_parser_accepts_multiple_data_dirs():
    train_critic = _load_train_critic_module()

    args = train_critic._build_arg_parser().parse_args(
        [
            "--data-dirs",
            "runs/task0",
            "runs/task1",
            "--output-dir",
            "runs/critic",
        ]
    )

    assert args.data_dir is None
    assert args.data_dirs == ["runs/task0", "runs/task1"]


def test_train_critic_parser_accepts_retrieval_progress_target():
    train_critic = _load_train_critic_module()

    args = train_critic._build_arg_parser().parse_args(
        [
            "--data-dir",
            "runs/data",
            "--output-dir",
            "runs/critic",
            "--target-mode",
            "retrieval_progress_blend",
            "--retrieval-k",
            "3",
        ]
    )

    assert args.target_mode == "retrieval_progress_blend"
    assert args.retrieval_k == 3


def test_train_critic_parser_accepts_vlm_hidden_task_features():
    train_critic = _load_train_critic_module()

    args = train_critic._build_arg_parser().parse_args(
        [
            "--data-dir",
            "runs/data",
            "--output-dir",
            "runs/critic",
            "--task-feature-source",
            "vlm_hidden",
            "--task-feature-key",
            "task_vlm_hidden",
        ]
    )

    assert args.task_feature_source == "vlm_hidden"
    assert args.task_feature_key == "task_vlm_hidden"
