import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_eval_policy_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_policy.py"
    spec = importlib.util.spec_from_file_location("eval_policy_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_critic_paths_supports_single_and_ensemble_paths():
    eval_policy = _load_eval_policy_module()

    single = eval_policy._resolve_critic_paths(
        SimpleNamespace(critic_path="one.pt", critic_paths=None)
    )
    ensemble = eval_policy._resolve_critic_paths(
        SimpleNamespace(critic_path=None, critic_paths=["one.pt", "two.pt"])
    )

    assert single == ["one.pt"]
    assert ensemble == ["one.pt", "two.pt"]


def test_infer_task_feature_source_uses_checkpoint_metadata():
    eval_policy = _load_eval_policy_module()
    critics = [
        SimpleNamespace(config=SimpleNamespace(task_feature_dim=960)),
        SimpleNamespace(config=SimpleNamespace(task_feature_dim=960)),
    ]
    checkpoints = [
        {"task_feature_source": "vlm_hidden"},
        {"task_feature_source": "vlm_hidden"},
    ]

    source = eval_policy._infer_task_feature_source(critics, checkpoints)

    assert source == "vlm_hidden"


def test_infer_task_feature_source_rejects_mismatched_ensemble_sources():
    eval_policy = _load_eval_policy_module()
    critics = [
        SimpleNamespace(config=SimpleNamespace(task_feature_dim=960)),
        SimpleNamespace(config=SimpleNamespace(task_feature_dim=960)),
    ]
    checkpoints = [
        {"task_feature_source": "vlm_hidden"},
        {"task_feature_source": "tokens"},
    ]

    with pytest.raises(ValueError, match="task feature source"):
        eval_policy._infer_task_feature_source(critics, checkpoints)
