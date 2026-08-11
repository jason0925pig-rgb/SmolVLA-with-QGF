import importlib.util
from pathlib import Path

import torch


def _load_precompute_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "precompute_task_features.py"
    spec = importlib.util.spec_from_file_location("precompute_task_features_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_precompute_episode_features_copies_episode_and_attaches_feature(tmp_path):
    precompute = _load_precompute_module()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    episodes_dir = input_dir / "episodes"
    episodes_dir.mkdir(parents=True)
    torch.save(
        {
            "state": torch.zeros(2, 1),
            "action_policy": torch.zeros(2, 1),
            "success": torch.tensor([False, True]),
            "task_tokens": torch.tensor([5, 7, 0]),
            "task_attention_mask": torch.tensor([True, True, False]),
        },
        episodes_dir / "episode_000000.pt",
    )
    (input_dir / "summary.json").write_text('{"ok": true}')

    written = precompute.precompute_episode_features(
        input_dir=input_dir,
        output_dir=output_dir,
        feature_key="task_vlm_hidden",
        feature_fn=lambda tokens, mask: torch.tensor([[0.1, 0.2, 0.3]]),
    )

    assert written == 1
    episode = torch.load(
        output_dir / "episodes" / "episode_000000.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.testing.assert_close(episode["task_vlm_hidden"], torch.tensor([0.1, 0.2, 0.3]))
    assert (output_dir / "summary.json").read_text() == '{"ok": true}'
