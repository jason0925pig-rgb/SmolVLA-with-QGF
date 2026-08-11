# Setup Plan

This machine is used to create and maintain the repository. The 8GB RTX 4070 machine should run model download, rollout, critic training, and evaluation.

For a more operational checklist, use `docs/agent_4070_libero_runbook.md`.

## Third-Party Repositories

Run the bootstrap script on the machine where experiments will run:

```bash
bash scripts/bootstrap_third_party.sh
```

Expected checkouts:

- `third_party/lerobot`
- `third_party/LIBERO`

Pinned commits used by the current local runs:

```text
lerobot: 6a788fbdb02cabfae60f7408636945df0b1eafa0
LIBERO: 8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO-plus: 4976dc30028e805ff8094b55501d532c48fec182
LIBERO-PRO: eafdb809426b13153aa1e4c42d6601844217dfec
```

## Python Environment

Create an environment with Python 3.10 or 3.11. Install this repo in editable mode:

```bash
pip install -e ".[dev,torch]"
```

Then install `LeRobot` and `LIBERO` according to their pinned upstream instructions. Do not assume their install commands until the checkout is inspected.

Preferred first route for this project is the LeRobot-managed LIBERO environment, because SmolVLA and LIBERO benchmark integration both live in the LeRobot stack. Keep the original `LIBERO` checkout for inspection, pinning, and fallback direct adapter work.

## Model

The base policy config points to:

```text
lerobot/smolvla_base
```

Store downloaded model artifacts under `checkpoints/` or the Hugging Face cache on the experiment machine.
