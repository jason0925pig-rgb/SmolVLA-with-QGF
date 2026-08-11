# Upstream Repositories

Keep clean upstream repos under `third_party/`. Do not copy their source into `src/guided_action_flow`.

## Expected Layout

```text
third_party/
├── lerobot/      # clean clone of Hugging Face LeRobot
└── LIBERO/       # clean clone of original LIBERO
```

## Clone Commands

Use:

```bash
bash scripts/bootstrap_third_party.sh
```

Equivalent manual commands:

```bash
mkdir -p third_party
git clone https://github.com/huggingface/lerobot.git third_party/lerobot
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
```

## Pinning Policy

After the first working run, record exact commits:

```bash
git -C third_party/lerobot rev-parse HEAD
git -C third_party/LIBERO rev-parse HEAD
```

Then update `docs/setup.md`.

## Why Keep LIBERO Separate If LeRobot Has LIBERO Support?

For the first experiment, use LeRobot's LIBERO support if possible because SmolVLA is a LeRobot model and the observation/action formatting is more likely to match.

Keep the original LIBERO repo because:

- it is the source of benchmark task definitions and simulator assumptions;
- it helps inspect task reward/success logic;
- it gives a fallback path if LeRobot's benchmark wrapper is insufficient.

Do not install the original LIBERO legacy environment into the same Python env until versions are inspected. Legacy LIBERO instructions have historically used older Python/Torch/CUDA stacks, while current LeRobot uses a newer stack.

## Upstream References

- LeRobot: https://github.com/huggingface/lerobot
- SmolVLA model: https://huggingface.co/lerobot/smolvla_base
- LeRobot installation docs: https://huggingface.co/docs/lerobot/installation
- LIBERO original repo: https://github.com/Lifelong-Robot-Learning/LIBERO
- LeRobot LIBERO docs: https://huggingface.co/docs/lerobot/libero

