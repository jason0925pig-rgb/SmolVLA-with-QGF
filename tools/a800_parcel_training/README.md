# A800 parcel-task SmolVLA training pipeline (as-run, 2026-08-25)

As-executed scripts for the 50-demo parcel-sorting checkpoint. See
`docs/SMOLVLA_PARCEL_50_TRAINING_RESULT_20260826.md` for the full result
report and `docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md` for the
original handoff these follow.

| file | role |
| --- | --- |
| `build_parcel_subset.py` | extract episodes 50-99 into a standalone LeRobot v3 dataset (re-indexed 0-49, stats aggregated from these 50 only) |
| `run_pipeline.sh` | subset -> verify -> train chain (GPU7, markers in PIPELINE.log) |
| `run_train_only.sh` | the training invocation actually used (init-dir method, 20k steps, seed 1000) |
| `post_train.sh` | validate all checkpoints -> select best -> SHA256 |
| `training_provenance.json` | full provenance: base revision, env, dataset, results, deployment |

Paths are user-specific (`~/parcel_smolvla/...` on the A800); adjust before reuse.
