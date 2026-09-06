# Screwdriver-into-box single-Q critic run.  Sourced by every pipeline step so
# no stage can drift from another.  Nothing here has a default anywhere in the
# pipeline: an unset variable is a hard error, by design.
export QGF_SSD_ROOT=/opt/qgf_real_robot
export QGF_TASK_KEY=screwdriver
export QGF_DATASET_ID=screwdriver_into_box_baseline50_20260906
export QGF_RUN_ID=screwdriver_into_box_single_q_45_5_20260906
export QGF_ORIN_EPISODES=/home/nvidia/work/telop/screwdriver_real_rollouts/episodes
# Established by access time, not by the launcher profile: the committed
# screwdriver profile names smolvla_20260904_screwdriver, which does not exist
# on the Orin.  model.safetensors under smolvla_20260904_screwdriver_into_box
# was read at 2026-09-06 12:38:59, when this collection session started; every
# other bundle's atime is 09-05 or older.  Visual tokens must come from the
# policy that produced the data.
export QGF_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260904_screwdriver_into_box
export QGF_BUNDLE_NAME=screwdriver_into_box
export QGF_EPISODE_FIRST=0
export QGF_EPISODE_LAST=49

# --- deployment stage (steps 8-10); harmless to have set earlier ---
export QGF_ORIN_DEPLOY_DIR=/home/nvidia/work/telop/models/qgf/screwdriver_into_box_single_q_45_5_20260906
export QGF_ROOT=/opt/qgf_real_robot
export QGF_RUN_MODE=qgf
export QGF_GRAD_CLIP_NORM=1.0
# The bundle records beta as the operator's choice, not as a trained constant.
# The offline probe on THIS critic says 0.01: all five admissibility gates pass
# there (median ratio 0.050).  0.5, which every other deployed task carries,
# measures 0.0011 here -- a thousandth of ||velocity||, which no robot run could
# tell apart from baseline.  QGF_BETA and SMOLVLA_QGF_BETA must agree; the Orin
# smoke test fails the run when they do not.
export QGF_BETA=0.01
export QGF_CRITIC_PATH=/opt/qgf_real_robot/runs/screwdriver_into_box_single_q_45_5_20260906/outputs/single_qcritic/critic_member_00.pt
export QGF_ORIN_REPO=/home/nvidia/work/telop/SmolVLA-with-QGF

# Handoff iron rule 1.  The extract and train scripts set these themselves; the
# other steps refuse to run unless they are already explicit, so set them here
# to the same values rather than letting any step guess.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
