#!/bin/bash
set -u
R=~/parcel_smolvla/20260825_parcel_50
export CUDA_VISIBLE_DEVICES=7
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/hf_cache
export TOKENIZERS_PARALLELISM=false
OUT=$R/outputs/train_20260825_parcel50_smolvla
echo "$(date "+%m-%d %H:%M:%S") TRAIN_START gpu=7 steps=20000 batch=64 seed=42" >> $R/PIPELINE.log
~/parcel_env/bin/lerobot-train   --dataset.repo_id=local/onearm_parcel_50   --dataset.root=$R/dataset   --policy.path=/home/zwwl_user2/parcel_smolvla/models/smolvla_parcel_init --policy.push_to_hub=false   --policy.device=cuda   --output_dir=$OUT   --batch_size=64   --steps=20000   --save_freq=5000   --seed=1000   --num_workers=8   --wandb.enable=false   > $R/train.log 2>&1
C=$?
[ $C -eq 0 ] && echo "$(date "+%m-%d %H:%M:%S") TRAIN_DONE" >> $R/PIPELINE.log || echo "$(date "+%m-%d %H:%M:%S") TRAIN_FAIL code=$C" >> $R/PIPELINE.log
