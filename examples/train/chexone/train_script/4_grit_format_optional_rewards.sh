#!/usr/bin/env bash
set -euo pipefail

# GRIT prompt style:
# think -> bbox -> rethink -> answer
SYSTEM_PROMPT='First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key "bbox_2d". Then, based on the thinking contents and coordinates, rethink between <rethink> and </rethink> and then answer the question after <answer>.'

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export WORLD_SIZE="${NUM_GPUS}"
echo "Detected ${NUM_GPUS} GPUs. Setting WORLD_SIZE=${WORLD_SIZE}"

: "${CKPT_PATH:?Please set CKPT_PATH to a model/checkpoint.}"
: "${TRAIN_JSON:?Please set TRAIN_JSON to your GRPO JSON/JSONL dataset.}"
LOSS_TYPE=${LOSS_TYPE:-grpo}  # set to dr_grpo for Dr. GRPO

# Default behavior: only GRIT format reward is optimized.
# Optional rewards (counting/iou/giou) can be enabled by extending reward_funcs + reward_weights.
#
# Example:
#   --reward_funcs external_grit_format_reward external_grit_counting_reward external_grit_iou_reward \
#   --reward_weights 1.0 0.2 0.1

NPROC_PER_NODE="${NUM_GPUS}" \
swift rlhf \
    --rlhf_type grpo \
    --model "${CKPT_PATH}" \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_grit_format_reward \
    --reward_weights 1.0 \
    --dataset "${TRAIN_JSON}" \
    --split_dataset_ratio 0.0 \
    --max_completion_length 2048 \
    --max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --save_steps 100 \
    --logging_steps 50 \
    --output_dir ./output/chexone-grit-format-only \
    --num_generations 8 \
    --temperature 1.0 \
    --system "${SYSTEM_PROMPT}" \
    --report_to wandb \
    --run_name chexone_grit_format_only \
    --loss_type "${LOSS_TYPE}"
