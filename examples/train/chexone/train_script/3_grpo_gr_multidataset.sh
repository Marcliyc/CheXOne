#!/usr/bin/env bash
set -euo pipefail

# GRPO-GR training recipe for CheXOne with UniRG + GRIT rewards.
# Datasets:
#   - MIMIC-CXR
#   - CheXpert-Plus
#   - RexGradient-160K
#   - PadChest-GR
#   - VinDr-CXR
#
# Environment variables you should set before running:
#   CKPT_PATH: model checkpoint from SFT stage
#   MIMIC_CXR_JSON, CHEXPERT_PLUS_JSON, REXGRADIENT_160K_JSON, PADCHEST_GR_JSON, VINDR_CXR_JSON
#
# Optional:
#   UNIRG_WEIGHT (default: 0.5)
#   GRIT_WEIGHT (default: 0.5)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export WORLD_SIZE="${NUM_GPUS}"
echo "Detected ${NUM_GPUS} GPUs. Setting WORLD_SIZE=${WORLD_SIZE}"

: "${CKPT_PATH:?Please set CKPT_PATH to your SFT checkpoint path.}"
: "${MIMIC_CXR_JSON:?Please set MIMIC_CXR_JSON.}"
: "${CHEXPERT_PLUS_JSON:?Please set CHEXPERT_PLUS_JSON.}"
: "${REXGRADIENT_160K_JSON:?Please set REXGRADIENT_160K_JSON.}"
: "${PADCHEST_GR_JSON:?Please set PADCHEST_GR_JSON.}"
: "${VINDR_CXR_JSON:?Please set VINDR_CXR_JSON.}"

per_device_train_batch_size=8
LOSS_TYPE=${LOSS_TYPE:-grpo}  # set to dr_grpo for Dr. GRPO

MAX_PIXELS=262144 \
NPROC_PER_NODE="${NUM_GPUS}" \
swift rlhf \
    --rlhf_type grpo \
    --model "${CKPT_PATH}" \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_format_reward_boxed external_grpo_gr_reward \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --vllm_tensor_parallel_size 4 \
    --offload_optimizer True \
    --offload_model True \
    --sleep_level 1 \
    --padding_free True \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --train_type full \
    --torch_dtype bfloat16 \
    --truncation_strategy delete \
    --dataset "${MIMIC_CXR_JSON}" \
              "${CHEXPERT_PLUS_JSON}" \
              "${REXGRADIENT_160K_JSON}" \
              "${PADCHEST_GR_JSON}" \
              "${VINDR_CXR_JSON}" \
    --split_dataset_ratio 0.0 \
    --max_completion_length 2048 \
    --max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${per_device_train_batch_size}" \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --output_dir ./output/chexone-grpo-gr-multidataset \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --system 'You are a helpful assistant.' \
    --report_to wandb \
    --run_name chexone_grpo_gr_multidataset \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --log_completions true \
    --sequence_parallel_size 1 \
    --dataloader_drop_last true \
    --freeze_llm False \
    --freeze_vit True \
    --freeze_aligner False \
    --loss_type "${LOSS_TYPE}"
