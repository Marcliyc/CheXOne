#!/usr/bin/env bash
set -euo pipefail

# Train CheXOne on ReXGradient using CheXOne's default report-reasoning prompt style:
#   "<image>Write the findings/impression ... Reason step by step and put final answer in \\boxed{}."
# This script is intended for datasets prepared with:
#   python examples/train/chexone/data_prep/prepare_cxr_datasets.py --prepare-rex --with-reasoning ...
#
# Reward setup:
# - Primary optimization reward is external_generation_with_reason_reward (RadCliQ-like composite reward).
# - "Evaluate all rewards" mode logs all available GRIT/UniRG rewards by adding them with zero weights.

: "${CKPT_PATH:=StanfordAIMI/CheXOne}"
: "${MODEL_TYPE:=qwen2_5_vl}"
: "${TRAIN_JSON:=data/prepared/chexone_rex_findings_grpo.jsonl}"
: "${USE_HF:=1}"
: "${LOSS_TYPE:=grpo}"
: "${PER_DEVICE_BATCH:=2}"
: "${GRAD_ACC:=16}"
: "${NUM_GENERATIONS:=8}"
: "${LR:=1e-6}"
: "${EVAL_ALL_REWARDS:=1}"

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
export WORLD_SIZE="${NUM_GPUS}"
VLLM_TP_SIZE=${VLLM_TP_SIZE:-${NUM_GPUS}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PLUGIN_PATH=${PLUGIN_PATH:-"${REPO_ROOT}/examples/train/grpo/plugin/plugin.py"}

if [[ ! -f "${PLUGIN_PATH}" ]]; then
  echo "[error] Plugin not found: ${PLUGIN_PATH}"
  exit 1
fi

if [[ "${EVAL_ALL_REWARDS}" == "1" ]]; then
  REWARD_FUNCS=(
    external_generation_with_reason_reward
    external_format_reward_boxed
    external_grit_reward
    external_unirg_reward
    external_grpo_gr_reward
    external_grit_format_reward
    external_grit_counting_reward
    external_grit_iou_reward
    external_grit_giou_reward
  )
  REWARD_WEIGHTS=(1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
else
  REWARD_FUNCS=(external_generation_with_reason_reward external_format_reward_boxed)
  REWARD_WEIGHTS=(1.0 0.0)
fi

# Keep system prompt simple; reasoning behavior is already in dataset user prompt.
SYSTEM_PROMPT=${SYSTEM_PROMPT:-'You are a helpful assistant.'}

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[error] No GPUs detected."
  exit 1
fi

echo "Detected ${NUM_GPUS} GPUs. WORLD_SIZE=${WORLD_SIZE}, VLLM_TP_SIZE=${VLLM_TP_SIZE}"
echo "Model=${CKPT_PATH}"
echo "Model type=${MODEL_TYPE}"
echo "Dataset=${TRAIN_JSON}"
echo "Plugin=${PLUGIN_PATH}"
echo "Rewards=${REWARD_FUNCS[*]}"
echo "Weights=${REWARD_WEIGHTS[*]}"

MAX_PIXELS=${MAX_PIXELS:-262144} \
NPROC_PER_NODE="${NUM_GPUS}" \
USE_HF="${USE_HF}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${CKPT_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --use_hf "${USE_HF}" \
  --external_plugins "${PLUGIN_PATH}" \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --reward_weights "${REWARD_WEIGHTS[@]}" \
  --dataset "${TRAIN_JSON}" \
  --split_dataset_ratio 0.01 \
  --eval_strategy steps \
  --eval_steps 200 \
  --use_vllm True \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.70 \
  --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
  --vllm_max_model_len 4096 \
  --attn_impl flash_attn \
  --torch_dtype bfloat16 \
  --max_completion_length 2048 \
  --max_length 4096 \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR}" \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --logging_steps 20 \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature 1.0 \
  --output_dir "${OUTPUT_DIR:-./output/chexone-rex160k-reasoning-all-rewards}" \
  --system "${SYSTEM_PROMPT}" \
  --report_to "${REPORT_TO:-wandb}" \
  --run_name "${RUN_NAME:-chexone_rex160k_reasoning_all_rewards}" \
  --loss_type "${LOSS_TYPE}" \
  --log_completions true
