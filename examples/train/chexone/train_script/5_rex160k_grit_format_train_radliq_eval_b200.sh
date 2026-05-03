#!/usr/bin/env bash
set -euo pipefail

# Train CheXOne on ReXGradient-160K with GRIT format reward,
# while tracking RadCliQ report reward as an eval metric.
#
# Key idea:
# - external_grit_format_reward has weight 1.0 (optimization target)
# - external_grit_reward has weight 0.0 (logged/monitored, not optimized)
#
# Usage:
#   CKPT_PATH=StanfordAIMI/CheXOne \
#   TRAIN_JSON=data/prepared/chexone_rex_findings_grpo.jsonl \
#   bash examples/train/chexone/train_script/5_rex160k_grit_format_train_radliq_eval_b200.sh
#
# Optional tuning knobs:
#   NUM_GPUS=8 VLLM_TP_SIZE=8 PER_DEVICE_BATCH=2 GRAD_ACC=16

: "${CKPT_PATH:=StanfordAIMI/CheXOne}"
: "${TRAIN_JSON:?Please set TRAIN_JSON to prepared ReXGradient-160K JSON/JSONL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_PLUGIN_PATH=""
for p in \
  "${REPO_ROOT}/examples/train/grpo/plugin/plugin.py" \
  "$(pwd)/examples/train/grpo/plugin/plugin.py" \
  "/opt/CheXOne/examples/train/grpo/plugin/plugin.py"; do
  if [[ -f "${p}" ]]; then
    DEFAULT_PLUGIN_PATH="${p}"
    break
  fi
done
PLUGIN_PATH="${PLUGIN_PATH:-${DEFAULT_PLUGIN_PATH}}"
if [[ -n "${PLUGIN_PATH}" && -f "${PLUGIN_PATH}" ]]; then
  # canonicalize to avoid cwd-dependent relative behavior inside swift.
  PLUGIN_PATH="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${PLUGIN_PATH}")"
fi
PLUGIN_DIR="$(dirname "${PLUGIN_PATH:-/dev/null}")"

if [[ ! -f "${PLUGIN_PATH}" || ! -d "${PLUGIN_DIR}" ]]; then
  echo "[error] Plugin not found: ${PLUGIN_PATH}"
  echo "[error] Tried defaults:"
  echo "        - ${REPO_ROOT}/examples/train/grpo/plugin/plugin.py"
  echo "        - $(pwd)/examples/train/grpo/plugin/plugin.py"
  echo "        - /opt/CheXOne/examples/train/grpo/plugin/plugin.py"
  echo "[error] Resolved plugin dir: ${PLUGIN_DIR}"
  echo "[error] If you are using Apptainer, make sure the host repo is bind-mounted."
  echo "        Example: apptainer exec -B /vast/projects/han91/scaling-medical-im/GRIT:/vast/projects/han91/scaling-medical-im/GRIT <image>.sif bash ..."
  echo "        Set PLUGIN_PATH to your plugin.py absolute path."
  exit 1
fi

SYSTEM_PROMPT=${SYSTEM_PROMPT:-'First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key "bbox_2d". Then, based on the thinking contents and coordinates, rethink between <rethink> and </rethink> and then answer the question after <answer>.'}

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
if [[ "${NUM_GPUS}" -lt 2 ]]; then
  echo "[warn] NUM_GPUS=${NUM_GPUS}. This script is intended for multi-GPU (e.g., B200) training."
fi

export WORLD_SIZE="${NUM_GPUS}"
VLLM_TP_SIZE=${VLLM_TP_SIZE:-${NUM_GPUS}}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACC=${GRAD_ACC:-16}
LOSS_TYPE=${LOSS_TYPE:-grpo}   # set dr_grpo if desired
OUTPUT_DIR=${OUTPUT_DIR:-./output/chexone-rex160k-grit-format-radcliq-monitor}
RUN_NAME=${RUN_NAME:-chexone_rex160k_grit_format_radcliq_monitor}

# B200-friendly defaults (adjust to your cluster policy)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}

echo "Detected ${NUM_GPUS} GPUs. WORLD_SIZE=${WORLD_SIZE}, VLLM_TP_SIZE=${VLLM_TP_SIZE}"
echo "Model=${CKPT_PATH}"
echo "Dataset=${TRAIN_JSON}"
echo "Plugin=${PLUGIN_PATH}"

MAX_PIXELS=${MAX_PIXELS:-262144} \
NPROC_PER_NODE="${NUM_GPUS}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${CKPT_PATH}" \
  --external_plugins "${PLUGIN_PATH}" \
  --reward_funcs external_grit_format_reward external_grit_reward \
  --reward_weights 1.0 0.0 \
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
  --learning_rate 1e-6 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --logging_steps 20 \
  --num_generations 8 \
  --temperature 1.0 \
  --output_dir "${OUTPUT_DIR}" \
  --system "${SYSTEM_PROMPT}" \
  --report_to wandb \
  --run_name "${RUN_NAME}" \
  --loss_type "${LOSS_TYPE}" \
  --log_completions true
