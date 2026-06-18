#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
ENV_PY=${ENV_PY:-/workspace/daeyong/conda_envs/3dmolt5/bin/python}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

seed=${seed:-42}
epochs=${epochs:-3}
max_steps=${max_steps:-0}
batch_size=${batch_size:-1}
grad_acc=${grad_acc:-16}
lr=${lr:-1e-5}
beta=${beta:-0.1}
delta=${delta:-0.0}
lambda_sft=${lambda_sft:-1.0}
lambda_copo=${lambda_copo:-0.2}
lambda_anchor=${lambda_anchor:-0.05}
max_source_length=${max_source_length:-512}
max_target_length=${max_target_length:-768}
num_workers=${num_workers:-2}
eval_steps=${eval_steps:-50}
save_steps=${save_steps:-50}
best_model_metric=${best_model_metric:-eval_loss}
logging_steps=${logging_steps:-5}
mixed_precision=${mixed_precision:-bf16}

ckpt_path=${ckpt_path:-model_checkpoint/3d-molt5-base-pubchem-des.bin}
train_pair_file=${train_pair_file:-data/dpo_chosen_rejected/train.jsonl}
eval_pair_file=${eval_pair_file:-data/dpo_chosen_rejected/val.jsonl}
ts=$(date +%y%m%d_%H%M%S)
out_dir=${out_dir:-checkpoints/mdpo_full/pubchem_des_mdpo_chosen_rejected_full_${ts}_lr${lr}_beta${beta}_ep${epochs}}
log_dir=${log_dir:-logs/mdpo_runs}
mkdir -p "${log_dir}"
log_file=${log_file:-${log_dir}/$(basename "${out_dir}").log}

exec > >(tee -a "${log_file}") 2>&1

echo "Logging to ${log_file}"
echo "Output dir: ${out_dir}"
echo "Training mode: mDPO full fine-tuning / modality-conflict E3FP"
echo "Base checkpoint: ${ckpt_path}"
echo "Train file: ${train_pair_file}"
echo "Eval file: ${eval_pair_file}"
echo "Effective batch size: $((batch_size * grad_acc))"

"${ENV_PY}" 3d_molt5/dpo_lora_train.py     --checkpoint_path "${ckpt_path}"     --output_dir "${out_dir}"     --tuning_mode full     --preference_loss mdpo     --prompt_style mdpo_instruction     --train_pair_file "${train_pair_file}"     --eval_pair_file "${eval_pair_file}"     --epochs "${epochs}"     --max_steps "${max_steps}"     --batch_size "${batch_size}"     --grad_accum_steps "${grad_acc}"     --learning_rate "${lr}"     --beta "${beta}"     --delta "${delta}"     --lambda_sft "${lambda_sft}"     --lambda_copo "${lambda_copo}"     --lambda_anchor "${lambda_anchor}"     --max_source_length "${max_source_length}"     --max_target_length "${max_target_length}"     --num_workers "${num_workers}"     --seed "${seed}"     --mixed_precision "${mixed_precision}"     --gradient_checkpointing     --logging_steps "${logging_steps}"     --eval_steps "${eval_steps}"     --save_steps "${save_steps}"     --best_model_metric "${best_model_metric}"
