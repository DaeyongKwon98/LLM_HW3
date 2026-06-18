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
batch_size=${batch_size:-2}
grad_acc=${grad_acc:-8}
lr=${lr:-5e-5}
beta=${beta:-0.1}
delta=${delta:-0.0}
lambda_sft=${lambda_sft:-1.0}
lambda_copo=${lambda_copo:-0.2}
lambda_anchor=${lambda_anchor:-0.05}
lora_r=${lora_r:-16}
lora_alpha=${lora_alpha:-32}
lora_dropout=${lora_dropout:-0.05}
max_source_length=${max_source_length:-512}
max_target_length=${max_target_length:-768}
num_workers=${num_workers:-2}
eval_steps=${eval_steps:-50}
save_steps=${save_steps:-50}
best_model_metric=${best_model_metric:-eval_loss}
logging_steps=${logging_steps:-5}

ckpt_path=${ckpt_path:-model_checkpoint/3d-molt5-base-pubchem-des.bin}
train_pair_file=${train_pair_file:-data/dpo_chosen_rejected/train.jsonl}
eval_pair_file=${eval_pair_file:-data/dpo_chosen_rejected/val.jsonl}
ts=$(date +%y%m%d_%H%M%S)
out_dir=${out_dir:-checkpoints/mdpo_lora/pubchem_cap_smiles_modality_conflict_${ts}_r${lora_r}_lr${lr}_beta${beta}_ep${epochs}}
log_dir=${log_dir:-logs/mdpo_runs}
mkdir -p "${log_dir}"
log_file=${log_file:-${log_dir}/$(basename "${out_dir}").log}

exec > >(tee -a "${log_file}") 2>&1

echo "Logging to ${log_file}"
echo "Output dir: ${out_dir}"
echo "Training mode: mDPO / CoPO modality-conflict LoRA with SMILES in prompt"
echo "Base checkpoint: ${ckpt_path}"
echo "Train file: ${train_pair_file}"
echo "Eval file: ${eval_pair_file}"

"${ENV_PY}" 3d_molt5/dpo_lora_train.py \
    --checkpoint_path "${ckpt_path}" \
    --output_dir "${out_dir}" \
    --preference_loss mdpo \
    --prompt_style mdpo_instruction \
    --train_pair_file "${train_pair_file}" \
    --eval_pair_file "${eval_pair_file}" \
    --epochs "${epochs}" \
    --max_steps "${max_steps}" \
    --batch_size "${batch_size}" \
    --grad_accum_steps "${grad_acc}" \
    --learning_rate "${lr}" \
    --beta "${beta}" \
    --delta "${delta}" \
    --lambda_sft "${lambda_sft}" \
    --lambda_copo "${lambda_copo}" \
    --lambda_anchor "${lambda_anchor}" \
    --lora_r "${lora_r}" \
    --lora_alpha "${lora_alpha}" \
    --lora_dropout "${lora_dropout}" \
    --max_source_length "${max_source_length}" \
    --max_target_length "${max_target_length}" \
    --num_workers "${num_workers}" \
    --seed "${seed}" \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --logging_steps "${logging_steps}" \
    --eval_steps "${eval_steps}" \
    --save_steps "${save_steps}" \
    --best_model_metric "${best_model_metric}" \
    --save_merged
