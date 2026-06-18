#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
ENV_PY=${ENV_PY:-/workspace/daeyong/conda_envs/3dmolt5/bin/python}

ckpt_path=${ckpt_path:-model_checkpoint/3d-molt5-base-pubchem-des.bin}
train_pair_file=${train_pair_file:-data/dpo_chosen_rejected/train.jsonl}
eval_pair_file=${eval_pair_file:-data/dpo_chosen_rejected/val.jsonl}
test_file=${test_file:-data/dpo_chosen_rejected/test.jsonl}

epochs=${epochs:-3}
lr=${lr:-1e-5}
beta=${beta:-0.1}
batch_size=${batch_size:-16}
grad_acc=${grad_acc:-1}
max_source_length=${max_source_length:-512}
max_target_length=${max_target_length:-768}
max_new_tokens=${max_new_tokens:-768}
infer_batch_size=${infer_batch_size:-4}

run_name=${run_name:-pubchem_des_mdpo_chosen_rejected_full_lr${lr}_ep${epochs}}
out_dir=${out_dir:-checkpoints/mdpo_full/${run_name}}
output_dir=${output_dir:-outputs/mdpo_full/${run_name}}
mkdir -p "${output_dir}"

out_dir="${out_dir}" ckpt_path="${ckpt_path}" train_pair_file="${train_pair_file}" eval_pair_file="${eval_pair_file}" epochs="${epochs}" lr="${lr}" beta="${beta}" batch_size="${batch_size}" grad_acc="${grad_acc}" max_source_length="${max_source_length}" max_target_length="${max_target_length}" bash 3d_molt5/finetune_scripts/mdpo_full_pubchem_des_chosen_rejected_ep3.sh

model_dir="${out_dir}/best_model" base_ckpt="${ckpt_path}" input_jsonl="${test_file}" pred_jsonl="${output_dir}/${run_name}_predictions.jsonl" metrics_json="${output_dir}/${run_name}_metrics.json" run_name="${run_name}" max_source_length="${max_source_length}" max_new_tokens="${max_new_tokens}" batch_size="${infer_batch_size}" bash 3d_molt5/finetune_scripts/infer_eval_mdpo_pubchem_cap_best.sh

echo "Checkpoint: ${out_dir}"
echo "Best model: ${out_dir}/best_model"
echo "Predictions: ${output_dir}/${run_name}_predictions.jsonl"
echo "Metrics: ${output_dir}/${run_name}_metrics.json"
