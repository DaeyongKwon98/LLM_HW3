#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
ENV_PY=${ENV_PY:-/workspace/daeyong/conda_envs/3dmolt5/bin/python}

model_dir=${model_dir:-checkpoints/mdpo_lora/pubchem_cap_modality_conflict_260617_150441_r16_lr5e-5_beta0.1_ep5/best_model}
base_ckpt=${base_ckpt:-model_checkpoint/3d-molt5-base-pubchem-cap.bin}
input_jsonl=${input_jsonl:-data/dpo_chosen_rejected/test.jsonl}
out_dir=${out_dir:-outputs/mdpo_lora}
run_name=${run_name:-pubchem_cap_modality_conflict_260617_150441_best_test}
pred_jsonl=${pred_jsonl:-${out_dir}/${run_name}_predictions.jsonl}
metrics_json=${metrics_json:-${out_dir}/${run_name}_metrics.json}

max_source_length=${max_source_length:-384}
max_new_tokens=${max_new_tokens:-768}
batch_size=${batch_size:-4}
num_beams=${num_beams:-1}
max_samples=${max_samples:-0}
dtype=${dtype:-bfloat16}

mkdir -p "${out_dir}"

"${ENV_PY}" 3d_molt5/dpo_lora_infer.py \
    --model_path "${model_dir}" \
    --base_checkpoint_path "${base_ckpt}" \
    --input_jsonl "${input_jsonl}" \
    --output_jsonl "${pred_jsonl}" \
    --preference_side chosen \
    --reference_field chosen_caption \
    --prompt_style mdpo_instruction \
    --max_source_length "${max_source_length}" \
    --max_new_tokens "${max_new_tokens}" \
    --batch_size "${batch_size}" \
    --num_beams "${num_beams}" \
    --max_samples "${max_samples}" \
    --dtype "${dtype}"

"${ENV_PY}" 3d_molt5/eval_caption_predictions.py \
    --predictions_jsonl "${pred_jsonl}" \
    --output_json "${metrics_json}"

echo "Predictions: ${pred_jsonl}"
echo "Metrics: ${metrics_json}"
