#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

ENV_PY=${ENV_PY:-/workspace/daeyong/conda_envs/3dmolt5/bin/python}
BASE_CKPT=${BASE_CKPT:-model_checkpoint/3d-molt5-base-pubchem-des.bin}
TRAIN_FILE=${TRAIN_FILE:-data/dpo_chosen_rejected/train.jsonl}
VAL_FILE=${VAL_FILE:-data/dpo_chosen_rejected/val.jsonl}
TEST_FILE=${TEST_FILE:-data/dpo_chosen_rejected/test.jsonl}

EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACC=${GRAD_ACC:-8}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-8}
MAX_SOURCE_LENGTH=${MAX_SOURCE_LENGTH:-512}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-768}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-50}
NUM_WORKERS=${NUM_WORKERS:-2}
SEED=${SEED:-42}

GPUS_CSV=${GPUS_CSV:-4,5}
RUN_PREFIX=${RUN_PREFIX:-pubchem_des_mdpo_smiles_sweep_$(date +%y%m%d_%H%M%S)}
SWEEP_DIR=${SWEEP_DIR:-outputs/mdpo_lora/sweeps/${RUN_PREFIX}}
LOG_DIR=${LOG_DIR:-logs/mdpo_runs}
CKPT_ROOT=${CKPT_ROOT:-checkpoints/mdpo_lora}

mkdir -p "${SWEEP_DIR}" "${LOG_DIR}" "${CKPT_ROOT}"
MANIFEST="${SWEEP_DIR}/manifest.csv"
SUMMARY_CSV="${SWEEP_DIR}/summary.csv"
SUMMARY_JSON="${SWEEP_DIR}/summary.json"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if [[ ${#GPUS[@]} -lt 1 ]]; then
    echo "GPUS_CSV must contain at least one GPU id" >&2
    exit 1
fi

echo "run_name,lora_r,lora_alpha,learning_rate,gpu,checkpoint_dir,best_model_dir,predictions_jsonl,metrics_json,best_metrics_json,train_log,infer_log" > "${MANIFEST}"

run_one() {
    local gpu="$1"
    local r="$2"
    local alpha="$3"
    local lr="$4"
    local tag="${RUN_PREFIX}_r${r}_a${alpha}_lr${lr}_ep${EPOCHS}"
    local checkpoint_dir="${CKPT_ROOT}/${tag}"
    local best_model_dir="${checkpoint_dir}/best_model"
    local train_log="${LOG_DIR}/${tag}.log"
    local infer_log="${LOG_DIR}/infer_${tag}.log"
    local predictions_jsonl="${SWEEP_DIR}/${tag}_predictions.jsonl"
    local metrics_json="${SWEEP_DIR}/${tag}_metrics.json"
    local best_metrics_json="${best_model_dir}/best_metrics.json"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    ENV_PY="${ENV_PY}" \
    ckpt_path="${BASE_CKPT}" \
    train_pair_file="${TRAIN_FILE}" \
    eval_pair_file="${VAL_FILE}" \
    out_dir="${checkpoint_dir}" \
    log_file="${train_log}" \
    epochs="${EPOCHS}" \
    batch_size="${BATCH_SIZE}" \
    grad_acc="${GRAD_ACC}" \
    lr="${lr}" \
    lora_r="${r}" \
    lora_alpha="${alpha}" \
    max_source_length="${MAX_SOURCE_LENGTH}" \
    max_target_length="${MAX_TARGET_LENGTH}" \
    eval_steps="${EVAL_STEPS}" \
    save_steps="${SAVE_STEPS}" \
    num_workers="${NUM_WORKERS}" \
    seed="${SEED}" \
    bash 3d_molt5/finetune_scripts/mdpo_lora_pubchem_cap_smiles_ep3.sh

    CUDA_VISIBLE_DEVICES="${gpu}" \
    ENV_PY="${ENV_PY}" \
    model_dir="${best_model_dir}" \
    base_ckpt="${BASE_CKPT}" \
    input_jsonl="${TEST_FILE}" \
    pred_jsonl="${predictions_jsonl}" \
    metrics_json="${metrics_json}" \
    run_name="${tag}" \
    max_source_length="${MAX_SOURCE_LENGTH}" \
    max_new_tokens="${MAX_NEW_TOKENS}" \
    batch_size="${INFER_BATCH_SIZE}" \
    bash 3d_molt5/finetune_scripts/infer_eval_mdpo_pubchem_cap_best.sh > "${infer_log}" 2>&1

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${tag}" "${r}" "${alpha}" "${lr}" "${gpu}" "${checkpoint_dir}" "${best_model_dir}" \
        "${predictions_jsonl}" "${metrics_json}" "${best_metrics_json}" "${train_log}" "${infer_log}" \
        >> "${MANIFEST}"
}

configs=(
    "16 32 1e-5"
    "16 32 5e-5"
    "16 32 1e-4"
    "32 64 1e-5"
    "32 64 5e-5"
    "32 64 1e-4"
    "64 128 1e-5"
    "64 128 5e-5"
    "64 128 1e-4"
)

for worker_idx in "${!GPUS[@]}"; do
    gpu="${GPUS[${worker_idx}]}"
    (
        for cfg_idx in "${!configs[@]}"; do
            if (( cfg_idx % ${#GPUS[@]} != worker_idx )); then
                continue
            fi
            read -r r alpha lr <<< "${configs[${cfg_idx}]}"
            run_one "${gpu}" "${r}" "${alpha}" "${lr}"
        done
    ) &
done

wait

"${ENV_PY}" 3d_molt5/summarize_mdpo_sweep.py \
    --manifest_csv "${MANIFEST}" \
    --output_csv "${SUMMARY_CSV}" \
    --output_json "${SUMMARY_JSON}"

echo "Manifest: ${MANIFEST}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Summary JSON: ${SUMMARY_JSON}"
