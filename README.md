# 3D-MolT5 DPO / mDPO LoRA Workflow

이 브랜치는 원본 3D-MolT5 코드를 기반으로, LogP/QM9 계열 preference data를 사용해 LoRA 방식의 DPO/mDPO 학습과 caption inference를 수행하기 위한 작업 공간이다.

핵심 진입점은 아래 파일들이다.

- `3d_molt5/dpo_lora_train.py`: base 3D-MolT5 checkpoint에 LoRA adapter를 붙여 DPO, mDPO, CoPO preference training을 수행한다.
- `3d_molt5/dpo_lora_infer.py`: 학습된 LoRA adapter 또는 merged checkpoint로 JSONL 입력에 대해 caption을 생성한다.
- `3d_molt5/eval_caption_predictions.py`: 생성 caption과 reference caption 사이의 BLEU/ROUGE/METEOR 등 caption metrics를 계산한다.
- `3d_molt5/summarize_mdpo_sweep.py`: sweep 결과 manifest와 metrics를 모아 CSV/JSON summary를 만든다.
- `3d_molt5/finetune_scripts/*.sh`: 위 Python entrypoint를 실제 경로와 기본 hyperparameter로 실행하는 wrapper script다.

## Directory Layout

```text
3d_molt5/
  dpo_lora_train.py                         # LoRA DPO/mDPO/CoPO training
  dpo_lora_infer.py                         # LoRA/merged model inference
  eval_caption_predictions.py               # caption metric evaluation
  summarize_mdpo_sweep.py                   # sweep result summarization
  finetune_scripts/
    mdpo_lora_pubchem_cap_smiles_ep3.sh     # single mDPO LoRA training run
    infer_eval_mdpo_pubchem_cap_best.sh     # inference + metric evaluation
    sweep_mdpo_pubchem_des_lora_lr.sh       # LoRA rank/lr sweep
  utils/                                    # 3D-MolT5 model/tokenizer utilities

data/
  dpo_chosen_rejected/
    train.jsonl
    val.jsonl
    test.jsonl
    stats.json
  build_chosen_rejected_splits.py           # positive/negative source files -> train/val/test pair splits
  build_dpo_caption_pairs_from_chosen_rejected.py

dict/
  selfies_dict.txt                          # molecule tokenizer 추가 token list

model_checkpoint/                           # base 3D-MolT5 checkpoints, git ignore 대상
checkpoints/                                # LoRA training outputs, git ignore 대상
outputs/                                    # inference/evaluation outputs, git ignore 대상
logs/                                       # run logs, git ignore 대상
```

## Environment

```bash
conda create -n 3dmolt5 python=3.8
conda activate 3dmolt5
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

실행 전 필요한 기본 파일은 다음과 같다.

- base checkpoint: 예: `model_checkpoint/3d-molt5-base-pubchem-des.bin`
- tokenizer molecule dictionary: `dict/selfies_dict.txt`
- preference data: `data/dpo_chosen_rejected/train.jsonl`, `val.jsonl`, `test.jsonl`

## Data Format

현재 주로 사용하는 입력은 `data/dpo_chosen_rejected/*.jsonl`이다. 각 row는 `logp` 또는 `qm9` task를 표현하며, chosen/rejected 3D condition을 가진다.

필수 또는 주요 field:

```json
{
  "id": "logp_000001",
  "task": "logp",
  "smiles": "...",
  "selfies": "...",
  "logP": 1.23,
  "chosen_caption": "reference caption",
  "rejected_caption": "reference caption or rejected caption",
  "chosen_e3fp": [[...], [...]],
  "rejected_e3fp": [[...], [...]],
  "chosen": {"e3fp": [[...]], "caption_en": "..."},
  "rejected": {"e3fp": [[...]], "caption_en": "..."}
}
```

QM9 task는 `logP` 대신 아래 field를 사용한다.

```json
{
  "task": "qm9",
  "homo": -0.23,
  "lumo": 0.04,
  "gap": 0.27
}
```

`dpo_lora_train.py`는 여러 입력 방식을 지원한다.

- `--train_pair_file`, `--eval_pair_file`: 명시적인 train/eval JSONL split. 현재 wrapper script가 사용하는 방식이다.
- `--caption_pair_file`: 하나의 caption-pair JSONL을 `--eval_ratio`로 train/eval split한다.
- `--positive_file`, `--negative_file`: 같은 molecule/property/caption을 공유하고 E3FP만 다른 aligned positive/negative JSONL pair를 사용한다.
- `--logp_positive`, `--logp_negative`, `--qm9_positive`, `--qm9_negative`: 기본 aligned source files를 사용한다.

## Build Data Splits

positive/negative source files가 `data/` 아래에 있을 때, chosen/rejected split은 다음처럼 만든다.

```bash
python data/build_chosen_rejected_splits.py \
  --data_dir data \
  --output_dir data/dpo_chosen_rejected \
  --seed 42
```

기본 입력 파일명:

- `data/logp_positive.jsonl`
- `data/logp_negative.jsonl`
- `data/qm9_positive.jsonl`
- `data/qm9_negative.jsonl`

생성 파일:

- `data/dpo_chosen_rejected/train.jsonl`
- `data/dpo_chosen_rejected/val.jsonl`
- `data/dpo_chosen_rejected/test.jsonl`
- `data/dpo_chosen_rejected/stats.json`

`data/build_dpo_caption_pairs_from_chosen_rejected.py`는 chosen/rejected split에서 caption-pair DPO data를 만드는 보조 스크립트다. 이 스크립트는 `training_data/build_dpo_caption_pairs_v2.py`를 import하므로, 해당 파일이 있는 환경에서만 동작한다.

```bash
python data/build_dpo_caption_pairs_from_chosen_rejected.py \
  --input_dir data/dpo_chosen_rejected \
  --output_dir data/dpo_caption_pairs_v2 \
  --splits train val test
```

## Training

가장 많이 쓰는 실행 방식은 wrapper script다.

```bash
bash 3d_molt5/finetune_scripts/mdpo_lora_pubchem_cap_smiles_ep3.sh
```

기본값:

- base checkpoint: `model_checkpoint/3d-molt5-base-pubchem-des.bin`
- train file: `data/dpo_chosen_rejected/train.jsonl`
- eval file: `data/dpo_chosen_rejected/val.jsonl`
- output dir: `checkpoints/mdpo_lora/pubchem_cap_smiles_modality_conflict_${ts}_r${lora_r}_lr${lr}_beta${beta}_ep${epochs}`
- log dir: `logs/mdpo_runs`
- preference loss: `mdpo`
- prompt style: `mdpo_instruction`
- LoRA target modules: `q,k,v,o,wi,wi_0,wi_1,wo`

GPU, checkpoint, hyperparameter는 environment variable로 override한다.

```bash
CUDA_VISIBLE_DEVICES=0 \
ckpt_path=model_checkpoint/3d-molt5-base-pubchem-des.bin \
train_pair_file=data/dpo_chosen_rejected/train.jsonl \
eval_pair_file=data/dpo_chosen_rejected/val.jsonl \
out_dir=checkpoints/mdpo_lora/my_run \
epochs=3 \
batch_size=2 \
grad_acc=8 \
lr=5e-5 \
lora_r=16 \
lora_alpha=32 \
bash 3d_molt5/finetune_scripts/mdpo_lora_pubchem_cap_smiles_ep3.sh
```

동일한 작업을 Python entrypoint로 직접 실행하면 다음과 같다.

```bash
python 3d_molt5/dpo_lora_train.py \
  --checkpoint_path model_checkpoint/3d-molt5-base-pubchem-des.bin \
  --output_dir checkpoints/mdpo_lora/my_run \
  --preference_loss mdpo \
  --prompt_style mdpo_instruction \
  --train_pair_file data/dpo_chosen_rejected/train.jsonl \
  --eval_pair_file data/dpo_chosen_rejected/val.jsonl \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --learning_rate 5e-5 \
  --beta 0.1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_source_length 512 \
  --max_target_length 768 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --save_merged
```

학습 전 데이터 shape만 확인하려면 `--dry_run_data`를 사용한다.

```bash
python 3d_molt5/dpo_lora_train.py \
  --checkpoint_path model_checkpoint/3d-molt5-base-pubchem-des.bin \
  --output_dir /tmp/dry_run \
  --preference_loss mdpo \
  --prompt_style mdpo_instruction \
  --train_pair_file data/dpo_chosen_rejected/train.jsonl \
  --eval_pair_file data/dpo_chosen_rejected/val.jsonl \
  --dry_run_data
```

한 번의 forward/backward smoke test만 수행하려면 `--smoke_test`를 사용한다.

## Preference Loss Modes

`--preference_loss`는 세 가지를 지원한다.

- `dpo`: chosen response와 rejected response의 likelihood ratio를 비교한다.
- `mdpo`: 같은 caption에 대해 chosen E3FP condition과 rejected E3FP condition의 likelihood 차이를 선호 학습한다.
- `copo`: mDPO 계열 condition preference에 SFT/anchor 항을 함께 사용한다.

주요 loss 관련 option:

- `--beta`: preference loss temperature.
- `--logprob_reduction`: `sum` 또는 `mean`.
- `--sft_loss_weight`: DPO mode에서 optional SFT loss weight.
- `--delta`: mDPO/CoPO anchor margin.
- `--lambda_sft`, `--lambda_copo`, `--lambda_anchor`: CoPO/mDPO 보조 loss weight.

## Training Outputs

`--output_dir` 아래에 다음 파일들이 저장된다.

```text
checkpoints/mdpo_lora/my_run/
  adapter_model.bin          # LoRA A/B weight 및 optional fp embedding weight
  adapter_config.json        # base checkpoint, LoRA 설정, data path, best metric metadata
  tokenizer files            # tokenizer_config.json, tokenizer.json, spiece.model 등
  pytorch_model_merged.bin   # --save_merged 사용 시 LoRA가 merge된 전체 model state_dict
  checkpoint-50/             # --save_steps 간격으로 저장되는 adapter checkpoint
  checkpoint-100/
  best_model/
    adapter_model.bin
    adapter_config.json
    best_metrics.json
    tokenizer files
```

`best_model/`은 `--best_model_metric` 기준으로 갱신된다. 기본값은 `eval_loss`이며, metric을 크게 만드는 기준이 필요하면 `--greater_is_better`를 추가한다.

## Inference

wrapper script로 best model inference와 caption metric 평가를 함께 실행한다.

```bash
bash 3d_molt5/finetune_scripts/infer_eval_mdpo_pubchem_cap_best.sh
```

기본값:

- model dir: `checkpoints/mdpo_lora/pubchem_cap_modality_conflict_260617_150441_r16_lr5e-5_beta0.1_ep5/best_model`
- base checkpoint: `model_checkpoint/3d-molt5-base-pubchem-cap.bin`
- input JSONL: `data/dpo_chosen_rejected/test.jsonl`
- output dir: `outputs/mdpo_lora`
- predictions: `${out_dir}/${run_name}_predictions.jsonl`
- metrics: `${out_dir}/${run_name}_metrics.json`

실제 run에서는 보통 경로를 override한다.

```bash
CUDA_VISIBLE_DEVICES=0 \
model_dir=checkpoints/mdpo_lora/my_run/best_model \
base_ckpt=model_checkpoint/3d-molt5-base-pubchem-des.bin \
input_jsonl=data/dpo_chosen_rejected/test.jsonl \
out_dir=outputs/mdpo_lora \
run_name=my_run_best_test \
bash 3d_molt5/finetune_scripts/infer_eval_mdpo_pubchem_cap_best.sh
```

Python entrypoint로 직접 inference만 실행할 수도 있다.

```bash
python 3d_molt5/dpo_lora_infer.py \
  --model_path checkpoints/mdpo_lora/my_run/best_model \
  --base_checkpoint_path model_checkpoint/3d-molt5-base-pubchem-des.bin \
  --input_jsonl data/dpo_chosen_rejected/test.jsonl \
  --output_jsonl outputs/mdpo_lora/my_run_best_test_predictions.jsonl \
  --preference_side chosen \
  --reference_field chosen_caption \
  --prompt_style mdpo_instruction \
  --max_source_length 512 \
  --max_new_tokens 768 \
  --batch_size 4 \
  --dtype bfloat16
```

`--model_path`는 다음 두 형태를 지원한다.

- directory: `adapter_model.bin` 또는 `pytorch_model_merged.bin`이 들어있는 directory.
- file: `.bin` file. 이 경우 merged model file로 취급한다.

Inference output JSONL row는 대략 다음 field를 가진다.

```json
{
  "id": "logp_000001",
  "task": "logp",
  "smiles": "...",
  "selfies": "...",
  "preference_side": "chosen",
  "prompt_style": "mdpo_instruction",
  "prompt": "...",
  "prediction": "generated caption",
  "reference": "reference caption",
  "logP": 1.23,
  "chosen_caption": "...",
  "rejected_caption": "..."
}
```

## Evaluation

`infer_eval_mdpo_pubchem_cap_best.sh`는 inference 후 아래 명령을 자동으로 실행한다.

```bash
python 3d_molt5/eval_caption_predictions.py \
  --predictions_jsonl outputs/mdpo_lora/my_run_best_test_predictions.jsonl \
  --output_json outputs/mdpo_lora/my_run_best_test_metrics.json
```

계산되는 주요 metric:

- `exact_match`
- `bleu`, `bleu_1`, `bleu_2`, `bleu_3`, `bleu_4`
- `rouge1_*`, `rouge2_*`, `rougeL_*`
- `meteor`
- `prediction_length_tokens`, `reference_length_tokens`
- `by_task`: task별 동일 metric

## Sweep

LoRA rank/alpha/lr sweep는 아래 script를 사용한다.

```bash
bash 3d_molt5/finetune_scripts/sweep_mdpo_pubchem_des_lora_lr.sh
```

주요 기본값:

- base checkpoint: `model_checkpoint/3d-molt5-base-pubchem-des.bin`
- train/val/test: `data/dpo_chosen_rejected/{train,val,test}.jsonl`
- checkpoint root: `checkpoints/mdpo_lora`
- sweep output: `outputs/mdpo_lora/sweeps/${RUN_PREFIX}`
- logs: `logs/mdpo_runs`
- GPUs: `GPUS_CSV=4,5`

예시 override:

```bash
GPUS_CSV=0,1 \
RUN_PREFIX=pubchem_des_mdpo_lora_sweep \
BASE_CKPT=model_checkpoint/3d-molt5-base-pubchem-des.bin \
TRAIN_FILE=data/dpo_chosen_rejected/train.jsonl \
VAL_FILE=data/dpo_chosen_rejected/val.jsonl \
TEST_FILE=data/dpo_chosen_rejected/test.jsonl \
bash 3d_molt5/finetune_scripts/sweep_mdpo_pubchem_des_lora_lr.sh
```

생성 파일:

```text
outputs/mdpo_lora/sweeps/${RUN_PREFIX}/
  manifest.csv
  summary.csv
  summary.json
  *_predictions.jsonl
  *_metrics.json
```

`summarize_mdpo_sweep.py`는 `manifest.csv`의 `metrics_json`과 `best_metrics_json`을 읽어 BLEU/ROUGE/METEOR, best eval metric, checkpoint path를 한 테이블로 모은다.

## Git / Artifact Policy

아래 경로는 용량이 크거나 재생성 가능한 산출물이므로 `.gitignore` 대상이다.

- `checkpoints/`
- `model_checkpoint/`
- `outputs/`
- `logs/`
- `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`, `pytorch_model*.bin`, `adapter_model*.bin`

Git에 올릴 때는 source code, shell scripts, README, 작은 설정 파일만 포함하고, checkpoint/output/log는 별도로 관리한다.

## Notes

- `dpo_lora_train.py`와 `dpo_lora_infer.py`는 `e3fp`가 이미 JSONL에 들어있다고 가정한다. SDF에서 E3FP를 새로 만드는 단계는 현재 LoRA 학습/추론 path에 포함되어 있지 않다.
- `prompt_style=mdpo_instruction`은 SMILES, SELFIES, property value, 3D E3FP condition을 모두 사용하는 현재 mDPO 실험용 prompt다.
- `max_source_length`는 prompt와 `<bom>...<eom>` 구간에 들어갈 E3FP span을 함께 고려해야 한다. 현재 wrapper는 training에서 512, inference에서 384 또는 override 값을 사용한다.
- `--save_merged`는 기본적으로 켜져 있다. 저장 공간을 줄이고 adapter만 저장하려면 `--no_save_merged`를 사용한다.
