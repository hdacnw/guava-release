#!/usr/bin/env bash
# Full-parameter SFT for Qwen3.5-VL checkpoints with ms-swift.
#
# Required:
#   DATASET=/absolute/path/to/training.jsonl
#
# Common overrides:
#   MODEL_PATH=Qwen/Qwen3.5-4B
#   OUTPUT_DIR=runs/my-sft
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   N_EPOCHS=3 PER_DEVICE_BS=2 GRAD_ACCUM=2 LR=1e-5
#   WANDB_MODE=disabled
#   ./scripts/run_sft_full.sh
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_PATH=${VENV_PATH:-$REPO_ROOT/.venv-train}
if [[ ! -x "$VENV_PATH/bin/swift" ]]; then
    echo "[run_sft_full] swift not found at $VENV_PATH/bin/swift" >&2
    exit 1
fi
export PATH="$VENV_PATH/bin:$PATH"
set -u

export TMPDIR=${TMPDIR:-/tmp/$USER/swift-tmp}
export TEMP=$TMPDIR
export TMP=$TMPDIR
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/$USER/triton_cache}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

export USE_HF=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export DS_SKIP_CUDA_CHECK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-512}

export WANDB_PROJECT=${WANDB_PROJECT:-Tooluse-VLA}
export WANDB_DIR=${WANDB_DIR:-$REPO_ROOT/runs/wandb}
mkdir -p "$WANDB_DIR"

N_EPOCHS=${N_EPOCHS:-3}
LR=${LR:-1e-5}
PER_DEVICE_BS=${PER_DEVICE_BS:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
MAX_LENGTH=${MAX_LENGTH:-10240}
SAVE_STEPS=${SAVE_STEPS:-25}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

DEEPSPEED=${DEEPSPEED:-zero3}
DS_ARGS=()
if [[ -n "$DEEPSPEED" ]]; then
    DS_ARGS=(--deepspeed "$DEEPSPEED")
fi

N_SAMPLES=${N_SAMPLES:-}
if [[ -n "$N_SAMPLES" ]]; then
    DATASET_SUFFIX="#${N_SAMPLES}"
    RUN_TAG="sft-full-n${N_SAMPLES}-lr${LR}"
else
    DATASET_SUFFIX=""
    RUN_TAG="sft-full-lr${LR}"
fi

OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/runs/qwen35-4b-train-${RUN_TAG}}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
DATASET=${DATASET:-}
if [[ -z "$DATASET" ]]; then
    echo "[run_sft_full] DATASET is required." >&2
    echo "Example: DATASET=/absolute/path/to/training.jsonl ./scripts/run_sft_full.sh" >&2
    exit 2
fi
case "${DATASET%%#*}" in
    /*|./*|../*|*.json|*.jsonl)
        if [[ ! -f "${DATASET%%#*}" ]]; then
            echo "[run_sft_full] dataset not found: ${DATASET%%#*}" >&2
            exit 2
        fi
        ;;
esac
mkdir -p "$OUTPUT_DIR"

SILENCE_TRITON_WARNINGS=${SILENCE_TRITON_WARNINGS:-1}
if [[ "$SILENCE_TRITON_WARNINGS" == "1" && -x "$HOME/.local/bin/gcc-no-cpp-warn" ]]; then
    export CC="$HOME/.local/bin/gcc-no-cpp-warn"
fi

cd "$REPO_ROOT"

"$VENV_PATH/bin/swift" sft \
    --model "$MODEL_PATH" \
    --tuner_type full \
    --dataset "${DATASET}${DATASET_SUFFIX}" \
    --torch_dtype bfloat16 \
    --num_train_epochs "$N_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --lr_scheduler_type cosine \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm 1.0 \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --add_non_thinking_prefix true \
    --use_liger_kernel true \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 5 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --save_strategy steps \
    --eval_strategy no \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --attn_impl ${ATTN_IMPL:-flash_attention_2} \
    "${DS_ARGS[@]}" \
    --report_to wandb \
    --run_name ${WANDB_RUN_NAME:-qwen35-4b-train-${RUN_TAG}}
