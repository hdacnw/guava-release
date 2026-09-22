#!/usr/bin/env bash
# Online GRPO fine-tuning via ms-swift, vLLM, and Guava simulation.
#
# "_full" suffix matches the v10_full.sh convention: full fine-tune (not
# LoRA). Override TUNER_TYPE=lora if you want LoRA instead — the rest of
# the wiring stays the same.
#
# Architecture (4 long-running processes):
#
#     GPU 0    SAM3 server         :8114    (perception, .venv-sam3)
#     GPU 0    sim_server          :8200    (mujoco + robosuite, .venv-eval,
#                                            shares GPU 0 with SAM3 for EGL)
#     GPU 1    swift rollout       :8001    (vLLM + GuavaMultiTurnScheduler,
#                                            .venv-train)
#     GPU 2-7  swift rlhf grpo               (GRPO trainer, .venv-train)
#
# Runtime environments:
#     VENV          — sim_server (defaults to .venv-eval)
#     VENV_SAM3     — SAM3 server (defaults to .venv-sam3)
#     VENV_TRAIN    — swift rollout + GRPO (vllm + ms-swift + flash-attn)
#
# Quick LoRA smoke (~3 min, single train GPU, 4 rollouts/step):
#     TUNER_TYPE=lora DEEPSPEED= \
#       TASK_CONFIGS="configs/rl/shell_game.yaml" \
#       PROMPTS_PER_TASK=2 NUM_GENERATIONS=2 MAX_TURNS=10 \
#       LORA_RANK=4 TRAIN_GPUS=2 PER_DEVICE_BS=2 \
#       ./scripts/run_grpo_full.sh
#
# Full-FT training defaults to the three released RL examples (Shell Game,
# Set Table, and Red Objects in Basket), K=8, and a 6-GPU ZeRO-3 trainer:
#     ./scripts/run_grpo_full.sh
#
# Override knobs (most common; see body for full list):
#     CHECKPOINT          — base model checkpoint (warm-start)
#     TASK_CONFIGS        — space-separated configs/*.yaml paths
#     PROMPTS_PER_TASK    — distinct (task, seed) prompts per task
#     NUM_GENERATIONS     — K rollouts per prompt (GRPO group size); MUST
#                           divide PER_DEVICE_BS × world_size (TRL constraint)
#     MAX_TURNS           — per-rollout turn cap
#     LR                  — learning rate (full FT defaults lower than LoRA)
#     PER_DEVICE_BS / GRAD_ACCUM
#     BETA                — GRPO KL penalty coefficient
#     TEMPERATURE         — sampling temperature for rollouts
#     GUAVA_LENGTH_PENALTY_FREE_TURNS — successful turns before length cost
#     GUAVA_LENGTH_PENALTY_PER_TURN   — deduction per excess successful turn
#     GUAVA_LENGTH_PENALTY_MAX        — maximum success-reward deduction
#     N_EPOCHS            — number of training epochs over the prompt set
#     MAX_STEPS           — explicit trainer steps; defaults to an automatic
#                           ceil(prompts / prompts-per-step) × N_EPOCHS value
#     TUNER_TYPE          — "full" (default) or "lora"
#     LORA_RANK / LORA_ALPHA / LORA_DROPOUT  — only when TUNER_TYPE=lora
#     TRAIN_GPUS          — comma list, e.g. "2,3,4,5,6,7"
#     DEEPSPEED           — "zero3" (default) | "" | zero2 | zero2_offload | zero3_offload
#     PREFLIGHT_KILL       — set to 1 to kill stale Guava/vLLM processes
#     AGGRESSIVE_CLEANUP   — set to 1 to pkill matching grandchildren on exit
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep guava pinned to this checkout even when an environment contains another
# editable installation.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── Venvs ────────────────────────────────────────────────────────────────── #
VENV=${VENV:-$REPO_ROOT/.venv-eval}
VENV_SAM3=${VENV_SAM3:-$REPO_ROOT/.venv-sam3}
VENV_TRAIN=${VENV_TRAIN:-$REPO_ROOT/.venv-train}
PY=$VENV/bin/python
PY_SAM3=$VENV_SAM3/bin/python
PY_TRAIN=$VENV_TRAIN/bin/python

for f in "$PY" "$PY_SAM3" "$PY_TRAIN"; do
    if [[ ! -x "$f" ]]; then
        echo "[run_grpo_full] missing python at $f" >&2
        exit 1
    fi
done
set -u

# .venv-train/bin on PATH so vLLM's EngineCore subprocess can locate `ninja`.
export PATH="$VENV_TRAIN/bin:$PATH"

# ── Tmp / cache / runtime env ────────────────────────────────────────────── #
export TMPDIR=${TMPDIR:-/tmp/$USER/swift-tmp}
export TEMP=$TMPDIR
export TMP=$TMPDIR
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/$USER/triton_cache}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

export USE_HF=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# MUJOCO_GL: egl (GPU offscreen render). EGL's display is process-global
# and not thread-safe, but we get parallelism by running N sim_server
# PROCESSES (each with its own EGL display); inside a process all sim ops
# serialize on a single-thread executor. See cap-x's parallel_eval pattern.
# Osmesa is the CPU fallback (set MUJOCO_GL=osmesa explicitly to use it).
export MUJOCO_GL=${MUJOCO_GL:-egl}

# Match the public v13b short SFT prompt.
export GUAVA_SYSTEM_PROMPT=${GUAVA_SYSTEM_PROMPT:-sft_v13b_short}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-512}
export GUAVA_LENGTH_PENALTY_FREE_TURNS=${GUAVA_LENGTH_PENALTY_FREE_TURNS:-16}
export GUAVA_LENGTH_PENALTY_PER_TURN=${GUAVA_LENGTH_PENALTY_PER_TURN:-0.025}
export GUAVA_LENGTH_PENALTY_MAX=${GUAVA_LENGTH_PENALTY_MAX:-0.10}
unset DISPLAY

export WANDB_PROJECT=${WANDB_PROJECT:-Tooluse-VLA-GRPO}
export WANDB_DIR=${WANDB_DIR:-$REPO_ROOT/runs/wandb}
mkdir -p "$WANDB_DIR"

# ── Model / data ─────────────────────────────────────────────────────────── #
CHECKPOINT=${CHECKPOINT:-AIcell/guava-v13b-qwen3.5-4b}

# Released RL-tuned task configs used by the reference training runs.
TASK_CONFIGS=${TASK_CONFIGS:-"\
configs/rl/shell_game.yaml \
configs/rl/set_table.yaml \
configs/rl/red_objects_in_basket.yaml"}
NORMALIZED_TASK_CONFIGS=""
for task_config in $TASK_CONFIGS; do
    if [[ "$task_config" != /* ]]; then
        task_config="$REPO_ROOT/$task_config"
    fi
    if [[ ! -f "$task_config" ]]; then
        echo "[run_grpo_full] missing task config: $task_config" >&2
        exit 2
    fi
    NORMALIZED_TASK_CONFIGS+="$task_config "
done
TASK_CONFIGS=${NORMALIZED_TASK_CONFIGS% }

# ── GRPO hyperparameters ─────────────────────────────────────────────────── #
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-10}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
START_SEED=${START_SEED:-2000}
MAX_TURNS=${MAX_TURNS:-25}
TEMPERATURE=${TEMPERATURE:-0.7}
BETA=${BETA:-0.04}                           # KL penalty coefficient
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}
MAX_LENGTH=${MAX_LENGTH:-8192}

# ── Optimizer / tuner / batch ────────────────────────────────────────────── #
# Defaults are tuned for FULL fine-tune of a 4B model on 6 H100 GPUs with
# ZeRO-3. Override TUNER_TYPE=lora + DEEPSPEED= for the LoRA path.
TUNER_TYPE=${TUNER_TYPE:-full}               # "full" | "lora"
LR=${LR:-5e-6}                                # full-FT defaults to lower than LoRA
LORA_RANK=${LORA_RANK:-8}                     # ignored when TUNER_TYPE=full
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
PER_DEVICE_BS=${PER_DEVICE_BS:-2}            # Must satisfy NUM_GENERATIONS | (BS × world_size)
GRAD_ACCUM=${GRAD_ACCUM:-2}                  # eff batch = 2 × 2 × 6 = 24
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
N_EPOCHS=${N_EPOCHS:-3}                       # epochs over the prompt set

# DeepSpeed ZeRO. Full-FT defaults to zero3 (shard params + grads + optimizer
# across GPUs — required to fit a 4B model + GRPO ref/old policies on 80GB).
DEEPSPEED=${DEEPSPEED:-zero3}
DS_ARGS=()
if [[ -n "$DEEPSPEED" ]]; then
    DS_ARGS=(--deepspeed "$DEEPSPEED")
fi

# ── GPU layout / ports ───────────────────────────────────────────────────── #
SAM3_GPU=${SAM3_GPU:-0}
ROLLOUT_GPU=${ROLLOUT_GPU:-1}
TRAIN_GPUS=${TRAIN_GPUS:-2,3,4,5,6,7}
SAM3_PORT=${SAM3_PORT:-8114}     # base port; SAM3 replicas land on SAM3_PORT + i
SIM_BASE_PORT=${SIM_BASE_PORT:-8200}
SIM_PORT=$SIM_BASE_PORT       # kept for back-compat (warmup, banner)
VLLM_PORT=${VLLM_PORT:-8001}
SERVER_WAIT_S=${SERVER_WAIT_S:-360}
# Number of independent sim_server PROCESSES. Each owns its own EGL display
# and engine cache; the rollout-side GuavaSimEnv round-robins across them.
# Memory cost ≈ NUM_SIM_SERVERS × |tasks| × ~400 MB engine; with 4×11 ≈ 17 GB
# on the SAM3 GPU (alongside SAM3's ~3 GB).
NUM_SIM_SERVERS=${NUM_SIM_SERVERS:-4}

# Number of SAM3 server replicas. A single SAM3 server queues under heavy
# concurrent load (~8+ clients) and a step of K=8 GRPO with 3 prompts/step
# sends ~24 concurrent /segment requests at episode start when all K rollouts
# of the same seed render the same first frame. Default 4 balances:
#   - GPU0 VRAM: NUM_SIM_SERVERS·~0.4 GB + NUM_SAM3·~3 GB = ~17 + 12 = 29 GB
#   - GPU0 compute: SAM3 forward ~200-500 ms; 4 concurrent on one H100 amortizes
#     CPU pre/post-processing, giving roughly 2-3× effective throughput.
# Set SAM3_GPUS to override placement (e.g. "0,0,1,1" puts 2 on GPU0 and 2 on
# GPU1 — only useful if vLLM's gpu_memory_utilization is dropped below 0.85).
NUM_SAM3_SERVERS=${NUM_SAM3_SERVERS:-4}
SAM3_GPUS=${SAM3_GPUS:-}

NUM_TRAIN_GPUS=$(awk -F, '{print NF}' <<< "$TRAIN_GPUS")
export NPROC_PER_NODE=${NPROC_PER_NODE:-$NUM_TRAIN_GPUS}

# GRPO's vLLM-backed dataloader is iterable, so Transformers requires an
# explicit positive max_steps. Derive the same number of steps that ordinary
# epoch-based training would take unless the caller supplies MAX_STEPS.
EFFECTIVE_BATCH=$(( PER_DEVICE_BS * NUM_TRAIN_GPUS * GRAD_ACCUM ))
if (( EFFECTIVE_BATCH % NUM_GENERATIONS != 0 )); then
    echo "[run_grpo_full] ERROR: NUM_GENERATIONS=$NUM_GENERATIONS must divide effective batch $EFFECTIVE_BATCH" >&2
    exit 1
fi
PROMPTS_PER_STEP=$(( EFFECTIVE_BATCH / NUM_GENERATIONS ))
NUM_TASKS=$(wc -w <<< "$TASK_CONFIGS")
TOTAL_PROMPTS=$(( NUM_TASKS * PROMPTS_PER_TASK ))
STEPS_PER_EPOCH=$(( (TOTAL_PROMPTS + PROMPTS_PER_STEP - 1) / PROMPTS_PER_STEP ))
MAX_STEPS=${MAX_STEPS:-$(( STEPS_PER_EPOCH * N_EPOCHS ))}

# ── Run tag / output dir ─────────────────────────────────────────────────── #
TS=$(date +%Y%m%d_%H%M%S)
if [[ "$TUNER_TYPE" == "lora" ]]; then
    RUN_TAG=${RUN_TAG:-grpo-lora-r${LORA_RANK}-lr${LR}-K${NUM_GENERATIONS}}
else
    RUN_TAG=${RUN_TAG:-grpo-fullft-lr${LR}-K${NUM_GENERATIONS}}
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/runs/${RUN_TAG}_${TS}}
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p "$LOG_DIR"

# ── Tuner-specific arg array ─────────────────────────────────────────────── #
TUNER_ARGS=(--tuner_type "$TUNER_TYPE")
if [[ "$TUNER_TYPE" == "lora" ]]; then
    TUNER_ARGS+=(
        --lora_rank "$LORA_RANK"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
        --target_modules all-linear
    )
fi

# ── Banner ───────────────────────────────────────────────────────────────── #
echo "[run_grpo_full] checkpoint:   $CHECKPOINT"
echo "[run_grpo_full] tasks:        $(echo "$TASK_CONFIGS" | tr ' ' '\n' | grep -c '\.yaml') configs"
echo "[run_grpo_full] grpo:         prompts/task=$PROMPTS_PER_TASK  K=$NUM_GENERATIONS  max_turns=$MAX_TURNS  beta=$BETA  temp=$TEMPERATURE"
echo "[run_grpo_full] length cost:  success-only, free=$GUAVA_LENGTH_PENALTY_FREE_TURNS  per_turn=$GUAVA_LENGTH_PENALTY_PER_TURN  cap=$GUAVA_LENGTH_PENALTY_MAX"
echo "[run_grpo_full] opt:          tuner=$TUNER_TYPE  lr=$LR  per_device_bs=$PER_DEVICE_BS  grad_accum=$GRAD_ACCUM  epochs=$N_EPOCHS"
echo "[run_grpo_full] steps:        prompts=$TOTAL_PROMPTS  prompts/step=$PROMPTS_PER_STEP  max_steps=$MAX_STEPS"
if [[ "$TUNER_TYPE" == "lora" ]]; then
    echo "[run_grpo_full] lora:         r=$LORA_RANK  alpha=$LORA_ALPHA  dropout=$LORA_DROPOUT"
fi
echo "[run_grpo_full] deepspeed:    ${DEEPSPEED:-disabled}"
if [[ -n "$SAM3_GPUS" ]]; then
    SAM3_GPU_BANNER="$SAM3_GPUS"
else
    # Build "<gpu>,<gpu>,..." by repeating $SAM3_GPU $NUM_SAM3_SERVERS times.
    # Used to be `yes "$SAM3_GPU" | head -n N | paste -sd, -` but `yes` exits
    # with SIGPIPE (rc=141) when head closes the pipe, and `set -eo pipefail`
    # then kills the whole script right here at the banner — confused half
    # of yesterday's "GRPO died instantly" failures. Use a plain for-loop.
    SAM3_GPU_BANNER=""
    for ((i=0; i<NUM_SAM3_SERVERS; i++)); do
        SAM3_GPU_BANNER+="${SAM3_GPU},"
    done
    SAM3_GPU_BANNER=${SAM3_GPU_BANNER%,}
fi
echo "[run_grpo_full] gpus:         SAM3:[$SAM3_GPU_BANNER] (×$NUM_SAM3_SERVERS)  rollout:$ROLLOUT_GPU  train:$TRAIN_GPUS"
SAM3_PORTS_BANNER=""
for ((s=0; s<NUM_SAM3_SERVERS; s++)); do
    SAM3_PORTS_BANNER+="$((SAM3_PORT + s)),"
done
SAM3_PORTS_BANNER=${SAM3_PORTS_BANNER%,}
SIM_PORTS_BANNER=""
for ((w=0; w<NUM_SIM_SERVERS; w++)); do
    SIM_PORTS_BANNER+="$((SIM_BASE_PORT + w)),"
done
SIM_PORTS_BANNER=${SIM_PORTS_BANNER%,}
echo "[run_grpo_full] ports:        SAM3:[$SAM3_PORTS_BANNER]  sim:[$SIM_PORTS_BANNER]  vllm:$VLLM_PORT"
echo "[run_grpo_full] output:       $OUTPUT_DIR"

# ── Generate prompt dataset ──────────────────────────────────────────────── #
# Each row is `{messages, env_config}`. The trainer (with
# --vllm_server_pass_dataset true) forwards every non-REQUEST_METADATA_FIELDS
# top-level key into data_dict; on the rollout side our guava_tool_scheduler
# reads data_dict['env_config'] to drive the sim.
PROMPT_FILE=$OUTPUT_DIR/grpo_prompts.jsonl
$PY - <<EOF
import json, os
tasks = """$TASK_CONFIGS""".split()
prompts_per_task = $PROMPTS_PER_TASK
start_seed = $START_SEED
rows = []
for t_idx, task in enumerate(tasks):
    for p in range(prompts_per_task):
        seed = start_seed + p + 10_000 * t_idx
        rows.append({
            "messages": [{"role": "user", "content": "rollout"}],
            "env_config": {
                "task_config": task,
                "seed": seed,
                "output_dir": "$OUTPUT_DIR/rollouts",
            },
        })
os.makedirs(os.path.dirname("$PROMPT_FILE"), exist_ok=True)
with open("$PROMPT_FILE", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} prompt rows → $PROMPT_FILE")
EOF

# ── Preflight: kill any leftover services from a previous run ────────────── #
# Stuck VLLM::EngineCore subprocesses commonly survive cleanup traps when the
# launcher is SIGKILLed, then pin GPU memory and break the next launch. Catch
# them by name before we touch any GPU.
preflight_kill() {
    local patterns=("VLLM::EngineCore" "swift.cli.rollout" "swift.cli.rlhf"
                    "guava.scripts.sim_server" "guava.scripts.sam3_server")
    local zombies=()
    for pat in "${patterns[@]}"; do
        while read -r pid; do
            [[ -n "$pid" ]] && zombies+=("$pid")
        done < <(pgrep -f "$pat" 2>/dev/null || true)
    done
    if (( ${#zombies[@]} > 0 )); then
        echo "[run_grpo_full] preflight: killing ${#zombies[@]} leftover process(es): ${zombies[*]}"
        kill -9 "${zombies[@]}" 2>/dev/null || true
        sleep 3
    fi
}
if [[ "${PREFLIGHT_KILL:-0}" == "1" ]]; then
    preflight_kill
fi

# Verify the GPUs we need are actually free.
for gpu in "$SAM3_GPU" "$ROLLOUT_GPU"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
    if [[ -n "$used" && "$used" -gt 2000 ]]; then
        echo "[run_grpo_full] ERROR: GPU $gpu has ${used} MiB in use." >&2
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i "$gpu" >&2
        echo "[run_grpo_full] Kill the offending process(es) above and retry." >&2
        exit 1
    fi
done

# ── Spawn long-running services ──────────────────────────────────────────── #
SERVER_PIDS=()
cleanup() {
    echo "[run_grpo_full] cleanup: killing ${#SERVER_PIDS[@]} background process(es)…"
    for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    sleep 3
    if [[ "${AGGRESSIVE_CLEANUP:-0}" == "1" ]]; then
        # Dedicated training nodes may opt into cleaning grandchildren that
        # outlive their launcher. This is intentionally off by default.
        pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
        pkill -9 -f "swift.cli.rollout" 2>/dev/null || true
        pkill -9 -f "guava.scripts.sim_server" 2>/dev/null || true
        pkill -9 -f "guava.scripts.sam3_server" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for() {
    local url=$1 deadline=$(( $(date +%s) + ${2:-180} ))
    while (( $(date +%s) < deadline )); do
        if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
        sleep 3
    done
    return 1
}

# 1) SAM3 server replicas — N processes on consecutive ports.
#    Each holds its own model copy on GPU; clients (SAM3Solver) round-robin
#    across the comma-separated GUAVA_SAM3_URLS env var, which child processes
#    (sim_server, swift rollout, trainer) all inherit.
SAM3_URLS=""
SAM3_PORTS=""
for ((s=0; s<NUM_SAM3_SERVERS; s++)); do
    port=$((SAM3_PORT + s))
    if [[ -n "$SAM3_GPUS" ]]; then
        gpu=$(awk -F, -v i=$((s+1)) '{print $i}' <<< "$SAM3_GPUS")
        [[ -z "$gpu" ]] && gpu=$SAM3_GPU   # underflow → fall back to default
    else
        gpu=$SAM3_GPU
    fi
    echo "[run_grpo_full] starting SAM3[$s] :$port on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
        "$PY_SAM3" -m guava.scripts.sam3_server --port "$port" --device cuda \
        >"$LOG_DIR/sam3_$s.log" 2>&1 &
    SERVER_PIDS+=("$!")
    SAM3_URLS+="http://127.0.0.1:$port,"
    SAM3_PORTS+="$port,"
done
SAM3_URLS=${SAM3_URLS%,}
SAM3_PORTS=${SAM3_PORTS%,}
# Exported so sim_server / swift rollout / GRPO trainer all see the same pool.
# SAM3Solver.__init__ reads this and ignores the single sam3_url in YAML if set.
export GUAVA_SAM3_URLS=$SAM3_URLS

# 2) sim_server replicas — N independent OS processes on consecutive ports.
#    Each owns one EGL display (process-global) and its own engine cache;
#    parallelism comes from running N of them, not from threading within one.
#    Share GPU 0 with SAM3 — compute footprint is tiny per process.
SIM_URLS=""
for ((w=0; w<NUM_SIM_SERVERS; w++)); do
    port=$((SIM_BASE_PORT + w))
    echo "[run_grpo_full] starting sim_server[$w] :$port on GPU $SAM3_GPU (EGL)"
    CUDA_VISIBLE_DEVICES=$SAM3_GPU \
        "$PY" -m guava.scripts.sim_server --port "$port" \
        >"$LOG_DIR/sim_server_$w.log" 2>&1 &
    SERVER_PIDS+=("$!")
    SIM_URLS+="http://127.0.0.1:$port,"
done
SIM_URLS=${SIM_URLS%,}                 # strip trailing comma
export GUAVA_SIM_SERVER_URLS=$SIM_URLS  # picked up by GuavaSimEnv on rollout side

# 3) swift rollout server (vLLM + multi-turn scheduler)
echo "[run_grpo_full] starting swift rollout :$VLLM_PORT on GPU $ROLLOUT_GPU"
env -u VLLM_PORT CUDA_VISIBLE_DEVICES=$ROLLOUT_GPU \
    "$PY_TRAIN" -m swift.cli.rollout \
        --model "$CHECKPOINT" \
        --port "$VLLM_PORT" \
        --vllm_use_async_engine true \
        --use_gym_env true \
        --gym_env guava_sim \
        --multi_turn_scheduler guava_tool_scheduler \
        --max_turns "$MAX_TURNS" \
        --external_plugins "$REPO_ROOT/guava/rl/swift_gym_env.py" \
        --temperature "$TEMPERATURE" \
        --vllm_data_parallel_size 1 \
        >"$LOG_DIR/rollout.log" 2>&1 &
SERVER_PIDS+=("$!")

echo "[run_grpo_full] waiting for $NUM_SAM3_SERVERS SAM3 server(s) …"
for ((s=0; s<NUM_SAM3_SERVERS; s++)); do
    port=$((SAM3_PORT + s))
    wait_for "http://127.0.0.1:$port/health" "$SERVER_WAIT_S" || {
        echo "SAM3[$s] :$port failed"; tail -30 "$LOG_DIR/sam3_$s.log" >&2; exit 1; }
done
for ((w=0; w<NUM_SIM_SERVERS; w++)); do
    port=$((SIM_BASE_PORT + w))
    echo "[run_grpo_full] waiting for sim_server[$w] :$port …"
    wait_for "http://127.0.0.1:$port/health" 60 || {
        echo "sim_server[$w] failed"; tail -30 "$LOG_DIR/sim_server_$w.log" >&2; exit 1; }
done
echo "[run_grpo_full] waiting for swift rollout (vLLM warming up) …"
wait_for "http://127.0.0.1:$VLLM_PORT/health/" "$SERVER_WAIT_S" || {
    echo "rollout failed"; tail -40 "$LOG_DIR/rollout.log" >&2; exit 1; }

# ── Prewarm sim_server replicas ──────────────────────────────────────────── #
# Each sim_server process has its own engine cache, so we must warm every
# (port, task) pair. First /reset triggers mujoco/robosuite/pybullet import
# (~2-3 min cold the very first time per process, ~2 s for subsequent tasks
# in the same process). Without this, iter 0's K parallel rollouts hit
# sim_server before any engine is built and the trainer's httpx call times
# out as "Server disconnected without sending a response".
#
# Parallelism: we fire warmups for each task in parallel across the N
# sim_server processes (different processes don't share any state, so this
# is safe). Tasks within the same process serialize on its single sim thread.
echo "[run_grpo_full] pre-warming $NUM_SIM_SERVERS sim_server(s) × $(echo "$TASK_CONFIGS" | wc -w) tasks"
WARMUP_DIR=$OUTPUT_DIR/warmup
mkdir -p "$WARMUP_DIR"
WARMUP_PIDS=()
for ((w=0; w<NUM_SIM_SERVERS; w++)); do
    port=$((SIM_BASE_PORT + w))
    (
        for cfg in $TASK_CONFIGS; do
            tag=$(basename "$cfg" .yaml)
            if ! curl --max-time 600 -sf -X POST "http://127.0.0.1:$port/reset" \
                -H "Content-Type: application/json" \
                -d "{\"task_config\": \"$cfg\", \"seed\": 0, \"output_dir\": \"$WARMUP_DIR\"}" \
                -o "$WARMUP_DIR/warmup_${w}_${tag}.json"; then
                echo "[run_grpo_full] WARN: warmup /reset failed for $cfg on port $port" >&2
            fi
        done
    ) &
    WARMUP_PIDS+=("$!")
done
for pid in "${WARMUP_PIDS[@]}"; do wait "$pid" || true; done
echo "[run_grpo_full] all services ready (sim_urls=$SIM_URLS)."

# ── Launch GRPO trainer ──────────────────────────────────────────────────── #
echo "[run_grpo_full] starting swift rlhf grpo on GPUs $TRAIN_GPUS"

cd "$REPO_ROOT"

# Use the `swift` CLI wrapper (not `python -m swift.cli.rlhf`) so it sees
# NPROC_PER_NODE and re-execs through torch.distributed.run, spawning one
# process per train GPU. Going through `python -m` gives us a single process
# trying to span all GPUs via device_map, which DeepSpeed rejects.
env -u VLLM_PORT CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
    "$VENV_TRAIN/bin/swift" rlhf \
        --rlhf_type grpo \
        --model "$CHECKPOINT" \
        "${TUNER_ARGS[@]}" \
        --freeze_vit true \
        --freeze_aligner true \
        --dataset "$PROMPT_FILE" \
        --external_plugins "$REPO_ROOT/guava/rl/swift_gym_env.py" \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host 127.0.0.1 \
        --vllm_server_port "$VLLM_PORT" \
        --vllm_server_pass_dataset true \
        --max_turns "$MAX_TURNS" \
        --num_generations "$NUM_GENERATIONS" \
        --beta "$BETA" \
        --temperature "$TEMPERATURE" \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --max_length "$MAX_LENGTH" \
        --num_train_epochs "$N_EPOCHS" \
        --max_steps "$MAX_STEPS" \
        --per_device_train_batch_size "$PER_DEVICE_BS" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --lr_scheduler_type "$LR_SCHEDULER" \
        --warmup_ratio "$WARMUP_RATIO" \
        --weight_decay "$WEIGHT_DECAY" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --torch_dtype bfloat16 \
        --gradient_checkpointing true \
        --output_dir "$OUTPUT_DIR" \
        --logging_steps 1 \
        --save_steps "${SAVE_STEPS:-50}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
        --eval_strategy no \
        --dataset_num_proc 2 \
        --dataloader_num_workers 2 \
        --attn_impl ${ATTN_IMPL:-flash_attention_2} \
        "${DS_ARGS[@]}" \
        --report_to wandb \
        --run_name "${WANDB_RUN_NAME:-$RUN_TAG}"

echo "[run_grpo_full] training finished. Output: $OUTPUT_DIR"
