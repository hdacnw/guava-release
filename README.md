# Guava

Guava is a harness and model distillation framework for agentic manipulation. It supports agent-controlled robot manipulation in simulation, a 15-task benchmark, and data collection in MuJoCo for opensource model fine-tuning.

## Installation

Require CUDA-capable GPU. Development and validation have used Ubuntu with Python 3.11 and an RTX 5090; other hardware may require different serving settings.

1. Install Git and `uv`, and obtain access to SAM3 weights [https://huggingface.co/facebook/sam3](SAM3)   using your own Hugging Face account.
2. Clone the repository and initialize the required submodules:
  ```bash
   git clone --branch pre-release https://github.com/hdacnw/guava-release.git
   cd guava-release
   git submodule update --init third_party/robosuite third_party/sam3
   python3 scripts/setup_sam3_compat.py
   python3 scripts/setup_sam3_compat.py --check
  ```

   
3. Sync the simulator and perception groups into separate environments:
  ```bash
   UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --locked --no-default-groups --group eval
  
   UV_PROJECT_ENVIRONMENT=.venv-sam3 uv sync --locked --no-default-groups --group sam3
   export MUJOCO_GL=egl
  ```
4. For local Qwen or Guava checkpoints, also create the serving environment:
  ```bash
   UV_PROJECT_ENVIRONMENT=.venv-serve uv sync --locked --no-default-groups --group serve
  ```

Dependencies are declared once in `pyproject.toml` and resolved in `uv.lock`
(validated with uv 0.11.24).
The `eval` group includes simulation, hosted-model clients, collection, and data
preparation; `sam3` provides perception; `serve` provides local vLLM inference.
These groups are mutually exclusive: SAM3 and vLLM require different Torch
versions. Do not use `--all-groups` or sync multiple groups into one environment.
The lockfile targets Linux x86-64 and Python 3.11. CUDA wheel sources are configured
in `pyproject.toml`; no extra package-index flags are needed. Local serving also
requires a host C++ compiler (for example, Ubuntu's `build-essential` package).
The serving group includes matching CUDA compiler/headers and Ninja for JIT builds.
The managed evaluation launcher also creates missing CUDA linker symlinks inside
the serving environment for FlashInfer. A caller-supplied `CUDA_HOME` is left
untouched and must point to a complete compatible toolkit.

Use the explicit environment paths above: a bare `uv sync` defaults to the `eval`
group in `.venv`, which may replace packages in an existing `.venv`. Likewise,
syncing an existing environment removes packages not needed by its selected group.
Use a fresh environment path if you need to preserve a previous setup. After
editing dependencies, run `uv lock` and commit both configuration and lockfile.
Fine-tuning frameworks and the optional GraspGen server are separate installations,
not part of these three runtime groups.

## Train Guava models

Training uses a fourth, isolated environment because ms-swift, DeepSpeed, vLLM,
and FlashAttention require a different Torch stack:

```bash
uv venv .venv-train --python 3.11
uv pip install --python .venv-train torch==2.11.0
uv pip install --python .venv-train --no-build-isolation -r requirements-train.txt
```

Run full-parameter SFT on a local JSONL file or a dataset supported by
ms-swift:

```bash
DATASET=/absolute/path/to/training.jsonl \
MODEL_PATH=Qwen/Qwen3.5-4B \
OUTPUT_DIR=runs/guava-sft \
./scripts/run_sft_full.sh
```

The default SFT configuration uses eight visible GPUs, global batch size 32
(`2 × 8 × gradient accumulation 2`), image-token limit 512, ZeRO-3, and saves
every 25 optimizer steps. All values can be overridden through environment
variables documented at the top of the script.

Online GRPO launches SAM3, multiple simulator workers, a vLLM rollout server,
and a six-GPU full-parameter trainer. Its released examples are Shell Game,
Set Table, and Red Objects in Basket:

```bash
CHECKPOINT=/absolute/path/to/sft-checkpoint \
NUM_GENERATIONS=8 MAX_STEPS=50 SAVE_STEPS=25 \
./scripts/run_grpo_full.sh
```

Successful trajectories receive binary task-owned reward. Successes longer
than 16 turns receive a `0.025` per-turn length penalty capped at `0.10`;
failures remain at zero. Override these values with
`GUAVA_LENGTH_PENALTY_FREE_TURNS`, `GUAVA_LENGTH_PENALTY_PER_TURN`, and
`GUAVA_LENGTH_PENALTY_MAX`.

## Quick start: evaluate a model

Hosted models use **OpenRouter by default**. Set `OPENROUTER_API_KEY` in your shell environment.

Preview a one-episode evaluation without starting services or making API calls:

```bash
.venv-eval/bin/python -m guava.inference.run_study \
  --models openai/gpt-5.4 --tasks can_in_bin --episodes 1 \
  --output runs/gpt-can-in-bin --dry-run
```

Remove `--dry-run` to execute. The launcher manages SAM3 automatically and also
starts vLLM when a local model is selected. Keep ports 8000 and 8114 available,
or use `--external-servers` with exactly one model to use your own services.

### Choose a model or provider

Use the exact model ID from your provider or Hugging Face model card—no
Guava-specific model aliases are required.


| Provider             | Example selection                                        |
| -------------------- | -------------------------------------------------------- |
| OpenRouter (default) | `--models openai/gpt-5.4`                                |
| Direct OpenAI        | `--provider openai --models gpt-5.4`                     |
| Local Qwen           | `--provider local --models Qwen/Qwen3.5-4B`              |
| Local Guava          | `--provider local --models AIcell/guava-v13b-qwen3.5-4b` |


For another hosted VLM, pass its provider model ID directly to `--models`.
Set `OPENROUTER_API_KEY` for OpenRouter or `OPENAI_API_KEY` for direct OpenAI.


Evaluate your own full checkpoint:

```bash
.venv-eval/bin/python -m guava.inference.run_study \
  --provider local --models /absolute/path/to/checkpoint --prompt-profile sft \
  --tasks can_in_bin --episodes 1 --output runs/custom-checkpoint
```

For a Hugging Face checkpoint, use `--provider local --models ORGANIZATION/MODEL`.
The listed Qwen/Guava checkpoints have pinned revisions. Other Hugging Face models
require `--revision COMMIT`, with one model per invocation when overriding a
revision.

Use `--prompt-profile sft` for short SFT prompt, or `--prompt-profile long` to explicitly select the long data generation prompt.

## Run the benchmark

```bash
.venv-eval/bin/python -m guava.inference.run_study \
  --provider local --models Qwen/Qwen3.5-4B AIcell/guava-v13b-qwen3.5-4b --episodes 30 --seed 42 \
  --output runs/benchmark --dry-run
```

Without `--tasks`, all 15 tasks run. With no model/provider flags, the default is
`openai/gpt-5.4` through OpenRouter. Choose a new output directory under
`runs/` or `data/`.


| Task ID                     | Task                                                             | Default turn cap |
| --------------------------- | ---------------------------------------------------------------- | ----------------: |
| `can_in_bin`                | Place the can in the box                                         | 25               |
| `apple_juice_order`         | Arrange apple and juice bottle by increasing size, left to right | 25               |
| `apple_juice_reverse_order` | Arrange apple and juice bottle by decreasing size, left to right | 25               |
| `remove_cube_from_tray`     | Remove the cube from the tray                                    | 25               |
| `push_basket`               | Push the basket left or right                                    | 25               |
| `close_drawer`              | Close the drawer                                                 | 25               |
| `pick_up_carrot`            | Pick up the carrot                                               | 25               |
| `tomato_near_potato`        | Move the tomato nearer the potato                                | 25               |
| `lemon_in_bin`              | Place the lemon in the bin                                       | 25               |
| `push_pot`                  | Push the pot left or right                                       | 25               |
| `cube_stack_reverse`        | Stack the green cube on the red cube                             | 25               |
| `bin_and_tray_simple`       | Put food in the bin and utensils on the tray                     | 30               |
| `set_table`                 | Set the table                                                    | 30               |
| `shell_game`                | Retrieve and lift the hidden cube                                | 30               |
| `red_objects_in_basket`     | Place all red objects in the basket                              | 30               |


`--max-turns N` overrides both action and decision caps. The default protocol uses full tool access, PCA grasping, motion speed 1, and 20 initial settling steps, all adjustable via per task `.yaml` configs. Shell game uses `sideview`; other tasks use `frontview`. Outputs include manifests, prompts, responses,
action logs, images, parser audits, and episode results.

## Data Collection

Start SAM3 in a separate terminal:

```bash
.venv-sam3/bin/python -m guava.scripts.sam3_server --port 8114 --device cuda
```

With `OPENROUTER_API_KEY` set, collect a single base trajectory:

```bash
.venv-eval/bin/python -m guava.scripts.collect_release \
  --task can_in_bin --trials 1 --seed 42 --max-turns 25 \
  --output data/collect/can_in_bin
```

Use `--model PROVIDER_MODEL_ID` to select another model, or `--provider openai`for direct OpenAI models. Add `--dry-run` to inspect settings. Change `--seed` for a
different set of initializations.

Generate perturbation and recovery branches from the collected checkpoints:

```bash
.venv-eval/bin/python -m guava.scripts.collect_branches \
  --config configs/can_in_bin.yaml \
  --baseline_dir data/collect/can_in_bin \
  --output_dir data/branches/can_in_bin --max-branches 3 \
  --perturbations grasp_failure object_moved_align drop_during_transport \
  --dryrun
```

Remove `--dryrun` to generate branches. Use `--trial_ids` and `--turns` to
select parent trials and intervention points. Available perturbations are
`grasp_failure`, `wrong_object`, `drop_during_transport`,
`object_moved_align`, `wrong_approach_side`, and `wrong_orientation`.

Branches restore simulator/controller state and preserve parent history.
 

### Filter and prepare data

```bash
.venv-eval/bin/python scripts/filter_trajectories.py \
  data/collect/can_in_bin data/branches/can_in_bin \
  --output data/can-in-bin-filter.json
```

Filtering writes an acceptance manifest; it does not modify source trajectories
or assemble a training dataset. It checks format, labels, images, duplicates,
action limits, and available provenance. Visual review is still necessary.

After assembling accepted records into `data/MY_DATASET/finetune_data.jsonl`,
use `scripts/convert_to_native.py MY_DATASET` to convert them to Qwen3.5 native  
format.  The default filter excludes known evaluation-only tasks from SFT data.

## Prompts and configuration

- **Data collection:** [shared long prompt](guava/collection/system_prompt.txt).
- **Fine-tune:** [short SFT prompt](configs/prompts/sft_v13b_short.txt).
- **Task scenes and settings:** [configs/](configs/).
- **Configuration schema:** [guava/config.py](guava/config.py).

Model-facing positions use metres in a table-aligned frame: tabletop `z=0`, with x/y aligned to the robot base. Tools perform the physical-frame conversion. PCA grasping is the default. Optional GraspGen requires a separate installation,  
weights, and running service; see [https://github.com/NVlabs/GraspGen](https://github.com/NVlabs/GraspGen).

## Record episode videos

Add `--record-video` to evaluation or collection commands. Use `--video-fps N`
to change the playback rate (default 20, supported range 1–60):

```bash
.venv-eval/bin/python -m guava.inference.run_study \
  --models openai/gpt-5.4 --tasks set_table --episodes 1 \
  --output runs/set-table-video --record-video

.venv-eval/bin/python -m guava.scripts.collect_release \
  --task can_in_bin --trials 1 --output data/collect/can-video \
  --record-video --video-fps 20
```

`collect_branches` accepts the same flags. Videos use the active observation
camera and include motion between tool calls, not just the saved observation
images. Playback follows simulation time; model/API waiting is omitted.
Recording adds rendering/encoding overhead but does not add physics steps.

- Evaluation: `episode.mp4` in each episode directory, including post-stop settling.
- Base collection: `videos/trial_NNNN.mp4` under the collection output directory.
- Perturbation collection: `videos_branch/trial_NNNN_branch_NNNN.mp4`, starting from
  the restored state and including the intervention (not the earlier parent video).

Each video has a `.video.json` sidecar with camera, FPS, frame count, and simulation
times. Recording is disabled by default. It uses ImageIO/FFmpeg from the evaluation
environment; ordinary Python exceptions finalize a partial clip, while a force-killed
process may leave an incomplete file. Existing videos are never overwritten.
For YAML-driven collection, set top-level `record_video: true` and `video_fps: 20`.

## Real World Deployment

This repository  **does not** include physical-robot operation, please adapt according to your own setup.
