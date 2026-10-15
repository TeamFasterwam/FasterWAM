# LIBERO and LIBERO-Plus

Faster-WAM is trained on LIBERO. The same checkpoint and normalization
statistics can then be evaluated on standard LIBERO or LIBERO-Plus. Follow the
sections below in order, starting with the `fasterwam` training environment from
[README.md](README.md#installation). Run all commands from the repository root.
The simulator environments are needed only for evaluation.

## Training data

Activate the training environment before downloading or preprocessing:

```bash
conda activate fasterwam
export PYTHONNOUSERSITE=1
source scripts/env.sh train
```

Download the preprocessed data from
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam).
Choose **one** format. The directly usable LeRobot v3.0 layout is:

```bash
huggingface-cli download yuanty/LIBERO-fastwam --repo-type dataset \
  --include "lerobot_v30/*" --local-dir data/libero_download
export LIBERO_DATA_ROOT="$PWD/data/libero_download/lerobot_v30"
```

Alternatively, download and extract the LeRobot v2.1 archives:

```bash
huggingface-cli download yuanty/LIBERO-fastwam --repo-type dataset \
  --include "*.tar.gz" --local-dir data/libero_mujoco3.3.2
(cd data/libero_mujoco3.3.2 && for f in *.tar.gz; do tar -xzf "$f"; done)
export LIBERO_DATA_ROOT="$PWD/data/libero_mujoco3.3.2"
```

Both formats contain the same samples. `LIBERO_DATA_ROOT` is their parent
folder, with this layout:

```text
$LIBERO_DATA_ROOT/
  libero_spatial_no_noops_lerobot/
  libero_object_no_noops_lerobot/
  libero_goal_no_noops_lerobot/
  libero_10_no_noops_lerobot/
```

Keep this variable exported for preprocessing and training. In a new shell,
export the path for the format you actually downloaded; the config's default
path is the v2.1 directory. If a download is interrupted, rerun the same command
to resume its partial files before extracting anything.

Prepare the training instruction embeddings once:

```bash
export LIBERO_TEXT_CACHE="$PWD/data/text_embeds_cache/libero"
CUDA_VISIBLE_DEVICES=0 python -m fasterwam.tools.precompute_text_embeds \
  data=libero_2cam +overwrite=false
```

The first invocation also downloads the T5 encoder and tokenizer. The standard
dataset has 40 distinct instructions. `+overwrite=false` skips existing
embeddings when continuing an interrupted run. Use the same cache path during
training. Evaluation has its own instructions and encodes them online by default.

The [dataset config](configs/data/libero_2cam.yaml) defines the two cameras,
video sampling and action/state transforms. Normalization statistics are
bundled in `configs/data/libero_stats.json`. If training on different data,
`data.train.pretrained_norm_stats=null` recomputes them.

## Training

Still in the training environment, with `LIBERO_DATA_ROOT` and
`LIBERO_TEXT_CACHE` exported, run the reported 8-GPU recipe:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh libero 8 \
  batch_size=8 gradient_accumulation_steps=2 num_epochs=10 \
  output_dir="$PWD/runs/libero/fasterwam"
```

This gives global batch size 128. For a different GPU count, adjust batch size
and accumulation as described in [README.md](README.md#training-and-checkpoints).
Use a new output directory for each experiment. For ablations, add
`model=<preset>` from [the preset table](docs/ablations.md); evaluation must use
the same preset and dimensions. [VAE precomputation](README_precompute_VAE.md)
is optional; the commands above encode videos during training.

For a short first run, use this **instead of** the full recipe:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh libero 2 \
  batch_size=1 gradient_accumulation_steps=1 max_steps=2 \
  num_workers=2 log_every=1 save_every=0 eval_every=0 \
  model.mot_checkpoint_mixed_attn=true data.train.strict_getitem=true \
  output_dir="$PWD/runs/libero/short"
```

This trains the full model for two optimizer steps and saves
`runs/libero/short/checkpoints/weights/step_000002.pt`, its full resume state,
and `runs/libero/short/dataset_stats.json`. Allow about 80 GiB for the saved
weights and state. The completion log contains `max_steps reached step=2`.
A two-step run checks execution, not task accuracy.

## Evaluation environments

Standard LIBERO and LIBERO-Plus both install a package named `libero`. Create
one new environment per benchmark. Do not install their upstream
`requirements.txt`: those pin versions that conflict with the model stack.

### Standard LIBERO

The following uses a checkout under `data/benchmarks/`, separate from the
training dataset, and an isolated config under `data/benchmark_config/`:

```bash
export LIBERO_ROOT="$PWD/data/benchmarks/LIBERO"
export LIBERO_CONFIG_PATH="$PWD/data/benchmark_config/libero"
mkdir -p data/benchmarks
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
git -C "$LIBERO_ROOT" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01

conda create -n fasterwam-libero python=3.10 -y
conda activate fasterwam-libero
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge "ffmpeg<8" -y
bash scripts/setup_env.sh libero
source scripts/env.sh libero
```

### LIBERO-Plus

LIBERO-Plus additionally requires ImageMagick and Fontconfig development
libraries (on Debian/Ubuntu: `sudo apt install libmagickwand-dev libfontconfig1-dev`).
Use this pinned revision in a separate environment:

```bash
export LIBERO_PLUS_ROOT="$PWD/data/benchmarks/LIBERO-plus"
export LIBERO_CONFIG_PATH="$PWD/data/benchmark_config/libero_plus"
mkdir -p data/benchmarks
git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_ROOT"
git -C "$LIBERO_PLUS_ROOT" checkout 4976dc30028e805ff8094b55501d532c48fec182

conda create -n fasterwam-liberoplus python=3.10 -y
conda activate fasterwam-liberoplus
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge "ffmpeg<8" -y
bash scripts/setup_env.sh libero_plus
source scripts/env.sh libero_plus

huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --local-dir data/libero_plus_assets
python -m zipfile -e data/libero_plus_assets/assets.zip \
  data/libero_plus_assets/unpacked
ASSETS_DIR="$(find data/libero_plus_assets/unpacked -type d -name assets -print -quit)"
mv -T "$ASSETS_DIR" "$LIBERO_PLUS_ROOT/libero/libero/assets"
```

The archive includes a nested directory prefix from the upstream machine. The
commands above locate and move the actual `assets/` directory into the fresh
simulator checkout. After extraction,
`$LIBERO_PLUS_ROOT/libero/libero/assets/` should directly contain
`articulated_objects/`, `scenes/`, `textures/`, `new_objects/`,
`stable_hope_objects/`, `stable_scanned_objects/` and `turbosquid_objects/`.
Avoid a nested `assets/assets/` directory. These simulator assets are separate
from the LIBERO training data. Allow about 20 GB for the archive and extracted
assets together.

### What the installer configures

The installer pins robosuite 1.4.0, MuJoCo 3.3.2, robomimic 0.2.0 and NumPy
1.26.4, then installs the selected benchmark in editable mode. It also applies
compatibility fixes for PyTorch's init-state loading, robosuite's per-user
temporary log, and LIBERO-Plus's missing package marker. It writes asset paths
to `$LIBERO_CONFIG_PATH/config.yaml` if that file does not exist.

Keep the matching config path exported in every evaluation shell. If you move
the simulator checkout, update the paths in that config or select a new config
directory and rerun setup. Existing configs are not overwritten.

`env.sh` selects EGL rendering. On a machine without EGL, standard single-model
workers can use `MUJOCO_GL=osmesa` if OSMesa is installed. Shared-server
multi-actor evaluation requires EGL. robomimic is installed without unused
training dependencies; `pip check` may therefore report missing `h5py`,
`tensorboard`, `tensorboardX` and `egl_probe` in these evaluation environments.

## Evaluation

In a new shell, return to the repository root and select the simulator and its
config. For standard LIBERO:

```bash
conda activate fasterwam-libero
export PYTHONNOUSERSITE=1
export LIBERO_ROOT="$PWD/data/benchmarks/LIBERO"
export LIBERO_CONFIG_PATH="$PWD/data/benchmark_config/libero"
source scripts/env.sh libero
```

Set `RUN_DIR` to your training output and `CHECKPOINT` to an actual saved weights
file, using the path printed by training. For example:

```bash
RUN_DIR="$PWD/runs/libero/fasterwam"
CHECKPOINT="$RUN_DIR/checkpoints/weights/step_NNNNNN.pt"  # replace NNNNNN
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh libero \
  ckpt="$CHECKPOINT" \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="$PWD/evaluate_results/libero/fasterwam"
```

This runs 50 episodes for each of the 40 standard tasks. To evaluate the two-step
training example on one task for one full episode, use:

```bash
mkdir -p data/eval_tasks
printf 'libero_spatial,0\n' > data/eval_tasks/libero_one_task.txt
RUN_DIR="$PWD/runs/libero/short"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh libero \
  ckpt="$RUN_DIR/checkpoints/weights/step_000002.pt" \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  MULTIRUN.task_file="$PWD/data/eval_tasks/libero_one_task.txt" \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir="$PWD/evaluate_results/libero/short"
```

For LIBERO-Plus, use the same LIBERO-trained checkpoint and statistics, but
activate its environment and config before launching:

```bash
conda activate fasterwam-liberoplus
export PYTHONNOUSERSITE=1
export LIBERO_PLUS_ROOT="$PWD/data/benchmarks/LIBERO-plus"
export LIBERO_CONFIG_PATH="$PWD/data/benchmark_config/libero_plus"
source scripts/env.sh libero_plus
RUN_DIR="$PWD/runs/libero/fasterwam"
CHECKPOINT="$RUN_DIR/checkpoints/weights/step_NNNNNN.pt"  # replace NNNNNN
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh libero_plus \
  ckpt="$CHECKPOINT" \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="$PWD/evaluate_results/libero_plus/fasterwam"
```

LIBERO-Plus runs each of its 10,030 perturbed tasks once. It uses the original
LIBERO training data and statistics; no additional Plus training data is needed
for this evaluation. Both launchers check that the active benchmark matches the
requested flavor. Use `model=<training preset>` and any training dimension
overrides when evaluating an ablation.

For a short check, this subset covers all seven perturbation categories in
`libero_spatial`, all five sensor-noise families, and one task from each of the
other three suites. The task indices are **zero-based** and refer to the pinned
LIBERO-Plus revision above; standard LIBERO task indices do not identify the
same tasks. Run this from the Plus environment after sourcing `env.sh`:

```bash
mkdir -p data/eval_tasks
cat > data/eval_tasks/libero_plus_short.txt <<'TASKS'
libero_spatial,0
libero_spatial,258
libero_spatial,608
libero_spatial,984
libero_spatial,1374
libero_spatial,1725
libero_spatial,2110
libero_spatial,1382
libero_spatial,1390
libero_spatial,1394
libero_spatial,1404
libero_object,0
libero_goal,0
libero_10,0
TASKS
RUN_DIR="$PWD/runs/libero/short"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh libero_plus \
  ckpt="$RUN_DIR/checkpoints/weights/step_000002.pt" \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  MULTIRUN.task_file="$PWD/data/eval_tasks/libero_plus_short.txt" \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir="$PWD/evaluate_results/libero_plus/short"
```

This runs 14 full episodes with the checkpoint from the two-step LIBERO training
example. It checks the evaluation workflow, not benchmark accuracy. For four
GPUs, use `CUDA_VISIBLE_DEVICES=0,1,2,3` and add `MULTIRUN.num_gpus=4`.

### GPU selection, subsets and results

- For two GPUs, set `CUDA_VISIBLE_DEVICES=0,1` and add `MULTIRUN.num_gpus=2`.
  Each worker keeps one model loaded; leave `MULTIRUN.max_tasks_per_gpu=1`.
- To run one suite, add `'MULTIRUN.task_suite_names=[libero_goal]'`. A task file
  accepts one `suite_name,task_id` pair per line; `EVALUATION.num_trials` sets
  episodes per task.
- Inference uses 10 denoising steps, a 32-step action chunk, replanning every
  10 actions and 30 initial settle steps. Keep these settings fixed for model
  comparisons. `seed` controls action noise; `EVALUATION.env_seed` defaults to
  the same value and can be set separately for the simulator.
- Use a fresh `EVALUATION.output_dir` for each evaluation. If omitted, the
  default includes the benchmark, model preset and timestamp.

The output contains `manager.log`, `manager_config.yaml`, `task_logs/`, per-task
JSON results, `summary.json`, `summary.csv`, and `failed_tasks.txt`. Videos are
under each suite's `videos/` directory; disable them with
`EVALUATION.save_rollout_video=false`. A zero task success rate is different from
a crashed worker: check that requested episodes are recorded and
`failed_tasks.txt` is empty.

Summarize a finished standard run:

```bash
python -m experiments.libero.summarize_libero \
  --output_dir "$PWD/evaluate_results/libero/fasterwam"
```

For the short example, use `evaluate_results/libero/short`; the summary reports
`PARTIAL` against the full suite because only one task was requested. In the
Plus environment, summarize the Plus output and its perturbation categories:

```bash
python -m experiments.libero.summarize_libero \
  --output_dir "$PWD/evaluate_results/libero_plus/fasterwam" \
  --benchmark_flavor libero_plus
python -m experiments.libero.summarize_by_disturbance \
  "$PWD/evaluate_results/libero_plus/fasterwam"
```

For the 14-episode example, use `evaluate_results/libero_plus/short` in both
commands. Its summary is also `PARTIAL` against the full benchmark.

## Faster evaluation with a prompt cache

By default the evaluator encodes task instructions with the T5 encoder online.
A precomputed prompt cache avoids loading T5 and is required for batching
requests from several simulator actors into one forward pass. Generate it in the
environment of the benchmark it belongs to.

Standard LIBERO:

```bash
conda activate fasterwam-libero
export LIBERO_ROOT="$PWD/data/benchmarks/LIBERO"
export LIBERO_CONFIG_PATH="$PWD/data/benchmark_config/libero"
source scripts/env.sh libero
python -m experiments.libero.precompute_task_prompts \
  --benchmark-flavor libero_standard --prepare-source \
  --source-dir ./data/eval_prompt_sources/libero \
  --output-dir ./data/eval_prompt_cache/libero
python -m experiments.libero.precompute_standard_task_embeddings \
  data=libero_2cam \
  'data.train.dataset_dirs=[./data/eval_prompt_sources/libero]' \
  data.train.text_embedding_cache_dir=./data/eval_prompt_cache/libero \
  +overwrite=false
python -m experiments.libero.precompute_task_prompts \
  --benchmark-flavor libero_standard --finalize \
  --source-dir ./data/eval_prompt_sources/libero \
  --output-dir ./data/eval_prompt_cache/libero
```

LIBERO-Plus:

```bash
conda activate fasterwam-liberoplus
export LIBERO_PLUS_ROOT="$PWD/data/benchmarks/LIBERO-plus"
export LIBERO_CONFIG_PATH="$PWD/data/benchmark_config/libero_plus"
source scripts/env.sh libero_plus
python -m experiments.libero.precompute_task_prompts \
  --benchmark-flavor libero_plus --prepare-source \
  --source-dir ./data/eval_prompt_sources/libero_plus \
  --output-dir ./data/eval_prompt_cache/libero_plus
python -m experiments.libero.precompute_liberoplus_task_embeddings \
  --source-dir ./data/eval_prompt_sources/libero_plus \
  --output-dir ./data/eval_prompt_cache/libero_plus --batch-size 1
python -m experiments.libero.precompute_task_prompts \
  --benchmark-flavor libero_plus --finalize \
  --source-dir ./data/eval_prompt_sources/libero_plus \
  --output-dir ./data/eval_prompt_cache/libero_plus
```

Evaluate with the cache and several simulator actors per GPU:

```bash
bash scripts/eval.sh libero_plus \
  ckpt=/absolute/path/to/step_NNNNNN.pt \
  EVALUATION.dataset_stats_path=/absolute/path/to/dataset_stats.json \
  EVALUATION.task_prompt_cache_dir=./data/eval_prompt_cache/libero_plus \
  model.load_text_encoder=false \
  MULTIRUN.sim_actors_per_gpu=[8] EVALUATION.policy_batch_size=8
```

Multiple actors and a shared model server require `MUJOCO_GL=egl`. Increase the
actor count or batch size to fit the available GPU memory and CPU cores.
