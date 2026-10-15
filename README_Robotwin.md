# RoboTwin 2.0

Start with the `fasterwam` training environment from
[README.md](README.md#installation), then follow the sections below in order.
Run every command from the repository root. Training and data preparation use
`fasterwam`; evaluation uses a separate `fasterwam-robotwin` environment.

## Training data

Activate the training environment, download the public
[yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
archives (about 80 GB), then extract all parts together:

```bash
conda activate fasterwam
export PYTHONNOUSERSITE=1
source scripts/env.sh train
huggingface-cli download yuanty/robotwin2.0-fastwam --repo-type dataset \
  --local-dir data/robotwin2.0
(cd data/robotwin2.0 && cat robotwin2.0.tar.gz.part-* | tar -xzf -)

export ROBOTWIN_DATA_ROOT="$PWD/data/robotwin2.0/robotwin2.0"
export ROBOTWIN_NORM_STATS="$PWD/data/robotwin2.0/dataset_stats.json"
```

Do not extract before the download completes. If a transfer fails, repeat the
same download command to resume partial files. The extracted layout is:

```text
data/robotwin2.0/
  dataset_stats.json
  robotwin2.0/
    meta/
    data/
    videos/
```

`ROBOTWIN_DATA_ROOT` points to the inner directory containing `meta/`;
`ROBOTWIN_NORM_STATS` points to the statistics file alongside it. Re-export
both in each training shell. Without the statistics variable, training computes
normalization statistics over the dataset before its first step.

The public download is LeRobot v2.1, with 27,500 episodes, three cameras
(`cam_high`, `cam_left_wrist`, `cam_right_wrist`), and 14-dimensional
joint/gripper actions and states. The loader also accepts data converted to
LeRobot v3.0, which the reported models used. Camera layout, temporal sampling
and normalization are specified in [robotwin.yaml](configs/data/robotwin.yaml).

## Training

### Full dataset

The data contains 921,032 distinct instructions. A BF16 text embedding takes
about 1 MiB (128 × 4096 values), so allow roughly **1 TB** for the full text
cache, in addition to the archives, extracted data, base models and training
outputs. For a short first run, skip this section and use the
[two-episode example](#short-training-on-two-episodes) below.

Prepare the text cache with the training environment and data paths above.
The first invocation downloads the T5 encoder and tokenizer. Set the GPU count
to match the selected devices:

```bash
export ROBOTWIN_TEXT_CACHE="$PWD/data/text_embeds_cache/robotwin"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  -m fasterwam.tools.precompute_text_embeds data=robotwin +overwrite=false
```

For one GPU, replace `torchrun --standalone --nproc_per_node=8` with `python`
and set `CUDA_VISIBLE_DEVICES=0`. `+overwrite=false` skips existing embeddings
when resuming. Keep the same cache path for full-dataset training.

Run the reported recipe, with global batch size 512 and five epochs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh robotwin 8 \
  batch_size=16 gradient_accumulation_steps=4 num_epochs=5 \
  output_dir="$PWD/runs/robotwin/fasterwam"
```

Lower `batch_size` and enable `model.mot_checkpoint_mixed_attn=true` when GPU
memory is limited; raise accumulation to preserve the global batch. Use a new
output directory for each experiment. For ablations, add `model=<preset>` from
[the preset table](docs/ablations.md). [VAE latent caching](README_precompute_VAE.md)
is optional; these commands encode videos during training.

### Short training on two episodes

For a two-step run, cache only the instructions referenced by real episodes 0
and 1. This preserves the videos, actions, states and per-frame instructions.
The following creates a separate metadata directory for text preprocessing;
the original dataset is read without modification. It requires the v2.1
layout downloaded above:

```bash
conda activate fasterwam
source scripts/env.sh train
export ROBOTWIN_DATA_ROOT="$PWD/data/robotwin2.0/robotwin2.0"
export ROBOTWIN_NORM_STATS="$PWD/data/robotwin2.0/dataset_stats.json"
export ROBOTWIN_TEXT_CACHE="$PWD/data/text_embeds_cache/robotwin_short"
python - <<'PY'
import json
import os
from pathlib import Path
import pyarrow.parquet as pq

root = Path(os.environ["ROBOTWIN_DATA_ROOT"])
info = json.loads((root / "meta/info.json").read_text())
assert info["codebase_version"] == "v2.1", "Use the v2.1 download recipe for this check"
task_ids = set()
for episode in [0, 1]:
    path = root / info["data_path"].format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode
    )
    task_ids.update(pq.read_table(path, columns=["task_index"])["task_index"].to_pylist())
records = []
with (root / "meta/tasks.jsonl").open() as handle:
    for line in handle:
        record = json.loads(line)
        if record["task_index"] in task_ids:
            records.append(record)
assert {r["task_index"] for r in records} == task_ids
destination = Path("data/subsets/robotwin_two_episodes/meta/tasks.jsonl")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text("".join(json.dumps(r) + "\n" for r in records))
print(f"Selected {len(records)} real instructions from episodes 0 and 1")
PY

CUDA_VISIBLE_DEVICES=0 python -m fasterwam.tools.precompute_text_embeds \
  data=robotwin data.val=null +overwrite=false \
  'data.train.dataset_dirs=[./data/subsets/robotwin_two_episodes]'

CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh robotwin 2 \
  batch_size=1 gradient_accumulation_steps=1 max_steps=2 \
  num_workers=2 log_every=1 save_every=0 eval_every=0 \
  model.mot_checkpoint_mixed_attn=true data.train.strict_getitem=true \
  '+data.train.episode_indices=[0,1]' data.val=null \
  output_dir="$PWD/runs/robotwin/short"
```

For the public dataset, this selects 157 instructions and 282 training windows.
The run saves `runs/robotwin/short/checkpoints/weights/step_000002.pt`, full
resume state, and `runs/robotwin/short/dataset_stats.json`. Allow about 80 GiB
for weights and state; the completion log contains `max_steps reached step=2`.
The full model is trained, but two optimizer steps do not establish task
accuracy. To later train on the full dataset, prepare and switch back to the
full text cache; this subset cache is insufficient for other episodes.

## Evaluation environment

Evaluation runs the policy inside the RoboTwin simulator. The benchmark code
is included under [third_party/RoboTwin](third_party/RoboTwin); its
[vendor notice](third_party/RoboTwin/README.vendor.md) records the upstream
revision and local changes. Rendering needs a Vulkan-capable NVIDIA driver.

Create a new Python environment. CuRobo compiles CUDA extensions, so its CUDA
compiler must match the PyTorch wheel's CUDA 12.8 version. Installing the
compiler into this environment avoids relying on the system's CUDA version:

```bash
conda create -n fasterwam-robotwin python=3.10 -y
conda activate fasterwam-robotwin
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge "ffmpeg<8" -y
conda install -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/setup_env.sh robotwin
source scripts/env.sh robotwin
```

The installer performs these steps:

1. Install Faster-WAM and the simulator's
   [pinned packages](third_party/RoboTwin/script/requirements.txt), including
   SAPIEN 3.0.0b1 and mplib 0.2.1.
2. Clone [CuRobo](https://github.com/NVlabs/curobo) v0.7.8 into
   `third_party/RoboTwin/envs/curobo` and compile it against the active PyTorch.
3. Pin `warp-lang==1.8.1`, `numpy==1.26.4` and `setuptools==69.5.1` after CuRobo,
   whose dependencies may otherwise upgrade them.
4. Download and extract simulator assets from
   [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
   if they are not installed. These are separate from the training videos.
5. Import the simulator packages and check Warp CUDA initialization and device
   allocation when a GPU is visible.

Warp 1.8.1 fixes the `cuDeviceGetUuid` initialization error seen with NVIDIA
R580 drivers and retains `warp.torch`, which CuRobo v0.7.8 needs
([upstream fix](https://github.com/NVIDIA/warp/releases/tag/v1.8.1)). Keep these
versions: Warp 1.13 removes that API, newer CuRobo releases change the planner
API, and mplib requires NumPy 1.x. This installer leaves mplib's screw-motion
planner unchanged, matching the Faster-WAM evaluation environment.

By default, evaluation uses the bundled RoboTwin. To use another checkout,
export `ROBOTWIN_ROOT=/absolute/path/to/RoboTwin` before setup and evaluation;
it must support the options in the bundled `script/eval_policy.py`. The runner
links `policy/fasterwam_policy` to this repository's
[policy adapter](experiments/robotwin/fasterwam_policy). Keep the checkout and
its assets at the installed location because asset configuration contains paths.

## Evaluation

In each new evaluation shell, return to the repository root and activate the
simulator environment. Training data and the training text cache are not needed;
the model encodes evaluation instructions online.

```bash
conda activate fasterwam-robotwin
export PYTHONNOUSERSITE=1
source scripts/env.sh robotwin
```

Set `RUN_DIR` to your training run and `CHECKPOINT` to the saved weights file
printed by training. To evaluate the full benchmark:

```bash
RUN_DIR="$PWD/runs/robotwin/fasterwam"
CHECKPOINT="$RUN_DIR/checkpoints/weights/step_NNNNNN.pt"  # replace NNNNNN
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh robotwin \
  ckpt="$CHECKPOINT" \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="$PWD/evaluate_results/robotwin/fasterwam"
```

The manager evaluates all 50 tasks, with 100 episodes per task in each of the
clean (`demo_clean`) and randomized (`demo_randomized`) settings. To evaluate
the two-step run on one task for one episode in each setting:

```bash
RUN_DIR="$PWD/runs/robotwin/short"
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh robotwin \
  ckpt="$RUN_DIR/checkpoints/weights/step_000002.pt" \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.task_name=click_alarmclock EVALUATION.eval_num_episodes=1 \
  EVALUATION.output_dir="$PWD/evaluate_results/robotwin/short"
```

Use `model=<training preset>` and matching dimension overrides for ablations.
Other useful options are:

| Option | Effect |
| --- | --- |
| `MULTIRUN.phases=clean` | Evaluate only the clean setting; default is `clean,random` |
| `EVALUATION.task_name=click_alarmclock` | Select one task instead of all 50 |
| `EVALUATION.eval_num_episodes=1` | Set the episode count per task and phase |
| `MULTIRUN.num_gpus=2` | Use two GPUs; also set `CUDA_VISIBLE_DEVICES=0,1` |
| `EVALUATION.full_frame_video=true` | Render and record every simulator frame |

The default action horizon is 32, with 24 actions executed before replanning
and 10 denoising steps. Keep these fixed for comparisons. The default
`EVALUATION.skip_get_obs_within_replan=true` skips camera reads between replans.
`EVALUATION.per_request_noise=true` draws reproducible noise per task, episode
and replan from `seed`.

Episode seeds follow RoboTwin: the first candidate is `100000 × (1 + seed)`,
and seeds whose expert demonstration fails are skipped. With `seed=42`, the
first candidate is 4,300,000; `seed=0` starts at 100,000.

### Outputs and resuming

Use a fresh output directory for each run. Without `EVALUATION.output_dir`,
results go to `evaluate_results/robotwin/<timestamp>/`. The output contains
manager/worker logs, `summary.csv`, `summary.json`, `failed_tasks.txt`,
`checkpoint.json`, and a directory per task with `_result_clean.txt`,
`_result_random.txt` and rollout videos. Check that both requested phase results
are present and `failed_tasks.txt` is empty; task failure is distinct from a
worker crash or an incomplete episode.

Print a summary after evaluation:

```bash
python -m experiments.robotwin.compile_results \
  "$PWD/evaluate_results/robotwin/fasterwam"
```

For the short example, use `evaluate_results/robotwin/short`. To resume an
interrupted evaluation, rerun the original command with
`resume=/absolute/path/to/eval_output_dir`, preserving the checkpoint,
statistics, model, phases, episode count and inference settings. Completed
phases are skipped; a partially completed phase is rerun.
