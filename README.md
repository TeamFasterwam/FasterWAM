# Faster-WAM

Faster-WAM is a world-model-centric world action model (WAM) for robot
manipulation, built on [Fast-WAM](https://github.com/yuantianyuan01/FastWAM). It
keeps the Wan2.2-TI2V-5B video backbone and replaces Fast-WAM's 30-layer action
expert with a single-layer action expert through three design choices:

- **Dock of Transformers (DoT)**: Lite KV-Fusion gives the action expert a
  learned per-head mixture of the native key–value caches of all video layers.
- **World-model-only conditioning**: language and proprioception condition only
  the video backbone; the action expert has no text cross-attention (TextCA).
- **Retracted 1D-RoPE**: the native 3D rotation of the fused video keys is
  removed and the keys are re-encoded on the action queries' 1D temporal axis.

This repository contains training and evaluation code for Faster-WAM, the
Fast-WAM baseline, and the architectural ablations reported in the paper, on
LIBERO, LIBERO-Plus and RoboTwin 2.0.

**License:** [PolyForm Noncommercial 1.0.0](LICENSE). See the
[license scope](#acknowledgements-and-license) below.

- [Model presets and ablations](docs/ablations.md)
- [LIBERO and LIBERO-Plus](README_Libero.md)
- [RoboTwin 2.0](README_Robotwin.md)
- [Optional VAE latent precomputation](README_precompute_VAE.md)

## Getting started

Clone the repository:

```bash
git clone https://github.com/TeamFasterwam/FasterWAM.git
cd FasterWAM
```

Run commands from this repository's root. Start with the training environment
below, then follow one benchmark guide in order:

1. [LIBERO / LIBERO-Plus](README_Libero.md): download data, prepare text
   embeddings, train, install the simulator, and evaluate the local checkpoint.
2. [RoboTwin 2.0](README_Robotwin.md): the same workflow, with a separate simulator
   environment and a CUDA build of CuRobo.

Each guide includes an optional two-step training example and a small evaluation
command. These use the full model and real data. They check that a setup runs;
use the full training and evaluation recipes for benchmark results.

## Installation

Use Linux x86-64, an NVIDIA GPU with BF16 support, and a driver compatible with
CUDA 12.8. The supplied training recipes were run on H200 GPUs. You also need
Git, a C++ compiler, `tar`, `unzip`, and EGL/Vulkan driver libraries for rendering.
Install [Miniconda](https://docs.anaconda.com/miniconda/install/) or
[Miniforge](https://github.com/conda-forge/miniforge). In a non-interactive SSH
shell, initialize Conda before activating environments:

```bash
source /path/to/conda/etc/profile.d/conda.sh
```

Create a new training environment. PyTorch is installed first from its CUDA
12.8 wheel index; FFmpeg is installed into the same environment for video decoding.

```bash
conda create -n fasterwam python=3.10 -y
conda activate fasterwam
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge "ffmpeg<8" -y
bash scripts/setup_env.sh train
source scripts/env.sh train
```

`setup_env.sh` installs this checkout in editable mode with pinned dependencies.
Keep these versions: `transformers==4.49.0` requires `huggingface-hub` below 1.0,
and TorchCodec needs FFmpeg 4–7. `PYTHONNOUSERSITE=1` excludes packages in
`~/.local`; clearing inherited Python paths avoids importing another checkout.

Evaluation runs the policy inside each simulator's Python environment, so
install Faster-WAM there as well, using the corresponding benchmark guide:

| Environment | Purpose | Installation guide |
| --- | --- | --- |
| `fasterwam` | Training and data preprocessing for both benchmarks | Commands above |
| `fasterwam-libero` | Standard LIBERO evaluation | [LIBERO](README_Libero.md#evaluation-environments) |
| `fasterwam-liberoplus` | LIBERO-Plus evaluation | [LIBERO-Plus](README_Libero.md#libero-plus) |
| `fasterwam-robotwin` | RoboTwin evaluation | [RoboTwin](README_Robotwin.md#evaluation-environment) |

LIBERO and LIBERO-Plus install the same Python package name and need separate
environments. All environments can share this checkout, downloaded base models
and training outputs. The original Fast-WAM repository is not required.

In each new shell, activate the appropriate environment, export the paths from
its benchmark guide, then source `scripts/env.sh <target>`. The launchers use
`python` and `accelerate` from the active environment. For example:

```bash
conda activate fasterwam
export PYTHONNOUSERSITE=1
source scripts/env.sh train
# Re-export LIBERO_DATA_ROOT / ROBOTWIN_DATA_ROOT and cache paths as needed.
```

## Data and base models

The benchmark guides provide the download, extraction and preprocessing commands
for the public [LIBERO](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) and
[RoboTwin](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam) datasets.
Both LeRobot v2.1 and v3.0 layouts are supported. Training needs a text embedding
cache; [VAE latent caching](README_precompute_VAE.md) is optional. Evaluation
encodes its instructions online by default and does not need the training data
or its text cache.

Keep these paths distinct:

| Variable | Contents |
| --- | --- |
| `LIBERO_DATA_ROOT` | Parent of the four LIBERO dataset directories |
| `LIBERO_TEXT_CACHE` | Precomputed LIBERO training instruction embeddings |
| `ROBOTWIN_DATA_ROOT` | Extracted RoboTwin dataset directory containing `meta/` |
| `ROBOTWIN_NORM_STATS` | RoboTwin's distributed `dataset_stats.json` |
| `ROBOTWIN_TEXT_CACHE` | Precomputed RoboTwin training instruction embeddings |
| `DIFFSYNTH_MODEL_BASE_PATH` | Wan base-model files; defaults to `<repo>/checkpoints` |

LIBERO normalization statistics are bundled in `configs/data/libero_stats.json`.
RoboTwin uses the statistics distributed with its dataset when
`ROBOTWIN_NORM_STATS` is set; otherwise it computes them from the dataset.
Evaluation should always use the `dataset_stats.json` written into the training
run directory.

On first use, the required Wan components are downloaded automatically from
[ModelScope](https://modelscope.cn), with no account required:

| Repository | Files |
| --- | --- |
| `Wan-AI/Wan2.2-TI2V-5B` | `diffusion_pytorch_model*.safetensors` (video DiT) |
| `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | `Wan2.2_VAE.safetensors`, `models_t5_umt5-xxl-enc-bf16.safetensors` |
| `Wan-AI/Wan2.1-T2V-1.3B` | `google/umt5-xxl/` (tokenizer) |

These files take about 32 GB. Multi-GPU training downloads missing files once
per node before the other processes load them. Keep the same model-base path
for preprocessing, training and evaluation, especially when starting new shells.

Allow space for downloaded archives, extracted videos, simulator assets and
training outputs. A full RoboTwin text cache alone needs roughly **1 TB**;
its guide also includes a small real-data subset. A two-GPU training checkpoint
with optimizer/resume state occupies about **80 GiB**, even for a two-step run.
Periodic checkpoints add to that storage.

## Training and checkpoints

After preparing the data and text cache in a benchmark guide, launch training
from the `fasterwam` environment. The reported recipes are:

```bash
# LIBERO: 10 epochs, global batch size 128.
bash scripts/train.sh libero 8 \
  batch_size=8 gradient_accumulation_steps=2 num_epochs=10

# RoboTwin: 5 epochs, global batch size 512.
bash scripts/train.sh robotwin 8 \
  batch_size=16 gradient_accumulation_steps=4 num_epochs=5
```

The GPU count is the second positional argument. Select physical GPUs using
`CUDA_VISIBLE_DEVICES` and match that count. Global batch size is
`batch_size × GPU processes × gradient_accumulation_steps`. To reduce memory,
lower `batch_size` and enable `model.mot_checkpoint_mixed_attn=true`; increase
accumulation if you need to preserve the global batch. BF16 and AdamW with
learning rate `1e-4`, weight decay `0.01` and cosine decay are the defaults.

Arguments after the GPU count are [Hydra](https://hydra.cc/) configuration
overrides. Quote list-valued arguments so the shell passes them unchanged.
Inspect a resolved configuration without loading models or data:

```bash
python scripts/train.py data=robotwin --cfg job --resolve
```

A run is written to `runs/<benchmark>/<RUN_ID>/`. `RUN_ID` defaults to a timestamp;
set it before launching, or pass `output_dir=/absolute/path/to/run`. Use a fresh
output directory for a new experiment. Each run contains:

```text
config.yaml                            Resolved configuration
dataset_stats.json                    Normalization statistics
checkpoints/weights/step_NNNNNN.pt      Policy weights for evaluation
checkpoints/state/step_NNNNNN/          Full state for resuming training
```

The final log prints both checkpoint paths. `save_every` controls periodic saves;
`save_every=0` disables periodic saves but still saves at normal completion.
`max_steps=N` stops at optimizer step N instead of the epoch-derived limit.
`eval_every=0` disables training-time validation; simulator evaluation is a
separate command. W&B logging is off by default (`wandb.enabled=true` enables it).

Resume with the full state directory, keeping the same dataset, model, GPU
layout and training settings. For example, to resume the LIBERO recipe above:

```bash
bash scripts/train.sh libero 8 \
  batch_size=8 gradient_accumulation_steps=2 num_epochs=10 \
  output_dir=/absolute/path/to/run \
  resume=/absolute/path/to/run/checkpoints/state/step_NNNNNN
```

`seed` is applied before model and dataset construction, with rank-specific
streams for training noise and data loading. `freeze_video_backbone=true`
trains only the action path. See [model presets](docs/ablations.md) for the
Fast-WAM baseline and architectural ablations. The Fast-WAM baseline needs a
separate ActionDiT initialization, created locally before training that preset:

```bash
python -m fasterwam.tools.preprocess_action_dit_backbone \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

## Evaluation

Activate the benchmark's simulator environment and follow its evaluation
section: [LIBERO](README_Libero.md#evaluation) or
[RoboTwin](README_Robotwin.md#evaluation). Pass the **weights file** and the
**statistics file from the same training run**. For example, after activating
and configuring the LIBERO environment:

```bash
bash scripts/eval.sh libero \
  ckpt=/absolute/path/to/run/checkpoints/weights/step_NNNNNN.pt \
  EVALUATION.dataset_stats_path=/absolute/path/to/run/dataset_stats.json
```

The model preset and dimension overrides must match training (`model=fasterwam`
by default). A checkpoint with incompatible model options or parameter shapes
is rejected. Keep the checkpoint's parent run directory, which records its
configuration and statistics. The benchmark guides explain how to select tasks,
distribute work across GPUs, limit episodes, and read summaries and videos.

## Launchers and multiple nodes

All launchers run directly on the current machine; Slurm is optional.

| Entry point | Usage |
| --- | --- |
| Environment installation | `bash scripts/setup_env.sh <train\|libero\|libero_plus\|robotwin>` |
| Environment variables | `source scripts/env.sh <train\|libero\|libero_plus\|robotwin>` |
| Training | `bash scripts/train.sh <benchmark> <gpus_per_node> [overrides...]` |
| Evaluation | `bash scripts/eval.sh <benchmark> [overrides...]` |
| Ablations | `bash scripts/ablation.sh <train\|eval> <benchmark> <preset\|all> [gpus_per_node for train] [overrides...]` |
| Optional Slurm training | `sbatch scripts/slurm_train.sh <benchmark> [overrides...]` |

Benchmarks are `libero`, `libero_plus` and `robotwin`; LIBERO-Plus uses LIBERO's
training recipe. Set `DRY_RUN=1` before a launcher to print its command without
running it. When running independent training jobs on one host, select different
GPUs and set a different `MASTER_PORT` for each job.

For multiple nodes, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` and
the same `RUN_ID` on every node, with a shared checkout, data, cache and output
location. Launch `train.sh` once per node. The optional Slurm template derives
these values from its allocation; supply your cluster's partition/account/QoS
and set `GPUS_PER_NODE` if needed.

## Troubleshooting

- **Interrupted dataset download:** rerun the same `huggingface-cli download`
  command. Preserve the download directory's `.cache` folder so partial files
  can be resumed; extract archives after the download succeeds.
- **Missing text embeddings:** activate `fasterwam`, export the same dataset
  and text-cache paths used for training, and rerun the preprocessing command
  with `+overwrite=false` to retain completed entries.
- **CUDA out of memory:** reduce the per-GPU batch and enable gradient
  checkpointing. Each evaluation worker loads the full model; keep one model
  worker per GPU before enabling the optional shared-server batching features.
- **FFmpeg/TorchCodec import error:** activate the intended environment and
  reinstall `"ffmpeg<8"` there. Do not rely on a different environment's FFmpeg.
- **Simulator import or rendering failure:** use the separate simulator
  environment and rerun its setup script. The LIBERO guide describes config
  and log paths; the RoboTwin guide explains the matching CUDA compiler and
  Warp version. The NVIDIA EGL/Vulkan driver must be available on the machine.

## Implementation layout

| Location | Responsibility |
| --- | --- |
| [Model factory](src/fasterwam/models/factory.py) | Validates and constructs the seven public presets |
| [DoT](src/fasterwam/models/wan22/dot.py) | Public KV-fusion and positional-interface choices |
| `src/fasterwam/models/wan22/` | Shared video/action experts, attention and cached inference |
| [Runtime](src/fasterwam/runtime.py), [trainer](src/fasterwam/trainer.py) | Dataset/model construction, distributed training and checkpoints |
| `src/fasterwam/datasets/`, `src/fasterwam/tools/` | Data loading, transforms and cache preparation |
| `configs/` | Model, dataset, training and evaluation settings |
| `experiments/` | Benchmark adapters, evaluation workers and result summaries |
| `scripts/` | Installation and launch entry points |
| `third_party/RoboTwin/` | Vendored RoboTwin simulator code |

## Acknowledgements and license

Faster-WAM is built on [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) and
uses code from Wan2.2, DiffSynth-Studio, LeRobot, PyTorch3D, OpenVLA and
RoboTwin 2.0.

Faster-WAM's original contributions are available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). This source-available license
permits noncommercial use and the other purposes specified in its terms.
Commercial use outside those permissions requires a separate license from the
Faster-WAM copyright holders.

Third-party components, including the inherited Fast-WAM code, retain their
original licenses and copyright notices, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Earlier versions distributed under MIT retain their original license.
