# VAE precomputation

Precompute Wan2.2 VAE latents once, then reuse them during training. Cached
training skips video decoding and VAE encoding for training batches. The
validation path still uses original videos, and the VAE remains available for
video reconstruction and evaluation.

This is optional and works with all seven public model presets when they share
the same dataset, preprocessing, and VAE configuration. Text embeddings are a
separate cache; prepare them as described in [README.md](README.md).

## Prepare the environment

Use the training environment from [README.md](README.md), a CUDA GPU, and the
processed LeRobot dataset. Run these commands from the repository root:

```bash
conda activate fasterwam
export PYTHONNOUSERSITE=1
source scripts/env.sh train
export LIBERO_VAE_CACHE="$PWD/data/vae_latents/libero"
```

Keep `LIBERO_DATA_ROOT` set to the dataset location selected in
[README_Libero.md](README_Libero.md#training-data), whether v2.1 or v3.0.
In a new shell, re-export that same path before starting precomputation.

The tool loads the VAE selected by the model configuration, without loading
the video DiT or text encoder. It uses BF16 encoding, batch size 1 per GPU,
and no VAE tiling. Both supplied dataset presets pin `video_backend=torchcodec`.

## Build a cache

For LIBERO on one GPU:

```bash
python -m fasterwam.tools.precompute_video_latents \
  data=libero_2cam \
  +vae_cache_output_dir="$LIBERO_VAE_CACHE" \
  +vae_cache_num_workers=4
```

For LIBERO on eight GPUs in one node:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m fasterwam.tools.precompute_video_latents \
  data=libero_2cam \
  +vae_cache_output_dir="$LIBERO_VAE_CACHE" \
  +vae_cache_num_workers=4
```

Each GPU processes a disjoint set of dataset indices. `vae_cache_num_workers`
is the number of decoder workers **per GPU**; use `0` when debugging decoding.
All ranks must access the same cache directory on a filesystem supporting
shared memory-mapped files and file locks.

For RoboTwin, use its dataset preset and a separate cache directory:

```bash
export ROBOTWIN_DATA_ROOT="$PWD/data/robotwin2.0/robotwin2.0"
export ROBOTWIN_NORM_STATS="$PWD/data/robotwin2.0/dataset_stats.json"
export ROBOTWIN_VAE_CACHE="$PWD/data/vae_latents/robotwin"
torchrun --standalone --nproc_per_node=8 \
  -m fasterwam.tools.precompute_video_latents \
  data=robotwin \
  +vae_cache_output_dir="$ROBOTWIN_VAE_CACHE" \
  +vae_cache_num_workers=4
```

Dataset construction also loads or computes action/state normalization
statistics. LIBERO uses its bundled statistics by default. Set
`ROBOTWIN_NORM_STATS` to the statistics distributed with the dataset, or let the dataset
compute them. `+vae_cache_norm_stats_path=/path/to/dataset_stats.json` overrides
the statistics path for the precomputation command.

To inspect the configuration without loading models or data, append
`--cfg job --resolve` to the Python command. A small encoding check can use
`+vae_cache_max_samples=16` with a separate output directory; this partial
dataset cache cannot be used for full-dataset training.

## Train with cached latents

After precomputation completes, use the same dataset overrides and enable
strict indexing:

```bash
bash scripts/train.sh libero 8 \
  mixed_precision=bf16 \
  data.train.video_latent_cache_dir="$LIBERO_VAE_CACHE" \
  data.train.strict_getitem=true

# RoboTwin:
bash scripts/train.sh robotwin 8 \
  mixed_precision=bf16 \
  data.train.video_latent_cache_dir="$ROBOTWIN_VAE_CACHE" \
  data.train.strict_getitem=true
```

`strict_getitem=true` prevents failed reads from silently substituting another
sample and pairing its actions with the wrong latents. Keep
`data.train.skip_padding_as_possible=false`, as in both supplied presets.
Cached training requires BF16; the runtime resolves and checks the VAE identity
automatically, so `video_latent_cache_vae_identity` needs no manual override.
Changing the training GPU count or model preset alone does not require rebuilding
the cache. Changes to the dataset, split, sample stride, temporal sampling,
camera processing, resolution, or VAE require a matching cache.

## Files and resuming

The cache contains:

| File | Purpose |
| --- | --- |
| `latents.bf16` | Memory-mapped BF16 latent tensors |
| `complete.u8` | One completion marker per sample |
| `manifest.in_progress.json` | Exact configuration and source identity for resuming |
| `manifest.json` | Published only after every sample is complete |
| `dataset_stats.json` | Action/state statistics loaded or computed during setup |
| `build.*.lock` | Prevent concurrent builders from writing the same cache |

Rerun the same command to resume an interrupted build. Keep the same GPU count,
source version, dataset, and options: incomplete builds require an exact
manifest match. Completed samples are skipped, with progress committed every
256 samples per rank and at normal completion.

Prefer a new directory when changing the cache recipe. The optional
`+vae_cache_overwrite=true` deletes and rebuilds the cache files in the selected
directory. Do not run it while training is reading those files.

Storage is approximately `sample_count × product(latent_shape) × 2` bytes,
plus the completion bitmap and manifests. The manifest records the latent
shape, dataset identity, preprocessing configuration, VAE checksum, and a
digest of the preprocessing source files. A cache is rejected after changes to
any of these, including edits to the dataset or preprocessing code.
