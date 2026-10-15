"""Precompute exact batch-1 Wan2.2 VAE latents for Faster-WAM training.

Launch with ``torchrun --standalone --nproc_per_node=8`` so each GPU encodes a
disjoint set of global dataset indices.
"""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import logging
import os
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fasterwam.datasets.video_latent_cache import (
    CACHE_SCHEMA_VERSION,
    COMPLETION_FILENAME,
    IN_PROGRESS_MANIFEST_FILENAME,
    LATENTS_FILENAME,
    MANIFEST_FILENAME,
    atomic_write_json,
    create_or_resume_cache_files,
    preprocess_spec_sha256,
    vae_implementation_sha256,
)
from fasterwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fasterwam.tools import config_dir
from fasterwam.utils import misc
from fasterwam.utils.config_resolvers import register_default_resolvers
from fasterwam.utils.logging_config import get_logger, setup_logging


register_default_resolvers()
logger = get_logger(__name__)
COMMIT_INTERVAL_SAMPLES = 256


class IndexedVideoDataset(Dataset):
    """Expose only global index and processed video for cache construction."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        """Return this rank's sample count."""

        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[int, torch.Tensor]:
        """Load one strict dataset sample and retain its global index."""

        index = self.indices[position]
        sample = self.dataset[index]
        if "video" not in sample:
            raise KeyError("Online precompute dataset did not return `video`.")
        return index, sample["video"]


def _init_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed and return rank, world size, and local rank."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Video latent precomputation requires CUDA.")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def _barrier(world_size: int) -> None:
    """Synchronize ranks when running distributed."""

    if world_size > 1:
        dist.barrier()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _broadcast_object(value: Any, rank: int, world_size: int) -> Any:
    """Broadcast a Python object from rank zero."""

    if world_size <= 1:
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _iter_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Provide a typed iterator wrapper for the rank-local DataLoader."""

    yield from loader


def _fsync_file(path: Path) -> None:
    """Flush a completed mmap backing file to local storage."""

    with path.open("rb") as file_handle:
        os.fsync(file_handle.fileno())


def _acquire_cache_build_lock(
    cache_dir: Path, *, rank: int, world_size: int
) -> list[Any]:
    """Hold a registration lock plus one live lease for every worker rank."""

    lock_handles = []
    error = None
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        registration_handle = (cache_dir / "build.registration.lock").open(
            "a+", encoding="utf-8"
        )
        try:
            fcntl.flock(
                registration_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            error = f"Another cache builder holds the lock for {cache_dir}."
            registration_handle.close()
        else:
            lock_handles.append(registration_handle)
            for lease_path in sorted(cache_dir.glob("build.rank*.lock")):
                probe = lease_path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    error = f"A live cache worker still holds {lease_path}."
                    probe.close()
                    break
                else:
                    probe.close()
    error = _broadcast_object(error, rank=rank, world_size=world_size)
    if error is not None:
        for handle in lock_handles:
            handle.close()
        raise RuntimeError(error)

    lease_handle = (cache_dir / f"build.rank{rank}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lease_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lease_handle.close()
        raise RuntimeError(f"Cache worker lease is already held for rank {rank}.") from error
    lock_handles.append(lease_handle)
    for handle in lock_handles:
        atexit.register(handle.close)
    return lock_handles


def _distributed_fail_fast(function):
    """Let torchrun terminate peer ranks promptly after any worker failure."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException:
            logger.exception("Fatal VAE cache worker failure; terminating this torchrun rank.")
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                os._exit(1)
            raise

    return wrapped


def _commit_completed_indices(
    *,
    cache_dir: Path,
    completion: torch.Tensor,
    indices: list[int],
) -> None:
    """Durably order latent payload writes before their completion markers."""

    if not indices:
        return
    _fsync_file(cache_dir / LATENTS_FILENAME)
    completion[indices] = 1
    _fsync_file(cache_dir / COMPLETION_FILENAME)


@hydra.main(config_path=config_dir(), config_name="train", version_base="1.3")
@_distributed_fail_fast
def main(cfg: DictConfig) -> None:
    """Build or resume a distributed batch-1 BF16 latent cache."""

    setup_logging(log_level=logging.INFO)
    rank, world_size, local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    if cfg.data is None or cfg.data.get("train") is None:
        raise ValueError("A composed `data.train` config is required.")
    if cfg.model is None:
        raise ValueError("A composed `model` config is required.")

    output_value = cfg.get("vae_cache_output_dir")
    if output_value is None:
        raise ValueError("Set `+vae_cache_output_dir=<absolute-or-project-relative-path>`.")
    cache_dir = Path(str(output_value)).expanduser().resolve()
    cache_build_locks = _acquire_cache_build_lock(
        cache_dir, rank=rank, world_size=world_size
    )
    # Dataset construction saves normalization statistics on rank zero.
    # Keep them with this cache, including on a checkout with no runs/ directory.
    misc.register_work_dir(cache_dir)
    overwrite = bool(cfg.get("vae_cache_overwrite", False))
    num_workers = int(cfg.get("vae_cache_num_workers", 4))
    if num_workers < 0:
        raise ValueError(f"vae_cache_num_workers must be nonnegative, got {num_workers}")

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    with open_dict(data_cfg):
        data_cfg.video_latent_cache_dir = None
        data_cfg.strict_getitem = True
        data_cfg.precompute_video_only = True
        if data_cfg.get("video_backend") is None:
            raise ValueError("Pin `data.train.video_backend` before precomputing latents.")
        norm_stats = cfg.get("vae_cache_norm_stats_path")
        if norm_stats is not None:
            data_cfg.pretrained_norm_stats = str(norm_stats)

    dataset = instantiate(data_cfg)
    dataset_sample_count = len(dataset)
    max_samples_value = cfg.get("vae_cache_max_samples")
    sample_count = dataset_sample_count
    if max_samples_value is not None:
        sample_count = min(dataset_sample_count, int(max_samples_value))
        if sample_count <= 0:
            raise ValueError(f"vae_cache_max_samples must be positive, got {max_samples_value}")
    rank_indices = list(range(rank, sample_count, world_size))
    existing_completion_path = cache_dir / COMPLETION_FILENAME
    if existing_completion_path.is_file() and not overwrite:
        if existing_completion_path.stat().st_size != sample_count:
            raise ValueError(
                "Existing completion bitmap size does not match the requested sample count: "
                f"{existing_completion_path.stat().st_size} vs {sample_count}"
            )
        existing_completion = torch.from_file(
            str(existing_completion_path),
            shared=False,
            size=sample_count,
            dtype=torch.uint8,
        )
        rank_indices = [index for index in rank_indices if int(existing_completion[index]) != 1]
        logger.info(
            "Rank %d resuming with %d incomplete samples.",
            rank,
            len(rank_indices),
        )
    indexed_dataset = IndexedVideoDataset(dataset, rank_indices)
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["in_order"] = False
    loader = DataLoader(indexed_dataset, **loader_kwargs)
    # Start decoder workers before initializing CUDA in the parent process.
    loader_iterator = iter(loader)

    model_id = str(cfg.model.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    tokenizer_model_id = str(
        cfg.model.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
    )
    redirect_common_files = bool(cfg.model.get("redirect_common_files", True))
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    # Rank 0 downloads missing VAE files before the other ranks read them.
    if rank != 0:
        _barrier(world_size)
    vae_config.download_if_necessary()
    if rank == 0:
        _barrier(world_size)
    vae_path = Path(str(vae_config.path)).resolve()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=str(device),
    ).eval()

    source_video_frames = len(dataset.video_sample_indices)
    height, width = (int(value) for value in dataset.video_size)
    latent_shape = [
        int(vae.model.z_dim),
        (source_video_frames - 1) // int(vae.temporal_downsample_factor) + 1,
        height // int(vae.upsampling_factor),
        width // int(vae.upsampling_factor),
    ]
    vae_sha256 = _broadcast_object(
        _sha256_file(vae_path) if rank == 0 else None,
        rank=rank,
        world_size=world_size,
    )
    preprocess_spec = dataset.video_preprocess_spec
    base_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_count": sample_count,
        "dataset_sample_count": dataset_sample_count,
        "latent_shape": latent_shape,
        "dtype": "bfloat16",
        "source_video_shape": [3, source_video_frames, height, width],
        "preprocess_spec": preprocess_spec,
        "preprocess_spec_sha256": preprocess_spec_sha256(preprocess_spec),
        "vae": {
            "path": str(vae_path),
            "sha256": vae_sha256,
            "model_id": model_id,
            "implementation_sha256": vae_implementation_sha256(),
            "encode_dtype": "bfloat16",
        },
        "encoder_batch_size_per_gpu": 1,
        "world_size": world_size,
    }

    if rank == 0:
        create_or_resume_cache_files(
            cache_dir=cache_dir,
            sample_count=sample_count,
            latent_shape=latent_shape,
            in_progress_manifest={**base_manifest, "status": "in_progress"},
            overwrite=overwrite,
        )
    _barrier(world_size)

    elements_per_sample = int(torch.tensor(latent_shape).prod().item())
    latent_storage = torch.from_file(
        str(cache_dir / LATENTS_FILENAME),
        shared=True,
        size=sample_count * elements_per_sample,
        dtype=torch.bfloat16,
    ).view(sample_count, *latent_shape)
    completion = torch.from_file(
        str(cache_dir / COMPLETION_FILENAME),
        shared=True,
        size=sample_count,
        dtype=torch.uint8,
    )

    processed = 0
    skipped = 0
    pending_completion_indices: list[int] = []
    progress = tqdm(
        total=len(rank_indices),
        desc=f"VAE cache rank {rank}/{world_size}",
        unit="sample",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    try:
        with torch.inference_mode():
            for indices, videos in _iter_loader(loader_iterator):
                index = int(indices.item())
                if int(completion[index].item()) == 1:
                    skipped += 1
                    progress.update(1)
                    continue
                videos = videos.to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
                latents = vae.encode(videos, device=str(device), tiled=False)
                if tuple(latents.shape) != (1, *latent_shape):
                    raise ValueError(
                        f"Unexpected VAE output for sample {index}: {tuple(latents.shape)} "
                        f"vs {(1, *latent_shape)}"
                    )
                latent_storage[index].copy_(
                    latents[0].detach().to(device="cpu", dtype=torch.bfloat16)
                )
                pending_completion_indices.append(index)
                if len(pending_completion_indices) >= COMMIT_INTERVAL_SAMPLES:
                    _commit_completed_indices(
                        cache_dir=cache_dir,
                        completion=completion,
                        indices=pending_completion_indices,
                    )
                    pending_completion_indices.clear()
                processed += 1
                progress.update(1)
    finally:
        progress.close()

    _commit_completed_indices(
        cache_dir=cache_dir,
        completion=completion,
        indices=pending_completion_indices,
    )

    _barrier(world_size)
    counts = torch.tensor([processed, skipped], device=device, dtype=torch.long)
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    if rank == 0:
        incomplete_count = int((completion != 1).sum().item())
        if incomplete_count:
            raise RuntimeError(f"Cache build ended with {incomplete_count} incomplete samples.")
        _fsync_file(cache_dir / LATENTS_FILENAME)
        _fsync_file(cache_dir / COMPLETION_FILENAME)
        final_manifest = {
            **base_manifest,
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
            },
            "processed_samples_this_run": int(counts[0].item()),
            "skipped_samples_this_run": int(counts[1].item()),
        }
        atomic_write_json(final_manifest, cache_dir / MANIFEST_FILENAME)
        (cache_dir / IN_PROGRESS_MANIFEST_FILENAME).unlink(missing_ok=True)
        logger.info(
            "Completed BF16 latent cache: %s (%d samples, shape=%s)",
            cache_dir,
            sample_count,
            latent_shape,
        )

    _barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()
    for cache_build_lock in reversed(cache_build_locks):
        cache_build_lock.close()


if __name__ == "__main__":
    main()
