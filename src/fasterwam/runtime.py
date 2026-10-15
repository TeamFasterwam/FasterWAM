"""Construct model and dataset components for training and inference."""

import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .trainer import Wan22Trainer, create_accelerator_from_cfg, resolve_freeze_video_backbone
from .utils.logging_config import get_logger, setup_logging
from .utils import misc
from .utils.pytorch_utils import set_global_seed

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    freeze_video_backbone: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam import FastWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        freeze_video_backbone=bool(freeze_video_backbone),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_decoupled(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    freeze_video_backbone: bool = False,
    decoupled: bool = True,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    kv_source_mode: str = "final_only",
):
    """Factory function for the Decoupled MoT variant (FastWAMDecoupled).

    Mirrors ``create_fastwam()`` but constructs ``FastWAMDecoupled`` with
    ``decoupled=True``, enabling asymmetric expert layer counts (e.g.,
    video=30, action=5). The action expert loads from
    ``action_dit_pretrained_path`` when provided, optionally with
    layer-selective initialization from ``kv_source_mapping``; otherwise it
    starts from random init.

    Args:
        model_id: HuggingFace model ID for the Wan2.2-TI2V-5B backbone.
        tokenizer_model_id: HuggingFace model ID for the tokenizer.
        video_dit_config: Video DiT architecture config dict.
        tokenizer_max_len: Maximum tokenizer sequence length.
        load_text_encoder: Whether to load the T5 text encoder.
        proprio_dim: Proprioceptive state dimension (None to disable).
        action_dit_config: Action DiT architecture config dict.
        action_dit_pretrained_path: Path to pretrained ActionDiT weights.
        skip_dit_load_from_pretrain: If True, skip pretrained DiT loads (for full checkpoint override).
        decoupled: Must be True for this factory (passed to from_wan22_pretrained).
        video_scheduler: Video scheduler config dict.
        action_scheduler: Action scheduler config dict.
        loss: Loss config dict with lambda_video and lambda_action.
        mot_checkpoint_mixed_attn: Whether to use gradient checkpointing in MoT.
        redirect_common_files: Whether to redirect common HuggingFace files.
        model_dtype: Model dtype for parameters.
        device: Device to place the model on.
        kv_source_mode: KV source routing mode for the action expert.

    Returns:
        FastWAMDecoupled instance with decoupled video/action experts.
    """
    from .models.wan22.fastwam_decoupled import FastWAMDecoupled

    if not decoupled:
        # This factory only builds decoupled models. Previously the flag was
        # accepted but ignored (decoupled=True was hardcoded downstream), so a
        # config setting decoupled=false silently got a decoupled model anyway.
        raise ValueError(
            "create_fastwam_decoupled only builds decoupled models; got "
            "decoupled=False. Use fasterwam.runtime.create_fastwam (model=fastwam) "
            "for the non-decoupled variant."
        )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    from .models.wan22.mot_decoupled import compute_kv_source_mapping

    layer_selected = action_dit_config.pop("layer_selected", None)
    if layer_selected is not None:
        action_dit_config["num_layers"] = len(layer_selected)
        kv_source_mapping = list(layer_selected)
        logger.info(f"layer_selected={layer_selected}, overriding num_layers={len(layer_selected)}, kv_source_mapping={kv_source_mapping}")
    else:
        kv_source_mapping = compute_kv_source_mapping(
            mode=kv_source_mode,
            video_num_layers=int(video_dit_config.get("num_layers", 30)),
            action_num_layers=int(action_dit_config.get("num_layers", 5)),
        )
        logger.info(f"KV source mode: {kv_source_mode}, mapping: {kv_source_mapping}")

    return FastWAMDecoupled.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        freeze_video_backbone=bool(freeze_video_backbone),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        decoupled=True,
        kv_source_mapping=kv_source_mapping,
        kv_source_mode=kv_source_mode,
    )


def _resolve_model_vae_identity(
    model_cfg: DictConfig, model_dtype: torch.dtype
) -> dict[str, str]:
    """Resolve and hash the exact VAE artifact selected by the model factory."""

    import hashlib

    from .datasets.video_latent_cache import vae_implementation_sha256
    from .models.wan22.helpers.loader import _resolve_configs

    if model_dtype != torch.bfloat16:
        raise ValueError(
            "The configured video latent cache was encoded with a BF16 VAE; "
            f"cached training requires mixed_precision=bf16, got {model_dtype}."
        )

    model_id = str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    tokenizer_model_id = str(
        model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
    )
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
    )
    vae_config.download_if_necessary()
    vae_path = Path(str(vae_config.path)).resolve()
    digest = hashlib.sha256()
    with vae_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "model_id": model_id,
        "sha256": digest.hexdigest(),
        "implementation_sha256": vae_implementation_sha256(),
        "encode_dtype": "bfloat16",
    }


def build_datasets(
    data_cfg: DictConfig,
    model_cfg: DictConfig | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
):
    resolved_data_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
    if (
        model_cfg is not None
        and resolved_data_cfg.train.get("video_latent_cache_dir") is not None
    ):
        resolved_data_cfg.train.video_latent_cache_vae_identity = _resolve_model_vae_identity(
            model_cfg, model_dtype
        )
    train_ds = instantiate(resolved_data_cfg.train)
    if resolved_data_cfg.get("val") is None:
        if resolved_data_cfg.train.get("video_latent_cache_dir") is None:
            val_ds = train_ds
        else:
            # Periodic evaluation needs original pixels for rollout and VAE
            # reconstruction metrics. Keep training video-free while building
            # one online-video dataset for the infrequent evaluation sample.
            online_val_cfg = OmegaConf.create(
                OmegaConf.to_container(resolved_data_cfg.train, resolve=True)
            )
            online_val_cfg.video_latent_cache_dir = None
            online_val_cfg.video_latent_cache_vae_identity = None
            online_val_cfg.strict_getitem = False
            online_val_cfg.is_training_set = False
            stats_path = resolved_data_cfg.train.get("pretrained_norm_stats")
            online_val_cfg.pretrained_norm_stats = (
                stats_path
                if stats_path is not None
                else os.path.join(misc.get_work_dir(), "dataset_stats.json")
            )
            logger.info(
                "Building an online-video validation dataset while training uses cached latents."
            )
            val_ds = instantiate(online_val_cfg)
    else:
        train_stats_path = resolved_data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = resolved_data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(resolved_data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    resolve_freeze_video_backbone(cfg)
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    accelerator = create_accelerator_from_cfg(cfg)
    model_device = str(accelerator.device)
    # Construct random action heads and datasets reproducibly on every rank.
    # The trainer subsequently selects rank-specific streams for training.
    set_global_seed(int(cfg.seed), rank_specific=False)
    # One process per node downloads missing backbone files before the others load them.
    with accelerator.local_main_process_first():
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data, cfg.model, model_dtype)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        accelerator=accelerator,
    )
    trainer.train()
