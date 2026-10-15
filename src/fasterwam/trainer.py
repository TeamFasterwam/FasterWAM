import logging
import json
import inspect
import os
import re
from collections.abc import Mapping
from datetime import timedelta
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .lr_schedules import WSDDecayLR, build_wsd_schedule
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def resolve_freeze_video_backbone(cfg: DictConfig, model=None) -> bool:
    """Resolve the training switch, accepting a model-only legacy config.

    Model presets inherit the top-level switch. Explicit conflicting values
    are rejected before model construction or optimizer initialization.
    """
    trainer_value = OmegaConf.select(cfg, "freeze_video_backbone", default=None)
    model_value = OmegaConf.select(cfg, "model.freeze_video_backbone", default=None)
    for name, value in (
        ("freeze_video_backbone", trainer_value),
        ("model.freeze_video_backbone", model_value),
    ):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean, got {value!r}")
    if trainer_value is not None and model_value is not None and trainer_value != model_value:
        raise ValueError(
            "Conflicting freeze_video_backbone and model.freeze_video_backbone values. "
            "Set freeze_video_backbone at the top level and let the model inherit it, "
            "or use matching values in legacy configs."
        )
    if trainer_value is not None:
        return trainer_value
    if model_value is not None:
        return model_value
    return bool(getattr(model, "freeze_video_backbone", False))


def create_accelerator_from_cfg(cfg: DictConfig) -> Accelerator:
    mixed_precision = str(cfg.mixed_precision).strip().lower()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {cfg.mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    distributed_timeout_minutes = int(
        OmegaConf.select(cfg, "distributed_timeout_minutes", default=60)
    )
    return Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(minutes=distributed_timeout_minutes))
        ],
    )


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig, accelerator=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        # WSD (warmup-stable-decay) schedule knobs. `wsd_branch_decay_steps`
        # turns this run into a decay branch off a full-state checkpoint; see
        # configs/train.yaml and fasterwam/lr_schedules.py.
        self.wsd_decay_fraction = float(OmegaConf.select(cfg, "wsd_decay_fraction", default=0.1))
        self.wsd_decay_shape = str(
            OmegaConf.select(cfg, "wsd_decay_shape", default="linear")
        ).strip().lower()
        branch_decay_steps = OmegaConf.select(cfg, "wsd_branch_decay_steps", default=None)
        self.wsd_branch_decay_steps = (
            int(branch_decay_steps) if branch_decay_steps is not None else None
        )
        if self.wsd_branch_decay_steps is not None and self.wsd_branch_decay_steps < 1:
            raise ValueError(
                f"wsd_branch_decay_steps must be >= 1 or null, got {self.wsd_branch_decay_steps}"
            )
        # Filled in by _apply_wsd_branch() on a branch resume and persisted in
        # trainer_state.json so an interrupted branch resumes as a continuation.
        self.lr_branch = None
        # End-to-end wsd only: the optimizer step at which the stable phase ends.
        # A full-state checkpoint is always written there so decay branches can
        # start from the exact end of the plateau.
        self.wsd_decay_start_step = None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_steps = self._normalize_save_steps(
            OmegaConf.select(cfg, "save_steps", default=[])
        )
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.freeze_video_backbone = resolve_freeze_video_backbone(cfg, model)
        
        self.resume = cfg.resume
        if self.wsd_branch_decay_steps is not None and not (
            self.resume and Path(str(self.resume)).is_dir()
        ):
            raise ValueError(
                "wsd_branch_decay_steps requires resume=<checkpoints/state/step_XXXXXX> "
                "(a full-state directory) to branch from; "
                f"got resume={self.resume!r}"
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.distributed_timeout_minutes = int(
            OmegaConf.select(cfg, "distributed_timeout_minutes", default=60)
        )

        self.accelerator = accelerator or create_accelerator_from_cfg(cfg)
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", {})
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f distributed_timeout_min=%d",
            self.accelerator.distributed_type,
            deepspeed_config.get("zero_optimization", {}).get("stage", "none"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
            self.distributed_timeout_minutes,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        # Initialization uses a shared seed in run_training; training noise and
        # data workers use separate streams on each distributed rank.
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(
            self.model,
            freeze_video_backbone=self.freeze_video_backbone,
        )

        # Log total/trainable/frozen param counts for visibility.
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        logger.info(
            "Total params: %.2fB | Trainable: %.2fB | Frozen: %.2fB",
            total_params / 1e9,
            trainable_params_count / 1e9,
            frozen_params_count / 1e9,
        )
        if getattr(self.model, 'freeze_video_backbone', False):
            video_params = sum(
                p.numel() for p in self.model.mot.mixtures["video"].parameters()
            )
            logger.info("Video backbone FROZEN: %.2fB params", video_params / 1e9)

        # Filter by requires_grad so frozen params (e.g., video expert when
        # freeze_video_backbone=True) are excluded from the optimizer, saving
        # DeepSpeed optimizer state memory.
        trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend([p for p in proprio_encoder.parameters() if p.requires_grad])
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self._log_lr_schedule(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        logger.info(
            "Full-state checkpoint schedule: save_every=%d save_steps=%s; "
            "WSD pre-decay and final checkpoints are always retained",
            self.save_every, sorted(self.save_steps),
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

        raw_model = self.accelerator.unwrap_model(self.model)
        mot = getattr(raw_model, "mot", None)
        if mot is not None:
            model_meta = {}
            for name in getattr(mot, "expert_order", []):
                expert = mot.mixtures[name]
                model_meta[f"{name}_num_params_B"] = round(
                    sum(p.numel() for p in expert.parameters()) / 1e9, 2
                )
            if hasattr(mot, "kv_source_mapping"):
                model_meta["kv_source_mapping"] = mot.kv_source_mapping
            if hasattr(mot, "video_num_layers"):
                model_meta["video_num_layers"] = mot.video_num_layers
            if hasattr(mot, "action_num_layers"):
                model_meta["action_num_layers"] = mot.action_num_layers
            wandb.config.update({"model_meta": model_meta}, allow_val_change=True)

        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)


    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )


    @staticmethod
    def _normalize_save_steps(steps) -> frozenset[int]:
        if steps is None:
            return frozenset()
        if not isinstance(steps, (list, tuple)) and not OmegaConf.is_list(steps):
            raise ValueError("save_steps must be a list of positive integer optimizer steps")
        if any(isinstance(step, bool) or not isinstance(step, int) or step < 1 for step in steps):
            raise ValueError("save_steps must contain only positive integer optimizer steps")
        return frozenset(steps)

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _opt_steps_per_epoch(self) -> int | None:
        if not hasattr(self.train_dataset, "__len__"):
            return None
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        return max(ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1)

    def _log_lr_schedule(self, scheduler_type, total_train_steps: int, warmup_steps: int):
        steps_per_epoch = self._opt_steps_per_epoch()
        if self.wsd_branch_decay_steps is not None:
            logger.info(
                "LR schedule: WSD decay branch pending (decay_steps=%d shape=%s floor=%.2e); "
                "lr_scheduler_type=%s is ignored, the decay is anchored to the resumed "
                "checkpoint's lr in load_training_state(). opt_steps_per_epoch=%s",
                self.wsd_branch_decay_steps,
                self.wsd_decay_shape,
                self.learning_rate * 0.01,
                scheduler_type,
                steps_per_epoch,
            )
            return
        scheduler_type = str(scheduler_type).strip().lower()
        if scheduler_type == "wsd":
            decay_steps = self._wsd_decay_steps(total_train_steps)
            logger.info(
                "LR schedule: wsd total_steps=%d warmup=%d stable=%d decay=%d (fraction=%.3f shape=%s) "
                "peak=%.2e floor=%.2e opt_steps_per_epoch=%s",
                total_train_steps,
                warmup_steps,
                total_train_steps - warmup_steps - decay_steps,
                decay_steps,
                self.wsd_decay_fraction,
                self.wsd_decay_shape,
                self.learning_rate,
                self.learning_rate * 0.01,
                steps_per_epoch,
            )
            return
        logger.info(
            "LR schedule: %s total_steps=%d warmup=%d peak=%.2e opt_steps_per_epoch=%s",
            scheduler_type,
            total_train_steps,
            warmup_steps,
            self.learning_rate,
            steps_per_epoch,
        )

    def _wsd_decay_steps(self, total_train_steps: int) -> int:
        if not (0.0 < self.wsd_decay_fraction < 1.0):
            raise ValueError(
                f"wsd_decay_fraction must be in (0, 1), got {self.wsd_decay_fraction}"
            )
        return max(int(round(int(total_train_steps) * self.wsd_decay_fraction)), 1)

    @staticmethod
    def _unwrap_scheduler(scheduler):
        # accelerator.prepare() wraps the torch scheduler in AcceleratedScheduler,
        # which exposes the original under `.scheduler`.
        return getattr(scheduler, "scheduler", scheduler)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        if self.wsd_branch_decay_steps is not None:
            # Decay branch. Starts at the peak lr; _apply_wsd_branch() re-anchors
            # it to the checkpoint's lr once the full state has been restored.
            return WSDDecayLR(
                self.optimizer,
                decay_steps=self.wsd_branch_decay_steps,
                end_lr=self.learning_rate * 0.01,
                shape=self.wsd_decay_shape,
            )
        if scheduler_type == "wsd":
            decay_steps = self._wsd_decay_steps(total_train_steps)
            self.wsd_decay_start_step = total_train_steps - decay_steps
            return build_wsd_schedule(
                self.optimizer,
                total_steps=total_train_steps,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_lr=self.learning_rate * 0.01,
                shape=self.wsd_decay_shape,
            )

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant', 'wsd']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    @staticmethod
    def _weights_checkpoint_beside_state_dir(state_dir: Path) -> Path | None:
        """Locate the weights checkpoint written alongside a full-state directory.

        ``save_checkpoint`` always writes ``checkpoints/weights/<step_tag>.pt``
        and ``checkpoints/state/<step_tag>/`` together, so the sibling weights
        file is this state directory's semantic-identity record.
        """
        candidate = state_dir.parent.parent / "weights" / f"{state_dir.name}.pt"
        return candidate if candidate.is_file() else None

    def _validate_full_state_resume_identity(self, state_dir: Path) -> None:
        """Check the model options before restoring a full training state.

        ``accelerator.load_state()`` restores tensors without the model options,
        and presets with different positional interfaces have identical shapes.
        The weights checkpoint saved beside every state directory is therefore
        loaded first through ``load_checkpoint``, which rejects a mismatch.
        """
        reference = self._weights_checkpoint_beside_state_dir(state_dir)
        if reference is None:
            expected = state_dir.parent.parent / "weights" / f"{state_dir.name}.pt"
            raise RuntimeError(
                f"No weights checkpoint beside state directory {state_dir} (expected "
                f"{expected}); cannot verify that it matches the configured model."
            )
        logger.info("Validating model options for full-state resume against %s", reference)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(reference), optimizer=None)

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self._validate_full_state_resume_identity(resume_path)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model, freeze_video_backbone: bool | None = None):
        if freeze_video_backbone is None:
            freeze_video_backbone = getattr(model, "freeze_video_backbone", False)
        freeze_video_backbone = bool(freeze_video_backbone)
        # Parameter freezing and the Decoupled compute path must use the same
        # resolved switch, including after evaluation restores training mode.
        model.freeze_video_backbone = freeze_video_backbone
        model.freeze_video_backbone_for_training = freeze_video_backbone

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        # Re-freeze video expert after blanket unfreeze when freeze_video_backbone is enabled.
        # This must come AFTER dit.requires_grad_(True) so only the video mixture is re-frozen.
        if freeze_video_backbone:
            model.mot.mixtures["video"].requires_grad_(False)
            model.mot.mixtures["video"].eval()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)


    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        eval_sample_is_unbatched = video.ndim == 4
        if eval_sample_is_unbatched:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt
        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        if self.lr_branch is not None:
            payload["lr_branch"] = dict(self.lr_branch)
        if self.wsd_decay_start_step is not None:
            payload["wsd_decay_start_step"] = int(self.wsd_decay_start_step)
            payload["wsd_phase"] = (
                "stable" if self.global_step <= self.wsd_decay_start_step else "decay"
            )
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self._apply_wsd_branch(payload)
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self._apply_wsd_branch({})
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def _apply_wsd_branch(self, payload: Mapping) -> None:
        """Anchor the decay branch to the checkpoint that was just restored.

        Runs after ``accelerator.load_state()`` inside ``load_training_state``.
        Two cases:

        * Fresh branch (the checkpoint's ``trainer_state.json`` carries no
          ``lr_branch``): the checkpoint's own scheduler state was ignored by
          :class:`WSDDecayLR`. Restart the decay at the lr that run last
          applied and stop after ``wsd_branch_decay_steps`` more steps.
        * Interrupted branch (``lr_branch`` present): the scheduler state was
          restored as-is, so keep it and recompute the stop step from the
          original branch point.
        """
        if self.wsd_branch_decay_steps is None:
            return
        inner = self._unwrap_scheduler(self.scheduler)
        if not isinstance(inner, WSDDecayLR):
            raise RuntimeError(
                "wsd_branch_decay_steps is set but the active scheduler is "
                f"{type(inner).__name__}, not WSDDecayLR"
            )

        prior = payload.get("lr_branch") if payload else None
        if prior:
            origin_step = int(prior["origin_step"])
            decay_steps = int(prior["decay_steps"])
            if decay_steps != self.wsd_branch_decay_steps:
                logger.warning(
                    "Resuming an interrupted WSD decay branch whose decay_steps=%d differs from "
                    "wsd_branch_decay_steps=%d; keeping the checkpoint's value.",
                    decay_steps,
                    self.wsd_branch_decay_steps,
                )
            self.lr_branch = dict(prior)
            self.max_steps = origin_step + decay_steps
            logger.info(
                "WSD decay branch resumed: origin_step=%d decay_steps=%d step=%d max_steps=%d "
                "lr=%s",
                origin_step,
                decay_steps,
                self.global_step,
                self.max_steps,
                inner.get_last_lr(),
            )
            return

        if inner.foreign_last_lr is not None:
            start_lrs = list(inner.foreign_last_lr)
            start_source = "checkpoint lr-scheduler state"
        else:
            start_lrs = [float(group["lr"]) for group in inner.optimizer.param_groups]
            start_source = "restored optimizer param_groups"
        inner.rebase(start_lrs, decay_steps=self.wsd_branch_decay_steps)
        self.lr_branch = {
            "origin_step": int(self.global_step),
            "decay_steps": int(self.wsd_branch_decay_steps),
            "start_lrs": [float(lr) for lr in start_lrs],
            "end_lr": float(inner.end_lr),
            "shape": str(inner.shape),
            "origin_state_dir": str(self.resume),
        }
        self.max_steps = int(self.global_step) + int(self.wsd_branch_decay_steps)
        logger.info(
            "WSD decay branch: origin_step=%d decay_steps=%d max_steps=%d start_lr=%s (from %s) "
            "end_lr=%.2e shape=%s",
            self.global_step,
            self.wsd_branch_decay_steps,
            self.max_steps,
            start_lrs,
            start_source,
            inner.end_lr,
            inner.shape,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    periodic_save = self.save_every > 0 and self.global_step % self.save_every == 0
                    explicit_save = self.global_step in self.save_steps
                    decay_start_save = (
                        self.wsd_decay_start_step is not None
                        and self.global_step == self.wsd_decay_start_step
                        and self.global_step < self.max_steps
                    )
                    # The final-step path below saves once even if it coincides
                    # with an explicit/periodic checkpoint.
                    if (periodic_save or explicit_save or decay_start_save) and self.global_step < self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                            if explicit_save:
                                logger.info("[ckpt] explicit full-state save_steps milestone=%d", self.global_step)
                            if decay_start_save:
                                logger.info(
                                    "[ckpt] WSD stable phase ends here: step=%d is the last "
                                    "peak-lr checkpoint (lr=%.2e); decay runs %d -> %d. Branch "
                                    "from state=%s to finish the stable model at other lengths.",
                                    self.global_step,
                                    current_lr,
                                    self.global_step,
                                    self.max_steps,
                                    ckpt_info["state_path"],
                                )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
