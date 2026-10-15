"""Run concurrent LIBERO simulator actors against one batched GPU model server."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import faulthandler
import hashlib
import json
import inspect
import logging
from multiprocessing.connection import Connection, wait
import os
import queue
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from experiments.libero.action_ensembler import ActionEnsembler
from experiments.libero.eval_libero_single import (
    _center_crop_resize,
    _denormalize_action,
    _extract_sim_state,
    _get_max_steps,
    _get_num_video_frames,
    _log_episode_stage_timing,
    _normalize_uint8_rgb_batch,
    _normalize_proprio,
    _resolve_dataset_stats_path,
    _resolve_environment_seed,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    save_rollout_video,
)
from experiments.libero.selective_render_env import build_selective_render_env_factory
from experiments.libero.task_prompt_cache import (
    load_task_prompt_context,
    load_task_prompt_context_for_task,
)
from experiments.libero.task_work_queue import (
    TaskLease,
    claim_next_task,
    complete_task,
    retry_or_fail_task,
)
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


INFERENCE_TIMEOUT_SECONDS = 600.0
QUEUE_POLL_SECONDS = 1.0
# How long a partially filled batch may wait for more requests. Only a backstop
# against a slow actor: the live-actor cap already dispatches as soon as no
# further request can arrive, so this never gates the normal drain path.
BATCH_COLLECT_TIMEOUT_SECONDS = 0.05
ACTOR_TERMINAL_EOF_TIMEOUT_SECONDS = 1.0
ACTOR_SHUTDOWN_TIMEOUT_SECONDS = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTOR_BOOTSTRAP_CODE = (
    "import sys; "
    "from experiments.libero.shared_model_batch import _actor_subprocess_entry; "
    "_actor_subprocess_entry(int(sys.argv[1]), int(sys.argv[2]))"
)
EPISODE_RETRY_EXCEPTIONS = (AttributeError,)
ACTOR_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    ValueError,
)


@dataclass
class _ActorRuntime:
    """Track one isolated actor subprocess and its duplex control channel."""

    actor_id: int
    render_gpu_id: int
    process: subprocess.Popen[bytes]
    connection: Connection


class _ConnectionQueueAdapter:
    """Provide the queue methods used by actor code over one duplex connection."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def put(self, message: dict[str, Any]) -> None:
        """Send one actor message to the model server."""
        self._connection.send(message)

    def get(self, timeout: float) -> dict[str, Any]:
        """Receive one model response or raise queue.Empty at the deadline."""
        if not self._connection.poll(timeout):
            raise queue.Empty
        response = self._connection.recv()
        if not isinstance(response, dict):
            raise RuntimeError(
                f"actor received non-dict model response: {type(response).__name__}"
            )
        return response


def _parent_import_path_entries() -> list[str]:
    """Return the parent worker's importable ``sys.path`` directories, in order.

    The worker itself is started as ``python experiments/libero/eval_libero_batch.py``,
    so ``sys.path[0]`` is ``<repo>/experiments/libero``. Actor children are started
    with ``python -c``, which seeds ``sys.path[0]`` from the working directory only,
    so unqualified sibling imports in the actor import chain (for example
    ``from action_ensembler import ActionEnsembler``) would otherwise fail.
    """
    entries: list[str] = []
    seen: set[str] = set()
    for raw_entry in sys.path:
        # An empty entry means "current directory"; the child already runs with
        # PROJECT_ROOT as its working directory, so it needs no PYTHONPATH entry.
        if not raw_entry:
            continue
        # Relative entries resolve against the parent's working directory, which
        # is the same resolution the parent performs for its own imports.
        resolved = os.path.abspath(raw_entry)
        # PYTHONPATH only carries directories. Zip imports (``python310.zip``)
        # and stale entries are dropped; the child rebuilds the interpreter's
        # own standard entries from its configuration.
        if not os.path.isdir(resolved):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        entries.append(resolved)
    return entries


def _actor_pythonpath(inherited_pythonpath: str | None) -> str:
    """Merge the parent's import path ahead of an inherited ``PYTHONPATH``."""
    entries = _parent_import_path_entries()
    seen = set(entries)
    for raw_entry in (inherited_pythonpath or "").split(os.pathsep):
        if not raw_entry:
            continue
        resolved = os.path.abspath(raw_entry)
        if resolved in seen:
            continue
        seen.add(resolved)
        entries.append(resolved)
    return os.pathsep.join(entries)


def _actor_environment(render_gpu_id: int) -> dict[str, str]:
    """Build an actor environment without mutating the model worker environment."""
    actor_env = os.environ.copy()
    device_id = str(render_gpu_id)
    # Robosuite validates both variables during import, before actor code runs.
    actor_env["CUDA_VISIBLE_DEVICES"] = device_id
    actor_env["MUJOCO_EGL_DEVICE_ID"] = device_id
    # ``python -c`` children do not inherit the parent's sys.path, so the actor
    # import chain is reproduced explicitly instead.
    actor_pythonpath = _actor_pythonpath(actor_env.get("PYTHONPATH"))
    if actor_pythonpath:
        actor_env["PYTHONPATH"] = actor_pythonpath
    return actor_env


def _launch_actor_subprocess(actor_id: int, render_gpu_id: int) -> _ActorRuntime:
    """Start an actor whose EGL ID exists before Python imports rendering modules."""
    # The inherited socket avoids multiprocessing spawn re-importing the parent main.
    parent_socket, child_socket = socket.socketpair()
    child_fd = child_socket.fileno()
    command = [
        sys.executable,
        "-c",
        ACTOR_BOOTSTRAP_CODE,
        str(child_fd),
        str(render_gpu_id),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=_actor_environment(render_gpu_id),
            pass_fds=(child_fd,),
        )
    except OSError:
        parent_socket.close()
        child_socket.close()
        raise
    child_socket.close()
    connection = Connection(parent_socket.detach())
    return _ActorRuntime(
        actor_id=actor_id,
        render_gpu_id=render_gpu_id,
        process=process,
        connection=connection,
    )


def _actor_subprocess_entry(actor_fd: int, render_gpu_id: int) -> None:
    """Receive actor initialization after spawn-time EGL binding is already active."""
    connection = Connection(actor_fd)
    try:
        # No rendering-dependent object is created before this binding assertion.
        actual_render_gpu_id = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        if actual_render_gpu_id != str(render_gpu_id):
            raise RuntimeError(
                "Actor EGL binding mismatch before rendering initialization: "
                f"expected {render_gpu_id}, got {actual_render_gpu_id!r}."
            )
        initialization = connection.recv()
        if not isinstance(initialization, dict):
            raise RuntimeError("actor initialization payload must be a dict")
        if int(initialization["render_gpu_id"]) != render_gpu_id:
            raise RuntimeError(
                "Actor initialization render GPU does not match spawn argument: "
                f"{initialization['render_gpu_id']!r} != {render_gpu_id}."
            )
        queue_adapter = _ConnectionQueueAdapter(connection)
        _actor_main(
            actor_id=int(initialization["actor_id"]),
            tasks=list(initialization["tasks"]),
            cfg_payload=dict(initialization["cfg_payload"]),
            action_horizon=int(initialization["action_horizon"]),
            input_w=int(initialization["input_w"]),
            input_h=int(initialization["input_h"]),
            request_queue=queue_adapter,
            response_queue=queue_adapter,
        )
    finally:
        connection.close()


@contextmanager
def _optional_cuda_timing_sync(enabled: bool) -> Iterator[None]:
    """Suppress timing-only CUDA barriers for one serialized policy request.

    Faster-WAM's inference path uses full-device synchronization to report stage
    timings. The optimized shared-model server is single-threaded, and the
    final action copy to CPU is already a completion barrier, so suppressing
    these intermediate timing barriers does not reorder model operations.
    """
    if not enabled:
        yield
        return

    original_synchronize = torch.cuda.synchronize

    def _no_op_synchronize(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    torch.cuda.synchronize = _no_op_synchronize
    try:
        yield
    finally:
        torch.cuda.synchronize = original_synchronize


def _prepare_actor_model_input(
    obs: dict[str, np.ndarray],
    cfg: DictConfig,
    processor: Any,
    *,
    input_w: int,
    input_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop images and normalize proprio on an actor CPU.

    The returned RGB array remains uint8. The model process therefore performs
    the same uint8-to-CUDA-BF16 conversion and scaling as the sequential
    evaluator, preserving the policy's numerical path.
    """
    images = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    num_cameras = int(processor.num_output_cameras)
    if len(image_meta) < num_cameras:
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={num_cameras}."
        )

    def _meta_to_hw(meta: dict[str, Any], camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(
                f"shape_meta.images[{camera_idx}].shape must be [C,H,W], "
                f"got {shape}"
            )
        return int(shape[1]), int(shape[2])

    concatenation = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(images["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(
            images["image"], width=primary_w, height=primary_h
        )
        wrist = _center_crop_resize(
            images["wrist_image"], width=wrist_w, height=wrist_h
        )
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(
            "LIBERO shared-model evaluation supports one or two output "
            f"cameras, got {num_cameras}."
        )

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    if (actual_h, actual_w) != (int(input_h), int(input_w)):
        raise ValueError(
            "Input image size mismatch after actor preprocessing: "
            f"got (H,W)=({actual_h},{actual_w}), expected "
            f"({int(input_h)},{int(input_w)})."
        )

    proprio = (
        _normalize_proprio(_extract_sim_state(obs), processor).detach().cpu().numpy()
    )
    return np.ascontiguousarray(rgb), np.ascontiguousarray(proprio)


def _predict_prepared_action_chunk(
    *,
    rgb: np.ndarray,
    proprio: np.ndarray,
    task_description: str,
    model: torch.nn.Module,
    processor: Any,
    cfg: DictConfig,
    action_horizon: int,
    model_device: str,
    prompt_context: tuple[torch.Tensor, torch.Tensor] | None,
    noise_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Run one policy request from actor-preprocessed batch-one inputs.

    ``noise_seed`` overrides the run-level seed for this one request and is set
    when ``per_request_noise`` is enabled.
    """
    image = _normalize_uint8_rgb_batch(
        rgb,
        device=model_device,
        dtype=model.torch_dtype,
    )
    proprio_tensor = torch.from_numpy(proprio)

    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    num_inference_steps = (
        int(cfg.get("eval_num_inference_steps", 20))
        if num_inference_steps_cfg is None
        else int(num_inference_steps_cfg)
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    infer_kwargs: dict[str, Any] = {
        "prompt": None if prompt_context is not None else prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio_tensor,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": (
            noise_seed
            if noise_seed is not None and uses_per_request_noise(cfg)
            else (None if cfg.get("seed") is None else int(cfg.seed))
        ),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if prompt_context is not None:
        context, context_mask = prompt_context
        infer_kwargs["context"] = context
        infer_kwargs["context_mask"] = context_mask
    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    disable_timing_sync = bool(
        cfg.EVALUATION.get("disable_inference_timing_sync", False)
    )
    with torch.no_grad(), _optional_cuda_timing_sync(disable_timing_sync):
        prediction = model.infer_action(**infer_kwargs)
    action = _denormalize_action(prediction["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    timing_ms = None if disable_timing_sync else prediction.get("timing_ms", None)
    return action, timing_ms


def uses_per_request_noise(cfg: DictConfig) -> bool:
    """Whether each inference request should draw its own action noise.

    Defaults to ``True``: every inference request draws its own noise, so a
    run estimates ``E_z[p(success)]``.

    With ``False`` the model rebuilds its generator from the run-level ``seed``
    on every request, so one frozen noise vector serves every denoising call of
    every episode and the run estimates ``p(success | z)`` for a single draw.
    Results under the two settings are not directly comparable.
    """
    return bool(cfg.EVALUATION.get("per_request_noise", True))


def derive_request_noise_seed(
    *,
    base_seed: int,
    suite_name: str,
    task_id: int,
    episode_idx: int,
    request_id: int,
) -> int:
    """Deterministic per-request action-noise seed.

    Keyed on request identity rather than on a counter so the draw is stable
    under the dynamic task queue, actor count and batch composition -- rerunning
    the same evaluation reproduces the same noise even though scheduling order
    does not repeat. ``hashlib`` is used rather than ``hash()`` because the
    latter is salted per process for strings.
    """
    payload = f"{base_seed}|{suite_name}|{task_id}|{episode_idx}|{request_id}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    # torch generators accept the full uint64 range; keep it positive for logs.
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


def batch_group_key(cfg: DictConfig) -> tuple:
    """Identify request groups that may share one batched forward pass.

    Every inference parameter that `_predict_prepared_action_batch` applies to
    the whole batch must appear here, so a group can never silently inherit the
    first request's value for something another member configured differently.
    """
    sigma_shift = cfg.EVALUATION.get("sigma_shift")
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    return (
        int(cfg.get("eval_num_inference_steps", 20))
        if num_inference_steps_cfg is None
        else int(num_inference_steps_cfg),
        None if sigma_shift is None else float(sigma_shift),
        None if cfg.get("seed") is None else int(cfg.seed),
        # Changes how `seed` is interpreted, so it must split batch groups.
        uses_per_request_noise(cfg),
        str(cfg.EVALUATION.get("rand_device", "cpu")),
        bool(cfg.EVALUATION.get("tiled", False)),
        bool(cfg.EVALUATION.get("binarize_gripper", False)),
        bool(cfg.EVALUATION.get("disable_inference_timing_sync", False)),
    )


def _predict_prepared_action_batch(
    *,
    rgb_batch: list[np.ndarray],
    proprio_batch: list[np.ndarray],
    prompt_contexts: list[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    processor: Any,
    cfg: DictConfig,
    action_horizon: int,
    model_device: str,
    request_seeds: list[int] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any] | None]:
    """Serve a homogeneous group of policy requests in one forward pass.

    ``request_seeds`` carries one action-noise seed per member and is required
    when the group's config enables ``per_request_noise``; otherwise the group
    shares the single run-level seed.

    ``cfg`` is the task config of any member: `batch_group_key()` guarantees all
    members agree on every parameter read here. Per-request text context still
    differs and is stacked, which is safe because the cache pads every prompt to
    ``tokenizer_max_len``.
    """
    batch_size = len(rgb_batch)
    assert batch_size == len(proprio_batch) == len(prompt_contexts) >= 1

    images = _normalize_uint8_rgb_batch(
        np.stack(rgb_batch),
        device=model_device,
        dtype=model.torch_dtype,
    )
    proprio_tensor = torch.from_numpy(np.stack(proprio_batch))
    if proprio_tensor.ndim == 3 and proprio_tensor.shape[1] == 1:
        proprio_tensor = proprio_tensor.squeeze(1)
    context = torch.cat([entry[0] for entry in prompt_contexts], dim=0)
    context_mask = torch.cat([entry[1] for entry in prompt_contexts], dim=0)
    (
        num_inference_steps,
        sigma_shift,
        seed,
        per_request_noise,
        rand_device,
        tiled,
        binarize_gripper,
        disable_timing_sync,
    ) = batch_group_key(cfg)

    # A scalar seed makes every slot share one noise draw; a per-slot list gives
    # each request its own. Both are slot-invariant (see `infer_action_batch`).
    if per_request_noise and seed is not None:
        if request_seeds is None or len(request_seeds) != batch_size:
            raise ValueError(
                "`per_request_noise` is enabled but this group carries "
                f"{0 if request_seeds is None else len(request_seeds)} request "
                f"seeds for {batch_size} slots."
            )
        seed_arg: Any = list(request_seeds)
    else:
        seed_arg = seed

    infer_kwargs: dict[str, Any] = {
        "input_image": images,
        "action_horizon": action_horizon,
        "context": context,
        "context_mask": context_mask,
        "proprio": proprio_tensor,
        "num_inference_steps": num_inference_steps,
        "sigma_shift": sigma_shift,
        "seed": seed_arg,
        "rand_device": rand_device,
        "tiled": tiled,
    }

    with torch.no_grad(), _optional_cuda_timing_sync(disable_timing_sync):
        prediction = model.infer_action_batch(**infer_kwargs)

    denormalized = _denormalize_action(prediction["action"], processor)
    actions: list[np.ndarray] = []
    for index in range(batch_size):
        action = denormalized[index]
        action[..., -1] = action[..., -1] * 2 - 1
        action = invert_gripper_action(action)
        if binarize_gripper:
            action[..., -1] = np.sign(action[..., -1])
        actions.append(action)
    timing_ms = None if disable_timing_sync else prediction.get("timing_ms", None)
    return actions, timing_ms


def partition_tasks(
    tasks: list[tuple[str, int]], num_actors: int
) -> list[list[tuple[str, int]]]:
    """Round-robin tasks across a fixed number of simulator actors."""
    assert num_actors > 0, f"num_actors must be positive, got {num_actors}"
    partitions: list[list[tuple[str, int]]] = [[] for _ in range(num_actors)]
    for task_index, task in enumerate(tasks):
        partitions[task_index % num_actors].append(task)
    return partitions


def _task_config(cfg: DictConfig, suite_name: str, task_id: int) -> DictConfig:
    task_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    )
    OmegaConf.update(task_cfg, "EVALUATION.task_suite_name", suite_name, force_add=True)
    OmegaConf.update(task_cfg, "EVALUATION.task_id", int(task_id), force_add=True)
    return task_cfg


def _run_actor_episode(
    *,
    actor_id: int,
    env: Any,
    initial_state: np.ndarray,
    suite_name: str,
    task_id: int,
    task_description: str,
    cfg: DictConfig,
    episode_idx: int,
    action_horizon: int,
    input_w: int,
    input_h: int,
    processor: Any,
    request_queue: Any,
    response_queue: Any,
) -> tuple[bool, list[dict[str, np.ndarray]]]:
    """Run one actor episode and request action chunks from the model server."""
    max_steps = _get_max_steps(suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    per_request_noise = uses_per_request_noise(cfg)
    base_seed = None if cfg.get("seed") is None else int(cfg.seed)
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    save_rollout = bool(cfg.EVALUATION.get("save_rollout_video", False))

    env.reset()
    obs = env.set_init_state(initial_state)
    ensembler = ActionEnsembler() if use_action_ensembler else None
    if ensembler is not None:
        ensembler.reset()

    replay_images: list[dict[str, np.ndarray]] = []
    pending_actions: list[list[float]] = []
    stage_timing_accum: dict[str, float] = {}
    inference_calls = 0
    request_id = 0
    control_step = 0
    done = False

    while control_step < max_steps + num_steps_wait:
        if control_step < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            control_step += 1
            continue

        if not pending_actions:
            rgb, proprio = _prepare_actor_model_input(
                obs,
                cfg,
                processor,
                input_w=input_w,
                input_h=input_h,
            )
            request_queue.put(
                {
                    "kind": "infer",
                    "actor_id": actor_id,
                    "request_id": request_id,
                    "suite_name": suite_name,
                    "task_id": int(task_id),
                    "task_description": task_description,
                    "rgb": rgb,
                    "proprio": proprio,
                    # Derived actor-side from request identity so the draw is
                    # reproducible even though batch composition is not.
                    "noise_seed": (
                        derive_request_noise_seed(
                            base_seed=base_seed,
                            suite_name=suite_name,
                            task_id=int(task_id),
                            episode_idx=episode_idx,
                            request_id=request_id,
                        )
                        if per_request_noise and base_seed is not None
                        else None
                    ),
                    "submitted_at": time.monotonic(),
                }
            )
            try:
                response = response_queue.get(timeout=INFERENCE_TIMEOUT_SECONDS)
            except queue.Empty as exc:
                raise RuntimeError(
                    f"actor {actor_id} timed out waiting for inference request {request_id}"
                ) from exc
            assert response["request_id"] == request_id, (
                f"actor {actor_id} expected request {request_id}, "
                f"received {response['request_id']}"
            )
            if not response["ok"]:
                raise RuntimeError(str(response["error"]))

            action_chunk = np.asarray(response["action_chunk"], dtype=np.float32)
            call_timing = response.get("timing_ms")
            if call_timing is not None:
                inference_calls += 1
                for timing_name, timing_value in call_timing.items():
                    if isinstance(timing_value, (int, float)):
                        stage_timing_accum[timing_name] = stage_timing_accum.get(
                            timing_name, 0.0
                        ) + float(timing_value)

            if ensembler is not None:
                ensembler.add_actions(action_chunk, control_step)
                pending_actions = [
                    ensembler.get_action(step).tolist()
                    for step in range(control_step, control_step + replan_steps)
                ]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            request_id += 1

        if save_rollout:
            replay_images.append(get_libero_image(obs))

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
        control_step += 1

    if inference_calls > 0:
        _log_episode_stage_timing(stage_timing_accum, inference_calls, episode_idx)
    return bool(done), replay_images


def _actor_main(
    actor_id: int,
    tasks: list[tuple[str, int]],
    cfg_payload: dict[str, Any],
    action_horizon: int,
    input_w: int,
    input_h: int,
    request_queue: Any,
    response_queue: Any,
) -> None:
    """Run one actor with a dedicated fatal-signal traceback file."""
    cfg = OmegaConf.create(cfg_payload)
    crash_dump_dir = Path(str(cfg.EVALUATION.output_dir)) / "actor_crash_dumps"
    crash_dump_dir.mkdir(parents=True, exist_ok=True)
    crash_dump_path = crash_dump_dir / f"gpu{int(cfg.gpu_id)}_actor{actor_id}.log"
    with crash_dump_path.open("a", encoding="utf-8", buffering=1) as crash_dump:
        crash_dump.write(
            f"actor_start pid={os.getpid()} actor_id={actor_id} "
            f"gpu_id={int(cfg.gpu_id)} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        faulthandler.enable(file=crash_dump, all_threads=True)
        try:
            _actor_main_inner(
                actor_id,
                tasks,
                cfg_payload,
                action_horizon,
                input_w,
                input_h,
                request_queue,
                response_queue,
            )
        finally:
            faulthandler.disable()


def _actor_main_inner(
    actor_id: int,
    tasks: list[tuple[str, int]],
    cfg_payload: dict[str, Any],
    action_horizon: int,
    input_w: int,
    input_h: int,
    request_queue: Any,
    response_queue: Any,
) -> None:
    """Own LIBERO environments and request GPU actions from the model server."""
    cfg = OmegaConf.create(cfg_payload)
    dataset_stats = load_dataset_stats_from_json(str(_resolve_dataset_stats_path(cfg)))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    get_env = (
        build_selective_render_env_factory(cfg)
        if bool(cfg.EVALUATION.get("skip_render_within_replan", True))
        else get_libero_env
    )
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suites: dict[str, Any] = {}
    output_base = Path(str(cfg.EVALUATION.output_dir))
    num_trials = int(cfg.EVALUATION.num_trials)
    max_episode_retries = int(cfg.EVALUATION.get("actor_episode_retries", 10))
    if max_episode_retries < 0:
        raise ValueError(
            "EVALUATION.actor_episode_retries must be non-negative, "
            f"got {max_episode_retries}"
        )
    gpu_id = int(cfg.gpu_id)
    completed = 0
    successes_total = 0
    episodes_total = 0

    batch_cfg = cfg.get("BATCH", {})
    dynamic_queue_value = batch_cfg.get("dynamic_task_queue_dir", None)
    dynamic_queue_dir = (
        Path(str(dynamic_queue_value)) if dynamic_queue_value is not None else None
    )
    task_queue_max_attempts = int(batch_cfg.get("task_queue_max_attempts", 1))
    if task_queue_max_attempts <= 0:
        raise ValueError(
            "BATCH.task_queue_max_attempts must be positive, "
            f"got {task_queue_max_attempts}"
        )
    worker_id = str(batch_cfg.get("worker_id", ""))
    if dynamic_queue_dir is not None and not worker_id:
        raise ValueError("BATCH.worker_id is required with dynamic task scheduling")
    queue_owner = f"{worker_id}_actor{actor_id}_pid{os.getpid()}"
    static_tasks = iter(tasks)

    while True:
        lease: TaskLease | None = None
        if dynamic_queue_dir is None:
            try:
                suite_name, task_id = next(static_tasks)
            except StopIteration:
                break
        else:
            lease = claim_next_task(dynamic_queue_dir, queue_owner)
            if lease is None:
                break
            suite_name, task_id = lease.suite_name, lease.task_id

        task_start = time.time()
        task_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
        task_successes = 0
        success_episodes: list[int] = []
        failure_episodes: list[int] = []
        task_episode_retries = 0
        task_error: str | None = None
        queue_disposition: str | None = None
        env = None
        try:
            task_cfg = _task_config(cfg, suite_name, task_id)
            if suite_name not in task_suites:
                task_suites[suite_name] = benchmark_dict[suite_name]()
            task_suite = task_suites[suite_name]
            task = task_suite.get_task(task_id)
            initial_states = list(task_suite.get_task_init_states(task_id))
            if not initial_states:
                raise RuntimeError("task has no initial states")
            while len(initial_states) < num_trials:
                initial_states.extend(
                    initial_states[: num_trials - len(initial_states)]
                )

            environment_seed = _resolve_environment_seed(task_cfg)
            env, task_description = get_env(
                task,
                LIBERO_ENV_RESOLUTION,
                environment_seed,
            )
            video_dir = output_base / suite_name / "videos"
            if bool(task_cfg.EVALUATION.get("save_rollout_video", False)):
                video_dir.mkdir(parents=True, exist_ok=True)

            for trial_idx in range(num_trials):
                retry_count = 0
                while True:
                    try:
                        success, replay_images = _run_actor_episode(
                            actor_id=actor_id,
                            env=env,
                            initial_state=initial_states[trial_idx],
                            suite_name=suite_name,
                            task_id=task_id,
                            task_description=task_description,
                            cfg=task_cfg,
                            episode_idx=trial_idx,
                            action_horizon=action_horizon,
                            input_w=input_w,
                            input_h=input_h,
                            processor=processor,
                            request_queue=request_queue,
                            response_queue=response_queue,
                        )
                        break
                    except EPISODE_RETRY_EXCEPTIONS:
                        if retry_count >= max_episode_retries:
                            raise
                        retry_count += 1
                        task_episode_retries += 1
                        logging.warning(
                            "Actor %d restarting %s task %d trial %d after "
                            "recoverable failure (retry %d/%d)\n%s",
                            actor_id,
                            suite_name,
                            task_id,
                            trial_idx,
                            retry_count,
                            max_episode_retries,
                            traceback.format_exc(),
                        )
                        if env is not None:
                            try:
                                env.close()
                            except Exception:
                                logging.warning(
                                    "Actor %d could not close the failed environment "
                                    "for %s task %d trial %d\n%s",
                                    actor_id,
                                    suite_name,
                                    task_id,
                                    trial_idx,
                                    traceback.format_exc(),
                                )
                        env, recovered_task_description = get_env(
                            task,
                            LIBERO_ENV_RESOLUTION,
                            environment_seed,
                        )
                        if recovered_task_description != task_description:
                            raise RuntimeError(
                                "Recovered environment changed task identity: "
                                f"expected {task_description!r}, got "
                                f"{recovered_task_description!r}"
                            )
                if success:
                    task_successes += 1
                    success_episodes.append(trial_idx)
                else:
                    failure_episodes.append(trial_idx)
                if bool(task_cfg.EVALUATION.get("save_rollout_video", False)):
                    save_rollout_video(
                        video_dir,
                        replay_images,
                        f"task{task_id}_trial{trial_idx}",
                        success=success,
                        task_description=task_description,
                    )
                request_queue.put(
                    {
                        "kind": "episode_done",
                        "actor_id": actor_id,
                        "suite_name": suite_name,
                        "task_id": int(task_id),
                        "trial_idx": trial_idx,
                        "success": bool(success),
                    }
                )

            result_dir = output_base / suite_name
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / f"gpu{gpu_id}_task{task_id}_results.json"
            result = {
                "task_suite": suite_name,
                "task_id": int(task_id),
                "task_description": task_description,
                "successes": task_successes,
                "total_episodes": num_trials,
                "gpu_id": gpu_id,
                "success_episodes": success_episodes,
                "failure_episodes": failure_episodes,
                "start_time": task_start_str,
                "duration": time.time() - task_start,
                "shared_model_actor": actor_id,
                "episode_retries": task_episode_retries,
                "environment_seed": environment_seed,
                "action_noise_seed": task_cfg.get("seed"),
                "task_prompt_cache_dir": task_cfg.EVALUATION.get("task_prompt_cache_dir", None),
            }
            result_file.write_text(json.dumps(result, indent=4), encoding="utf-8")
            completed += 1
            successes_total += task_successes
            episodes_total += num_trials
        except ACTOR_EXCEPTIONS as exc:
            task_error = f"{type(exc).__name__}: {exc}"
            logging.error(
                "Actor %d failed %s task %d\n%s",
                actor_id,
                suite_name,
                task_id,
                traceback.format_exc(),
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    logging.warning(
                        "Actor %d could not close the environment for %s task %d\n%s",
                        actor_id,
                        suite_name,
                        task_id,
                        traceback.format_exc(),
                    )
            if lease is not None:
                if task_error is None:
                    complete_task(lease)
                    queue_disposition = "completed"
                else:
                    queue_disposition = retry_or_fail_task(
                        lease, task_error, task_queue_max_attempts
                    )
            request_queue.put(
                {
                    "kind": "task_done",
                    "actor_id": actor_id,
                    "suite_name": suite_name,
                    "task_id": int(task_id),
                    "successes": task_successes,
                    "episodes": num_trials if task_error is None else 0,
                    "error": task_error,
                    "queue_disposition": queue_disposition,
                }
            )

    request_queue.put(
        {
            "kind": "actor_done",
            "actor_id": actor_id,
            "completed": completed,
            "successes": successes_total,
            "episodes": episodes_total,
        }
    )


def _load_prompt_context(
    *,
    model: torch.nn.Module,
    cfg: DictConfig,
    suite_name: str,
    task_id: int,
    task_description: str,
    model_device: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    cache_dir = cfg.EVALUATION.get("task_prompt_cache_dir", None)
    cache_text_context = bool(cfg.EVALUATION.get("cache_text_context", False))
    prompt = DEFAULT_PROMPT.format(task=task_description)
    if cache_dir is not None:
        cache_kwargs = {
            "context_len": int(cfg.model.tokenizer_max_len),
            "context_dim": int(cfg.model.video_dit_config.text_dim),
            "model_id": str(cfg.model.model_id),
            "device": model_device,
            "dtype": model.torch_dtype,
        }
        if bool(cfg.EVALUATION.get("task_prompt_cache_by_task", False)):
            return load_task_prompt_context_for_task(
                prompt,
                cache_dir,
                suite_name=suite_name,
                task_id=task_id,
                tokenizer_model_id=str(cfg.model.tokenizer_model_id),
                **cache_kwargs,
            )
        return load_task_prompt_context(prompt, cache_dir, **cache_kwargs)
    if cache_text_context:
        with torch.no_grad():
            return model.encode_prompt(prompt)
    return None


def _poll_actor_exitcodes(
    actors: list[_ActorRuntime],
    actor_done_ids: set[int],
    ready_fds: set[int],
) -> None:
    """Detect actors that exit before reporting completion on every service iteration."""
    unexpected_exits: list[str] = []
    for actor in actors:
        return_code = actor.process.poll()
        if return_code is None or actor.actor_id in actor_done_ids:
            continue
        # A ready channel may still contain the final actor_done sent just before exit.
        # ``ready_fds`` is only the snapshot ``wait()`` took before this poll, so an actor
        # that sent actor_done and exited inside that window is missing from it; the
        # non-blocking re-check below keeps such an actor out of the failure set. Any
        # pending frame or EOF is then classified by the service loop on the next
        # iteration, which either consumes actor_done or raises on EOF-before-done.
        if actor.connection.fileno() in ready_fds or actor.connection.poll(0.0):
            continue
        # An exited actor whose channel is silent can never be classified by the loop,
        # so it stays a hard failure and keeps liveness detection bounded.
        unexpected_exits.append(
            f"actor={actor.actor_id} render_gpu={actor.render_gpu_id} rc={return_code}"
        )
    if unexpected_exits:
        raise RuntimeError(
            "LIBERO actor process exited before actor_done: "
            + ", ".join(unexpected_exits)
        )


def _require_actor_terminal_eof(actor: _ActorRuntime) -> None:
    """Require EOF immediately after an actor's terminal protocol frame."""
    # Waiting for EOF closes the race where another buffered frame arrives just
    # after actor_done, while the timeout prevents a faulty actor from hanging service.
    if not actor.connection.poll(ACTOR_TERMINAL_EOF_TIMEOUT_SECONDS):
        raise RuntimeError(
            "LIBERO actor did not close its connection after actor_done: "
            f"actor={actor.actor_id} render_gpu={actor.render_gpu_id}."
        )
    try:
        trailing_message = actor.connection.recv()
    except EOFError:
        return

    if (
        isinstance(trailing_message, dict)
        and trailing_message.get("kind") == "actor_done"
        and trailing_message.get("actor_id") == actor.actor_id
    ):
        raise ValueError(f"duplicate actor_done from actor {actor.actor_id}")
    trailing_kind = (
        trailing_message.get("kind", "<missing>")
        if isinstance(trailing_message, dict)
        else type(trailing_message).__name__
    )
    raise ValueError(
        f"actor {actor.actor_id} sent {trailing_kind!r} frame after actor_done"
    )


def _shutdown_actor_subprocesses(actors: list[_ActorRuntime]) -> None:
    """Reap actors, then terminate or kill any process that remains alive."""
    deadline = time.monotonic() + ACTOR_SHUTDOWN_TIMEOUT_SECONDS
    for actor in actors:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            actor.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            break

    # Cleanup is bounded across the whole actor set rather than per process.
    alive_actors = [actor for actor in actors if actor.process.poll() is None]
    for actor in alive_actors:
        actor.process.terminate()
    terminate_deadline = time.monotonic() + ACTOR_SHUTDOWN_TIMEOUT_SECONDS
    for actor in alive_actors:
        remaining = max(0.0, terminate_deadline - time.monotonic())
        try:
            actor.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            actor.process.kill()
            actor.process.wait(timeout=ACTOR_SHUTDOWN_TIMEOUT_SECONDS)
    for actor in actors:
        actor.connection.close()


@dataclass
class _PendingRequest:
    """One collected `infer` message awaiting dispatch in a batch group."""

    actor: "_ActorRuntime"
    request_id: int
    task_key: tuple[str, int]
    task_description: str
    rgb: np.ndarray
    proprio: np.ndarray
    noise_seed: int | None = None


@dataclass
class _ServedBatch:
    """Outcome of serving one request group."""

    service_seconds: float
    timing_ms: dict[str, Any] | None


def _select_batch_group(
    pending: list[_PendingRequest],
    task_cfg_cache: dict[tuple[str, int], DictConfig],
    minimum: int,
    cap: int,
) -> list[_PendingRequest] | None:
    """Largest dispatchable group of mutually compatible pending requests.

    Requests from different tasks batch freely -- the prompt cache pads every
    context to ``tokenizer_max_len`` -- but requests configured with different
    inference parameters must not share a forward pass, so they are bucketed by
    `batch_group_key()`. Returns ``None`` when no bucket has reached ``minimum``.
    """
    buckets: dict[tuple, list[_PendingRequest]] = {}
    for request in pending:
        key = batch_group_key(task_cfg_cache[request.task_key])
        buckets.setdefault(key, []).append(request)
    if not buckets:
        return None
    largest = max(buckets.values(), key=len)
    if len(largest) < minimum:
        return None
    return largest[:cap]


def _serve_request_group(
    *,
    group: list[_PendingRequest],
    model: torch.nn.Module,
    processor: Any,
    task_cfg_cache: dict[tuple[str, int], DictConfig],
    prompt_context_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor] | None],
    action_horizon: int,
    model_device: str,
) -> _ServedBatch:
    """Run one group through the model and reply to each member's actor."""
    started = time.monotonic()
    group_cfg = task_cfg_cache[group[0].task_key]
    contexts = [prompt_context_cache[request.task_key] for request in group]
    # A batched forward needs precomputed context; without it the text encoder
    # would have to run per request, which the eval server does not load. Tiled
    # VAE encoding remains on infer_action() because infer_action_batch()
    # currently rejects it.
    batchable = (
        len(group) > 1
        and callable(getattr(model, "infer_action_batch", None))
        and bool(getattr(model, "supports_batched_action_inference", False))
        and all(entry is not None for entry in contexts)
        and not bool(group_cfg.EVALUATION.get("tiled", False))
    )

    timing_ms: dict[str, Any] | None = None
    try:
        if batchable:
            actions, timing_ms = _predict_prepared_action_batch(
                rgb_batch=[request.rgb for request in group],
                proprio_batch=[request.proprio for request in group],
                prompt_contexts=contexts,  # type: ignore[arg-type]
                model=model,
                processor=processor,
                cfg=group_cfg,
                action_horizon=action_horizon,
                model_device=model_device,
                request_seeds=(
                    [request.noise_seed for request in group]
                    if all(request.noise_seed is not None for request in group)
                    else None
                ),
            )
        else:
            actions = []
            for request in group:
                action, timing_ms = _predict_prepared_action_chunk(
                    rgb=request.rgb,
                    proprio=request.proprio,
                    task_description=request.task_description,
                    model=model,
                    processor=processor,
                    cfg=task_cfg_cache[request.task_key],
                    action_horizon=action_horizon,
                    model_device=model_device,
                    prompt_context=prompt_context_cache[request.task_key],
                    noise_seed=request.noise_seed,
                )
                actions.append(action)
        responses: list[dict[str, Any]] = [
            {
                "request_id": request.request_id,
                "ok": True,
                "action_chunk": action,
                "timing_ms": timing_ms,
            }
            for request, action in zip(group, actions, strict=True)
        ]
    except ACTOR_EXCEPTIONS as exc:
        # One bad request fails its whole group; each actor still gets a reply so
        # none of them block until their inference timeout.
        error = f"{type(exc).__name__}: {exc}"
        responses = [
            {"request_id": request.request_id, "ok": False, "error": error}
            for request in group
        ]

    service_seconds = time.monotonic() - started
    for request, response in zip(group, responses, strict=True):
        request.actor.connection.send(response)
    return _ServedBatch(service_seconds=service_seconds, timing_ms=timing_ms)


def run_shared_model_batch(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    processor: Any,
    tasks: list[tuple[str, int]],
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    actor_render_gpu_ids: list[int],
) -> dict[str, Any]:
    """Serve batched model calls while isolated actors advance environments."""
    assert tasks, "shared-model evaluation requires at least one task"
    assert not bool(
        cfg.EVALUATION.get("visualize_future_video", False)
    ), "shared-model evaluation does not support visualize_future_video"
    batch_cfg = cfg.get("BATCH", {})
    dynamic_task_queue = bool(batch_cfg.get("dynamic_task_queue", False))
    if dynamic_task_queue and batch_cfg.get("dynamic_task_queue_dir", None) is None:
        raise ValueError(
            "BATCH.dynamic_task_queue_dir is required when "
            "BATCH.dynamic_task_queue=true"
        )
    actor_count = min(len(actor_render_gpu_ids), len(tasks))
    assert actor_count > 0, "shared-model evaluation requires at least one actor"
    effective_actor_gpu_ids = list(actor_render_gpu_ids[:actor_count])
    partitions = (
        [[] for _ in range(actor_count)]
        if dynamic_task_queue
        else partition_tasks(tasks, actor_count)
    )
    cfg_payload = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    assert isinstance(cfg_payload, dict)

    # Each subprocess receives its EGL device in execve(), before Python imports
    # MuJoCo, robosuite, LIBERO utilities, or this module in the child interpreter.
    actors: list[_ActorRuntime] = []
    try:
        for actor_id in range(actor_count):
            actors.append(
                _launch_actor_subprocess(actor_id, effective_actor_gpu_ids[actor_id])
            )
        for actor in actors:
            actor.connection.send(
                {
                    "actor_id": actor.actor_id,
                    "render_gpu_id": actor.render_gpu_id,
                    "tasks": partitions[actor.actor_id],
                    "cfg_payload": cfg_payload,
                    "action_horizon": action_horizon,
                    "input_w": input_w,
                    "input_h": input_h,
                }
            )
    except (OSError, BrokenPipeError):
        _shutdown_actor_subprocesses(actors)
        raise

    task_cfg_cache: dict[tuple[str, int], DictConfig] = {}
    prompt_context_cache: dict[
        tuple[str, int], tuple[torch.Tensor, torch.Tensor] | None
    ] = {}
    inference_requests = 0
    inference_service_seconds = 0.0
    queue_wait_seconds = 0.0
    # Per-stage model timings, populated only when EVALUATION.disable_inference_timing_sync
    # is false; the model omits them otherwise because the timing barriers are suppressed.
    stage_timing_accum: dict[str, float] = {}
    stage_timing_calls = 0
    policy_batch_size = max(1, int(cfg.EVALUATION.get("policy_batch_size", 1)))
    pending_requests: list[_PendingRequest] = []
    pending_since: float | None = None
    batch_size_histogram: dict[int, int] = {}
    completed_tasks = 0
    successes_total = 0
    episodes_total = 0
    actor_done_ids: set[int] = set()
    actor_errors: list[str] = []
    service_start = time.monotonic()
    progress = tqdm(
        total=(
            None
            if dynamic_task_queue
            else len(tasks) * int(cfg.EVALUATION.num_trials)
        ),
        desc="Shared LIBERO trials",
        unit="trial",
        disable=not bool(cfg.EVALUATION.get("progress_bar", True)),
        dynamic_ncols=True,
    )
    actors_by_fd = {actor.connection.fileno(): actor for actor in actors}

    try:
        while len(actor_done_ids) < actor_count:
            active_connections = [
                actor.connection
                for actor in actors
                if actor.actor_id not in actor_done_ids
            ]
            # While a batch is filling, never block past its collection deadline.
            poll_timeout = QUEUE_POLL_SECONDS
            if pending_since is not None:
                poll_timeout = max(
                    0.0,
                    min(
                        poll_timeout,
                        pending_since + BATCH_COLLECT_TIMEOUT_SECONDS - time.monotonic(),
                    ),
                )
            ready_connections = wait(active_connections, timeout=poll_timeout)
            ready_fds = {connection.fileno() for connection in ready_connections}
            _poll_actor_exitcodes(actors, actor_done_ids, ready_fds)

            # Process every ready actor so one chatty actor cannot starve liveness checks.
            for connection in ready_connections:
                actor = actors_by_fd[connection.fileno()]
                try:
                    message = connection.recv()
                except EOFError as exc:
                    return_code = actor.process.poll()
                    raise RuntimeError(
                        "LIBERO actor connection closed before actor_done: "
                        f"actor={actor.actor_id} render_gpu={actor.render_gpu_id} "
                        f"rc={return_code}."
                    ) from exc
                if not isinstance(message, dict):
                    raise ValueError(
                        f"actor {actor.actor_id} sent non-dict message "
                        f"{type(message).__name__}"
                    )

                kind = str(message["kind"])
                message_actor_id = int(message["actor_id"])
                if message_actor_id != actor.actor_id:
                    raise ValueError(
                        f"actor channel {actor.actor_id} carried actor_id={message_actor_id}"
                    )
                if kind == "infer":
                    suite_name = str(message["suite_name"])
                    task_id = int(message["task_id"])
                    task_key = (suite_name, task_id)
                    queue_wait_seconds += max(
                        0.0, time.monotonic() - float(message["submitted_at"])
                    )
                    if task_key not in task_cfg_cache:
                        task_cfg_cache[task_key] = _task_config(cfg, suite_name, task_id)
                        prompt_context_cache[task_key] = _load_prompt_context(
                            model=model,
                            cfg=task_cfg_cache[task_key],
                            suite_name=suite_name,
                            task_id=task_id,
                            task_description=str(message["task_description"]),
                            model_device=model_device,
                        )
                    pending_requests.append(
                        _PendingRequest(
                            actor=actor,
                            request_id=int(message["request_id"]),
                            task_key=task_key,
                            task_description=str(message["task_description"]),
                            rgb=message["rgb"],
                            proprio=message["proprio"],
                            noise_seed=message.get("noise_seed"),
                        )
                    )
                    if pending_since is None:
                        pending_since = time.monotonic()
                elif kind == "task_done":
                    task_key = (str(message["suite_name"]), int(message["task_id"]))
                    queue_disposition = message.get("queue_disposition")
                    # A requeued task may already have been claimed by another actor.
                    # Retain its cache until a terminal completion/failure message so
                    # that cross-connection message ordering cannot invalidate an
                    # in-flight retry on this model server.
                    if queue_disposition != "requeued":
                        task_cfg_cache.pop(task_key, None)
                        prompt_context_cache.pop(task_key, None)
                    if message.get("error") is None:
                        completed_tasks += 1
                        successes_total += int(message["successes"])
                        episodes_total += int(message["episodes"])
                    elif queue_disposition == "requeued":
                        logging.warning(
                            "Actor %d requeued %s task %d after error: %s",
                            int(message["actor_id"]),
                            task_key[0],
                            task_key[1],
                            message["error"],
                        )
                    else:
                        actor_errors.append(
                            f"{task_key[0]} task {task_key[1]}: {message['error']}"
                        )
                elif kind == "episode_done":
                    progress.update(1)
                    progress.set_postfix(
                        actor=int(message["actor_id"]),
                        task=f"{message['suite_name']}:{int(message['task_id'])}",
                        success=int(bool(message["success"])),
                        refresh=False,
                    )
                elif kind == "actor_done":
                    actor_id = int(message["actor_id"])
                    if actor_id in actor_done_ids:
                        raise ValueError(f"duplicate actor_done from actor {actor_id}")
                    _require_actor_terminal_eof(actor)
                    actor_done_ids.add(actor_id)
                else:
                    raise ValueError(f"unknown shared-model message kind: {kind}")

            # An actor is strictly synchronous: it blocks on its response after
            # sending, so it can never have more than one request outstanding.
            # Once we hold one request per live actor, no further request can
            # arrive and waiting longer would deadlock -- hence the live-actor
            # cap. The deadline only guards against a slow actor stalling a
            # batch that is otherwise ready.
            while pending_requests:
                live_actors = max(1, actor_count - len(actor_done_ids))
                # Holding one request per live actor means no further request can
                # arrive; waiting past that point would deadlock.
                saturated = len(pending_requests) >= live_actors
                collection_expired = (
                    pending_since is not None
                    and time.monotonic() - pending_since >= BATCH_COLLECT_TIMEOUT_SECONDS
                )
                minimum = 1 if (saturated or collection_expired) else policy_batch_size
                group = _select_batch_group(
                    pending_requests, task_cfg_cache, minimum, policy_batch_size
                )
                if group is None:
                    break

                served = _serve_request_group(
                    group=group,
                    model=model,
                    processor=processor,
                    task_cfg_cache=task_cfg_cache,
                    prompt_context_cache=prompt_context_cache,
                    action_horizon=action_horizon,
                    model_device=model_device,
                )
                inference_service_seconds += served.service_seconds
                inference_requests += len(group)
                batch_size_histogram[len(group)] = (
                    batch_size_histogram.get(len(group), 0) + 1
                )
                if served.timing_ms is not None:
                    stage_timing_calls += 1
                    for stage_name, stage_value in served.timing_ms.items():
                        if isinstance(stage_value, (int, float)):
                            stage_timing_accum[stage_name] = (
                                stage_timing_accum.get(stage_name, 0.0)
                                + float(stage_value)
                            )
                for request in group:
                    pending_requests.remove(request)
                pending_since = (
                    time.monotonic() if pending_requests else None
                )
    finally:
        progress.close()
        _shutdown_actor_subprocesses(actors)

    failed_exit_codes = {
        f"actor-{actor.actor_id}": actor.process.returncode
        for actor in actors
        if actor.process.returncode != 0
    }
    if failed_exit_codes:
        raise RuntimeError(f"LIBERO actor exit failures: {failed_exit_codes}")

    service_wall_seconds = time.monotonic() - service_start
    batch_cfg = cfg.get("BATCH", {})
    requested_actor_counts = list(
        batch_cfg.get("requested_sim_actors_per_gpu", [])
    )
    selected_gpu_ids = list(batch_cfg.get("selected_gpu_ids", []))
    server_gpu = batch_cfg.get("server_gpu", None)
    actor_device_histogram = {
        str(gpu_id): count
        for gpu_id, count in sorted(Counter(effective_actor_gpu_ids).items())
    }
    stats = {
        "actor_processes": actor_count,
        "requested_sim_actors_per_gpu": requested_actor_counts,
        "requested_actor_count": sum(requested_actor_counts),
        "selected_gpu_ids": selected_gpu_ids,
        "effective_actor_render_gpu_ids": effective_actor_gpu_ids,
        "actor_device_histogram": actor_device_histogram,
        "server_gpu": None if server_gpu is None else int(server_gpu),
        "dynamic_task_queue": dynamic_task_queue,
        "completed_tasks": completed_tasks,
        "successes": successes_total,
        "episodes": episodes_total,
        "inference_requests": inference_requests,
        "service_wall_seconds": service_wall_seconds,
        "inference_service_seconds": inference_service_seconds,
        "inference_server_busy_fraction": (
            inference_service_seconds / service_wall_seconds
            if service_wall_seconds > 0
            else 0.0
        ),
        "mean_queue_wait_ms": (
            queue_wait_seconds * 1000.0 / inference_requests
            if inference_requests > 0
            else 0.0
        ),
        "actor_errors": actor_errors,
        "policy_batch_size": policy_batch_size,
        "task_prompt_cache_dir": cfg.EVALUATION.get("task_prompt_cache_dir", None),
        "batch_size_histogram": {
            str(size): count for size, count in sorted(batch_size_histogram.items())
        },
        "mean_realized_batch_size": (
            sum(size * count for size, count in batch_size_histogram.items())
            / sum(batch_size_histogram.values())
            if batch_size_histogram
            else 0.0
        ),
        "batch_dispatches": sum(batch_size_histogram.values()),
        "actor_side_input_preprocessing": True,
        "cuda_timing_sync_disabled": bool(
            cfg.EVALUATION.get("disable_inference_timing_sync", False)
        ),
        "stage_timing_calls": stage_timing_calls,
        "mean_stage_timing_ms": (
            {
                stage_name: stage_total / stage_timing_calls
                for stage_name, stage_total in sorted(stage_timing_accum.items())
            }
            if stage_timing_calls > 0
            else {}
        ),
    }
    stats_path = Path(str(cfg.EVALUATION.output_dir)) / "shared_model_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if actor_errors:
        preview = "; ".join(actor_errors[:5])
        remaining = len(actor_errors) - 5
        if remaining > 0:
            preview = f"{preview}; ... and {remaining} more"
        raise RuntimeError(
            f"Shared-model actors reported {len(actor_errors)} task failures. "
            f"Diagnostics: {stats_path}. {preview}"
        )
    return stats
