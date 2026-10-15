"""Batch LIBERO evaluation that loads the model once for multiple tasks.

The default path evaluates tasks sequentially in one process. When topology
routing requires process isolation, CPU simulator actors submit serialized policy
requests to the same GPU model process.

Writes per-task result JSONs in the same format as eval_libero_single.py
(consumed by run_libero_manager_v2.py and summarize_by_disturbance.py).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, flag_override

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import experiments.libero.eval_libero_single as eval_libero_single
from experiments.libero.eval_libero_single import (
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_task,
)
from experiments.libero.selective_render_env import build_selective_render_env_factory
from experiments.libero.shared_model_batch import run_shared_model_batch
from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fasterwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark

_LEGACY_GET_LIBERO_ENV = eval_libero_single.get_libero_env

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver(
    "split", lambda s, idx: s.split("/")[int(idx)], replace=True
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_PHYSICAL_GPU_ID = 64


def _resolve_actor_render_gpu_ids(cfg: DictConfig) -> list[int]:
    """Validate explicit physical render GPU IDs received from the manager."""
    batch_cfg = cfg.get("BATCH", {})
    if "concurrent_envs" in batch_cfg:
        raise ValueError(
            "BATCH.concurrent_envs was removed; use BATCH.actor_render_gpu_ids=[...] "
            "instead."
        )
    # The explicit list is the only supported manager-to-worker actor transport.
    raw_gpu_ids = batch_cfg.get("actor_render_gpu_ids", [])
    if not OmegaConf.is_list(raw_gpu_ids):
        raise ValueError(
            "BATCH.actor_render_gpu_ids must be a list[int], "
            f"got {raw_gpu_ids!r}."
        )
    if "actor_render_gpu_ids" in batch_cfg and len(raw_gpu_ids) == 0:
        raise ValueError(
            "BATCH.actor_render_gpu_ids must be non-empty when explicitly configured."
        )
    gpu_ids: list[int] = []
    for position, raw_gpu_id in enumerate(raw_gpu_ids):
        if isinstance(raw_gpu_id, bool) or not isinstance(raw_gpu_id, int):
            raise ValueError(
                f"BATCH.actor_render_gpu_ids[{position}] must be a canonical integer "
                f"in [0, {MAX_PHYSICAL_GPU_ID}], got {raw_gpu_id!r}."
            )
        if not 0 <= raw_gpu_id <= MAX_PHYSICAL_GPU_ID:
            raise ValueError(
                f"BATCH.actor_render_gpu_ids[{position}] must be in "
                f"[0, {MAX_PHYSICAL_GPU_ID}], got {raw_gpu_id!r}."
            )
        gpu_ids.append(raw_gpu_id)
    return gpu_ids


def _configure_selective_rendering(cfg: DictConfig) -> bool:
    """Install or restore the LIBERO environment factory for this batch worker."""
    enabled = bool(cfg.EVALUATION.get("skip_render_within_replan", True))
    eval_libero_single.get_libero_env = (
        build_selective_render_env_factory(cfg) if enabled else _LEGACY_GET_LIBERO_ENV
    )
    return enabled


def _parse_task_file(path: Path) -> list[tuple[str, int]]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        suite_name, task_id_str = line.split(",", 1)
        tasks.append((suite_name.strip(), int(task_id_str.strip())))
    return tasks


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml"
)
def main(cfg: DictConfig) -> None:
    """Evaluate assigned LIBERO tasks with one persistent model instance."""
    overall_start = time.time()
    PartialState().config = cfg
    actor_render_gpu_ids = _resolve_actor_render_gpu_ids(cfg)

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)
    if _configure_selective_rendering(cfg):
        logging.info(
            "Selective LIBERO rendering enabled: wait_steps=%s replan_steps=%s visualize_future=%s",
            cfg.EVALUATION.get("num_steps_wait", 5),
            cfg.EVALUATION.get("replan_steps", 5),
            cfg.EVALUATION.get("visualize_future_video", False),
        )

    # ---- Determine task list ----
    task_file_path = cfg.get("BATCH", {}).get("task_file")
    if not task_file_path:
        task_list = [
            (str(cfg.EVALUATION.task_suite_name), int(cfg.EVALUATION.task_id))
        ]
    else:
        task_file = Path(
            os.path.expanduser(os.path.expandvars(str(task_file_path)))
        )
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        task_list = _parse_task_file(task_file)

    logging.info("Batch eval: %d tasks to evaluate", len(task_list))

    # ---- Load model ONCE ----
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    load_time = time.time() - overall_start
    logging.info("Model loaded in %.1fs. Starting %d tasks.", load_time, len(task_list))

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])

    benchmark_dict = benchmark.get_benchmark_dict()
    output_base = Path(cfg.EVALUATION.output_dir)
    num_trials = int(cfg.EVALUATION.num_trials)
    gpu_id = int(cfg.gpu_id)

    effective_actor_count = min(len(actor_render_gpu_ids), len(task_list))
    effective_actor_gpu_ids = actor_render_gpu_ids[:effective_actor_count]
    batch_cfg = cfg.get("BATCH", {})
    one_server_mode = batch_cfg.get("server_gpu", None) is not None
    dynamic_task_queue = bool(batch_cfg.get("dynamic_task_queue", False))
    use_actor_subprocess = (
        dynamic_task_queue
        or one_server_mode
        or effective_actor_count > 1
        or bool(effective_actor_gpu_ids and effective_actor_gpu_ids[0] != gpu_id)
    )
    if use_actor_subprocess:
        policy_batch_size = max(1, int(cfg.EVALUATION.get("policy_batch_size", 1)))
        logging.info(
            "Shared-model evaluation enabled with %d simulator actors on physical "
            "render GPUs %s and policy batch size %d.",
            effective_actor_count,
            effective_actor_gpu_ids,
            policy_batch_size,
        )
        shared_stats = run_shared_model_batch(
            cfg=cfg,
            model=model,
            processor=processor,
            tasks=task_list,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            actor_render_gpu_ids=actor_render_gpu_ids,
        )
        total_time = time.time() - overall_start
        print(
            f"\nShared-model batch complete: {shared_stats['successes']}/"
            f"{shared_stats['episodes']} across {shared_stats['completed_tasks']} tasks "
            f"in {total_time:.1f}s (model load: {load_time:.1f}s, "
            f"actors: {shared_stats['actor_processes']}, policy batch size: {shared_stats['policy_batch_size']}, "
            f"server busy: {shared_stats['inference_server_busy_fraction']:.1%})",
            flush=True,
        )
        return

    # ---- Evaluate tasks ----
    completed = 0
    total_successes = 0
    total_episodes = 0
    failed_tasks: list[tuple[str, int]] = []

    for idx, (suite_name, task_id) in enumerate(task_list):
        print(
            f"\n[{idx + 1}/{len(task_list)}] {suite_name} task_id={task_id}",
            flush=True,
        )
        task_start = time.time()
        task_start_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with flag_override(cfg, "struct", False):
            cfg.EVALUATION.task_suite_name = suite_name
            cfg.EVALUATION.task_id = task_id

        video_dir = output_base / suite_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_dir = output_base / suite_name / "predicted_videos"
        if bool(cfg.EVALUATION.get("visualize_future_video", False)):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)

        task_suite = benchmark_dict[suite_name]()
        task = task_suite.get_task(task_id)
        initial_states = list(task_suite.get_task_init_states(task_id))
        if len(initial_states) == 0:
            logging.error(
                "No initial states for %s task_id=%d, skipping", suite_name, task_id
            )
            failed_tasks.append((suite_name, task_id))
            continue
        while len(initial_states) < num_trials:
            initial_states.extend(
                initial_states[: (num_trials - len(initial_states))]
            )

        # Per-task crash isolation: a failure in one task (e.g. simulator
        # crash, OOM on a single rollout) must not kill the worker's whole
        # remaining partition. Log the failure with full traceback and move
        # on. Success counters only increment after the JSON write succeeds.
        try:
            task_results = run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )

            results = {
                "task_suite": suite_name,
                "task_id": task_id,
                "task_description": task_results.get("task_description"),
                "successes": task_results.get("successes", 0),
                "total_episodes": num_trials,
                "gpu_id": gpu_id,
                "success_episodes": task_results.get("success_episodes", []),
                "failure_episodes": task_results.get("failure_episodes", []),
                "start_time": task_start_str,
                "duration": time.time() - task_start,
            }
            if "future_video_psnr_mean" in task_results:
                results["future_video_psnr_mean"] = task_results[
                    "future_video_psnr_mean"
                ]

            result_dir = output_base / suite_name
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / f"gpu{gpu_id}_task{task_id}_results.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)

            completed += 1
            total_successes += results["successes"]
            total_episodes += num_trials
            print(
                f"  -> {results['successes']}/{num_trials} successes "
                f"({results['duration']:.1f}s) "
                f"[cumulative: {total_successes}/{total_episodes}]",
                flush=True,
            )
        except Exception:
            logging.exception(
                "Task %s task_id=%s crashed, skipping", suite_name, task_id
            )
            failed_tasks.append((suite_name, task_id))
            continue

    total_time = time.time() - overall_start
    print(
        f"\nBatch complete: {total_successes}/{total_episodes} "
        f"across {completed} tasks in {total_time:.1f}s "
        f"(model load: {load_time:.1f}s)",
        flush=True,
    )
    if failed_tasks:
        failed_summary = ", ".join(
            f"{suite_name}:{task_id}" for suite_name, task_id in failed_tasks
        )
        raise RuntimeError(
            f"{len(failed_tasks)} of {len(task_list)} assigned tasks failed: "
            f"{failed_summary}"
        )


if __name__ == "__main__":
    main()
