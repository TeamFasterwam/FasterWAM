"""LIBERO evaluation manager using persistent model workers.

Each worker loads the model once. It evaluates assigned tasks sequentially or
serves concurrent simulator actors through ``eval_libero_batch.py``.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from experiments.libero.task_work_queue import (
    initialize_task_queue,
    queue_state_counts,
    resume_task_queue,
)

BATCH_ENTRY = PROJECT_ROOT / "experiments" / "libero" / "eval_libero_batch.py"
TERMINATE_TIMEOUT_SEC = 30
SHUTDOWN_POLL_INTERVAL_SEC = 0.05
POLL_INTERVAL_SEC = 5
MAX_PHYSICAL_GPU_ID = 64
FORCE_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


@dataclass
class TaskSpec:
    """Identify one LIBERO benchmark task."""

    suite_name: str
    task_id: int


@dataclass
class GpuWorker:
    """Track one persistent model worker and its assigned simulator actors."""

    gpu_id: int
    tasks: list[TaskSpec]
    actor_render_gpu_ids: list[int]
    task_file: Path
    log_file: Path
    worker_id: str = ""
    process: subprocess.Popen[str] | None = None
    process_group_id: int | None = None
    log_fh: Any = None


def _is_blocked_override(raw_override: str) -> bool:
    key, _, value = raw_override.partition("=")
    key = key.lstrip("+~")
    blocked = {
        "data",
        "ckpt",
        "gpu_id",
        "EVALUATION",
        "MULTIRUN",
        "BATCH",
        "hydra",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.num_trials",
        "EVALUATION.output_dir",
    }
    if key in blocked:
        return True
    if key == "model" and value.lstrip().startswith("{"):
        return True
    return (
        key.startswith("MULTIRUN.")
        or key.startswith("hydra.")
        or key.startswith("BATCH.")
    )


def _collect_worker_overrides() -> list[str]:
    return [
        ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)
    ]


def _is_random_init_checkpoint(ckpt: object) -> bool:
    if ckpt is None:
        return False
    return str(ckpt).strip().lower() in {"random", "random_init", "random-init", "none", "null"}


def _resolve_worker_data_choice() -> str:
    data_choice = HydraConfig.get().runtime.choices.get("data")
    if data_choice is None or str(data_choice).strip() == "":
        raise ValueError(
            "Hydra data choice is empty. Pass data=... on the command line."
        )
    return str(data_choice)


def _worker_config_arguments() -> list[str]:
    """Preserve the manager's benchmark config and data recipe in workers."""
    config_name = HydraConfig.get().job.config_name
    if config_name is None or str(config_name).strip() == "":
        raise ValueError("Hydra config name is empty; cannot configure evaluation workers.")
    return [f"--config-name={config_name}", f"data={_resolve_worker_data_choice()}"]


def _worker_environment(gpu_id: int) -> dict[str, str]:
    """Bind CUDA and MuJoCo EGL to the worker's assigned physical GPU."""
    env = os.environ.copy()
    device_id = str(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = device_id
    env["MUJOCO_EGL_DEVICE_ID"] = device_id
    return env


def _canonical_gpu_id(value: object, key: str) -> int:
    """Validate one canonical physical GPU ID in the supported range."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{key} must be a canonical integer in [0, {MAX_PHYSICAL_GPU_ID}], "
            f"got {value!r}."
        )
    if not 0 <= value <= MAX_PHYSICAL_GPU_ID:
        raise ValueError(
            f"{key} must be in [0, {MAX_PHYSICAL_GPU_ID}], got {value!r}."
        )
    return value


def _canonical_positive_int(value: object, key: str) -> int:
    """Validate one non-boolean positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    return value


def _canonical_bool(value: object, key: str) -> bool:
    """Validate one strict boolean without accepting truthy strings."""
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false, got {value!r}.")
    return value


def _resolve_selected_gpu_ids(num_gpus: int, cuda_visible_devices: str) -> list[int]:
    """Resolve selected physical IDs, preserving non-contiguous user ordering."""
    if cuda_visible_devices:
        # Keep physical ordering exactly as supplied for positional actor counts.
        raw_gpu_ids = cuda_visible_devices.split(",")
        gpu_ids: list[int] = []
        for raw_gpu_id in raw_gpu_ids:
            if not raw_gpu_id.isdigit() or (
                len(raw_gpu_id) > 1 and raw_gpu_id.startswith("0")
            ):
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES must contain canonical comma-separated "
                    f"physical GPU IDs, got {cuda_visible_devices!r}."
                )
            gpu_ids.append(
                _canonical_gpu_id(int(raw_gpu_id), "CUDA_VISIBLE_DEVICES")
            )
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must not contain duplicate physical GPU IDs, "
                f"got {cuda_visible_devices!r}."
            )
        if num_gpus > len(gpu_ids):
            raise ValueError(
                f"MULTIRUN.num_gpus={num_gpus} exceeds the "
                f"{len(gpu_ids)} IDs in CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}."
            )
        return gpu_ids[:num_gpus]
    if num_gpus > MAX_PHYSICAL_GPU_ID + 1:
        raise ValueError(
            f"MULTIRUN.num_gpus must not exceed {MAX_PHYSICAL_GPU_ID + 1}, "
            f"got {num_gpus}."
        )
    return list(range(num_gpus))


def _resolve_actor_counts(raw_counts: object, gpu_ids: list[int]) -> list[int]:
    """Resolve per-selected-GPU actor counts while keeping direct defaults legacy-safe."""
    if raw_counts is None:
        return [1] * len(gpu_ids)
    if not OmegaConf.is_list(raw_counts):
        raise ValueError(
            "MULTIRUN.sim_actors_per_gpu must be null or a list[int] aligned "
            f"to selected physical GPUs {gpu_ids}, got {raw_counts!r}."
        )
    # Hydra must preserve list structure; scalar aliases are intentionally rejected.
    counts = list(raw_counts)
    if len(counts) != len(gpu_ids):
        raise ValueError(
            "MULTIRUN.sim_actors_per_gpu length must equal the selected GPU count: "
            f"got {counts!r} for selected physical GPUs {gpu_ids}."
        )
    validated_counts: list[int] = []
    for position, count in enumerate(counts):
        key = f"MULTIRUN.sim_actors_per_gpu[{position}]"
        validated_counts.append(_canonical_gpu_id(count, key))
    if sum(validated_counts) <= 0:
        raise ValueError(
            "MULTIRUN.sim_actors_per_gpu total must be positive, "
            f"got {validated_counts!r}."
        )
    return validated_counts


def _resolve_server_gpu(raw_server_gpu: object, gpu_ids: list[int]) -> int | None:
    """Validate the optional physical GPU hosting the single model server."""
    if raw_server_gpu is None:
        return None
    server_gpu = _canonical_gpu_id(raw_server_gpu, "MULTIRUN.server_gpu")
    if server_gpu not in gpu_ids:
        raise ValueError(
            f"MULTIRUN.server_gpu={server_gpu} must be one of the selected "
            f"physical GPU IDs {gpu_ids}."
        )
    return server_gpu


def _validate_opt_in_topology(actor_counts: list[int], server_gpu: int | None) -> None:
    """Require EGL rendering for asymmetric or single-server topology."""
    asymmetric = len(set(actor_counts)) > 1
    # Symmetric one-model-per-GPU topology works with any renderer.
    if server_gpu is None and not asymmetric:
        return
    mujoco_gl = os.environ.get("MUJOCO_GL", "")
    if mujoco_gl != "egl":
        raise ValueError(
            "Single-server or asymmetric MULTIRUN topology requires "
            f"MUJOCO_GL=egl; got MULTIRUN.sim_actors_per_gpu={actor_counts!r}, "
            f"MULTIRUN.server_gpu={server_gpu!r}, MUJOCO_GL={mujoco_gl!r}."
        )


def _apportion_actor_counts(actor_counts: list[int], task_count: int) -> list[int]:
    """Clip actor slots with deterministic largest-remainder apportionment."""
    if task_count < 0:
        raise ValueError(f"task_count must be non-negative, got {task_count}.")
    total_slots = sum(actor_counts)
    if task_count >= total_slots:
        return list(actor_counts)
    if task_count == 0:
        return [0] * len(actor_counts)

    # Integer numerators avoid floating-point ties in proportional apportionment.
    apportioned = [count * task_count // total_slots for count in actor_counts]
    remainders = [count * task_count % total_slots for count in actor_counts]
    remaining = task_count - sum(apportioned)
    ranked_positions = sorted(
        range(len(actor_counts)), key=lambda position: (-remainders[position], position)
    )
    for position in ranked_positions[:remaining]:
        apportioned[position] += 1
    return apportioned


def _expand_actor_gpu_ids(gpu_ids: list[int], actor_counts: list[int]) -> list[int]:
    """Expand positional actor counts into explicit physical render GPU IDs."""
    return [
        gpu_id
        for gpu_id, actor_count in zip(gpu_ids, actor_counts, strict=True)
        for _ in range(actor_count)
    ]


def _plan_worker_topology(
    tasks: list[TaskSpec],
    gpu_ids: list[int],
    max_workers_per_gpu: int,
    actor_counts: list[int],
    server_gpu: int | None,
) -> list[tuple[int, list[TaskSpec], list[int]]]:
    """Assign tasks and explicit actor render devices to model workers."""
    if server_gpu is not None:
        effective_counts = _apportion_actor_counts(actor_counts, len(tasks))
        return [
            (
                server_gpu,
                list(tasks),
                _expand_actor_gpu_ids(gpu_ids, effective_counts),
            )
        ]

    active_gpu_slots = [
        (gpu_id, actor_count)
        for gpu_id, actor_count in zip(gpu_ids, actor_counts, strict=True)
        if actor_count > 0
    ]
    # Zero-capacity GPUs contribute no worker slots; active slots retain the
    # configured per-GPU worker multiplier and receive a fresh round-robin split.
    partitions = _partition_tasks(
        tasks, len(active_gpu_slots) * max_workers_per_gpu
    )
    planned_workers: list[tuple[int, list[TaskSpec], list[int]]] = []
    worker_index = 0
    for gpu_id, actor_count in active_gpu_slots:
        for _ in range(max_workers_per_gpu):
            worker_tasks = partitions[worker_index]
            worker_index += 1
            if not worker_tasks:
                continue
            effective_count = _apportion_actor_counts(
                [actor_count], len(worker_tasks)
            )[0]
            planned_workers.append(
                (gpu_id, worker_tasks, [gpu_id] * effective_count)
            )
    return planned_workers


def _hydra_int_list(values: list[int]) -> str:
    """Serialize an integer list as a shell-independent Hydra override value."""
    return "[" + ",".join(str(value) for value in values) + "]"


def _build_task_list(task_suite_names: list[str]) -> list[TaskSpec]:
    benchmark_dict = benchmark.get_benchmark_dict()
    specs: list[TaskSpec] = []
    for suite_name in task_suite_names:
        suite = benchmark_dict[suite_name]()
        n_tasks = int(suite.n_tasks)
        print(f"\n{suite_name}: {n_tasks} tasks")
        for task_id in range(n_tasks):
            specs.append(TaskSpec(suite_name=suite_name, task_id=task_id))
    print(f"\nTotal tasks: {len(specs)}")
    return specs


def _load_task_list(path: Path, task_suite_names: list[str]) -> list[TaskSpec]:
    """Load and validate an explicit suite/task subset for this manager run."""
    resolved_path = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if not resolved_path.is_file():
        raise FileNotFoundError(f"MULTIRUN.task_file not found: {resolved_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    selected_suites = set(task_suite_names)
    suite_sizes: dict[str, int] = {}
    specs: list[TaskSpec] = []
    seen: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"{resolved_path}:{line_number}: expected suite_name,task_id"
            )
        suite_name, raw_task_id = parts
        if suite_name not in benchmark_dict:
            raise ValueError(
                f"{resolved_path}:{line_number}: unknown benchmark suite {suite_name!r}"
            )
        if suite_name not in selected_suites:
            raise ValueError(
                f"{resolved_path}:{line_number}: suite {suite_name!r} is not in "
                f"MULTIRUN.task_suite_names={task_suite_names!r}"
            )
        if not raw_task_id.isdigit() or (
            len(raw_task_id) > 1 and raw_task_id.startswith("0")
        ):
            raise ValueError(
                f"{resolved_path}:{line_number}: task_id must be a canonical "
                f"non-negative integer, got {raw_task_id!r}"
            )
        task_id = int(raw_task_id)
        if suite_name not in suite_sizes:
            suite_sizes[suite_name] = int(benchmark_dict[suite_name]().n_tasks)
        if task_id >= suite_sizes[suite_name]:
            raise ValueError(
                f"{resolved_path}:{line_number}: task_id {task_id} is out of range "
                f"for {suite_name} with {suite_sizes[suite_name]} tasks"
            )
        identity = (suite_name, task_id)
        if identity in seen:
            raise ValueError(
                f"{resolved_path}:{line_number}: duplicate task {suite_name},{task_id}"
            )
        seen.add(identity)
        specs.append(TaskSpec(suite_name=suite_name, task_id=task_id))

    if not specs:
        raise ValueError(f"MULTIRUN.task_file contains no tasks: {resolved_path}")
    print(f"\nLoaded {len(specs)} tasks from {resolved_path}")
    return specs


def _partition_tasks(tasks: list[TaskSpec], num_workers: int) -> list[list[TaskSpec]]:
    """Round-robin partition tasks across workers."""
    partitions: list[list[TaskSpec]] = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        partitions[i % num_workers].append(task)
    return partitions


def _write_task_file(path: Path, tasks: list[TaskSpec]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(f"{t.suite_name},{t.task_id}\n")


def _result_file_for_task(output_dir: Path, spec: TaskSpec, gpu_id: int) -> Path:
    return output_dir / spec.suite_name / f"gpu{gpu_id}_task{spec.task_id}_results.json"


def _update_progress_from_results(pbar: tqdm, pending_result_files: set[Path]) -> None:
    completed_files = [path for path in pending_result_files if path.exists()]
    for path in completed_files:
        pending_result_files.remove(path)
    if completed_files:
        pbar.update(len(completed_files))
    pbar.set_postfix_str(f"remaining={len(pending_result_files)}", refresh=False)


def _update_progress_from_task_queue(
    pbar: tqdm, queue_dir: Path, previous_terminal_count: int
) -> int:
    """Advance manager progress from globally visible queue transitions."""
    counts = queue_state_counts(queue_dir)
    terminal_count = counts["completed"] + counts["failed"]
    if terminal_count > previous_terminal_count:
        pbar.update(terminal_count - previous_terminal_count)
    pbar.set_postfix(
        pending=counts["pending"],
        running=counts["running"],
        failed=counts["failed"],
        refresh=False,
    )
    return terminal_count


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a POSIX process group still contains any process."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _worker_tree_running(worker: GpuWorker) -> bool:
    """Return whether a worker leader or any owned process-group member remains."""
    process = worker.process
    if process is None:
        return False
    process.poll()
    if os.name == "posix" and worker.process_group_id is not None:
        return _process_group_exists(worker.process_group_id)
    return process.returncode is None


def _signal_worker_tree(
    worker: GpuWorker, signal_number: int, *, force: bool = False
) -> None:
    """Signal one owned worker tree, falling back to the direct process."""
    process = worker.process
    if process is None:
        return
    if os.name == "posix" and worker.process_group_id is not None:
        try:
            os.killpg(worker.process_group_id, signal_number)
        except ProcessLookupError:
            return
        return
    if process.poll() is not None:
        return
    if force:
        process.kill()
    else:
        process.terminate()


def _wait_for_worker_trees(workers: list[GpuWorker], timeout: float) -> bool:
    """Wait within one shared deadline for every worker process tree to exit."""
    deadline = time.monotonic() + timeout
    # Poll leaders to reap them while also waiting for orphanable group members.
    while any(_worker_tree_running(worker) for worker in workers):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(SHUTDOWN_POLL_INTERVAL_SEC, remaining))
    return True


@contextmanager
def _manager_sigterm_handler() -> Iterator[None]:
    """Convert the manager's first SIGTERM into a cleanup-preserving exit."""
    previous_handler = signal.getsignal(signal.SIGTERM)
    termination_requested = False

    def _handle_sigterm(signal_number: int, frame: Any) -> None:
        del frame
        nonlocal termination_requested
        # Ignore repeat SIGTERM while the first one is unwinding cleanup.
        if termination_requested:
            return
        termination_requested = True
        raise SystemExit(128 + signal_number)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _shutdown_workers(
    workers: list["GpuWorker"], log: Callable[[str], None] = print
) -> None:
    """Terminate, reap, and close every launched worker process tree.

    POSIX workers own sessions whose process groups include their actor children.
    Cleanup signals all live groups together, waits within a shared TERM deadline,
    escalates remaining groups to KILL, and polls worker leaders to reap zombies.
    Direct-process fallback preserves behavior on non-POSIX platforms and for
    callers that did not launch an owned group.
    """
    active_workers = [worker for worker in workers if _worker_tree_running(worker)]
    for worker in active_workers:
        process = worker.process
        assert process is not None
        log(f"terminating worker tree on GPU {worker.gpu_id} (pid={process.pid})")
        _signal_worker_tree(worker, signal.SIGTERM)

    if active_workers and not _wait_for_worker_trees(
        active_workers, TERMINATE_TIMEOUT_SEC
    ):
        remaining_workers = [
            worker for worker in active_workers if _worker_tree_running(worker)
        ]
        for worker in remaining_workers:
            log(
                f"  worker tree on GPU {worker.gpu_id} did not stop in "
                f"{TERMINATE_TIMEOUT_SEC}s; sending SIGKILL"
            )
            _signal_worker_tree(worker, FORCE_KILL_SIGNAL, force=True)
        if not _wait_for_worker_trees(remaining_workers, TERMINATE_TIMEOUT_SEC):
            for worker in remaining_workers:
                if _worker_tree_running(worker):
                    log(
                        f"  worker tree on GPU {worker.gpu_id} still did not exit "
                        "after SIGKILL"
                    )

    for worker in workers:
        process = worker.process
        if process is not None:
            process.poll()
        file_handle = worker.log_fh
        if file_handle is not None and not file_handle.closed:
            file_handle.close()


def _installed_benchmark_flavor() -> str:
    """LIBERO-Plus ships task_classification.json next to its benchmark module."""
    marker = Path(benchmark.__file__).resolve().parent / "task_classification.json"
    return "libero_plus" if marker.is_file() else "libero_standard"


def _check_benchmark_flavor(cfg: DictConfig) -> None:
    """Both benchmarks install `libero`; refuse to evaluate in the wrong environment."""
    expected = str(cfg.EVALUATION.get("benchmark_flavor", "libero_standard"))
    installed = _installed_benchmark_flavor()
    if installed != expected:
        raise RuntimeError(
            f"This configuration evaluates {expected}, but the active environment "
            f"provides {installed} ({benchmark.__file__}). Activate the matching "
            "environment (see README_Libero.md)."
        )


def run_manager(cfg: DictConfig) -> None:
    """Launch persistent LIBERO model workers for the resolved topology."""
    _check_benchmark_flavor(cfg)
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if not _is_random_init_checkpoint(cfg.ckpt):
        ckpt_path = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not BATCH_ENTRY.exists():
        raise FileNotFoundError(f"Batch eval entry not found: {BATCH_ENTRY}")

    worker_config_arguments = _worker_config_arguments()
    manager_cfg = cfg.MULTIRUN

    if "concurrent_envs_per_gpu" in manager_cfg:
        raise ValueError(
            "MULTIRUN.concurrent_envs_per_gpu was removed; use "
            "MULTIRUN.sim_actors_per_gpu=[...] instead."
        )
    num_gpus = int(manager_cfg.num_gpus)
    if num_gpus <= 0:
        raise ValueError("MULTIRUN.num_gpus must be > 0.")
    max_workers_per_gpu = int(manager_cfg.max_tasks_per_gpu)
    if max_workers_per_gpu <= 0:
        raise ValueError("MULTIRUN.max_tasks_per_gpu must be > 0.")
    dynamic_task_queue = _canonical_bool(
        manager_cfg.get("dynamic_task_queue", False),
        "MULTIRUN.dynamic_task_queue",
    )
    resume_dynamic_task_queue = _canonical_bool(
        manager_cfg.get("resume_dynamic_task_queue", False),
        "MULTIRUN.resume_dynamic_task_queue",
    )
    if resume_dynamic_task_queue and not dynamic_task_queue:
        raise ValueError(
            "MULTIRUN.resume_dynamic_task_queue requires dynamic_task_queue=true"
        )
    task_queue_max_attempts = _canonical_positive_int(
        manager_cfg.get("task_queue_max_attempts", 1),
        "MULTIRUN.task_queue_max_attempts",
    )

    gpu_ids = _resolve_selected_gpu_ids(
        num_gpus, os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    actor_counts = _resolve_actor_counts(
        manager_cfg.get("sim_actors_per_gpu"), gpu_ids
    )
    server_gpu = _resolve_server_gpu(manager_cfg.get("server_gpu"), gpu_ids)
    _validate_opt_in_topology(actor_counts, server_gpu)
    if server_gpu is not None and max_workers_per_gpu != 1:
        raise ValueError(
            "MULTIRUN.server_gpu requires MULTIRUN.max_tasks_per_gpu=1, "
            f"got {max_workers_per_gpu}."
        )
    if max_workers_per_gpu > 1 and any(count > 1 for count in actor_counts):
        raise ValueError(
            "Concurrent simulator actors require MULTIRUN.max_tasks_per_gpu=1; "
            f"got MULTIRUN.sim_actors_per_gpu={actor_counts!r}."
        )

    output_dir = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task_log_dir = output_dir / "task_logs"
    task_log_dir.mkdir(parents=True, exist_ok=True)
    batch_files_dir = output_dir / "batch_files"
    batch_files_dir.mkdir(parents=True, exist_ok=True)

    manager_log = output_dir / "manager.log"
    failed_tasks_file = output_dir / "failed_tasks.txt"

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    task_suite_names = list(manager_cfg.task_suite_names)
    configured_task_file = manager_cfg.get("task_file")
    all_tasks = (
        _load_task_list(Path(str(configured_task_file)), task_suite_names)
        if configured_task_file is not None
        else _build_task_list(task_suite_names)
    )
    extra_overrides = _collect_worker_overrides()
    num_trials = int(cfg.EVALUATION.num_trials)
    if num_trials <= 0:
        raise ValueError("EVALUATION.num_trials must be > 0.")

    if bool(manager_cfg.get("create_only", False)):
        print("create_only=True, exiting after task list creation.")
        return

    planned_workers = _plan_worker_topology(
        all_tasks,
        gpu_ids,
        max_workers_per_gpu,
        actor_counts,
        server_gpu,
    )
    dynamic_queue_dir: Path | None = None
    if dynamic_task_queue:
        dynamic_queue_dir = output_dir / "dynamic_task_queue"
        queue_tasks = [(task.suite_name, task.task_id) for task in all_tasks]
        if resume_dynamic_task_queue:
            stale_count = resume_task_queue(dynamic_queue_dir, queue_tasks)
            print(
                f"Resuming dynamic task queue; requeued {stale_count} stale leases.",
                flush=True,
            )
        else:
            initialize_task_queue(dynamic_queue_dir, queue_tasks)

    def log(msg: str) -> None:
        """Write one timestamped manager message to stdout and the run log."""
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---- Materialize workers from the validated topology plan ----
    workers: list[GpuWorker] = []
    gpu_slot_counts: dict[int, int] = {}
    for gpu_id, tasks, actor_render_gpu_ids in planned_workers:
        slot = gpu_slot_counts.get(gpu_id, 0)
        gpu_slot_counts[gpu_id] = slot + 1
        worker_id = f"gpu{gpu_id}_w{slot}"
        task_file = batch_files_dir / f"gpu{gpu_id}_w{slot}_tasks.txt"
        _write_task_file(task_file, tasks)
        log_file = task_log_dir / f"gpu{gpu_id}_w{slot}_batch.log"
        workers.append(
            GpuWorker(
                worker_id=worker_id,
                gpu_id=gpu_id,
                tasks=tasks,
                actor_render_gpu_ids=actor_render_gpu_ids,
                task_file=task_file,
                log_file=log_file,
            )
        )

    log(
        f"manager start: {len(all_tasks)} tasks, {len(workers)} workers "
        f"({max_workers_per_gpu}/GPU), selected_gpu_ids={gpu_ids}, "
        f"num_trials={num_trials}, sim_actors_per_gpu={actor_counts}, "
        f"server_gpu={server_gpu}, dynamic_task_queue={dynamic_task_queue}, "
        f"task_queue_max_attempts={task_queue_max_attempts}"
    )
    for w in workers:
        log(
            f"  GPU {w.gpu_id}: {len(w.tasks)} tasks, "
            f"actor_render_gpu_ids={w.actor_render_gpu_ids}"
        )
    print(f"Checkpoint: {cfg.ckpt}")
    print(f"Output: {output_dir}")

    failure_messages: list[str] = []
    with _manager_sigterm_handler():
        try:
            for w in workers:
                cmd = [
                    sys.executable,
                    str(BATCH_ENTRY),
                    *worker_config_arguments,
                    f"ckpt={str(cfg.ckpt)}",
                    f"gpu_id={w.gpu_id}",
                    f"EVALUATION.num_trials={num_trials}",
                    f"EVALUATION.output_dir={str(output_dir)}",
                    f"+BATCH.task_file={str(w.task_file)}",
                    "+BATCH.actor_render_gpu_ids="
                    f"{_hydra_int_list(w.actor_render_gpu_ids)}",
                    "+BATCH.requested_sim_actors_per_gpu="
                    f"{_hydra_int_list(actor_counts)}",
                    f"+BATCH.selected_gpu_ids={_hydra_int_list(gpu_ids)}",
                    f"+BATCH.server_gpu={'null' if server_gpu is None else server_gpu}",
                ]
                if dynamic_queue_dir is not None:
                    cmd.extend(
                        [
                            f"+BATCH.dynamic_task_queue_dir={dynamic_queue_dir}",
                            f"+BATCH.worker_id={w.worker_id}",
                            "+BATCH.dynamic_task_queue=true",
                            f"+BATCH.task_queue_max_attempts={task_queue_max_attempts}",
                        ]
                    )
                cmd.extend(extra_overrides)

                log(f"launch GPU {w.gpu_id}: {' '.join(cmd)}")
                w.log_fh = w.log_file.open("w", encoding="utf-8")
                env = _worker_environment(w.gpu_id)
                w.process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    stdout=w.log_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=os.name == "posix",
                )
                if os.name == "posix":
                    # Actor children inherit this worker-owned process group.
                    w.process_group_id = w.process.pid

            # ---- Wait for all workers ----
            with tqdm(
                total=len(all_tasks),
                desc="LIBERO tasks",
                unit="task",
                dynamic_ncols=True,
            ) as pbar:
                if dynamic_queue_dir is not None:
                    terminal_count = _update_progress_from_task_queue(
                        pbar, dynamic_queue_dir, 0
                    )
                    while any(w.process.poll() is None for w in workers):
                        time.sleep(POLL_INTERVAL_SEC)
                        terminal_count = _update_progress_from_task_queue(
                            pbar, dynamic_queue_dir, terminal_count
                        )
                    _update_progress_from_task_queue(
                        pbar, dynamic_queue_dir, terminal_count
                    )
                else:
                    pending_result_files = {
                        _result_file_for_task(output_dir, spec, w.gpu_id)
                        for w in workers
                        for spec in w.tasks
                    }
                    _update_progress_from_results(pbar, pending_result_files)
                    while any(w.process.poll() is None for w in workers):
                        time.sleep(POLL_INTERVAL_SEC)
                        _update_progress_from_results(pbar, pending_result_files)
                    _update_progress_from_results(pbar, pending_result_files)
        finally:
            # SIGTERM, Ctrl-C, launch errors, and normal exits share tree cleanup.
            _shutdown_workers(workers, log)

    # ---- Collect results ----
    for w in workers:
        rc = w.process.returncode

        if rc != 0:
            msg = f"FAILED: GPU {w.gpu_id} rc={rc} log={w.log_file}"
            failure_messages.append(msg)
            log(msg)
            print(f"\n--- Tail of failed worker log: {w.log_file} ---")
            try:
                lines = w.log_file.read_text(encoding="utf-8").splitlines()
                for line in lines[-40:]:
                    print(line)
            except OSError as e:
                print(f"(could not read log file: {e})")
            print("--- End of log ---\n")
        else:
            log(f"GPU {w.gpu_id} exited successfully ({len(w.tasks)} assigned tasks)")

    # ---- Parse all result JSONs ----
    task_results: dict[str, dict[int, float | None]] = {}
    for spec in all_tasks:
        task_results.setdefault(spec.suite_name, {})[spec.task_id] = None

    failed_records: list[dict[str, Any]] = []
    for spec in all_tasks:
        found = False
        for w in workers:
            result_file = _result_file_for_task(output_dir, spec, w.gpu_id)
            if result_file.exists():
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, ValueError) as exc:
                    log(f"WARNING: corrupt result JSON {result_file}: {exc}")
                    continue
                successes = data.get("successes", 0)
                task_results[spec.suite_name][spec.task_id] = float(successes) / float(
                    num_trials
                )
                found = True
                break
        if not found:
            assigned_worker = (
                None
                if dynamic_queue_dir is not None
                else next(
                    (worker for worker in workers if spec in worker.tasks),
                    None,
                )
            )
            failed_records.append(
                {
                    "suite": spec.suite_name,
                    "task_id": spec.task_id,
                    "gpu_id": (
                        assigned_worker.gpu_id if assigned_worker is not None else -1
                    ),
                    "return_code": (
                        assigned_worker.process.returncode
                        if assigned_worker is not None
                        else -1
                    ),
                }
            )

    # ---- Write failed-tasks list ----
    suite_names = list(manager_cfg.task_suite_names)
    with failed_tasks_file.open("w", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(
                f"{rec['suite']},task_id={rec['task_id']},gpu={rec['gpu_id']},"
                f"rc={rec['return_code']}\n"
            )

    log(f"per-task result JSONs written under: {output_dir}")

    # ---- Print summary ----
    for suite in suite_names:
        if suite not in task_results:
            continue
        rates = [v for v in task_results[suite].values() if v is not None]
        if rates:
            mean = sum(rates) / len(rates) * 100
            total_s = sum(1 for v in rates if v and v > 0)
            print(
                f"\n{suite}: {mean:.2f}% success rate ({total_s}/{len(rates)} tasks with successes)"
            )

    if failed_records:
        failed_summary = ", ".join(
            f"{rec['suite']}:{rec['task_id']}@gpu{rec['gpu_id']}"
            for rec in failed_records
        )
        msg = (
            f"FAILED: {len(failed_records)} assigned task result(s) missing or "
            f"corrupt: {failed_summary}; details={failed_tasks_file}"
        )
        failure_messages.append(msg)
        log(msg)

    if dynamic_queue_dir is not None:
        queue_counts = queue_state_counts(dynamic_queue_dir)
        if (
            queue_counts["completed"] != len(all_tasks)
            or queue_counts["pending"] != 0
            or queue_counts["running"] != 0
            or queue_counts["failed"] != 0
        ):
            msg = (
                "FAILED: dynamic task queue did not complete cleanly: "
                f"expected={len(all_tasks)}, states={queue_counts}"
            )
            failure_messages.append(msg)
            log(msg)

    if failure_messages:
        raise RuntimeError("\n".join(failure_messages))

    log("manager finished successfully")


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml"
)
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint for the shared-model LIBERO manager."""
    run_manager(cfg)


if __name__ == "__main__":
    main()
