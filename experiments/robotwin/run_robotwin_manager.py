import csv
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem



def _prepare_output_directory(cfg: DictConfig, ckpt_path: Path) -> Path:
    """Honor the requested directory and verify checkpoint identity on resume."""
    resuming = cfg.resume is not None
    output = _resolve_path(
        str(cfg.resume if resuming else cfg.EVALUATION.output_dir), base=PROJECT_ROOT
    )
    if resuming and not output.is_dir():
        raise FileNotFoundError(f"Resume path does not exist or is not a directory: {output}")
    identity_path = output / "checkpoint.json"
    if identity_path.exists():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        previous = _resolve_path(recorded["checkpoint"], base=PROJECT_ROOT)
        if previous != ckpt_path.resolve():
            raise ValueError(f"Checkpoint mismatch for evaluation directory: {output}")
    elif resuming and output.parent.name != _resolve_ckpt_tag(ckpt_path):
        # Older runs identified their checkpoint by the parent directory name.
        raise ValueError(f"Cannot verify resume checkpoint: missing {identity_path}")
    output.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(
        json.dumps({"checkpoint": str(ckpt_path.resolve())}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "resume",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks(robotwin_root: Path) -> list[str]:
    task_file = robotwin_root / "task_config" / "_eval_step_limit.yml"
    if not task_file.exists():
        raise FileNotFoundError(f"Task list file not found: {task_file}")
    with task_file.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {task_file}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _scan_completed_results(
    run_output_dir: Path,
    tasks: list[str],
    enabled_phases: list[str],
) -> tuple[dict[str, dict[str, float | None]], dict[str, int]]:
    """Scan an existing run directory for completed phase results.

    For each (task, phase) pair, checks if the corresponding result file
    exists and is parseable. Returns two dicts:
      - scanned_rates: {task: {phase: float_or_None}} with parsed success rates
      - start_phase_index: {task: int} where int is the index of the first
        incomplete phase. If all phases are complete, value = len(enabled_phases).

    Malformed or unparseable result files are logged as warnings and treated
    as incomplete (None rate, phase scanning stops for that task).
    """
    scanned_rates: dict[str, dict[str, float | None]] = {
        task: {p: None for p in enabled_phases} for task in tasks
    }
    start_phase_index: dict[str, int] = {task: 0 for task in tasks}

    for task in tasks:
        # Walk phases in order; stop at the first incomplete one.
        for idx, phase in enumerate(enabled_phases):
            result_file = run_output_dir / task / _phase_result_filename(phase)
            if not result_file.exists():
                # Phase not started; this is where the task should resume.
                break
            try:
                rate = _parse_success_rate(result_file)
            except (ValueError, FileNotFoundError) as exc:
                logger.warning(
                    "[resume] Malformed result file %s: %s. "
                    "Treating phase '%s' of task '%s' as incomplete.",
                    result_file,
                    exc,
                    phase,
                    task,
                )
                break
            scanned_rates[task][phase] = rate
            start_phase_index[task] = idx + 1

    return scanned_rates, start_phase_index


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env:
        gpu_ids = [int(g) for g in cuda_env.split(",")][:num_gpus]
    else:
        gpu_ids = list(range(num_gpus))

    enabled_phases = [p.strip() for p in str(cfg.MULTIRUN.phases).split(",")]
    for p in enabled_phases:
        if p not in ("clean", "random"):
            raise ValueError(f"Invalid phase '{p}' in MULTIRUN.phases. Expected 'clean' and/or 'random'.")

    run_output_dir = _prepare_output_directory(cfg, ckpt_path)
    output_dir = run_output_dir
    resume_path = run_output_dir if cfg.resume is not None else None

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks(robotwin_root)
    else:
        tasks = [str(task_name_cfg)]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {p: None for p in enabled_phases} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []
    progress_bar_enabled = bool(cfg.EVALUATION.get("progress_bar", True))
    progress: tqdm | None = None

    # Per-task starting phase index. Default 0 (start from first phase).
    # When resuming, completed phases are skipped by setting the start index
    # to the first incomplete phase for each task.
    start_phase_index: dict[str, int] = {task: 0 for task in tasks}

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        if progress is not None:
            progress.write(line)
        else:
            print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def refresh_progress_postfix() -> None:
        if progress is None:
            return
        progress.set_postfix(
            running=len(running_states),
            pending_tasks=len(pending_tasks),
            failed=len(failed_records),
            refresh=True,
        )

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            # Use the per-task start phase index (non-zero only when resuming
            # a task that was partially completed in a previous run).
            first_phase = enabled_phases[start_phase_index.get(task_name, 0)]
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase=first_phase))
            refresh_progress_postfix()

    def write_outputs() -> None:
        phase_means = {p: _mean_or_none([task_rates[t][p] for t in tasks]) for p in enabled_phases}

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name"] + [f"{p}_success_rate" for p in enabled_phases])
            for task in tasks:
                writer.writerow([task] + [task_rates[task][p] for p in enabled_phases])
            writer.writerow(["__overall__"] + [phase_means[p] for p in enabled_phases])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    **{f"{p}_success_rate": _to_jsonable(task_rates[task][p]) for p in enabled_phases},
                }
                for task in tasks
            ],
            "overall": {
                f"{p}_mean_success_rate": _to_jsonable(phase_means[p]) for p in enabled_phases
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    # --- Resume scan + skip logic (Subtasks 4-5) ---
    if resume_path is not None:
        log(f"[resume] Resuming from {resume_path}")
        scanned_rates, scanned_indices = _scan_completed_results(
            run_output_dir, tasks, enabled_phases,
        )
        # Pre-populate task_rates and start_phase_index from scan results.
        for task in tasks:
            start_phase_index[task] = scanned_indices[task]
            for phase in enabled_phases:
                rate = scanned_rates[task][phase]
                if rate is not None:
                    task_rates[task][phase] = rate

        # Filter pending_tasks: remove tasks where all phases are complete.
        fully_complete = {
            task for task in tasks
            if scanned_indices[task] >= len(enabled_phases)
        }
        pending_tasks = deque(t for t in tasks if t not in fully_complete)

        # Compute counts for logging.
        n_skip = len(fully_complete)
        n_partial = sum(
            1 for t in tasks
            if 0 < scanned_indices[t] < len(enabled_phases)
        )
        n_remaining = sum(
            1 for t in tasks if scanned_indices[t] == 0
        )
        log(
            f"[resume] {n_skip} tasks fully complete, "
            f"{n_partial} partially complete, {n_remaining} remaining"
        )

        for task in tasks:
            if task in fully_complete:
                log(f"[resume] SKIP task={task} (all phases complete)")
            elif scanned_indices[task] > 0:
                start_phase = enabled_phases[scanned_indices[task]]
                log(f"[resume] PARTIAL task={task}: starting from phase={start_phase}")

        if len(pending_tasks) == 0:
            log("[resume] All tasks already complete, regenerating summary")

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir}"
    )

    has_failure = False
    failure_message = ""

    try:
        if progress_bar_enabled:
            progress = tqdm(
                total=len(tasks) * len(enabled_phases),
                desc="RoboTwin eval",
                unit="phase",
                dynamic_ncols=True,
            )
            completed_phases = sum(
                1
                for task in tasks
                for phase in enabled_phases
                if task_rates[task][phase] is not None
            )
            if completed_phases:
                progress.update(completed_phases)
            refresh_progress_postfix()

        # Launch initial tasks for each GPU up to capacity.
        for gpu_id in gpu_ids:
            try_launch_pending(gpu_id)

        while len(running_states) > 0:
            progressed = False
            for state in list(running_states):
                gpu_id = state.gpu_id
                return_code = state.process.poll()
                if return_code is None:
                    continue
                progressed = True
                running_states.remove(state)
                refresh_progress_postfix()

                if return_code != 0:
                    has_failure = True
                    failure_message = (
                        f"worker failed: task={state.task_name}, phase={state.phase}, "
                        f"gpu={gpu_id}, return_code={return_code}"
                    )
                    failed_records.append(
                        {
                            "task_name": state.task_name,
                            "phase": state.phase,
                            "gpu_id": gpu_id,
                            "return_code": return_code,
                            "reason": "process_failed",
                        }
                    )
                    refresh_progress_postfix()
                    log(failure_message)
                    terminate_all_running()
                    running_states.clear()
                    break

                result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
                try:
                    success_rate = _parse_success_rate(result_file)
                except Exception as exc:
                    has_failure = True
                    failure_message = (
                        f"result parse failed: task={state.task_name}, phase={state.phase}, "
                        f"gpu={gpu_id}, error={repr(exc)}"
                    )
                    failed_records.append(
                        {
                            "task_name": state.task_name,
                            "phase": state.phase,
                            "gpu_id": gpu_id,
                            "return_code": return_code,
                            "reason": "result_parse_failed",
                        }
                    )
                    refresh_progress_postfix()
                    log(failure_message)
                    terminate_all_running()
                    running_states.clear()
                    break

                task_rates[state.task_name][state.phase] = success_rate
                if progress is not None:
                    progress.update(1)
                refresh_progress_postfix()
                log(
                    f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                    f"success_rate={success_rate:.4f}"
                )

                phase_idx = enabled_phases.index(state.phase)
                if phase_idx + 1 < len(enabled_phases):
                    running_states.append(launch_phase(
                        task_name=state.task_name,
                        gpu_id=gpu_id,
                        phase=enabled_phases[phase_idx + 1],
                    ))
                    refresh_progress_postfix()
                    continue

                try_launch_pending(gpu_id)

            if has_failure:
                break
            if not progressed:
                time.sleep(POLL_INTERVAL_SEC)

        # Mark not started tasks when failure happened.
        if has_failure:
            for task_name in pending_tasks:
                failed_records.append(
                    {
                        "task_name": task_name,
                        "phase": "not_started",
                        "gpu_id": -1,
                        "return_code": -1,
                        "reason": "aborted_not_started",
                    }
                )
            refresh_progress_postfix()

        write_outputs()
        log(f"summary saved: {summary_csv} and {summary_json}")
    finally:
        if progress is not None:
            progress.close()
            progress = None

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
