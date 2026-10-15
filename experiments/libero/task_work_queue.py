"""Filesystem-backed task claiming for same-host LIBERO evaluation workers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Iterable


QUEUE_VERSION = 1
_QUEUE_STATES = ("pending", "running", "completed", "failed")
_OWNER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class TaskLease:
    """One task atomically claimed by a simulator actor."""

    queue_dir: Path
    claim_path: Path
    task_index: int
    suite_name: str
    task_id: int
    attempt: int
    owner: str


def initialize_task_queue(
    queue_dir: Path, tasks: Iterable[tuple[str, int]]
) -> None:
    """Create a new queue containing each unique task exactly once."""
    task_list = [(str(suite_name), int(task_id)) for suite_name, task_id in tasks]
    if not task_list:
        raise ValueError("dynamic LIBERO task queue requires at least one task")
    if len(set(task_list)) != len(task_list):
        raise ValueError(f"dynamic LIBERO task queue contains duplicates: {task_list!r}")

    queue_dir.mkdir(parents=True, exist_ok=False)
    for state in _QUEUE_STATES:
        (queue_dir / state).mkdir()

    manifest = {
        "version": QUEUE_VERSION,
        "task_count": len(task_list),
        "tasks": [
            {
                "task_index": task_index,
                "suite_name": suite_name,
                "task_id": task_id,
            }
            for task_index, (suite_name, task_id) in enumerate(task_list)
        ],
    }
    _write_json(queue_dir / "manifest.json", manifest)
    for task_index, (suite_name, task_id) in enumerate(task_list):
        _write_json(
            queue_dir / "pending" / _task_filename(task_index),
            {
                "task_index": task_index,
                "suite_name": suite_name,
                "task_id": task_id,
                "attempts_completed": 0,
            },
        )


def resume_task_queue(
    queue_dir: Path, tasks: Iterable[tuple[str, int]]
) -> int:
    """Validate an existing queue and requeue leases left by dead workers."""
    task_list = [(str(suite_name), int(task_id)) for suite_name, task_id in tasks]
    manifest = load_manifest(queue_dir)
    manifest_tasks = [
        (str(record["suite_name"]), int(record["task_id"]))
        for record in manifest["tasks"]
    ]
    if manifest_tasks != task_list:
        raise ValueError("existing dynamic LIBERO task queue does not match task list")

    requeued = 0
    for claim_path in sorted((queue_dir / "running").glob("*.json")):
        payload = _read_task_payload(claim_path)
        destination = queue_dir / "pending" / _task_filename(payload["task_index"])
        if destination.exists():
            raise FileExistsError(
                f"cannot requeue stale task because pending record exists: {destination}"
            )
        claim_path.rename(destination)
        requeued += 1
    return requeued


def claim_next_task(queue_dir: Path, owner: str) -> TaskLease | None:
    """Atomically claim the next pending task, or return ``None`` when empty."""
    _validate_owner(owner)
    pending_dir = queue_dir / "pending"
    running_dir = queue_dir / "running"
    for pending_path in sorted(pending_dir.glob("*.json")):
        claim_path = running_dir / f"{pending_path.stem}__{owner}.json"
        try:
            pending_path.rename(claim_path)
        except FileNotFoundError:
            continue

        payload = _read_task_payload(claim_path)
        return TaskLease(
            queue_dir=queue_dir,
            claim_path=claim_path,
            task_index=payload["task_index"],
            suite_name=payload["suite_name"],
            task_id=payload["task_id"],
            attempt=payload["attempts_completed"] + 1,
            owner=owner,
        )
    return None


def complete_task(lease: TaskLease) -> None:
    """Move a successfully evaluated task into the completed state."""
    payload = _read_task_payload(lease.claim_path)
    payload["attempts_completed"] = lease.attempt
    payload["completed_by"] = lease.owner
    _replace_json(lease.claim_path, payload)
    lease.claim_path.rename(
        lease.queue_dir / "completed" / _task_filename(lease.task_index)
    )


def retry_or_fail_task(
    lease: TaskLease, error: str, max_attempts: int
) -> str:
    """Requeue a failed task within its attempt budget, otherwise mark it failed."""
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")

    payload = _read_task_payload(lease.claim_path)
    payload["attempts_completed"] = lease.attempt
    payload["last_error"] = str(error)
    payload["last_owner"] = lease.owner
    _replace_json(lease.claim_path, payload)

    if lease.attempt < max_attempts:
        destination = lease.queue_dir / "pending" / _task_filename(lease.task_index)
        disposition = "requeued"
    else:
        destination = lease.queue_dir / "failed" / _task_filename(lease.task_index)
        disposition = "failed"
    lease.claim_path.rename(destination)
    return disposition


def queue_state_counts(queue_dir: Path) -> dict[str, int]:
    """Count task records in every queue state."""
    return {
        state: sum(1 for _ in (queue_dir / state).glob("*.json"))
        for state in _QUEUE_STATES
    }


def load_manifest(queue_dir: Path) -> dict[str, object]:
    """Load and validate the immutable queue manifest."""
    manifest_path = queue_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"queue manifest must be a JSON object: {manifest_path}")
    if manifest.get("version") != QUEUE_VERSION:
        raise ValueError(
            f"unsupported queue version {manifest.get('version')!r} in {manifest_path}"
        )
    task_count = manifest.get("task_count")
    tasks = manifest.get("tasks")
    if not isinstance(task_count, int) or task_count <= 0:
        raise ValueError(f"invalid queue task_count {task_count!r} in {manifest_path}")
    if not isinstance(tasks, list) or len(tasks) != task_count:
        raise ValueError(
            f"queue manifest task list does not match task_count={task_count}"
        )
    return manifest


def _task_filename(task_index: int) -> str:
    if task_index < 0:
        raise ValueError(f"task_index must be non-negative, got {task_index}")
    return f"{task_index:06d}.json"


def _validate_owner(owner: str) -> None:
    if not owner or _OWNER_PATTERN.fullmatch(owner) is None:
        raise ValueError(
            "queue owner must contain only letters, digits, dot, underscore, or dash; "
            f"got {owner!r}"
        )


def _read_task_payload(path: Path) -> dict[str, int | str]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"queue task record must be a JSON object: {path}")

    task_index = payload.get("task_index")
    suite_name = payload.get("suite_name")
    task_id = payload.get("task_id")
    attempts_completed = payload.get("attempts_completed")
    if (
        isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or task_index < 0
        or not isinstance(suite_name, str)
        or not suite_name
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id < 0
        or isinstance(attempts_completed, bool)
        or not isinstance(attempts_completed, int)
        or attempts_completed < 0
    ):
        raise ValueError(f"invalid queue task record {payload!r} in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")


def _replace_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{id(payload)}.tmp"
    )
    try:
        with temporary_path.open("x", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True)
            file_handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
