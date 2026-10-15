"""Load evaluation-only LIBERO task-prompt embeddings."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


DEFAULT_TASK_PROMPT_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "eval_prompt_cache" / "libero"
)
TASK_IDENTITY_MANIFEST_FORMAT = "fasterwam_liberoplus_task_identity_t5_cache_v1"
TASK_IDENTITY_PAYLOAD_FORMAT = "fasterwam_liberoplus_task_identity_t5_payload_v1"


def model_id_to_encoder_id(model_id: str) -> str:
    """Convert a model identifier to the cache filename identifier."""
    model_name = str(model_id).split("/")[-1]
    encoder_id = re.sub(r"[^a-z0-9]+", "", model_name.lower())
    return encoder_id or "textenc"


def task_prompt_cache_path(
    prompt: str,
    cache_dir: str | Path,
    *,
    context_len: int,
    model_id: str,
) -> Path:
    """Return the training-compatible cache path for one formatted prompt."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    encoder_id = model_id_to_encoder_id(model_id)
    filename = f"{prompt_hash}.t5_len{context_len}.{encoder_id}.pt"
    return Path(cache_dir).expanduser().resolve() / filename


def load_task_prompt_context(
    prompt: str,
    cache_dir: str | Path,
    *,
    context_len: int,
    context_dim: int,
    model_id: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and validate one cached task-prompt context for inference."""
    cache_path = task_prompt_cache_path(
        prompt,
        cache_dir,
        context_len=context_len,
        model_id=model_id,
    )
    return _load_context_payload(
        cache_path,
        context_len=context_len,
        context_dim=context_dim,
        device=device,
        dtype=dtype,
    )


@lru_cache(maxsize=8)
def _load_task_manifest(
    cache_dir: str,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    """Load one task-identity manifest and index it by suite and task ID."""
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing task-prompt manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid task-prompt manifest tasks: {manifest_path}")
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for record in tasks:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid task-prompt manifest record: {record!r}")
        key = (str(record.get("suite")), int(record.get("task_id")))
        if key in index:
            raise ValueError(f"Duplicate task identity in {manifest_path}: {key}")
        index[key] = record
    return payload, index


def load_task_prompt_context_for_task(
    prompt: str,
    cache_dir: str | Path,
    *,
    suite_name: str,
    task_id: int,
    context_len: int,
    context_dim: int,
    model_id: str,
    tokenizer_model_id: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one task-identity cache and verify that its prompt matches."""
    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    manifest, task_index = _load_task_manifest(str(resolved_cache_dir))
    expected_manifest_values: dict[str, object] = {
        "format": TASK_IDENTITY_MANIFEST_FORMAT,
        "benchmark_flavor": "libero_plus",
        "model_id": model_id,
        "tokenizer_model_id": tokenizer_model_id,
        "context_len": context_len,
        "context_dim": context_dim,
    }
    for key, expected_value in expected_manifest_values.items():
        actual_value = manifest.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Task-prompt manifest mismatch for {key}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    record = task_index.get((suite_name, int(task_id)))
    if record is None:
        raise KeyError(f"Missing task identity in prompt manifest: {(suite_name, task_id)}")
    expected_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if record.get("prompt_sha256") != expected_prompt_hash:
        raise ValueError(
            f"Prompt hash mismatch for {(suite_name, task_id)} in {resolved_cache_dir}"
        )
    cache_file = str(record.get("cache_file", ""))
    if Path(cache_file).name != cache_file:
        raise ValueError(f"Invalid task cache filename: {cache_file!r}")
    return _load_context_payload(
        resolved_cache_dir / cache_file,
        context_len=context_len,
        context_dim=context_dim,
        device=device,
        dtype=dtype,
        expected_task_identity=(suite_name, int(task_id)),
        expected_prompt_hash=expected_prompt_hash,
        expected_payload_metadata={
            "format": TASK_IDENTITY_PAYLOAD_FORMAT,
            "model_id": model_id,
            "tokenizer_model_id": tokenizer_model_id,
            "context_len": context_len,
            "context_dim": context_dim,
        },
    )


def _load_context_payload(
    cache_path: Path,
    *,
    context_len: int,
    context_dim: int,
    device: str | torch.device,
    dtype: torch.dtype,
    expected_task_identity: tuple[str, int] | None = None,
    expected_prompt_hash: str | None = None,
    expected_payload_metadata: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and validate a standard or task-identity context payload."""
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing LIBERO evaluation task-prompt cache: {cache_path}. "
            "Run the benchmark-specific task-prompt precompute wrapper first."
        )

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or "context" not in payload
        or "mask" not in payload
    ):
        raise ValueError(
            f"Invalid task-prompt cache payload in {cache_path}: expected context/mask."
        )
    if expected_task_identity is not None:
        actual_task_identity = (payload.get("suite"), payload.get("task_id"))
        if actual_task_identity != expected_task_identity:
            raise ValueError(
                f"Task identity mismatch in {cache_path}: "
                f"expected {expected_task_identity}, got {actual_task_identity}"
            )
    if expected_prompt_hash is not None and payload.get("prompt_sha256") != expected_prompt_hash:
        raise ValueError(f"Prompt hash mismatch inside task cache: {cache_path}")
    if expected_payload_metadata is not None:
        for key, expected_value in expected_payload_metadata.items():
            actual_value = payload.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"Task-prompt payload mismatch for {key} in {cache_path}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )
    context = payload["context"]
    context_mask = payload["mask"]
    if not isinstance(context, torch.Tensor) or not isinstance(
        context_mask, torch.Tensor
    ):
        raise TypeError(f"Invalid task-prompt cache tensors in {cache_path}.")
    if tuple(context.shape) != (context_len, context_dim):
        raise ValueError(
            f"Cached context shape mismatch in {cache_path}: "
            f"expected {(context_len, context_dim)}, got {tuple(context.shape)}."
        )
    if tuple(context_mask.shape) != (context_len,):
        raise ValueError(
            f"Cached mask shape mismatch in {cache_path}: "
            f"expected {(context_len,)}, got {tuple(context_mask.shape)}."
        )

    context_mask = context_mask.bool()
    context = context.clone()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)

    context = context.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    context_mask = context_mask.unsqueeze(0).to(
        device=device,
        dtype=torch.bool,
        non_blocking=True,
    )
    return context, context_mask
