"""Encode every LIBERO-Plus task into a distinct evaluation cache file."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import TypedDict

import torch
import torch.distributed as dist
from tqdm import tqdm

from experiments.libero.task_prompt_cache import TASK_IDENTITY_PAYLOAD_FORMAT
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fasterwam.tools.precompute_text_embeds import (
    DEFAULT_CONTEXT_LEN,
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_MODEL_ID,
    HuggingfaceTokenizer,
    _atomic_torch_save,
    _init_distributed,
    _load_registered_model,
    _resolve_configs,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONTEXT_DIM = 4096
DEFAULT_PRECOMPUTE_BATCH_SIZE = 1


class TaskEmbeddingRecord(TypedDict):
    """One task row and its unique physical cache destination."""

    suite: str
    task_id: int
    task: str
    cache_file: str


def read_task_embedding_records(source_dir: Path) -> list[TaskEmbeddingRecord]:
    """Read all source rows without deduplicating repeated instructions."""
    tasks_path = source_dir / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing LIBERO-Plus task source: {tasks_path}")

    records: list[TaskEmbeddingRecord] = []
    cache_files: set[str] = set()
    for line_number, line in enumerate(tasks_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        required = {"suite", "task_id", "task", "cache_file"}
        missing = sorted(required.difference(payload))
        if missing:
            raise KeyError(f"Missing fields {missing} at {tasks_path}:{line_number}")
        cache_file = str(payload["cache_file"])
        if Path(cache_file).name != cache_file:
            raise ValueError(f"cache_file must be a filename at {tasks_path}:{line_number}")
        if cache_file in cache_files:
            raise ValueError(f"Duplicate task cache filename: {cache_file}")
        cache_files.add(cache_file)
        records.append(
            {
                "suite": str(payload["suite"]),
                "task_id": int(payload["task_id"]),
                "task": str(payload["task"]),
                "cache_file": cache_file,
            }
        )
    return records


def _expected_payload_metadata(
    record: TaskEmbeddingRecord,
    *,
    prompt: str,
    model_id: str,
    tokenizer_model_id: str,
    context_len: int,
    context_dim: int,
) -> dict[str, object]:
    """Return the identity and encoder provenance required in one cache payload."""
    return {
        "format": TASK_IDENTITY_PAYLOAD_FORMAT,
        "suite": record["suite"],
        "task_id": record["task_id"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model_id": model_id,
        "tokenizer_model_id": tokenizer_model_id,
        "context_len": context_len,
        "context_dim": context_dim,
    }


def _cache_matches_record(
    cache_path: Path,
    record: TaskEmbeddingRecord,
    *,
    model_id: str,
    tokenizer_model_id: str,
    context_len: int,
    context_dim: int,
) -> bool:
    """Return whether an existing payload is safe to reuse for this exact task."""
    if not cache_path.is_file():
        return False
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, ValueError):
        LOGGER.warning("Regenerating unreadable task cache: %s", cache_path)
        return False
    if not isinstance(payload, dict):
        return False
    prompt = DEFAULT_PROMPT.format(task=record["task"])
    expected_metadata = _expected_payload_metadata(
        record,
        prompt=prompt,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        context_len=context_len,
        context_dim=context_dim,
    )
    if any(payload.get(key) != value for key, value in expected_metadata.items()):
        return False
    context = payload.get("context")
    mask = payload.get("mask")
    return (
        isinstance(context, torch.Tensor)
        and tuple(context.shape) == (context_len, context_dim)
        and isinstance(mask, torch.Tensor)
        and tuple(mask.shape) == (context_len,)
    )


def parse_args() -> argparse.Namespace:
    """Parse the task-identity precompute arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=(DEFAULT_MODEL_ID,), default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--tokenizer-model-id",
        choices=(DEFAULT_TOKENIZER_MODEL_ID,),
        default=DEFAULT_TOKENIZER_MODEL_ID,
    )
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--context-dim", type=int, default=DEFAULT_CONTEXT_DIM)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_PRECOMPUTE_BATCH_SIZE,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Encode all source rows one-to-one, distributed across visible GPUs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_task_embedding_records(source_dir)
    if len(records) != 10030:
        raise ValueError(f"Expected 10030 LIBERO-Plus rows, found {len(records)}")
    if args.context_len <= 0 or args.context_dim <= 0 or args.batch_size <= 0:
        raise ValueError("context-len, context-dim, and batch-size must be positive")

    is_distributed, rank, world_size, local_rank = _init_distributed()
    rank_records = records[rank::world_size] if is_distributed else records
    if not args.overwrite:
        rank_records = [
            record
            for record in rank_records
            if not _cache_matches_record(
                output_dir / record["cache_file"],
                record,
                model_id=str(args.model_id),
                tokenizer_model_id=str(args.tokenizer_model_id),
                context_len=args.context_len,
                context_dim=args.context_dim,
            )
        ]

    local_count = len(rank_records)
    total_count = local_count
    if is_distributed:
        count_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        count_tensor = torch.tensor([local_count], device=count_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        total_count = int(count_tensor.item())
    if rank == 0:
        LOGGER.info(
            "LIBERO-Plus task rows=%d, world_size=%d, rows_to_encode=%d",
            len(records),
            world_size,
            total_count,
        )
    if total_count == 0:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("LIBERO-Plus task embedding generation requires CUDA")

    device = f"cuda:{local_rank}" if is_distributed else "cuda"
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=str(args.model_id),
        tokenizer_model_id=str(args.tokenizer_model_id),
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=args.context_len,
        clean="whitespace",
    )

    with tqdm(
        total=len(rank_records),
        desc=f"Encoding Plus tasks (rank {rank}/{world_size})",
        unit="task",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as progress:
        with torch.no_grad():
            for start in range(0, len(rank_records), args.batch_size):
                batch_records = rank_records[start : start + args.batch_size]
                prompts = [DEFAULT_PROMPT.format(task=record["task"]) for record in batch_records]
                token_ids, masks = tokenizer(
                    prompts,
                    return_mask=True,
                    add_special_tokens=True,
                )
                token_ids = token_ids.to(device)
                masks = masks.to(device=device, dtype=torch.bool)
                contexts = text_encoder(token_ids, masks)
                if tuple(contexts.shape[1:]) != (args.context_len, args.context_dim):
                    raise ValueError(
                        "Text encoder output shape mismatch: "
                        f"expected (*, {args.context_len}, {args.context_dim}), "
                        f"got {tuple(contexts.shape)}"
                    )
                for index, record in enumerate(batch_records):
                    prompt = prompts[index]
                    payload = {
                        "context": contexts[index]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous(),
                        "mask": masks[index]
                        .detach()
                        .to(device="cpu", dtype=torch.bool)
                        .contiguous(),
                        **_expected_payload_metadata(
                            record,
                            prompt=prompt,
                            model_id=str(args.model_id),
                            tokenizer_model_id=str(args.tokenizer_model_id),
                            context_len=args.context_len,
                            context_dim=args.context_dim,
                        ),
                    }
                    _atomic_torch_save(payload, output_dir / record["cache_file"])
                progress.update(len(batch_records))

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
