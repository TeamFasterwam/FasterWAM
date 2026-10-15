"""Prepare prompt inputs and validate evaluation cache contents and encoder metadata."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Literal, TypedDict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.task_prompt_cache import (  # noqa: E402
    DEFAULT_TASK_PROMPT_CACHE_DIR,
    TASK_IDENTITY_MANIFEST_FORMAT,
    load_task_prompt_context,
    load_task_prompt_context_for_task,
    model_id_to_encoder_id,
    task_prompt_cache_path,
)
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from libero.libero import benchmark  # noqa: E402


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_CONTEXT_DIM = 4096
STANDARD_PRECOMPUTE_BATCH_SIZE = 1
PLUS_PRECOMPUTE_BATCH_SIZE = 1
STANDARD_PRECOMPUTE_SCRIPT = "experiments/libero/precompute_standard_task_embeddings.py"
PLUS_PRECOMPUTE_SCRIPT = "experiments/libero/precompute_liberoplus_task_embeddings.py"
STANDARD_FLAVOR = "libero_standard"
PLUS_FLAVOR = "libero_plus"
BenchmarkFlavor = Literal["libero_standard", "libero_plus"]
EVALUATION_SUITES = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
STANDARD_EXPECTED_TASK_COUNTS = {suite_name: 10 for suite_name in EVALUATION_SUITES}
PLUS_EXPECTED_TASK_COUNTS = {
    "libero_10": 2519,
    "libero_goal": 2591,
    "libero_spatial": 2402,
    "libero_object": 2518,
}
PLUS_EXPECTED_TOTAL_TASKS = 10030
DEFAULT_PLUS_TASK_PROMPT_CACHE_DIR = PROJECT_ROOT / "data" / "eval_prompt_cache" / "libero_plus"
DEFAULT_PROMPT_INPUT_DIR = (
    Path(os.environ.get("TMPDIR", "/tmp")) / "fasterwam_libero_prompts"
)
DEFAULT_PLUS_PROMPT_INPUT_DIR = (
    Path(os.environ.get("TMPDIR", "/tmp")) / "fasterwam_liberoplus_prompts"
)


class TaskPromptRecord(TypedDict):
    """Canonical LIBERO task and its formatted Faster-WAM prompt."""

    suite: str
    task_id: int
    task_description: str
    prompt: str


class _RequiredManifestTaskRecord(TypedDict):
    """Required fields shared by standard and Plus manifest task records."""

    suite: str
    task_id: int
    task_description: str
    cache_file: str


class ManifestTaskRecord(_RequiredManifestTaskRecord, total=False):
    """Task identity plus the Plus-only prompt hash persisted in the manifest."""

    prompt_sha256: str


def _expected_task_counts(benchmark_flavor: BenchmarkFlavor) -> dict[str, int]:
    """Return the required task counts for one benchmark flavor."""
    if benchmark_flavor == STANDARD_FLAVOR:
        return STANDARD_EXPECTED_TASK_COUNTS
    if benchmark_flavor == PLUS_FLAVOR:
        return PLUS_EXPECTED_TASK_COUNTS
    raise ValueError(f"Unsupported benchmark flavor: {benchmark_flavor!r}")


def _precompute_contract(benchmark_flavor: BenchmarkFlavor) -> tuple[str, int]:
    """Return the generator and encoding batch size for one benchmark flavor."""
    if benchmark_flavor == STANDARD_FLAVOR:
        return STANDARD_PRECOMPUTE_SCRIPT, STANDARD_PRECOMPUTE_BATCH_SIZE
    if benchmark_flavor == PLUS_FLAVOR:
        return PLUS_PRECOMPUTE_SCRIPT, PLUS_PRECOMPUTE_BATCH_SIZE
    raise ValueError(f"Unsupported benchmark flavor: {benchmark_flavor!r}")


def collect_libero_prompts(benchmark_flavor: BenchmarkFlavor) -> list[TaskPromptRecord]:
    """Collect every task prompt for standard LIBERO or LIBERO-Plus."""
    benchmark_dict = benchmark.get_benchmark_dict()
    records: list[TaskPromptRecord] = []
    expected_counts = _expected_task_counts(benchmark_flavor)

    for suite_name in EVALUATION_SUITES:
        if suite_name not in benchmark_dict:
            raise ValueError(f"Missing LIBERO suite {suite_name!r} for {benchmark_flavor}.")
        with contextlib.redirect_stdout(io.StringIO()):
            task_suite = benchmark_dict[suite_name]()
        actual_count = int(task_suite.n_tasks)
        expected_count = expected_counts[suite_name]
        if actual_count != expected_count:
            raise ValueError(
                f"Wrong benchmark package for {benchmark_flavor}: suite={suite_name} "
                f"has {actual_count} tasks, expected {expected_count}."
            )
        for task_id in range(int(task_suite.n_tasks)):
            task_description = str(task_suite.get_task(task_id).language)
            prompt = DEFAULT_PROMPT.format(task=task_description)
            records.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "prompt": prompt,
                }
            )

    expected_total = sum(expected_counts.values())
    if len(records) != expected_total:
        raise ValueError(
            f"Expected {expected_total} {benchmark_flavor} task prompts, "
            f"found {len(records)}."
        )
    return records


def collect_standard_libero_prompts() -> list[TaskPromptRecord]:
    """Collect the 40 canonical standard-LIBERO task prompts."""
    records = collect_libero_prompts(STANDARD_FLAVOR)
    prompts = [record["prompt"] for record in records]
    if len(set(prompts)) != len(prompts):
        raise ValueError("Standard LIBERO task prompts must remain unique.")
    return records


def collect_liberoplus_prompts() -> list[TaskPromptRecord]:
    """Collect all 10,030 LIBERO-Plus tasks without prompt deduplication."""
    records = collect_libero_prompts(PLUS_FLAVOR)
    if len(records) != PLUS_EXPECTED_TOTAL_TASKS:
        raise ValueError(
            f"Expected {PLUS_EXPECTED_TOTAL_TASKS} LIBERO-Plus tasks, found {len(records)}."
        )
    return records


def _atomic_write_text(output_path: Path, content: str) -> None:
    """Atomically replace one UTF-8 text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, output_path)


def write_prompt_input_dataset(
    source_dir: Path,
    records: list[TaskPromptRecord],
    output_dir: Path,
    benchmark_flavor: BenchmarkFlavor = STANDARD_FLAVOR,
) -> Path:
    """Write the tasks.jsonl input consumed by the text-embedding precompute tool."""
    tasks_path = source_dir / "meta" / "tasks.jsonl"
    lines = [
        json.dumps(
            {
                "task": record["task_description"],
                "suite": record["suite"],
                "task_id": record["task_id"],
                "cache_file": _cache_filename(output_dir, record, benchmark_flavor),
            }
        )
        for record in records
    ]
    _atomic_write_text(tasks_path, "\n".join(lines) + "\n")
    LOGGER.info("Prepared %d prompt inputs at %s.", len(records), tasks_path)
    return tasks_path


def _build_manifest_tasks(
    output_dir: Path,
    records: list[TaskPromptRecord],
    benchmark_flavor: BenchmarkFlavor = STANDARD_FLAVOR,
) -> list[ManifestTaskRecord]:
    """Build the ordered manifest task records used for writing and validation."""
    manifest_tasks: list[ManifestTaskRecord] = []
    for record in records:
        manifest_record: ManifestTaskRecord = {
            "suite": record["suite"],
            "task_id": record["task_id"],
            "task_description": record["task_description"],
            "cache_file": _cache_filename(output_dir, record, benchmark_flavor),
        }
        if benchmark_flavor == PLUS_FLAVOR:
            manifest_record["prompt_sha256"] = hashlib.sha256(
                record["prompt"].encode("utf-8")
            ).hexdigest()
        manifest_tasks.append(manifest_record)
    return manifest_tasks


def _cache_filename(
    output_dir: Path,
    record: TaskPromptRecord,
    benchmark_flavor: BenchmarkFlavor,
) -> str:
    """Return the physical cache filename for one benchmark task."""
    if benchmark_flavor == STANDARD_FLAVOR:
        return task_prompt_cache_path(
            record["prompt"],
            output_dir,
            context_len=DEFAULT_CONTEXT_LEN,
            model_id=DEFAULT_MODEL_ID,
        ).name
    if benchmark_flavor == PLUS_FLAVOR:
        encoder_id = model_id_to_encoder_id(DEFAULT_MODEL_ID)
        return (
            f"{record['suite']}.task_{record['task_id']:04d}."
            f"t5_len{DEFAULT_CONTEXT_LEN}.{encoder_id}.pt"
        )
    raise ValueError(f"Unsupported benchmark flavor: {benchmark_flavor!r}")


def write_manifest(
    output_dir: Path,
    records: list[TaskPromptRecord],
    benchmark_flavor: BenchmarkFlavor = STANDARD_FLAVOR,
) -> Path:
    """Record the encoder settings and task mapping used to create the cache."""
    is_plus = benchmark_flavor == PLUS_FLAVOR
    generator, encoding_batch_size = _precompute_contract(benchmark_flavor)
    manifest = {
        "format": (
            TASK_IDENTITY_MANIFEST_FORMAT
            if is_plus
            else "fasterwam_t5_task_prompt_cache_v1"
        ),
        "benchmark_flavor": benchmark_flavor,
        "generator": generator,
        "encoding_batch_size": encoding_batch_size,
        "model_id": DEFAULT_MODEL_ID,
        "tokenizer_model_id": DEFAULT_TOKENIZER_MODEL_ID,
        "context_len": DEFAULT_CONTEXT_LEN,
        "context_dim": DEFAULT_CONTEXT_DIM,
        "num_prompts": len(records),
        "tasks": _build_manifest_tasks(output_dir, records, benchmark_flavor),
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def verify_task_prompt_manifest(
    output_dir: Path,
    records: list[TaskPromptRecord],
    benchmark_flavor: BenchmarkFlavor = STANDARD_FLAVOR,
) -> None:
    """Validate manifest provenance and its exact ordered benchmark task mapping."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing task-prompt manifest: {manifest_path}.")
    manifest_data: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        raise ValueError(f"Invalid task-prompt manifest root in {manifest_path}.")
    manifest: dict[str, object] = manifest_data
    is_plus = benchmark_flavor == PLUS_FLAVOR
    generator, encoding_batch_size = _precompute_contract(benchmark_flavor)
    expected_manifest_values: dict[str, object] = {
        "format": (
            TASK_IDENTITY_MANIFEST_FORMAT
            if is_plus
            else "fasterwam_t5_task_prompt_cache_v1"
        ),
        "generator": generator,
        "encoding_batch_size": encoding_batch_size,
        "model_id": DEFAULT_MODEL_ID,
        "tokenizer_model_id": DEFAULT_TOKENIZER_MODEL_ID,
        "context_len": DEFAULT_CONTEXT_LEN,
        "context_dim": DEFAULT_CONTEXT_DIM,
        "num_prompts": len(records),
    }
    if is_plus:
        expected_manifest_values["benchmark_flavor"] = PLUS_FLAVOR
    for key, expected_value in expected_manifest_values.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Task-prompt manifest mismatch for {key}: "
                f"expected {expected_value!r}, got {manifest.get(key)!r}."
            )
    expected_manifest_tasks = _build_manifest_tasks(output_dir, records, benchmark_flavor)
    if manifest.get("tasks") != expected_manifest_tasks:
        raise ValueError(
            f"Task-prompt manifest tasks do not match the {benchmark_flavor} registry: "
            f"{manifest_path}."
        )
    LOGGER.info("Verified %d task-prompt manifest records in %s.", len(records), output_dir)


def verify_task_prompt_cache(
    output_dir: Path,
    records: list[TaskPromptRecord],
    benchmark_flavor: BenchmarkFlavor = STANDARD_FLAVOR,
) -> None:
    """Validate every cached embedding payload after validating its manifest."""
    verify_task_prompt_manifest(output_dir, records, benchmark_flavor)
    is_plus = benchmark_flavor == PLUS_FLAVOR

    for record in records:
        if is_plus:
            load_task_prompt_context_for_task(
                record["prompt"],
                output_dir,
                suite_name=record["suite"],
                task_id=record["task_id"],
                context_len=DEFAULT_CONTEXT_LEN,
                context_dim=DEFAULT_CONTEXT_DIM,
                model_id=DEFAULT_MODEL_ID,
                tokenizer_model_id=DEFAULT_TOKENIZER_MODEL_ID,
                device="cpu",
                dtype=torch.bfloat16,
            )
        else:
            load_task_prompt_context(
                record["prompt"],
                output_dir,
                context_len=DEFAULT_CONTEXT_LEN,
                context_dim=DEFAULT_CONTEXT_DIM,
                model_id=DEFAULT_MODEL_ID,
                device="cpu",
                dtype=torch.bfloat16,
            )
    LOGGER.info(
        "Verified %d task-prompt caches in %s.",
        len(records),
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-flavor",
        choices=(STANDARD_FLAVOR, PLUS_FLAVOR),
        default=STANDARD_FLAVOR,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--prepare-source", action="store_true")
    action_group.add_argument("--finalize", action="store_true")
    action_group.add_argument("--verify-only", action="store_true")
    action_group.add_argument("--verify-manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Prepare prompt inputs, finalize metadata, or verify the cache."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    benchmark_flavor: BenchmarkFlavor = args.benchmark_flavor
    default_output_dir = (
        DEFAULT_PLUS_TASK_PROMPT_CACHE_DIR
        if benchmark_flavor == PLUS_FLAVOR
        else DEFAULT_TASK_PROMPT_CACHE_DIR
    )
    default_source_dir = (
        DEFAULT_PLUS_PROMPT_INPUT_DIR
        if benchmark_flavor == PLUS_FLAVOR
        else DEFAULT_PROMPT_INPUT_DIR
    )
    output_dir = (args.output_dir or default_output_dir).expanduser().resolve()
    source_dir = (args.source_dir or default_source_dir).expanduser().resolve()
    records = (
        collect_liberoplus_prompts()
        if benchmark_flavor == PLUS_FLAVOR
        else collect_standard_libero_prompts()
    )
    if args.prepare_source:
        write_prompt_input_dataset(source_dir, records, output_dir, benchmark_flavor)
        return
    if args.finalize:
        write_manifest(output_dir, records, benchmark_flavor)
    if args.verify_manifest_only:
        verify_task_prompt_manifest(output_dir, records, benchmark_flavor)
        return
    verify_task_prompt_cache(output_dir, records, benchmark_flavor)


if __name__ == "__main__":
    main()
