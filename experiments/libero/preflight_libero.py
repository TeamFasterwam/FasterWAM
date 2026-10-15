"""Check the active LIBERO benchmark and available initial states before evaluation.

Standard LIBERO and LIBERO-Plus share suite names, so check the installed
benchmark's task counts and marker files. Also reject insufficient initial
states instead of silently repeating them to meet the requested trial count.
Checks read benchmark metadata without changing its configuration or assets.

Run ``python -m experiments.libero.preflight_libero`` from the repository root.
LIBERO is imported lazily so configuration tools can import this module without
installing the simulator. The CLI exits nonzero if a check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)

# Standard LIBERO has 10 tasks per suite; Plus has 2402 or more.
STANDARD_LIBERO_N_TASKS: int = 10

# Match the standard evaluation defaults in configs/sim_libero.yaml.
DEFAULT_NUM_TRIALS: int = 50

DEFAULT_TASK_SUITE_NAMES: list[str] = [
    "libero_10",
    "libero_goal",
    "libero_spatial",
    "libero_object",
]

# This marker ships only with LIBERO-Plus.
LIBERO_PLUS_MARKER_FILENAME: str = "task_classification.json"


class PreflightError(RuntimeError):
    """Raised when a preflight check fails.

    Carrying a dedicated exception type lets the CLI entry point distinguish an
    expected preflight failure (report cleanly, exit nonzero) from an unexpected
    bug.
    """


def _import_benchmark() -> "ModuleType":
    """Lazily import and return ``libero.libero.benchmark``.

    The import is deferred so configuration tools do not need the simulator.

    Returns:
        The imported ``libero.libero.benchmark`` module.

    Raises:
        PreflightError: If ``libero`` cannot be imported. Failing to import
            ``libero`` is itself a preflight failure when run as a CLI.
    """
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise PreflightError(
            "Could not import 'libero.libero.benchmark'. LIBERO does not appear "
            "to be installed in the active environment. Install/activate the "
            f"LIBERO eval env before running preflight. Underlying error: {exc}"
        ) from exc
    return benchmark


def _instantiate_suite(benchmark: "ModuleType", suite_name: str) -> Any:
    """Instantiate one benchmark suite by name, suppressing its stdout banner.

    LIBERO's suite constructors print a banner to stdout; we redirect that to a
    throwaway buffer so preflight output stays clean. This is read-only: it only
    constructs the in-memory benchmark object.

    Args:
        benchmark: The ``libero.libero.benchmark`` module.
        suite_name: Suite key, for example ``"libero_10"``.

    Returns:
        The instantiated task-suite object.

    Raises:
        PreflightError: If the suite name is not registered.
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        available = ", ".join(sorted(benchmark_dict.keys()))
        raise PreflightError(
            f"Suite '{suite_name}' is not a registered LIBERO benchmark. "
            f"Available suites: {available}"
        )
    # Suppress the constructor's stdout banner (read-only construction).
    with contextlib.redirect_stdout(io.StringIO()):
        suite = benchmark_dict[suite_name]()
    return suite


def _warn_if_libero_plus_installed(benchmark: "ModuleType") -> None:
    """Emit a warning if the LIBERO-plus marker file is present (read-only).

    Checks for ``task_classification.json`` in the installed benchmark package
    directory. This is a secondary, informational signal only; the authoritative
    check is the per-suite task count. The file is never opened or modified,
    only tested for existence.

    Args:
        benchmark: The ``libero.libero.benchmark`` module.
    """
    marker_path = Path(benchmark.__file__).parent / LIBERO_PLUS_MARKER_FILENAME
    if marker_path.exists():
        logger.warning(
            "LIBERO-plus marker file found at %s. LIBERO-plus appears to be "
            "installed. Standard-LIBERO eval numbers may not be comparable.",
            marker_path,
        )


def assert_benchmark_is_standard(suite_names: list[str]) -> None:
    """Assert every requested suite is standard LIBERO (10 tasks each).

    For each suite name, instantiate it via
    ``benchmark.get_benchmark_dict()[suite]()`` and require
    ``n_tasks == STANDARD_LIBERO_N_TASKS``. A non-standard count (for example the
    2402+ tasks of LIBERO-plus) fails the check. Every offending suite is
    reported, not just the first. A secondary read-only warning is emitted if the
    LIBERO-plus marker file is present.

    Args:
        suite_names: Suite keys to validate, for example ``["libero_10", ...]``.

    Raises:
        PreflightError: If any suite is missing or does not have exactly
            ``STANDARD_LIBERO_N_TASKS`` tasks.
    """
    benchmark = _import_benchmark()
    _warn_if_libero_plus_installed(benchmark)

    problems: list[str] = []
    # Check every suite so the report is complete rather than first-failure-only.
    for suite_name in suite_names:
        suite = _instantiate_suite(benchmark, suite_name)
        n_tasks = int(suite.n_tasks)
        if n_tasks != STANDARD_LIBERO_N_TASKS:
            problems.append(
                f"  - suite '{suite_name}': n_tasks={n_tasks} "
                f"(expected {STANDARD_LIBERO_N_TASKS} for standard LIBERO)"
            )
        else:
            logger.info(
                "Suite '%s' OK: n_tasks=%d (standard LIBERO).", suite_name, n_tasks
            )

    if problems:
        raise PreflightError(
            "Benchmark guard FAILED: the active benchmark is not standard "
            "LIBERO. Offending suites:\n" + "\n".join(problems)
        )


def assert_init_states(suite_names: list[str], num_trials: int) -> None:
    """Assert every task in every suite has at least ``num_trials`` init states.

    For each suite and each ``task_id`` in ``range(n_tasks)``, require
    ``len(task_suite.get_task_init_states(task_id)) >= num_trials``. Every
    offending ``(suite, task_id, count)`` triple is reported, not just the first,
    then the check fails. This guards against the batch worker silently padding
    init states by repetition when a task has fewer than ``num_trials`` states
    (Source: eval_libero_batch.py:153-156), which would produce non-comparable
    numbers with no error.

    Args:
        suite_names: Suite keys to validate.
        num_trials: Minimum required init states per task.

    Raises:
        PreflightError: If any task has fewer than ``num_trials`` init states.
    """
    benchmark = _import_benchmark()

    problems: list[str] = []
    for suite_name in suite_names:
        suite = _instantiate_suite(benchmark, suite_name)
        n_tasks = int(suite.n_tasks)
        # Report EVERY offending (suite, task_id, count), not just the first.
        for task_id in range(n_tasks):
            init_states = suite.get_task_init_states(task_id)
            count = len(init_states)
            if count < num_trials:
                problems.append(
                    f"  - suite '{suite_name}' task_id={task_id}: "
                    f"{count} init states (need >= {num_trials})"
                )
        logger.info(
            "Suite '%s': checked init states for %d tasks (need >= %d each).",
            suite_name,
            n_tasks,
            num_trials,
        )

    if problems:
        raise PreflightError(
            "Init-state guard FAILED: the following tasks have fewer than "
            f"{num_trials} init states, so the batch worker would silently pad "
            "them by repetition:\n" + "\n".join(problems)
        )


def run_preflight(suite_names: list[str], num_trials: int) -> int:
    """Run all preflight checks and return a process exit code.

    Runs the benchmark guard first, then the init-state guard. The init-state
    guard is SKIPPED when the benchmark guard fails: on a non-standard benchmark
    (for example LIBERO-plus, which enumerates ~2402+ tasks per suite) the answer
    is already moot, and running it anyway would trigger thousands of
    ``get_task_init_states`` disk reads for no benefit. The skip is reported
    explicitly and still counts as an overall FAIL (nonzero exit).

    Args:
        suite_names: Suite keys to validate.
        num_trials: Minimum required init states per task.

    Returns:
        ``0`` if all checks pass, ``1`` if any check fails.
    """
    exit_code = 0
    benchmark_ok = True

    # Check 1: benchmark identity (standard LIBERO vs LIBERO-plus/other).
    try:
        assert_benchmark_is_standard(suite_names)
        logger.info("PASS: benchmark guard - all suites are standard LIBERO.")
    except PreflightError as exc:
        exit_code = 1
        benchmark_ok = False
        logger.error("FAIL: benchmark guard.\n%s", exc)

    # Check 2: init-state sufficiency. Only meaningful once we know the active
    # benchmark IS standard LIBERO. If the benchmark guard already failed, skip
    # this check: on a LIBERO-plus machine it would enumerate ~2402 tasks/suite x
    # get_task_init_states disk reads to answer a question that is already moot.
    if benchmark_ok:
        try:
            assert_init_states(suite_names, num_trials)
            logger.info(
                "PASS: init-state guard - every task has >= %d init states.",
                num_trials,
            )
        except PreflightError as exc:
            exit_code = 1
            logger.error("FAIL: init-state guard.\n%s", exc)
    else:
        # Overall verdict is already FAIL; make the skip loud and explicit so the
        # operator knows the init-state guard did NOT run (it was not silently
        # passed). Fix the benchmark identity first, then re-run preflight.
        logger.error(
            "SKIP: init-state guard skipped because the benchmark guard failed. "
            "On a non-standard benchmark this check would read init states for "
            "thousands of tasks to answer a moot question. Fix the benchmark "
            "identity (install/activate standard LIBERO) and re-run preflight."
        )

    if exit_code == 0:
        logger.info("PREFLIGHT PASSED: all checks green.")
    else:
        logger.error("PREFLIGHT FAILED: see FAIL lines above.")
    return exit_code


def _parse_suite_names(raw_values: list[str]) -> list[str]:
    """Flatten space- or comma-separated suite names into a clean list.

    Accepts both ``--task_suite_names a b c`` and ``--task_suite_names a,b,c``
    (and any mix), splitting on commas and dropping empty tokens.

    Args:
        raw_values: Raw argv tokens for the ``--task_suite_names`` option.

    Returns:
        Ordered list of non-empty suite names.
    """
    names: list[str] = []
    for token in raw_values:
        # Each token may itself be a comma-separated group.
        for part in token.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the preflight module."""
    parser = argparse.ArgumentParser(
        prog="preflight_libero",
        description=(
            "Fail loudly before a LIBERO eval launch if the active benchmark is "
            "not standard LIBERO, or if any task lacks enough init states."
        ),
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=(
            "Minimum init states required per task "
            f"(default: {DEFAULT_NUM_TRIALS})."
        ),
    )
    parser.add_argument(
        "--task_suite_names",
        nargs="+",
        default=list(DEFAULT_TASK_SUITE_NAMES),
        help=(
            "Suites to check, space or comma separated "
            f"(default: {' '.join(DEFAULT_TASK_SUITE_NAMES)})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parse args, run preflight, return an exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on all-pass, nonzero on any failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    suite_names = _parse_suite_names(args.task_suite_names)
    if not suite_names:
        logger.error("No task suite names provided after parsing --task_suite_names.")
        return 1

    logger.info(
        "Running LIBERO preflight: suites=%s, num_trials=%d",
        suite_names,
        args.num_trials,
    )
    # A missing/broken libero surfaces here as a PreflightError -> nonzero exit.
    try:
        return run_preflight(suite_names, args.num_trials)
    except PreflightError as exc:
        logger.error("PREFLIGHT FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
