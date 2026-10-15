"""Summarize LIBERO results using the full requested episode budget.

Read per-task JSON results from an evaluation directory and report task, suite,
and overall success rates. Missing or failed tasks contribute no successes while
retaining their requested episodes in the denominator.

Usage:
    python experiments/libero/summarize_libero.py --output_dir <eval_output_dir>"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

# Standard-LIBERO suites each contain exactly 10 tasks (README_Libero.md:353-358).
STANDARD_SUITE_N_TASKS: int = 10

STANDARD_BENCHMARK_FLAVOR = "libero_standard"
PLUS_BENCHMARK_FLAVOR = "libero_plus"
BenchmarkFlavor = Literal["libero_standard", "libero_plus"]

# Canonical LIBERO-Plus counts, also enforced by
# the LIBERO-Plus benchmark installation.
PLUS_EXPECTED_TASK_COUNTS: dict[str, int] = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}

# Fallback expected task count used when the libero benchmark cannot be imported.
FALLBACK_N_TASKS: int = 10

# Default per-task episode budget (num_trials) used only when a suite has no
# result JSONs from which the real value can be inferred.
DEFAULT_NUM_TRIALS: int = 50

# The four canonical standard-LIBERO suite names, used to order output when
# suites are discovered on disk.
KNOWN_SUITE_NAMES: list[str] = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]

logger = logging.getLogger(__name__)


def _collect_results(output_dir: Path, suite_name: str) -> dict[int, dict[str, Any]]:
    """Read all per-task result JSONs for a suite.

    Walks <output_dir>/<suite_name>/ for files named gpu*_results.json, skipping any
    file that cannot be parsed as JSON (logged as a warning, not fatal). Deduplicates
    by task_id: when two files claim the same task_id (e.g. a RETRY written into the
    same dir by a different gpu worker), the NEWEST file by mtime wins so the retry
    supersedes the original. Every such collision is logged loudly with both
    filenames and which one was kept. Mirrors the collection walk in
    summarize_by_disturbance.py:70-98 minus the perturbation-category layer.

    Args:
        output_dir: Root evaluation output directory.
        suite_name: Suite subdirectory name (e.g. "libero_spatial").

    Returns:
        Mapping of task_id (int) -> parsed result dict.
    """
    suite_dir = output_dir / suite_name
    if not suite_dir.exists():
        return {}

    results: dict[int, dict[str, Any]] = {}
    # Track the winning file path per task_id so a later duplicate can be compared
    # by mtime and the collision can be reported with both filenames.
    winning_paths: dict[int, Path] = {}
    for filename in sorted(suite_dir.iterdir()):
        # Only consider the worker output pattern gpu{id}_task{id}_results.json.
        if not filename.name.startswith("gpu") or not filename.name.endswith(
            "_results.json"
        ):
            continue
        try:
            with filename.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as exc:
            # Corrupt or partially written JSON must not abort the whole summary.
            logger.warning("Skipping corrupt result JSON %s: %s", filename, exc)
            continue

        # Prefer the task_id recorded inside the JSON; fall back to parsing it out
        # of the filename (gpu{gpu_id}_task{task_id}_results.json) if absent.
        task_id = data.get("task_id")
        if task_id is None:
            parts = filename.name.split("_")
            task_id = int(parts[1].replace("task", ""))
        else:
            task_id = int(task_id)

        if task_id not in results:
            # First file seen for this task_id.
            results[task_id] = data
            winning_paths[task_id] = filename
            continue

        # Duplicate task_id: a retry (possibly from a different gpu worker) was
        # written into the same dir. Keep the NEWEST file by mtime so the retry
        # supersedes the original. On an EXACT mtime tie the later-encountered file
        # (deterministic sorted-walk order) wins: coarse-resolution filesystems can
        # stamp a fast retry with the same mtime as the original, and keeping the
        # stale original in that case would reintroduce the very bug this fixes.
        # Either way, log the collision loudly.
        existing_path = winning_paths[task_id]
        existing_mtime = existing_path.stat().st_mtime
        new_mtime = filename.stat().st_mtime
        if new_mtime >= existing_mtime:
            results[task_id] = data
            winning_paths[task_id] = filename
            kept, dropped = filename, existing_path
        else:
            kept, dropped = existing_path, filename
        logger.warning(
            "DUPLICATE result JSONs for task_id=%d in suite '%s': %s and %s. "
            "Kept the NEWEST by mtime (%s); dropped %s. A retry supersedes the "
            "original -- verify this was an intended re-run.",
            task_id,
            suite_name,
            existing_path.name,
            filename.name,
            kept.name,
            dropped.name,
        )

    return results


def _get_expected_n_tasks(suite_name: str) -> int:
    """Return the expected number of tasks for a suite from the libero benchmark.

    Imports libero LAZILY so this module stays importable in environments without
    libero installed. If the import
    fails, falls back to the standard suite size with a logged warning.

    Args:
        suite_name: Suite name registered in the libero benchmark dict.

    Returns:
        Expected task count for the suite, or FALLBACK_N_TASKS if libero or the
        suite is unavailable.
    """
    try:
        # Lazy import: keep the module import-clean where libero is absent.
        from libero.libero import benchmark
    except ImportError as exc:
        logger.warning(
            "libero not importable (%s); falling back to expected n_tasks=%d for '%s'",
            exc,
            FALLBACK_N_TASKS,
            suite_name,
        )
        return FALLBACK_N_TASKS

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        logger.warning(
            "Suite '%s' not in benchmark dict; falling back to expected n_tasks=%d",
            suite_name,
            FALLBACK_N_TASKS,
        )
        return FALLBACK_N_TASKS

    # Instantiating the suite prints benchmark banners to stdout; that is harmless
    # here so it is left as-is (unlike summarize_by_disturbance which suppresses it).
    suite = benchmark_dict[suite_name]()
    return int(suite.n_tasks)


def _infer_num_trials(
    results: dict[int, dict[str, Any]],
    suite_name: str,
    num_trials_fallback: int,
) -> int:
    """Infer the per-task episode budget (num_trials) for a suite.

    Every task JSON records total_episodes == num_trials (eval_libero_batch.py:182).
    All tasks in a suite must agree; a disagreement indicates mixed eval runs and is
    a hard error. When the suite has no result JSONs, the CLI fallback is used.

    Args:
        results: Collected per-task result dicts for the suite.
        suite_name: Suite name, used only for error messages.
        num_trials_fallback: Value to use when no JSONs are present.

    Returns:
        The inferred (or fallback) num_trials for the suite.

    Raises:
        ValueError: If tasks report differing total_episodes values.
    """
    if not results:
        return num_trials_fallback

    observed: set[int] = set()
    for data in results.values():
        observed.add(int(data.get("total_episodes", num_trials_fallback)))

    if len(observed) > 1:
        # Mixed num_trials across tasks means the denominator is ill-defined.
        raise ValueError(
            f"Suite '{suite_name}' has inconsistent total_episodes across tasks: "
            f"{sorted(observed)}. All tasks must share one num_trials value. "
            "This usually means result JSONs from different eval runs were mixed "
            "in the same output directory."
        )

    return observed.pop()


def _discover_suites(output_dir: Path, task_suite_names: list[str] | None) -> list[str]:
    """Determine which suites to report on.

    A suite is reported if its subdirectory under output_dir contains at least one
    gpu*_results.json file. Any suite named explicitly via --task_suite_names is also
    reported (even if empty, so missing runs are visible). Ordering favors the known
    canonical suite order, then any remaining suites alphabetically.

    Args:
        output_dir: Root evaluation output directory.
        task_suite_names: Optional explicit list of suites to force-report.

    Returns:
        Ordered list of suite names to summarize.
    """
    discovered: set[str] = set()
    if output_dir.exists():
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            # A suite dir qualifies only if it holds worker result files.
            has_results = any(
                f.name.startswith("gpu") and f.name.endswith("_results.json")
                for f in child.iterdir()
            )
            if has_results:
                discovered.add(child.name)

    if task_suite_names:
        discovered.update(task_suite_names)

    # Canonical suites first (in fixed order), then any extras alphabetically.
    ordered = [s for s in KNOWN_SUITE_NAMES if s in discovered]
    extras = sorted(s for s in discovered if s not in KNOWN_SUITE_NAMES)
    return ordered + extras


def summarize_libero(
    output_dir: str,
    num_trials_fallback: int = DEFAULT_NUM_TRIALS,
    task_suite_names: list[str] | None = None,
    benchmark_flavor: BenchmarkFlavor = STANDARD_BENCHMARK_FLAVOR,
) -> dict[str, Any]:
    """Aggregate LIBERO results and write summary.json and summary.csv.

    Computes per-task, per-suite, and overall success rates with a completion-aware
    denominator: each suite's denominator is always n_tasks_expected * num_trials, so
    crashed or missing tasks depress the rate instead of vanishing from it.

    Args:
        output_dir: Root evaluation output directory containing per-suite subdirs.
        num_trials_fallback: Episode budget assumed for suites with no result JSONs.
        task_suite_names: Optional suites to force-report even when empty.
        benchmark_flavor: Benchmark whose canonical task counts should be validated.

    Returns:
        The full JSON payload dict that is also written to disk.

    Raises:
        FileNotFoundError: If output_dir does not exist.
        ValueError: If a suite's tasks report inconsistent total_episodes.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    suites = _discover_suites(output_path, task_suite_names)
    if not suites:
        logger.warning("No suite result directories found in %s", output_dir)
        print(f"No suite result directories found in {output_dir}")
        return {}

    per_task_rows: list[list[str]] = []  # for CSV: per-task detail rows
    suite_payloads: dict[str, Any] = {}

    # Overall (cross-suite) accumulators using the completion-aware denominator.
    overall_successes = 0
    overall_expected_episodes = 0
    overall_completed_tasks = 0
    overall_expected_tasks = 0
    any_suspect = False

    benchmark_label = (
        "LIBERO-PLUS"
        if benchmark_flavor == PLUS_BENCHMARK_FLAVOR
        else "STANDARD-LIBERO"
    )
    print(f"\n{'=' * 70}")
    print(f"{benchmark_label} RESULTS SUMMARY (completion-aware denominator)")
    print(f"{'=' * 70}")

    for suite_name in suites:
        results = _collect_results(output_path, suite_name)
        n_tasks_expected = _get_expected_n_tasks(suite_name)
        num_trials = _infer_num_trials(results, suite_name, num_trials_fallback)

        canonical_n_tasks = (
            PLUS_EXPECTED_TASK_COUNTS.get(suite_name)
            if benchmark_flavor == PLUS_BENCHMARK_FLAVOR
            else STANDARD_SUITE_N_TASKS
        )
        # SANITY GUARD: compare the active suite against the selected benchmark's
        # canonical count. Unknown Plus suites remain suspect because there is no
        # canonical count against which to validate them.
        is_suspect = (
            canonical_n_tasks is None or n_tasks_expected != canonical_n_tasks
        )
        if is_suspect:
            logger.warning(
                "SUSPECT SUITE '%s': active n_tasks=%d != canonical %s count %s. "
                "Numbers below are flagged SUSPECT.",
                suite_name,
                n_tasks_expected,
                benchmark_flavor,
                canonical_n_tasks if canonical_n_tasks is not None else "UNKNOWN",
            )

        expected_episodes = n_tasks_expected * num_trials
        completed_tasks = len(results)

        # W3(a) GUARD: more completed tasks than expected means the denominator
        # (n_tasks_expected * num_trials) is SMALLER than the number of episodes
        # actually summed, so the reported rate can exceed 100%. This is reachable
        # via the ImportError fallback n_tasks=10 while summarizing a LIBERO-plus
        # dir. Flag SUSPECT and explain; do not silently rescale the denominator.
        if completed_tasks > n_tasks_expected:
            is_suspect = True
            logger.warning(
                "SUSPECT SUITE '%s': completed_tasks=%d exceeds expected "
                "n_tasks=%d, so the completion-aware denominator "
                "(%d tasks x %d trials = %d) undercounts the episodes summed and "
                "the success rate can exceed 100%%. This usually means a "
                "non-standard (e.g. LIBERO-plus) result dir is being summarized "
                "with the standard fallback of %d tasks. Numbers are kept as-is.",
                suite_name,
                completed_tasks,
                n_tasks_expected,
                n_tasks_expected,
                num_trials,
                expected_episodes,
                STANDARD_SUITE_N_TASKS,
            )

        # Sum successes over present tasks only; the denominator already accounts
        # for the missing tasks via expected_episodes.
        suite_successes = 0
        for task_id in sorted(results.keys()):
            data = results[task_id]
            successes = int(data.get("successes", 0))
            total_episodes = int(data.get("total_episodes", num_trials))
            # W3(b) GUARD: a task cannot succeed more often than it was run. A
            # corrupt/inflated count (successes > total_episodes) is impossible;
            # flag the whole suite SUSPECT and keep the RAW numbers (do not clamp)
            # so the corruption stays visible in the output.
            if successes > total_episodes:
                is_suspect = True
                logger.warning(
                    "SUSPECT SUITE '%s' task_id=%d: successes=%d exceeds "
                    "total_episodes=%d. A task cannot succeed more times than it "
                    "ran; raw numbers are kept (not clamped) but the suite is "
                    "flagged SUSPECT.",
                    suite_name,
                    task_id,
                    successes,
                    total_episodes,
                )
            task_rate = successes / total_episodes if total_episodes > 0 else 0.0
            suite_successes += successes
            per_task_rows.append(
                [
                    suite_name,
                    str(task_id),
                    str(successes),
                    str(total_episodes),
                    f"{task_rate:.4f}",
                ]
            )

        suite_rate = (
            suite_successes / expected_episodes if expected_episodes > 0 else 0.0
        )

        # Completion status label for the terminal line.
        if completed_tasks >= n_tasks_expected:
            status = "COMPLETE"
        else:
            status = "PARTIAL"
        suspect_tag = " SUSPECT" if is_suspect else ""

        print(
            f"{suite_name}: {suite_rate * 100:.2f}% "
            f"({suite_successes}/{expected_episodes}) "
            f"[{status}: {completed_tasks}/{n_tasks_expected} tasks]{suspect_tag}"
        )

        suite_payloads[suite_name] = {
            "success_rate": round(suite_rate, 4),
            "successes": suite_successes,
            "expected_episodes": expected_episodes,
            "num_trials": num_trials,
            "completed_tasks": completed_tasks,
            "expected_tasks": n_tasks_expected,
            "status": status,
            "suspect": is_suspect,
        }

        # Accumulate into the overall totals. is_suspect is only final HERE, after
        # the non-standard-count, over-completion (W3a), and inflated-success (W3b)
        # checks have all run, so propagate it to the overall flag at this point.
        any_suspect = any_suspect or is_suspect
        overall_successes += suite_successes
        overall_expected_episodes += expected_episodes
        overall_completed_tasks += completed_tasks
        overall_expected_tasks += n_tasks_expected

    overall_rate = (
        overall_successes / overall_expected_episodes
        if overall_expected_episodes > 0
        else 0.0
    )
    overall_status = (
        "COMPLETE" if overall_completed_tasks >= overall_expected_tasks else "PARTIAL"
    )
    overall_suspect_tag = " SUSPECT" if any_suspect else ""

    print(f"{'-' * 70}")
    print(
        f"ALL SUITES COMBINED: {overall_rate * 100:.2f}% "
        f"({overall_successes}/{overall_expected_episodes}) "
        f"[{overall_status}: {overall_completed_tasks}/{overall_expected_tasks} tasks]"
        f"{overall_suspect_tag}"
    )
    print(f"{'=' * 70}")

    overall_payload = {
        "success_rate": round(overall_rate, 4),
        "successes": overall_successes,
        "expected_episodes": overall_expected_episodes,
        "completed_tasks": overall_completed_tasks,
        "expected_tasks": overall_expected_tasks,
        "status": overall_status,
        "suspect": any_suspect,
    }

    full_payload: dict[str, Any] = {
        "per_suite": suite_payloads,
        "overall": overall_payload,
    }

    # Write summary.json.
    json_path = output_path / "summary.json"
    json_path.write_text(
        json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write summary.csv: per-task detail rows followed by per-suite and overall
    # aggregate rows (task_id left blank for aggregate rows).
    csv_path = output_path / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "level",
                "suite",
                "task_id",
                "successes",
                "total_episodes",
                "success_rate",
                "completed_tasks",
                "expected_tasks",
                "status",
                "suspect",
            ]
        )
        for row in per_task_rows:
            # row = [suite, task_id, successes, total_episodes, success_rate]
            writer.writerow(["task", *row, "", "", "", ""])
        for suite_name, payload in suite_payloads.items():
            writer.writerow(
                [
                    "suite",
                    suite_name,
                    "",
                    str(payload["successes"]),
                    str(payload["expected_episodes"]),
                    f"{payload['success_rate']:.4f}",
                    str(payload["completed_tasks"]),
                    str(payload["expected_tasks"]),
                    payload["status"],
                    str(payload["suspect"]),
                ]
            )
        writer.writerow(
            [
                "overall",
                "ALL_SUITES",
                "",
                str(overall_payload["successes"]),
                str(overall_payload["expected_episodes"]),
                f"{overall_payload['success_rate']:.4f}",
                str(overall_payload["completed_tasks"]),
                str(overall_payload["expected_tasks"]),
                overall_payload["status"],
                str(overall_payload["suspect"]),
            ]
        )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")

    return full_payload


def _parse_suite_names(raw_values: list[str] | None) -> list[str] | None:
    """Flatten space- or comma-separated suite names into a clean list.

    Accepts both ``--task_suite_names a b c`` and ``--task_suite_names a,b,c`` (and
    any mix), splitting on commas and dropping empty tokens. This deliberately
    DUPLICATES ``preflight_libero._parse_suite_names`` rather than importing it, so
    the two modules stay import-independent (summarize_libero must import cleanly in
    environments where preflight_libero / libero are absent). Keep the two copies in
    sync if the accepted syntax ever changes.

    Args:
        raw_values: Raw argv tokens for ``--task_suite_names`` (``None`` when the
            flag was omitted entirely).

    Returns:
        Ordered list of non-empty suite names, or ``None`` when ``raw_values`` is
        ``None`` (flag omitted -> discover suites from disk).
    """
    if raw_values is None:
        return None
    names: list[str] = []
    for token in raw_values:
        # Each token may itself be a comma-separated group.
        for part in token.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def main() -> None:
    """Parse CLI arguments and run the LIBERO summarizer."""
    # Configure the module logger once, at the entry point, so importing this
    # module does not hijack the root logging config of a host process.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Summarize LIBERO eval results with a completion-aware denominator."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root evaluation output directory (contains per-suite subdirs with result JSONs).",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Fallback per-task episode budget, used only for suites with no result JSONs.",
    )
    parser.add_argument(
        "--task_suite_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional suites to force-report even when they have no result JSONs.",
    )
    parser.add_argument(
        "--benchmark_flavor",
        choices=(STANDARD_BENCHMARK_FLAVOR, PLUS_BENCHMARK_FLAVOR),
        default=STANDARD_BENCHMARK_FLAVOR,
        help="Benchmark flavor used to validate canonical per-suite task counts.",
    )
    args = parser.parse_args()
    # Flatten comma/space forms so this CLI matches preflight_libero's advertised
    # syntax (--task_suite_names a,b,c and a b c both work).
    summarize_libero(
        output_dir=args.output_dir,
        num_trials_fallback=args.num_trials,
        task_suite_names=_parse_suite_names(args.task_suite_names),
        benchmark_flavor=args.benchmark_flavor,
    )


if __name__ == "__main__":
    main()
