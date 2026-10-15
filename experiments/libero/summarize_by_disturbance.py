"""Summarize LIBERO-plus evaluation results grouped by perturbation category.

Reads per-task result JSONs from an evaluation output directory, maps each task
to its perturbation category via task_classification.json (shipped with
LIBERO-plus), and reports per-category success rates.

Usage:
    python experiments/libero/summarize_by_disturbance.py <eval_output_dir>

Outputs:
    - Terminal table with per-category breakdown
    - <output_dir>/disturbance_summary.csv
    - <output_dir>/disturbance_summary.json
"""

import argparse
import contextlib
import csv
import io
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from libero.libero import benchmark


def _load_classification() -> dict[str, list[dict[str, Any]]]:
    """Load task_classification.json from the LIBERO-plus benchmark package."""
    path = Path(benchmark.__file__).parent / "task_classification.json"
    if not path.exists():
        raise FileNotFoundError(
            f"task_classification.json not found at {path}. "
            "This script requires LIBERO-plus (not standard LIBERO)."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_task_id_to_category(
    suite_name: str,
    classification: dict[str, list[dict[str, Any]]],
) -> dict[int, str]:
    """Map task_id (0-based benchmark index) -> perturbation category string."""
    if suite_name not in classification:
        return {}

    name_to_cat = {
        entry["name"]: entry["category"] for entry in classification[suite_name]
    }

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        return {}

    with contextlib.redirect_stdout(io.StringIO()):
        suite = benchmark_dict[suite_name]()
    mapping: dict[int, str] = {}
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        cat = name_to_cat.get(task.name)
        if cat is not None:
            mapping[task_id] = cat

    return mapping


def _collect_results(
    output_dir: Path,
    suite_name: str,
) -> dict[int, dict[str, Any]]:
    """Read all per-task result JSONs for a suite. Returns {task_id: result_dict}."""
    suite_dir = output_dir / suite_name
    if not suite_dir.exists():
        return {}

    results: dict[int, dict[str, Any]] = {}
    for filename in sorted(suite_dir.iterdir()):
        if not filename.name.startswith("gpu") or not filename.name.endswith("_results.json"):
            continue
        try:
            with filename.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as exc:
            logging.warning("Skipping corrupt result JSON %s: %s", filename, exc)
            continue
        task_id = data.get("task_id")
        if task_id is None:
            parts = filename.name.split("_")
            task_id = int(parts[1].replace("task", ""))
        else:
            task_id = int(task_id)
        if task_id not in results:
            results[task_id] = data

    return results


def summarize_by_disturbance(output_dir: str) -> None:
    """Main entry point: aggregate results by perturbation category."""
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    classification = _load_classification()

    all_suites = [
        "libero_spatial", "libero_object", "libero_goal",
        "libero_10",
    ]
    discovered_suites = [
        s for s in all_suites if (output_path / s).exists()
    ]
    if not discovered_suites:
        print(f"No suite result directories found in {output_dir}")
        return

    full_payload: dict[str, Any] = {}
    csv_rows: list[list[str]] = []

    benchmark_dict = benchmark.get_benchmark_dict()

    # Accumulators for combined cross-suite aggregation
    combined_successes: dict[str, int] = defaultdict(int)
    combined_episodes: dict[str, int] = defaultdict(int)
    combined_tasks: dict[str, int] = defaultdict(int)
    combined_expected: dict[str, int] = defaultdict(int)
    combined_uncategorized: int = 0
    contributing_suites: list[str] = []  # suites that produced >= 1 result file

    for suite_name in discovered_suites:
        id_to_cat = _build_task_id_to_category(suite_name, classification)
        results = _collect_results(output_path, suite_name)

        # Expected task counts from benchmark
        expected_total = 0
        if suite_name in benchmark_dict:
            with contextlib.redirect_stdout(io.StringIO()):
                expected_total = int(benchmark_dict[suite_name]().n_tasks)
        cat_expected: dict[str, int] = defaultdict(int)
        for cat in id_to_cat.values():
            cat_expected[cat] += 1

        if not results:
            print(f"\n{suite_name}: no result files found (expected {expected_total})")
            continue

        contributing_suites.append(suite_name)

        # Group by category
        cat_successes: dict[str, int] = defaultdict(int)
        cat_episodes: dict[str, int] = defaultdict(int)
        cat_tasks: dict[str, int] = defaultdict(int)
        uncategorized_count = 0

        for task_id, data in sorted(results.items()):
            category = id_to_cat.get(task_id, "Uncategorized")
            if category == "Uncategorized":
                uncategorized_count += 1
            successes = int(data.get("successes", 0))
            episodes = int(data.get("total_episodes", 1))
            cat_successes[category] += successes
            cat_episodes[category] += episodes
            cat_tasks[category] += 1

        # Accumulate into combined cross-suite totals
        for cat in cat_successes:
            combined_successes[cat] += cat_successes[cat]
            combined_episodes[cat] += cat_episodes[cat]
            combined_tasks[cat] += cat_tasks[cat]
        for cat in cat_expected:
            combined_expected[cat] += cat_expected[cat]
        combined_uncategorized += uncategorized_count

        # Compute per-category rates
        cat_rates: dict[str, float] = {}
        for cat in sorted(cat_successes.keys()):
            if cat_episodes[cat] > 0:
                cat_rates[cat] = cat_successes[cat] / cat_episodes[cat]

        # Overall suite rate
        total_succ = sum(cat_successes.values())
        total_ep = sum(cat_episodes.values())
        suite_rate = total_succ / total_ep if total_ep > 0 else 0.0

        # Completion status
        is_complete = len(results) >= expected_total if expected_total > 0 else True
        status = "COMPLETE" if is_complete else "PARTIAL"
        pct_done = len(results) / expected_total * 100 if expected_total > 0 else 100.0

        # Print
        print(f"\n{'=' * 70}")
        print(f"{suite_name}: {suite_rate * 100:.2f}% success rate  "
              f"[{status}: {len(results)}/{expected_total} tasks ({pct_done:.0f}%)]")
        print(f"{'=' * 70}")
        print(f"  {'Category':<25} {'Success Rate':>14} {'Successes':>12} {'Progress':>12}")
        print(f"  {'-' * 25} {'-' * 14} {'-' * 12} {'-' * 12}")

        for cat in sorted(cat_rates.keys()):
            rate = cat_rates[cat]
            succ = cat_successes[cat]
            ep = cat_episodes[cat]
            n_done = cat_tasks[cat]
            n_expected = cat_expected.get(cat, n_done)
            progress = f"{n_done}/{n_expected}"
            print(f"  {cat:<25} {rate * 100:>13.2f}% {succ:>5}/{ep:<5} {progress:>12}")
            csv_rows.append([suite_name, cat, f"{rate:.4f}", str(succ), str(ep),
                             str(n_done), str(n_expected), status])

        csv_rows.append([suite_name, "__overall__", f"{suite_rate:.4f}",
                         str(total_succ), str(total_ep),
                         str(len(results)), str(expected_total), status])

        if uncategorized_count > 0:
            print(f"\n  WARNING: {uncategorized_count} tasks had no category mapping")

        # Build JSON payload
        full_payload[suite_name] = {
            "status": status,
            "overall_success_rate": round(suite_rate, 4),
            "total_successes": total_succ,
            "total_episodes": total_ep,
            "num_tasks_completed": len(results),
            "num_tasks_expected": expected_total,
            "per_category": {
                cat: {
                    "success_rate": round(cat_rates[cat], 4),
                    "successes": cat_successes[cat],
                    "episodes": cat_episodes[cat],
                    "num_tasks_completed": cat_tasks[cat],
                    "num_tasks_expected": cat_expected.get(cat, cat_tasks[cat]),
                }
                for cat in sorted(cat_rates.keys())
            },
        }

    # --- Combined cross-suite aggregation ---
    # Pool successes and episodes across all discovered suites per category
    if combined_episodes:
        combined_rates: dict[str, float] = {}
        for cat in sorted(combined_successes.keys()):
            if combined_episodes[cat] > 0:
                combined_rates[cat] = combined_successes[cat] / combined_episodes[cat]
            else:
                combined_rates[cat] = 0.0

        combined_total_succ = sum(combined_successes.values())
        combined_total_ep = sum(combined_episodes.values())
        combined_overall_rate = (
            combined_total_succ / combined_total_ep if combined_total_ep > 0 else 0.0
        )
        combined_total_tasks = sum(combined_tasks.values())
        combined_total_expected = sum(combined_expected.values())

        print(f"\n{'=' * 70}")
        print(
            f"ALL SUITES COMBINED: {combined_overall_rate * 100:.2f}% success rate  "
            f"[{combined_total_tasks} tasks across {len(contributing_suites)} suites]"
        )
        print(f"{'=' * 70}")
        print(
            f"  {'Category':<25} {'Success Rate':>14} "
            f"{'Successes':>12} {'Tasks':>8} {'Episodes':>10}"
        )
        print(f"  {'-' * 25} {'-' * 14} {'-' * 12} {'-' * 8} {'-' * 10}")

        for cat in sorted(combined_rates.keys()):
            rate = combined_rates[cat]
            succ = combined_successes[cat]
            ep = combined_episodes[cat]
            n_tasks = combined_tasks[cat]
            print(
                f"  {cat:<25} {rate * 100:>13.2f}% "
                f"{succ:>5}/{ep:<5} {n_tasks:>8} {ep:>10}"
            )
            csv_rows.append([
                "ALL_SUITES", cat, f"{rate:.4f}", str(succ), str(ep),
                str(n_tasks), str(combined_expected.get(cat, n_tasks)), "COMBINED",
            ])

        # Overall combined row
        csv_rows.append([
            "ALL_SUITES", "__overall__", f"{combined_overall_rate:.4f}",
            str(combined_total_succ), str(combined_total_ep),
            str(combined_total_tasks), str(combined_total_expected), "COMBINED",
        ])

        if combined_uncategorized > 0:
            print(
                f"\n  WARNING: {combined_uncategorized} tasks across all suites "
                f"had no category mapping"
            )

        # Add combined section to JSON payload
        full_payload["ALL_SUITES"] = {
            "status": "COMBINED",
            "overall_success_rate": round(combined_overall_rate, 4),
            "total_successes": combined_total_succ,
            "total_episodes": combined_total_ep,
            "num_tasks_completed": combined_total_tasks,
            "num_tasks_expected": combined_total_expected,
            "suites_included": contributing_suites,
            "per_category": {
                cat: {
                    "success_rate": round(combined_rates[cat], 4),
                    "successes": combined_successes[cat],
                    "episodes": combined_episodes[cat],
                    "num_tasks_completed": combined_tasks[cat],
                    "num_tasks_expected": combined_expected.get(
                        cat, combined_tasks[cat]
                    ),
                }
                for cat in sorted(combined_rates.keys())
            },
        }

    # Write CSV
    csv_path = output_path / "disturbance_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "category", "success_rate", "successes", "episodes",
                         "completed", "expected", "status"])
        writer.writerows(csv_rows)

    # Write JSON
    json_path = output_path / "disturbance_summary.json"
    json_path.write_text(
        json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize LIBERO-plus results by perturbation category."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Root evaluation output directory (contains suite subdirs with result JSONs).",
    )
    args = parser.parse_args()
    summarize_by_disturbance(args.output_dir)


if __name__ == "__main__":
    main()
