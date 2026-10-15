"""Aggregate RoboTwin task results and compare evaluation runs.

Read per-task result files, report success rates, and print a summary table.
Pass multiple directories to compare evaluation runs.

Usage:
    python experiments/robotwin/compile_results.py <results_dir> [<other_dir> ...]"""

import argparse
import json
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMING_BLOCK_RE = re.compile(
    r"Faster-WAM Inference Timing\s+\(ep\s+(?P<episode>\d+),\s+avg over\s+"
    r"(?P<calls>\d+)\s+calls\).*?"
    r"\|\s*Total\s+(?P<total_ms>[0-9]+(?:\.[0-9]+)?)\s*ms\s+100%\s*\|",
    re.DOTALL,
)
TASK_NAME_RE = re.compile(r"Task Name:\s*(?P<task_name>\S+)")
PHASE_RE = re.compile(r"\|\s*fasterwam_policy\s*\|\s*(?P<phase>demo_(?:clean|randomized))\s*\|")

PHASE_NAME_MAP = {
    "demo_clean": "clean",
    "demo_randomized": "random",
}


def _parse_result_file(path: Path) -> float | None:
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None
    try:
        return float(lines[-1])
    except ValueError:
        return None


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _weighted_mean(vals: list[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _, weight in vals)
    if total_weight == 0:
        return None
    return sum(val * weight for val, weight in vals) / total_weight


def _task_name_from_log_path(path: Path) -> str:
    match = re.match(r"^eval_(.+)_\d{8}_\d{6}\.log$", path.name)
    if match:
        return match.group(1)
    return path.stem


def _parse_timing_log(path: Path) -> list[dict]:
    text = ANSI_RE.sub("", path.read_text(errors="ignore"))

    task_match = TASK_NAME_RE.search(text)
    task_name = task_match.group("task_name") if task_match else _task_name_from_log_path(path)

    phase_match = PHASE_RE.search(text)
    phase = PHASE_NAME_MAP.get(phase_match.group("phase"), "unknown") if phase_match else "unknown"

    rows = []
    for match in TIMING_BLOCK_RE.finditer(text):
        rows.append(
            {
                "task_name": task_name,
                "phase": phase,
                "episode": int(match.group("episode")),
                "calls": int(match.group("calls")),
                "total_ms": float(match.group("total_ms")),
                "log_file": path.name,
            }
        )
    return rows


def _summarize_timing_rows(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    per_task: dict[str, list[dict]] = {}
    per_phase: dict[str, list[dict]] = {}
    per_task_phase: dict[tuple[str, str], list[dict]] = {}

    for row in rows:
        per_task.setdefault(row["task_name"], []).append(row)
        per_phase.setdefault(row["phase"], []).append(row)
        per_task_phase.setdefault((row["task_name"], row["phase"]), []).append(row)

    def summarize(group_rows: list[dict]) -> dict:
        totals = [r["total_ms"] for r in group_rows]
        weighted = [(r["total_ms"], r["calls"]) for r in group_rows]
        task_groups: dict[str, list[dict]] = {}
        for r in group_rows:
            task_groups.setdefault(r["task_name"], []).append(r)
        task_equal_call_weighted_vals = [
            _weighted_mean([(r["total_ms"], r["calls"]) for r in task_rows])
            for task_rows in task_groups.values()
        ]
        task_equal_call_weighted_vals = [
            val for val in task_equal_call_weighted_vals if val is not None
        ]
        return {
            "tasks": len(task_groups),
            "episodes": len(group_rows),
            "calls": sum(r["calls"] for r in group_rows),
            "episode_mean_total_ms": _mean(totals),
            "call_weighted_total_ms": _weighted_mean(weighted),
            "task_equal_call_weighted_total_ms": _mean(task_equal_call_weighted_vals),
        }

    phases = {phase: summarize(phase_rows) for phase, phase_rows in sorted(per_phase.items())}
    tasks = {}
    for task_name, task_rows in sorted(per_task.items()):
        tasks[task_name] = summarize(task_rows)
        phase_rows = {
            phase: summarize(rows_for_phase)
            for (task, phase), rows_for_phase in sorted(per_task_phase.items())
            if task == task_name
        }
        tasks[task_name]["phases"] = phase_rows

    return {
        "overall": summarize(rows),
        "phases": phases,
        "tasks": tasks,
    }


def compile_timing(results_dir: Path) -> dict | None:
    rows = []
    for log_path in sorted(results_dir.glob("eval_*.log")):
        rows.extend(_parse_timing_log(log_path))
    return _summarize_timing_rows(rows)


def compile_single(results_dir: Path) -> dict:
    task_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith("_")],
        key=lambda d: d.name,
    )

    phases = set()
    results: dict[str, dict[str, float | None]] = {}

    for task_dir in task_dirs:
        task_name = task_dir.name
        results[task_name] = {}
        for result_file in sorted(task_dir.glob("_result_*.txt")):
            phase = result_file.stem.replace("_result_", "")
            phases.add(phase)
            results[task_name][phase] = _parse_result_file(result_file)

    phases_sorted = sorted(phases)

    for task_name in results:
        for phase in phases_sorted:
            results[task_name].setdefault(phase, None)

    completed = {}
    means = {}
    for phase in phases_sorted:
        vals = [results[t][phase] for t in results if results[t][phase] is not None]
        means[phase] = sum(vals) / len(vals) if vals else None
        completed[phase] = len(vals)

    return {
        "dir": str(results_dir),
        "phases": phases_sorted,
        "results": results,
        "means": means,
        "completed": completed,
        "total_tasks": len(results),
        "timing": compile_timing(results_dir),
    }


def print_table(data: dict) -> None:
    phases = data["phases"]
    results = data["results"]
    means = data["means"]
    completed = data["completed"]

    print(f"\n{'=' * 70}")
    print(f"  {data['dir']}")
    print(f"{'=' * 70}")

    header = f"{'Task':<35}" + "".join(f"{p:>15}" for p in phases)
    print(header)
    print("-" * len(header))

    for task_name in sorted(results.keys()):
        row = f"{task_name:<35}"
        for phase in phases:
            val = results[task_name].get(phase)
            if val is not None:
                row += f"{val * 100:>14.1f}%"
            else:
                row += f"{'—':>15}"
        print(row)

    print("-" * len(header))
    row = f"{'OVERALL':<35}"
    for phase in phases:
        m = means[phase]
        c = completed[phase]
        total = data["total_tasks"]
        if m is not None:
            row += f"{m * 100:>10.1f}% ({c}/{total})"
        else:
            row += f"{'—':>15}"
    print(row)
    print()

    print_timing_summary(data)


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2f} ms ({val / 1000:.4f} s)"


def print_timing_summary(data: dict) -> None:
    timing = data.get("timing")
    if not timing:
        return

    overall = timing["overall"]
    print("Inference timing:")
    print(f"  tasks={overall['tasks']} episodes={overall['episodes']} calls={overall['calls']}")
    print(f"  task-equal call-weighted avg: {_fmt_ms(overall['task_equal_call_weighted_total_ms'])}")
    print(f"  call-weighted avg:            {_fmt_ms(overall['call_weighted_total_ms'])}")
    print(f"  episode avg:                  {_fmt_ms(overall['episode_mean_total_ms'])}")
    for phase, phase_timing in timing["phases"].items():
        print(
            f"  {phase:<6} task-equal call-weighted: "
            f"{_fmt_ms(phase_timing['task_equal_call_weighted_total_ms'])}"
        )
    print()


def _fmt_val(val: float | None, w: int) -> str:
    if val is not None:
        return f"{val:>{w - 1}.1f}%"
    return f"{'—':>{w}}"


def _fmt_delta(a: float | None, b: float | None, w: int) -> str:
    if a is not None and b is not None:
        d = a - b
        sign = "+" if d > 0 else ""
        s = f"{sign}{d:.1f}%"
        return f"{s:>{w}}"
    return f"{'—':>{w}}"


def print_runs_comparison(all_data: list[dict]) -> None:
    d1 = all_data[0]
    d2 = all_data[1]
    phases = d1["phases"]
    r1 = d1["results"]
    r2 = d2["results"]

    all_tasks = sorted(set(list(r1.keys()) + list(r2.keys())))

    col_w = 9
    sep = "  |"

    print(f"\n{'=' * 100}")
    print(f"  R1: {d1['dir']}")
    print(f"  R2: {d2['dir']}")
    print(f"{'=' * 100}")

    col_labels = ["R1", "R2", "R2-R1"]

    phase_header = f"{'':<30}"
    header = f"{'Task':<30}"
    for i, phase in enumerate(phases):
        phase_label = f"--- {phase} ---"
        phase_header += f"{phase_label:^{col_w * len(col_labels)}}"
        for cl in col_labels:
            header += f"{cl:>{col_w}}"
        if i < len(phases) - 1:
            phase_header += sep
            header += sep
    print(phase_header)
    print(header)
    print("-" * len(header))

    for task_name in all_tasks:
        row = f"{task_name:<30}"
        for i, phase in enumerate(phases):
            v1_raw = r1.get(task_name, {}).get(phase)
            v2_raw = r2.get(task_name, {}).get(phase)
            v1 = v1_raw * 100 if v1_raw is not None else None
            v2 = v2_raw * 100 if v2_raw is not None else None
            row += _fmt_val(v1, col_w)
            row += _fmt_val(v2, col_w)
            row += _fmt_delta(v2, v1, col_w)
            if i < len(phases) - 1:
                row += sep
        print(row)

    print("-" * len(header))

    row = f"{'OVERALL':<30}"
    for i, phase in enumerate(phases):
        m1 = d1["means"].get(phase)
        m2 = d2["means"].get(phase)
        m1_pct = m1 * 100 if m1 is not None else None
        m2_pct = m2 * 100 if m2 is not None else None
        row += _fmt_val(m1_pct, col_w)
        row += _fmt_val(m2_pct, col_w)
        row += _fmt_delta(m2_pct, m1_pct, col_w)
        if i < len(phases) - 1:
            row += sep
    print(row)

    row = f"{'COMPLETED':<30}"
    for i, phase in enumerate(phases):
        c1 = d1["completed"].get(phase, 0)
        t1 = d1["total_tasks"]
        c2 = d2["completed"].get(phase, 0)
        t2 = d2["total_tasks"]
        row += f"{f'{c1}/{t1}':>{col_w}}"
        row += f"{f'{c2}/{t2}':>{col_w}}"
        row += f"{'':>{col_w}}"
        if i < len(phases) - 1:
            row += sep
    print(row)
    print()


def print_all_timing_summaries(all_data: list[dict]) -> None:
    if not any(data.get("timing") for data in all_data):
        return
    print(f"\n{'=' * 70}")
    print("  Inference Timing Summary")
    print(f"{'=' * 70}")
    for data in all_data:
        print(f"{data['dir']}")
        print_timing_summary(data)


def main():
    parser = argparse.ArgumentParser(description="Compile RoboTwin eval results")
    parser.add_argument("dirs", nargs="+", type=Path, help="Result directories to compile")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    all_data = []
    for d in args.dirs:
        if not d.exists():
            print(f"Warning: {d} does not exist, skipping", file=sys.stderr)
            continue
        data = compile_single(d)
        all_data.append(data)

    if args.json:
        print(json.dumps(all_data, indent=2, default=str))
    elif len(all_data) >= 2:
        print_runs_comparison(all_data)
        print_all_timing_summaries(all_data)
    else:
        for data in all_data:
            print_table(data)


if __name__ == "__main__":
    main()
