"""Summarize final hardware-run pose errors by task and model.

The archive layout is ``<root>/<task>/<model directory>/<files>``.  Each model
directory contains a goal-pose YAML file and a set of state/action text logs.
Position and angular errors are computed between the goal pose and the last
recorded ``state: [...]`` in each log.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
import zipfile


DEFAULT_ZIP = Path("/anvme/workspace/v106be10-valpa-robolab/.cache/hw_4tasks"
".zip")
GOAL_FILENAMES = {"goal_pose.yaml", "goal_pose-1.yaml"}
STATE_RE = re.compile(r"^state:\s*(\[[^\n]+\])", re.MULTILINE)
DISTANCE_THRESHOLD = 0.05
ANGLE_THRESHOLD_DEGREES = 12.0
SUCCESS_ONLY_ON_LAST_STATE = False


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    values_mean = mean(values)
    return math.sqrt(sum((value - values_mean) ** 2 for value in values) / (len(values) - 1))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rpy_to_quat_wxyz(rpy: list[float]) -> list[float]:
    """Convert fixed-axis roll, pitch, yaw (radians) to a WXYZ quaternion."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def quat_angle_error_degrees_wxyz(current: list[float], target: list[float]) -> float:
    dot = abs(sum(current_value * target_value for current_value, target_value in zip(current, target)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def angular_error_degrees(current_rpy: list[float], target_rpy: list[float]) -> float:
    return quat_angle_error_degrees_wxyz(
        rpy_to_quat_wxyz(current_rpy), rpy_to_quat_wxyz(target_rpy)
    )


def parse_goal_pose(text: str, source: str) -> list[float]:
    """Parse the six scalar entries beneath GOAL_POSE without a YAML dependency."""
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "GOAL_POSE:")
    except StopIteration as error:
        raise ValueError(f"{source}: missing GOAL_POSE") from error

    values: list[float] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("-"):
            if values:
                break
            continue
        values.append(float(stripped[1:].strip()))

    if len(values) != 6:
        raise ValueError(f"{source}: expected 6 GOAL_POSE values, found {len(values)}")
    return values


def parse_states(text: str, source: str) -> list[list[float]]:
    matches = STATE_RE.findall(text)
    if not matches:
        raise ValueError(f"{source}: no state entries")

    states: list[list[float]] = []
    for match in matches:
        state = ast.literal_eval(match)
        if not isinstance(state, list) or len(state) < 6:
            raise ValueError(f"{source}: each state must contain at least 6 values")
        states.append([float(value) for value in state[:6]])
    return states


def model_name(directory: str, task: str) -> str:
    prefix = f"videos_{task}_"
    return directory[len(prefix) :] if directory.startswith(prefix) else directory


def metrics_row(
    task: str,
    model: str,
    position_errors: list[float],
    angle_errors: list[float],
    successes: int,
) -> dict[str, object]:
    runs = len(position_errors)
    return {
        "task": task,
        "model": model,
        "runs": runs,
        "successes": successes,
        "success_rate": successes / runs if runs else 0.0,
        "position_error_mean": mean(position_errors),
        "position_error_std": sample_std(position_errors),
        "angle_error_mean_deg": mean(angle_errors),
        "angle_error_std_deg": sample_std(angle_errors),
    }


def summarize(
    zip_path: Path,
    tasks_filter: list[str] | None = None,
    verbose_runs: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped_files: dict[tuple[str, str], dict[str, object]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            parts = Path(item.filename).parts
            if len(parts) < 3:
                continue
            task, directory, filename = parts[-3], parts[-2], parts[-1]
            if tasks_filter is not None and task not in tasks_filter:
                continue
            model = model_name(directory, task)
            group = grouped_files.setdefault((task, model), {"goal": None, "logs": []})
            if filename in GOAL_FILENAMES:
                group["goal"] = item
            elif filename.endswith(".txt"):
                group["logs"].append(item)

        task_rows: list[dict[str, object]] = []
        all_results: dict[str, dict[str, object]] = {}
        for (task, model), files in sorted(grouped_files.items()):
            goal_item = files["goal"]
            log_items = files["logs"]
            if goal_item is None or not log_items:
                continue

            goal_text = archive.read(goal_item).decode("utf-8")
            goal_pose = parse_goal_pose(goal_text, goal_item.filename)
            position_errors: list[float] = []
            angle_errors: list[float] = []
            successes = 0

            for log_item in sorted(log_items, key=lambda item: item.filename):
                log_text = archive.read(log_item).decode("utf-8")
                states = parse_states(log_text, log_item.filename)
                final_pose = states[-1]
                position_error = euclidean_distance(final_pose[:3], goal_pose[:3])
                angle_error = angular_error_degrees(final_pose[3:6], goal_pose[3:6])
                states_to_check = [final_pose] if SUCCESS_ONLY_ON_LAST_STATE else states
                successful = any(
                    euclidean_distance(state[:3], goal_pose[:3]) < DISTANCE_THRESHOLD
                    and angular_error_degrees(state[3:6], goal_pose[3:6])
                    < ANGLE_THRESHOLD_DEGREES
                    for state in states_to_check
                )
                position_errors.append(position_error)
                angle_errors.append(angle_error)
                if successful:
                    successes += 1
                if verbose_runs:
                    status = "success" if successful else "failed"
                    print(
                        f"{task} | {model} | {Path(log_item.filename).name} | {status} | "
                        f"pos_err={position_error:.6f} m | ang_err={angle_error:.3f} deg"
                    )

            task_rows.append(
                metrics_row(task, model, position_errors, angle_errors, successes)
            )
            model_results = all_results.setdefault(
                model, {"position": [], "angle": [], "successes": 0}
            )
            model_results["position"].extend(position_errors)
            model_results["angle"].extend(angle_errors)
            model_results["successes"] += successes

    overall_rows = [
        metrics_row(
            "all tasks",
            model,
            results["position"],
            results["angle"],
            results["successes"],
        )
        for model, results in sorted(all_results.items())
    ]
    return task_rows, overall_rows


def print_table(rows: list[dict[str, object]]) -> None:
    print(
        f"{'Model':<12} {'Runs':>4} {'Succ':>4} {'SR':>6} "
        f"{'Pos Mean (m)':>14} {'Pos Std (m)':>13} "
        f"{'Ang Mean (deg)':>15} {'Ang Std (deg)':>14}"
    )
    print("-" * 90)
    for row in rows:
        print(
            f"{row['model']:<12} {row['runs']:>4} "
            f"{row['successes']:>4} {row['success_rate']:>6.2f} "
            f"{row['position_error_mean']:>14.6f} {row['position_error_std']:>13.6f} "
            f"{row['angle_error_mean_deg']:>15.3f} {row['angle_error_std_deg']:>14.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize final position and angular errors in hardware state/action logs."
    )
    parser.add_argument(
        "--zip", type=Path, default=DEFAULT_ZIP, help=f"Input archive (default: {DEFAULT_ZIP})"
    )
    parser.add_argument("--verbose-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_filter = [
        "bottle",
        # "flower",
        "fruit",
        "spoon",
    ]
    print(
        "success thresholds: "
        f"position error < {DISTANCE_THRESHOLD} m, "
        f"angle error < {ANGLE_THRESHOLD_DEGREES} deg"
    )
    print(
        "success evaluation: "
        f"{'last state only' if SUCCESS_ONLY_ON_LAST_STATE else 'any state'}"
    )
    task_rows, overall_rows = summarize(args.zip, tasks_filter, args.verbose_runs)
    tasks = sorted({str(row["task"]) for row in task_rows})
    for task in tasks:
        print(f"\nTask: {task}")
        print_table([row for row in task_rows if row["task"] == task])

    print("\nAll tasks")
    print_table(overall_rows)


if __name__ == "__main__":
    main()
