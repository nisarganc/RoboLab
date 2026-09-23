"""Summarize final hardware-run pose errors by predictor.

Each experiment directory in the input archive contains one ``goal_pose-1.yaml``
and a set of state/action text logs.  Position and angular errors are computed
between the goal pose and the last recorded ``state: [...]`` in each log.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
import zipfile


DEFAULT_ZIP = Path("/anvme/workspace/v106be10-valpa-robolab/.cache/output_hw/hw_3tasks.zip")
GOAL_FILENAME = "goal_pose-1.yaml"
STATE_RE = re.compile(r"^state:\s*(\[[^\n]+\])", re.MULTILINE)


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


def parse_last_state(text: str, source: str) -> list[float]:
    matches = STATE_RE.findall(text)
    if not matches:
        raise ValueError(f"{source}: no state entries")
    state = ast.literal_eval(matches[-1])
    if not isinstance(state, list) or len(state) < 6:
        raise ValueError(f"{source}: final state must contain at least 6 values")
    return [float(value) for value in state[:6]]


def predictor_name(directory: str) -> str:
    prefix = "videos_pickup_"
    return directory[len(prefix) :] if directory.startswith(prefix) else directory


def summarize(zip_path: Path, verbose_runs: bool = False) -> list[dict[str, object]]:
    grouped_files: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            parts = Path(item.filename).parts
            if len(parts) < 3:
                continue
            directory, filename = parts[-2], parts[-1]
            group = grouped_files.setdefault(directory, {"goal": None, "logs": []})
            if filename == GOAL_FILENAME:
                group["goal"] = item
            elif filename.endswith(".txt"):
                group["logs"].append(item)

        rows: list[dict[str, object]] = []
        for directory, files in sorted(grouped_files.items()):
            goal_item = files["goal"]
            log_items = files["logs"]
            if goal_item is None or not log_items:
                continue

            goal_text = archive.read(goal_item).decode("utf-8")
            goal_pose = parse_goal_pose(goal_text, goal_item.filename)
            position_errors: list[float] = []
            angle_errors: list[float] = []

            for log_item in sorted(log_items, key=lambda item: item.filename):
                log_text = archive.read(log_item).decode("utf-8")
                final_pose = parse_last_state(log_text, log_item.filename)
                position_error = euclidean_distance(final_pose[:3], goal_pose[:3])
                angle_error = angular_error_degrees(final_pose[3:6], goal_pose[3:6])
                position_errors.append(position_error)
                angle_errors.append(angle_error)
                if verbose_runs:
                    print(
                        f"{predictor_name(directory)} | {Path(log_item.filename).name} | "
                        f"pos_err={position_error:.6f} m | ang_err={angle_error:.3f} deg"
                    )

            rows.append(
                {
                    "predictor": predictor_name(directory),
                    "runs": len(position_errors),
                    "position_error_mean": mean(position_errors),
                    "position_error_std": sample_std(position_errors),
                    "angle_error_mean_deg": mean(angle_errors),
                    "angle_error_std_deg": sample_std(angle_errors),
                }
            )
    return rows


def print_table(rows: list[dict[str, object]]) -> None:
    print(
        f"{'Predictor':<12} {'Runs':>4} {'Pos Mean (m)':>14} {'Pos Std (m)':>13} "
        f"{'Ang Mean (deg)':>15} {'Ang Std (deg)':>14}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['predictor']:<12} {row['runs']:>4} "
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
    print_table(summarize(args.zip, args.verbose_runs))


if __name__ == "__main__":
    main()
