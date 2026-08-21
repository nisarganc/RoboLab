#!/usr/bin/env python3
"""Summarize angled reach evaluation runs.

Success is computed against the saved goal end-effector pose using the same
quaternion angular-distance check as ``angled_reach_object``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import zipfile

import h5py


ZIP_PATHS = [
    Path("output_angledreach/dual_dinov3_roboarena_angledreach.zip"),
    Path("output_angledreach/dual_dinov3_angledreach.zip"),
    Path("output_angledreach/ind_dinov3_angledreach.zip"),
    Path("output_angledreach/right_dinov3_angledreach.zip"),
    Path("output_angledreach/wrist_dinov3_angledreach.zip"),
    Path("output_angledreach/dual_vjepa_angledreach.zip"),
    Path("output_angledreach/ind_vjepa_angledreach.zip"),
    Path("output_angledreach/right_vjepa_angledreach.zip"),
    Path("output_angledreach/wrist_vjepa_angledreach.zip"),
]

ANGLED_REACH_TASKS = [
    "AngledReachKetchupTask",
    "AngledReachDrillTask",
    "AngledReachCartoonTask",
]

DISTANCE_THRESHOLD = 0.05
ANGLE_THRESHOLD_DEGREES = 15.0


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    values_mean = mean(values)
    return math.sqrt(sum((x - values_mean) ** 2 for x in values) / (len(values) - 1))


def euclidean_distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def normalize_quat_wxyz(quat) -> list[float]:
    norm = math.sqrt(sum(float(x) ** 2 for x in quat))
    if norm == 0:
        raise ValueError("zero-length quaternion")
    return [float(x) / norm for x in quat]


def quat_angle_error_degrees_wxyz(current_quat, target_quat) -> float:
    current = normalize_quat_wxyz(current_quat)
    target = normalize_quat_wxyz(target_quat)
    dot = abs(sum(c * t for c, t in zip(current, target)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def model_variant_from_zip(zip_path: Path) -> str:
    name = zip_path.stem
    suffix = "_angledreach"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def load_goal_pose(task: str) -> tuple[list[float], list[float]]:
    status_path = Path("assets") / "wm_tasks" / task / "status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status_data = json.load(handle)

    goal_pose = status_data["last_ee_pose"]
    return goal_pose[:3], goal_pose[3:7]


def empty_task_stats() -> dict[str, object]:
    return {
        "total_runs": 0,
        "successful_runs": 0,
        "success_rate": 0.0,
        "steps": [],
        "steps_mean": 0.0,
        "steps_std": 0.0,
        "path_lengths": [],
        "path_length_mean": 0.0,
        "path_length_std": 0.0,
        "goal_distances": [],
        "goal_distance_mean": 0.0,
        "goal_distance_std": 0.0,
        "angle_errors": [],
        "angle_error_mean": 0.0,
        "angle_error_std": 0.0,
    }


def update_summary_stats(stats: dict[str, object]) -> None:
    stats["success_rate"] = (
        stats["successful_runs"] / stats["total_runs"]
        if stats["total_runs"]
        else 0.0
    )
    stats["steps_mean"] = mean(stats["steps"])
    stats["steps_std"] = sample_std(stats["steps"])
    stats["path_length_mean"] = mean(stats["path_lengths"])
    stats["path_length_std"] = sample_std(stats["path_lengths"])
    stats["goal_distance_mean"] = mean(stats["goal_distances"])
    stats["goal_distance_std"] = sample_std(stats["goal_distances"])
    stats["angle_error_mean"] = mean(stats["angle_errors"])
    stats["angle_error_std"] = sample_std(stats["angle_errors"])


def run_result(position, orientation, goal_pos, goal_quat) -> dict[str, float | int | bool]:
    path_length = 0.0
    first_success = None
    final_distance = 0.0
    final_angle_error = 0.0

    # first_position = position[0, :]
    # print(f"start position: {first_position}, goal position: {goal_pos}")
    # goal_distance = euclidean_distance(first_position, goal_pos)
    # print(f"initial distance to goal: {goal_distance:.4f}")

    for step in range(position.shape[0]):
        current_position = position[step, :]
        current_orientation = orientation[step, :]

        if step > 0:
            path_length += euclidean_distance(current_position, position[step - 1, :])

        distance = euclidean_distance(current_position, goal_pos)
        angle_error = quat_angle_error_degrees_wxyz(current_orientation, goal_quat)
        final_distance = distance
        final_angle_error = angle_error

        if distance < DISTANCE_THRESHOLD and angle_error < ANGLE_THRESHOLD_DEGREES:
            first_success = step
            break

    if first_success is None:
        step = position.shape[0]
    else:
        step = first_success

    return {
        "successful": first_success is not None,
        "step": step,
        "distance": final_distance,
        "angle_error": final_angle_error,
        "path_length": path_length,
    }


def print_task_table(model_variant: str, tasks_statistics: dict[str, dict[str, object]]) -> None:
    print(f"\n{model_variant}")
    print(
        f"{'Task':<32} {'Runs':>4} {'Succ':>4} {'SR':>6} "
        f"{'Step Mean':>10} {'Step Std':>9} "
        f"{'Path Mean':>10} {'Path Std':>9} "
        f"{'Dist Mean':>10} {'Dist Std':>9} "
        f"{'Ang Mean':>9} {'Ang Std':>8}"
    )
    print("-" * 143)
    for task, stats in tasks_statistics.items():
        print(
            f"{task:<32} {stats['total_runs']:>4} {stats['successful_runs']:>4} {stats['success_rate']:>6.2f} "
            f"{stats['steps_mean']:>10.2f} {stats['steps_std']:>9.2f} "
            f"{stats['path_length_mean']:>10.4f} {stats['path_length_std']:>9.4f} "
            f"{stats['goal_distance_mean']:>10.4f} {stats['goal_distance_std']:>9.4f} "
            f"{stats['angle_error_mean']:>9.2f} {stats['angle_error_std']:>8.2f}"
        )

    total_runs = sum(stats["total_runs"] for stats in tasks_statistics.values())
    successful_runs = sum(stats["successful_runs"] for stats in tasks_statistics.values())
    total_success_rate = successful_runs / total_runs if total_runs else 0.0
    print(f"total success rate: {total_success_rate:.2f}")


def main() -> None:
    print(
        "success thresholds: "
        f"distance < {DISTANCE_THRESHOLD} m, angle deviation < {ANGLE_THRESHOLD_DEGREES} deg"
    )

    for zip_path in ZIP_PATHS:
        if not zip_path.exists():
            print(f"\nSkipping missing archive: {zip_path}")
            continue

        model_variant = model_variant_from_zip(zip_path)
        tasks_statistics = {task: empty_task_stats() for task in ANGLED_REACH_TASKS}

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            for item in zip_file.infolist():
                if item.is_dir() or not item.filename.endswith(".hdf5"):
                    continue

                parts = item.filename.split("/")
                filename = parts[-1]
                if len(parts) < 3 or parts[1] not in tasks_statistics or not filename.startswith("run_"):
                    continue

                task = parts[1]
                goal_pos, goal_quat = load_goal_pose(task)

                with zip_file.open(item) as file:
                    with h5py.File(file, "r") as hdf5_file:
                        demo = hdf5_file["data"]["demo_0"]
                        position = demo["ee_pose"]["position"]
                        orientation = demo["ee_pose"]["orientation"]
                        result = run_result(position, orientation, goal_pos, goal_quat)

                stats = tasks_statistics[task]
                stats["total_runs"] += 1
                stats["steps"].append(result["step"])
                stats["goal_distances"].append(result["distance"])
                stats["angle_errors"].append(result["angle_error"])
                stats["path_lengths"].append(result["path_length"])

                if result["successful"]:
                    stats["successful_runs"] += 1
                    print(
                        f"file {item.filename}: reached goal at step {result['step']}, "
                        f"distance: {result['distance']:.4f}, angle: {result['angle_error']:.2f} deg"
                    )
                else:
                    print(
                        f"file {item.filename}: did not reach goal, "
                        f"final distance: {result['distance']:.4f}, "
                        f"final angle: {result['angle_error']:.2f} deg"
                    )

        for stats in tasks_statistics.values():
            update_summary_stats(stats)

        print_task_table(model_variant, tasks_statistics)


if __name__ == "__main__":
    main()
