"""Report three-phase pose errors and success rates for angled pickup runs.

Each 130-step trajectory is evaluated at the end of its three fixed phases:
reach (index 59), grasp (index 69), and lift (index 129).  The corresponding
goals are ``last_ee_pose``, ``last_ee_pose_2``, and ``last_ee_pose_3`` from the
task's status.json.  Angled-reach success uses the same thresholds as
eval_summary.py and is true when they are jointly met at any of the first 60
steps.  Error aggregates include every run, whether successful or not.  Lift
success additionally requires the target object to remain held at the end.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path, PurePosixPath
import zipfile

import h5py

from eval_summary import (
    euclidean_distance,
    mean,
    quat_angle_error_degrees_wxyz,
    sample_std,
)
ANGLE_THRESHOLD_DEGREES = 20.0
DISTANCE_THRESHOLD = 0.15

FINAL_DISTANCE_THRESHOLD = 0.20
FINAL_ANGLE_THRESHOLD_DEGREES = 350.0

# Maximum allowed change in the target object's position relative to the end
# effector over the lift phase.
MAX_OBJECT_EE_RELATIVE_MOTION = 0.2

RESULTS_ROOT = Path(
    "/anvme/workspace/v106be10-valpa-robolab/.cache/output_angledpickup"
)
DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"

# Edit this list to select tasks. Set it to None to include every task found.
TASKS_FILTER: list[str] | None = [
    # "AngledPickupBananaTask",
    "AngledPickupKetchupTask",
    "AngledPickupLemonTask",
    "AngledPickupLizardFigurineTask",
    "AngledPickupSoftScrubBottleTask",
]

PHASES = (
    ("reach", 59, "last_ee_pose"),
    ("grasp", 69, "last_ee_pose_2"),
    ("lift", 129, "last_ee_pose_3"),
)

# SEMANTIC_SR = {
#     "dual_dinov3_angledpickup": {
#         "AngledPickupKetchupTask": {
#             0: (True, True),
#             1: (True, True),
#             2: (True, True),
#             3: (True, False),
#         },
#         "AngledPickupLemonTask": {},
#         "AngledPickupLizardFigurineTask": {},
#         "AngledPickupSoftScrubBottleTask": {},
#     },
#     "ind_dinov3_angledpickup": {},
#     "wrist_dinov3_angledpickup": {}
# }

Row = dict[str, object]
Stats = dict[str, object]


def model_from_zip(zip_path: Path) -> str:
    suffix = "_angledpickup"
    return zip_path.stem[: -len(suffix)] if zip_path.stem.endswith(suffix) else zip_path.stem


def resolve_zip_paths(patterns: list[str] | None) -> list[Path]:
    if patterns is None:
        return sorted(RESULTS_ROOT.glob("*.zip"))

    paths: list[Path] = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if path.is_file():
            paths.append(path)
        else:
            paths.extend(Path(item) for item in glob.glob(str(RESULTS_ROOT / pattern)))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def iter_run_files(zip_file: zipfile.ZipFile):
    for item in zip_file.infolist():
        if item.is_dir() or not item.filename.endswith(".hdf5"):
            continue
        parts = item.filename.split("/")
        if len(parts) >= 3 and parts[-1].startswith("run_"):
            yield item, parts[-2]


def load_phase_goals(task: str, assets_root: Path) -> dict[str, tuple[list[float], list[float]]]:
    status_path = assets_root / "wm_tasks" / task / "status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)

    goals: dict[str, tuple[list[float], list[float]]] = {}
    for phase, _, pose_key in PHASES:
        pose = status[pose_key]
        goals[phase] = (pose[:3], pose[3:7])
    return goals


def load_target_object(
    zip_file: zipfile.ZipFile,
    run_item: zipfile.ZipInfo,
) -> str:
    """Read the target rigid-object name from the cached task configuration."""
    config_name = str(PurePosixPath(run_item.filename).with_name("env_cfg.json"))
    try:
        config = json.loads(zip_file.read(config_name))
        target_object = config["goal"]["object"]
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read goal.object from {config_name}"
        ) from error
    if not isinstance(target_object, str) or not target_object:
        raise ValueError(f"invalid goal.object in {config_name}: {target_object!r}")
    return target_object


def load_object_position(demo: h5py.Group, target_object: str):
    """Return per-step target positions, preferring the full rigid pose."""
    rigid_objects = demo["states"]["rigid_object"]
    if target_object in rigid_objects:
        return rigid_objects[target_object]["root_pose"][:, :3]

    # Older recordings may contain bbox centroids but not full scene state.
    centroids = demo["bbox"]["centroid"]
    if target_object in centroids:
        return centroids[target_object][:, :3]
    raise KeyError(f"target object {target_object!r} is absent from the recording")


def object_held_at_end(object_position, ee_position) -> tuple[bool, float, float]:
    """Infer a final grasp from lift motion and stable object-to-EE coupling."""
    grasp_endpoint = PHASES[1][1]
    lift_endpoint = PHASES[2][1]
    required_steps = lift_endpoint + 1
    if (
        int(object_position.shape[0]) < required_steps
        or int(ee_position.shape[0]) < required_steps
    ):
        raise ValueError(
            f"expected at least {required_steps} object and EE poses, found "
            f"{object_position.shape[0]} and {ee_position.shape[0]}"
        )

    lift_distance = float(
        object_position[lift_endpoint, 2]
        - object_position[grasp_endpoint, 2]
    )

    relative_positions = [
        [
            float(object_position[index, axis] - ee_position[index, axis])
            for axis in range(3)
        ]
        for index in range(grasp_endpoint + 1, lift_endpoint + 1)
    ]
    relative_motion = max(
        euclidean_distance(relative_position, relative_positions[-1])
        for relative_position in relative_positions
    )
    held = relative_motion <= MAX_OBJECT_EE_RELATIVE_MOTION
    return held, lift_distance, relative_motion


def empty_stats() -> Stats:
    return {
        "runs": 0,
        "reach_successes": 0,
        "lift_successes": 0,
        **{
            f"{phase}_{metric}": []
            for phase, _, _ in PHASES
            for metric in ("position_errors", "angle_errors")
        },
    }


def analyze_run(position, orientation, object_position, goals) -> dict[str, object]:
    required_steps = PHASES[-1][1] + 1
    num_steps = int(position.shape[0])
    if num_steps < required_steps or int(orientation.shape[0]) < required_steps:
        raise ValueError(f"expected at least {required_steps} poses, found {num_steps}")

    reach_goal_pos, reach_goal_quat = goals["reach"]
    reach_success = False
    for index in range(PHASES[0][1] + 1):
        position_error = euclidean_distance(position[index, :], reach_goal_pos)
        angle_error = quat_angle_error_degrees_wxyz(
            orientation[index, :], reach_goal_quat
        )
        if (
            position_error < DISTANCE_THRESHOLD
            and angle_error < ANGLE_THRESHOLD_DEGREES
        ):
            reach_success = True
            break

    result: dict[str, object] = {"reach_success": reach_success}
    for phase, endpoint, _ in PHASES:
        goal_pos, goal_quat = goals[phase]
        result[f"{phase}_position_error"] = euclidean_distance(
            position[endpoint, :], goal_pos
        )
        result[f"{phase}_angle_error"] = quat_angle_error_degrees_wxyz(
            orientation[endpoint, :], goal_quat
        )

    object_held, object_lift_distance, object_ee_relative_motion = (
        object_held_at_end(object_position, position)
    )
    result["object_held_at_end"] = object_held
    result["object_lift_distance"] = object_lift_distance
    result["object_ee_relative_motion"] = object_ee_relative_motion
    print(f"Object held at end: {object_held}")
    result["lift_success"] = (
        object_held
        and result["lift_position_error"] < FINAL_DISTANCE_THRESHOLD
        and result["lift_angle_error"] < FINAL_ANGLE_THRESHOLD_DEGREES
    )
    return result


def add_result(stats: Stats, result: dict[str, object]) -> None:
    stats["runs"] += 1
    if result["reach_success"]:
        stats["reach_successes"] += 1
    if result["lift_success"]:
        stats["lift_successes"] += 1
    for phase, _, _ in PHASES:
        stats[f"{phase}_position_errors"].append(
            result[f"{phase}_position_error"]
        )
        stats[f"{phase}_angle_errors"].append(result[f"{phase}_angle_error"])


def make_row(model: str, task: str, stats: Stats) -> Row:
    runs = int(stats["runs"])
    row: Row = {
        "model": model,
        "task": task,
        "runs": runs,
        "reach_successes": stats["reach_successes"],
        "lift_successes": stats["lift_successes"],
        "angled_reach_success_rate": (
            int(stats["reach_successes"]) / runs if runs else 0.0
        ),
        "lift_success_rate": (
            int(stats["lift_successes"]) / runs if runs else 0.0
        ),
    }
    for phase, _, _ in PHASES:
        position_errors = stats[f"{phase}_position_errors"]
        angle_errors = stats[f"{phase}_angle_errors"]
        row[f"{phase}_position_error_mean"] = mean(position_errors)
        row[f"{phase}_position_error_std"] = sample_std(position_errors)
        row[f"{phase}_angle_error_mean_deg"] = mean(angle_errors)
        row[f"{phase}_angle_error_std_deg"] = sample_std(angle_errors)
    return row


def summarize(
    zip_paths: list[Path],
    assets_root: Path,
    tasks_filter: set[str] | None,
    verbose_runs: bool,
) -> list[Row]:
    rows: list[Row] = []
    for zip_path in zip_paths:
        model = model_from_zip(zip_path)
        by_task: dict[str, Stats] = {}
        model_stats = empty_stats()

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            target_objects: dict[str, str] = {}
            for item, task in iter_run_files(zip_file):
                if tasks_filter is not None and task not in tasks_filter:
                    continue
                goals = load_phase_goals(task, assets_root)
                if task not in target_objects:
                    target_objects[task] = load_target_object(zip_file, item)
                target_object = target_objects[task]

                with zip_file.open(item) as run_file:
                    with h5py.File(run_file, "r") as hdf5_file:
                        demo = hdf5_file["data"]["demo_0"]
                        ee_pose = demo["ee_pose"]
                        object_position = load_object_position(demo, target_object)
                        print(run_file.name)
                        result = analyze_run(
                            ee_pose["position"],
                            ee_pose["orientation"],
                            object_position,
                            goals,
                        )

                task_stats = by_task.setdefault(task, empty_stats())
                add_result(task_stats, result)
                add_result(model_stats, result)

                if verbose_runs:
                    phase_text = " | ".join(
                        f"{phase}: pos={result[f'{phase}_position_error']:.6f}, "
                        f"ang={result[f'{phase}_angle_error']:.3f} deg"
                        for phase, _, _ in PHASES
                    )
                    print(
                        f"{model} | {task} | {item.filename} | "
                        f"reach_success={result['reach_success']} | "
                        f"object_held={result['object_held_at_end']} | "
                        f"object_lift={result['object_lift_distance']:.4f} m | "
                        f"relative_motion="
                        f"{result['object_ee_relative_motion']:.4f} m | "
                        f"lift_success={result['lift_success']} | {phase_text}"
                    )

        for task, stats in sorted(by_task.items()):
            rows.append(make_row(model, task, stats))
        if model_stats["runs"]:
            rows.append(make_row(model, "ALL_TASKS", model_stats))
    return rows


def print_table(rows: list[Row]) -> None:
    print(
        f"{'Model':<22} {'Task':<38} {'Runs':>4} {'Reach SR':>9} {'Lift SR':>9} "
        f"{'Reach Pos':>21} {'Reach Ang':>19} "
        f"{'Grasp Pos':>21} {'Grasp Ang':>19} "
        f"{'Lift Pos':>21} {'Lift Ang':>19}"
    )
    print(
        f"{'':<22} {'':<38} {'':>4} {'':>9} {'':>9} "
        + " ".join(
            f"{'mean ± std':>21} {'mean ± std (deg)':>19}" for _ in PHASES
        )
    )
    print("-" * 215)
    for row in rows:
        phase_columns = []
        for phase, _, _ in PHASES:
            phase_columns.append(
                f"{row[f'{phase}_position_error_mean']:.6f} ± "
                f"{row[f'{phase}_position_error_std']:.6f}"
            )
            phase_columns.append(
                f"{row[f'{phase}_angle_error_mean_deg']:.3f} ± "
                f"{row[f'{phase}_angle_error_std_deg']:.3f}"
            )
        print(
            f"{row['model']:<22} {row['task']:<38} {row['runs']:>4} "
            f"{row['angled_reach_success_rate']:>8.2%} "
            f"{row['lift_success_rate']:>8.2%} "
            + " ".join(
                f"{value:>{21 if index % 2 == 0 else 19}}"
                for index, value in enumerate(phase_columns)
            )
        )


def write_csv(rows: list[Row], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        dest="zip_patterns",
        action="append",
        default=None,
        help="Zip path or glob relative to the hard-coded results root; repeatable.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task to include; repeatable. Overrides TASKS_FILTER when supplied.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--verbose-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_paths = resolve_zip_paths(args.zip_patterns)
    if not zip_paths:
        raise SystemExit(f"No zip archives found in {RESULTS_ROOT}")
    tasks_filter = set(args.task) if args.task else (
        set(TASKS_FILTER) if TASKS_FILTER is not None else None
    )

    print(f"Found {len(zip_paths)} archive(s)")
    print(
        "Angled-reach success thresholds: "
        f"position error < {DISTANCE_THRESHOLD} m, "
        f"angle error < {ANGLE_THRESHOLD_DEGREES} deg"
    )
    print(
        "Lift success thresholds: "
        f"position error < {FINAL_DISTANCE_THRESHOLD} m, "
        f"angle error < {FINAL_ANGLE_THRESHOLD_DEGREES} deg, "
        f"object/EE relative motion <= "
        f"{MAX_OBJECT_EE_RELATIVE_MOTION} m over the lift phase"
    )
    rows = summarize(zip_paths, args.assets_root, tasks_filter, args.verbose_runs)
    print_table(rows)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
