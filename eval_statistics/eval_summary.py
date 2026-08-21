"""Summarize pickup position and angle error by model and task.

For each run, this script scans the trajectory to determine whether the success
criteria were ever satisfied. Position and angle errors are always computed at
the final recorded timestep, so the reported means do not depend on which
timestep first crossed the success threshold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
import zipfile

import h5py


DEFAULT_ZIP_GLOB = "/anvme/workspace/v106be10-valpa-robolab/.cache/output_angledreach/*.zip"

DISTANCE_THRESHOLD = 0.05
ANGLE_THRESHOLD_DEGREES = 12.0

TASK_NAME_MAP = {
    "ReachAppleTask": "Apple",
    "ReachBagelTask": "Bagel",
    "ReachBananaTask": "Banana",
    "ReachCeramicMugTask": "Ceramic Mug",
    "ReachCoffeeCanTask": "Coffee Can",
    "ReachCoffeePotTask": "Coffee Pot",
    "ReachOrangeJuiceCartonTask": "Juice Carton",
    "ReachOrangeTask": "Orange",
    "ReachPitcherTask": "White Pitcher",
    "ReachSpoonBigTask": "Large Spoon",
    "ReachYogurtCupTask": "Yogurt Cup",
    "AngledReachBananaTask": "Banana",
    "AngledReachCartoonTask": "Orange Juice Carton 2",
    "AngledReachCartoon2Task": "Orange Juice Carton",
    "AngledReachDrillTask": "Cordless Drill",
    "AngledReachKetchupTask": "Ketchup Bottle",
    "AngledReachMacaroniTask": "Macaroni Carton",
    "AngledReachMarkerTask": "Dry-Erase Marker",
}

MODEL_NAME_MAP  = {
    "right_vjepa": r"SV$_{\mathbf{VJ}}$",
    "right_dinov3": r"SV$_{\mathbf{D3}}$",
    "wrist_vjepa": r"WV$_{\mathbf{VJ}}$",
    "wrist_dinov3": r"WV$_{\mathbf{D3}}$",
    "ind_vjepa": r"DI$_{\mathbf{VJ}}$",
    "ind_dinov3": r"DI$_{\mathbf{D3}}$",
    "dual_vjepa": r"DJ$_{\mathbf{VJ}}$",
    "dual_dinov3": r"DJ$_{\mathbf{D3}}$",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
    suffix = "_pickup"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def load_goal_pose(task: str, assets_root: Path) -> tuple[list[float], list[float]]:
    status_path = assets_root / "wm_tasks" / task / "status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status_data = json.load(handle)

    goal_pose = status_data["last_ee_pose"]
    return goal_pose[:3], goal_pose[3:7]


def run_result(position, orientation, goal_pos, goal_quat) -> dict[str, float | int | bool]:
    """Return success over the trajectory and errors at the final valid index."""
    num_steps = int(position.shape[0])
    if num_steps == 0:
        raise ValueError("empty ee_pose trajectory")

    selected_index = num_steps - 1
    successful = False

    for index in range(num_steps):
        pos_err = euclidean_distance(position[index, :], goal_pos)
        ang_err = quat_angle_error_degrees_wxyz(orientation[index, :], goal_quat)

        if pos_err < DISTANCE_THRESHOLD and ang_err < ANGLE_THRESHOLD_DEGREES:
            successful = True
            # selected_index = index
            break

    # Always report the last recorded pose. The threshold determines success,
    # but it does not determine which pose contributes to the error means.
    selected_pos_err = euclidean_distance(position[selected_index, :], goal_pos)
    selected_ang_err = quat_angle_error_degrees_wxyz(orientation[selected_index, :], goal_quat)

    return {
        "successful": successful,
        "selected_index": selected_index,
        "position_error": selected_pos_err,
        "angle_error": selected_ang_err,
    }


def empty_task_stats() -> dict[str, object]:
    return {
        "total_runs": 0,
        "successful_runs": 0,
        "success_rate": 0.0,
        "selected_indices": [],
        "selected_index_mean": 0.0,
        "selected_index_std": 0.0,
        "position_errors": [],
        "position_error_mean": 0.0,
        "position_error_std": 0.0,
        "angle_errors": [],
        "angle_error_mean": 0.0,
        "angle_error_std": 0.0,
    }


def update_summary_stats(stats: dict[str, object]) -> None:
    stats["success_rate"] = (
        stats["successful_runs"] / stats["total_runs"] if stats["total_runs"] else 0.0
    )
    stats["selected_index_mean"] = mean(stats["selected_indices"])
    stats["selected_index_std"] = sample_std(stats["selected_indices"])
    stats["position_error_mean"] = mean(stats["position_errors"])
    stats["position_error_std"] = sample_std(stats["position_errors"])
    stats["angle_error_mean"] = mean(stats["angle_errors"])
    stats["angle_error_std"] = sample_std(stats["angle_errors"])


def iter_run_files(zip_file: zipfile.ZipFile):
    for item in zip_file.infolist():
        if item.is_dir() or not item.filename.endswith(".hdf5"):
            continue
        parts = item.filename.split("/")
        filename = parts[-1]
        if len(parts) >= 3 and filename.startswith("run_"):
            yield item, parts[1]


def resolve_zip_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    return sorted(dict.fromkeys(paths))


def summarize(
    zip_paths: list[Path],
    assets_root: Path,
    tasks_filter: set[str] | None,
    verbose_runs: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for zip_path in zip_paths:
        if not zip_path.exists():
            print(f"Skipping missing archive: {zip_path}")
            continue

        model_variant = model_variant_from_zip(zip_path)
        tasks_statistics: dict[str, dict[str, object]] = {}
        model_statistics = empty_task_stats()

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            for item, task in iter_run_files(zip_file):
                if tasks_filter is not None and task not in tasks_filter:
                    continue

                if task not in tasks_statistics:
                    tasks_statistics[task] = empty_task_stats()

                goal_pos, goal_quat = load_goal_pose(task, assets_root)

                with zip_file.open(item) as file:
                    with h5py.File(file, "r") as hdf5_file:
                        demo = hdf5_file["data"]["demo_0"]
                        position = demo["ee_pose"]["position"]
                        orientation = demo["ee_pose"]["orientation"]
                        result = run_result(position, orientation, goal_pos, goal_quat)

                stats = tasks_statistics[task]
                stats["total_runs"] += 1
                stats["selected_indices"].append(result["selected_index"])
                stats["position_errors"].append(result["position_error"])
                stats["angle_errors"].append(result["angle_error"])
                if result["successful"]:
                    stats["successful_runs"] += 1

                model_statistics["total_runs"] += 1
                model_statistics["selected_indices"].append(result["selected_index"])
                model_statistics["position_errors"].append(result["position_error"])
                model_statistics["angle_errors"].append(result["angle_error"])
                if result["successful"]:
                    model_statistics["successful_runs"] += 1

                if verbose_runs:
                    status = "success" if result["successful"] else "failed"
                    print(
                        f"{model_variant} | {task} | {item.filename} | {status} | "
                        f"idx={result['selected_index']} | "
                        f"pos_err={result['position_error']:.6f} | "
                        f"ang_err={result['angle_error']:.3f} deg"
                    )

        for task, stats in sorted(tasks_statistics.items()):
            update_summary_stats(stats)
            rows.append(
                {
                    "model": model_variant,
                    "task": task,
                    "runs": stats["total_runs"],
                    "successes": stats["successful_runs"],
                    "success_rate": stats["success_rate"],
                    "selected_index_mean": stats["selected_index_mean"],
                    "selected_index_std": stats["selected_index_std"],
                    "position_error_mean": stats["position_error_mean"],
                    "position_error_std": stats["position_error_std"],
                    "angle_error_mean_deg": stats["angle_error_mean"],
                    "angle_error_std_deg": stats["angle_error_std"],
                }
            )

        if model_statistics["total_runs"]:
            update_summary_stats(model_statistics)
            rows.append(
                {
                    "model": model_variant,
                    "task": "ALL_TASKS",
                    "runs": model_statistics["total_runs"],
                    "successes": model_statistics["successful_runs"],
                    "success_rate": model_statistics["success_rate"],
                    "selected_index_mean": model_statistics["selected_index_mean"],
                    "selected_index_std": model_statistics["selected_index_std"],
                    "position_error_mean": model_statistics["position_error_mean"],
                    "position_error_std": model_statistics["position_error_std"],
                    "angle_error_mean_deg": model_statistics["angle_error_mean"],
                    "angle_error_std_deg": model_statistics["angle_error_std"],
                }
            )

    return rows


def print_table(rows: list[dict[str, object]]) -> None:
    print(
        f"{'Model':<30} {'Task':<32} {'Runs':>4} {'Succ':>4} {'SR':>6} "
        f"{'Pos Mean':>10} {'Pos Std':>10} {'Ang Mean':>10} {'Ang Std':>10}"
    )
    print("-" * 130)
    for row in rows:
        print(
            f"{row['model']:<30} {row['task']:<32} "
            f"{row['runs']:>4} {row['successes']:>4} {row['success_rate']:>6.2f} "
            f"{row['position_error_mean']:>10.6f} {row['position_error_std']:>10.6f} "
            f"{row['angle_error_mean_deg']:>10.3f} {row['angle_error_std_deg']:>10.3f}"
        )


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_success_rate_heatmap(
    rows: list[dict[str, object]],
    output_path: Path,
    task_order: list[str] | None = None,
) -> None:
    """Write a model-by-task heatmap of success rates."""
    if not rows:
        return

    # Import plotting only when requested so the statistics code can still be
    # imported in environments where matplotlib is unavailable.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    available_models = {str(row["model"]) for row in rows}
    models = []
    model_order = [
        "right_vjepa",
        "wrist_vjepa",
        "ind_vjepa",
        "dual_vjepa",
        "right_dinov3",
        "wrist_dinov3",
        "ind_dinov3",
        "dual_dinov3",
    ]
    for model in model_order:
        if model in available_models:
            models.append(model)
    models.extend(sorted(available_models.difference(models)))
    available_tasks = {
        str(row["task"]) for row in rows if row["task"] != "ALL_TASKS"
    }
    if not available_tasks:
        return
    tasks = (
        [task for task in task_order if task in available_tasks]
        if task_order is not None
        else sorted(available_tasks)
    )
    tasks.extend(sorted(available_tasks.difference(tasks)))
    rates = {
        (str(row["model"]), str(row["task"])): float(row["success_rate"])
        for row in rows
        if row["task"] != "ALL_TASKS"
    }
    matrix = [[rates.get((model, task), math.nan) for task in tasks] for model in models]

    figure, axis = plt.subplots(figsize=(1280 / 300, 720 / 300), dpi=300)
    color_map = plt.get_cmap("Greens").copy()
    color_map.set_bad(color="#d9d9d9")
    axis.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect="auto")

    task_labels = [str(index + 1) for index, _ in enumerate(tasks)]
    model_labels = []
    for model in models:
        display_name = MODEL_NAME_MAP.get(model, model)
        model_labels.append(display_name)
    header_style = {
        "facecolor": "#f2f2f2",
        "edgecolor": "#c4c4c4",
        "linewidth": 0.6,
    }
    axis.add_patch(Rectangle((-1.5, -1.5), 1.0, 1.0, **header_style))
    for task_index, task_label in enumerate(task_labels):
        axis.add_patch(Rectangle((task_index - 0.5, -1.5), 1.0, 1.0, **header_style))
        axis.text(
            task_index,
            -1.0,
            task_label,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans",
            fontsize=6.0,
            fontweight="bold",
            clip_on=True,
        )
    for model_index, model_label in enumerate(model_labels):
        axis.add_patch(Rectangle((-1.5, model_index - 0.5), 1.0, 1.0, **header_style))
        axis.text(
            -1.0,
            model_index,
            model_label,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans",
            fontsize=6.5,
            fontweight="bold",
            clip_on=True,
        )

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(-1.5, len(tasks) - 0.5)
    axis.set_ylim(len(models) - 0.5, -1.5)
    for spine in axis.spines.values():
        spine.set_edgecolor("#c4c4c4")
        spine.set_linewidth(0.6)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    for model_index, model in enumerate(models):
        for task_index, task in enumerate(tasks):
            rate = rates.get((model, task))
            label = "N/A" if rate is None else f"{rate:.0%}"
            text_color = "white" if rate is not None and rate >= 0.6 else "black"
            axis.text(task_index, model_index, label, ha="center", va="center", color=text_color, fontsize=4.8, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        dest="zip_patterns",
        action="append",
        default=None,
        help="Zip path or glob. Can be repeated. Default: output_pickup/*.zip",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("assets"),
        help="Root containing wm_tasks/<Task>/status.json. Default: assets",
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=None,
        help="Task name to include. Can be repeated. Default: all tasks found in zips",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("pickup_error_summary.csv"),
        help="CSV output path. Default: pickup_error_summary.csv",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "all_models_success_rate_heatmap.pdf",
        help="Success-rate heatmap output path.",
    )
    parser.add_argument("--verbose-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = args.zip_patterns or [DEFAULT_ZIP_GLOB]
    zip_paths = resolve_zip_paths(patterns)
    # tasks_filter = set(args.tasks) if args.tasks else None
    # tasks_filter = ["ReachCoffeePotTask",
    #                 "ReachCoffeeCanTask",
    #                 # "ReachSpoonBigTask",
    #                 "ReachBananaTask",
    #                 "ReachOrangeJuiceCartonTask",
    #                 "ReachYogurtCupTask",
    #                 "ReachAppleTask",
    #                 "ReachOrangeTask",
    #                 "ReachBagelTask",
    #                 "ReachPitcherTask",
    #                 "ReachCeramicMugTask"
    # ]
    tasks_filter = [
                    # "AngledReachMacaroniTask",
                    "AngledReachBananaTask",
                    "AngledReachDrillTask",
                    # "AngledReachMarkerTask",
                    "AngledReachKetchupTask",
                    "AngledReachCartoon2Task"
                    ]

    print(
        "success thresholds: "
        f"position error < {DISTANCE_THRESHOLD} m, "
        f"angle error < {ANGLE_THRESHOLD_DEGREES} deg"
    )


    # statistics
    rows = summarize(zip_paths, args.assets_root, tasks_filter, args.verbose_runs)
    print_table(rows)

    # write_csv(rows, args.csv)
    # print(f"\nWrote CSV: {args.csv}")

    # headmap of success rates by model and task
    # write_success_rate_heatmap(rows, args.heatmap, tasks_filter)
    # print(f"\nWrote heatmap: {args.heatmap}")


if __name__ == "__main__":
    main()
