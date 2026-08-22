"""Summarize background/distractor evaluation results.

The script prints three tables:

1. Per-model, per-task metrics plus an aggregate row for each model.
2. Per-model metrics aggregated over all tasks, grouped by background.
3. Per-model metrics aggregated over all tasks, grouped by object count.

As in ``eval_summary.py``, success is determined over the full trajectory while
position and angle errors are measured at the final recorded timestep.  All
aggregates are computed directly from episodes (rather than by averaging task,
background, or object-count summaries).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Iterable

import h5py

from eval_summary import (
    empty_task_stats,
    load_goal_pose,
    run_result,
    update_summary_stats,
)


RESULTS_ROOT = Path(
    "/anvme/workspace/v106be10-valpa-robolab/.cache/ouput_bgdistractor"
)
DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"

# Edit this list to select tasks. Set it to None to include every task found.
TASKS_FILTER: list[str] | None = [
    # "AngledReachBananaTask",
    "AngledReachCartoonTask",
    "AngledReachDrillTask",
    "AngledReachKetchupTask",
    "AngledReachMacaroniTask",
    # "AngledReachMarkerTask",
]

Row = dict[str, object]
Stats = dict[str, object]



def model_from_experiment(experiment_dir: Path) -> str:
    """Extract the model name from an experiment directory name."""
    match = re.match(r"(?P<model>.+?)_angledreach(?:_|$)", experiment_dir.name)
    return match.group("model") if match else experiment_dir.name


def discover_experiments(results_root: Path, patterns: list[str] | None) -> list[Path]:
    """Find experiment directories containing an episode_results.jsonl file."""
    if patterns:
        candidates: list[Path] = []
        for pattern in patterns:
            path = Path(pattern).expanduser()
            if path.is_dir():
                candidates.append(path)
            else:
                candidates.extend(results_root.glob(pattern))
    else:
        candidates = list(results_root.iterdir()) if results_root.is_dir() else []

    return sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_dir() and (path / "episode_results.jsonl").is_file()
        }
    )


def iter_episode_records(experiment_dir: Path) -> Iterable[tuple[int, dict[str, object]]]:
    results_path = experiment_dir / "episode_results.jsonl"
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def episode_hdf5_path(experiment_dir: Path, episode: dict[str, object]) -> Path:
    env_name = str(episode["env_name"])
    run_number = int(episode.get("run", 0))
    env_dir = experiment_dir / env_name
    run_path = env_dir / f"run_{run_number}.hdf5"
    if run_path.is_file():
        return run_path

    # Some exports retain only the concatenated single-run filename.
    data_path = env_dir / "data.hdf5"
    if data_path.is_file():
        return data_path
    raise FileNotFoundError(f"no run HDF5 found in {env_dir}")

@cache

def goal_pose(task: str, assets_root: Path) -> tuple[list[float], list[float]]:
    """Load a goal pose, accepting an explicitly marked draft when necessary."""
    try:
        return load_goal_pose(task, assets_root)
    except FileNotFoundError:
        draft_path = assets_root / "wm_tasks" / task / "status_draft.json"
        if not draft_path.is_file():
            raise
        with draft_path.open("r", encoding="utf-8") as handle:
            pose = json.load(handle)["last_ee_pose"]
        print(f"Using draft goal pose: {draft_path}", file=sys.stderr)
        return pose[:3], pose[3:7]


def add_result(stats: Stats, result: dict[str, float | int | bool]) -> None:
    stats["total_runs"] += 1
    stats["selected_indices"].append(result["selected_index"])
    stats["position_errors"].append(result["position_error"])
    stats["angle_errors"].append(result["angle_error"])
    if result["successful"]:
        stats["successful_runs"] += 1


def make_row(model: str, group: str, value: object, stats: Stats) -> Row:
    update_summary_stats(stats)
    return {
        "model": model,
        group: value,
        "runs": stats["total_runs"],
        "successes": stats["successful_runs"],
        "success_rate": stats["success_rate"],
        "position_error_mean": stats["position_error_mean"],
        "position_error_std": stats["position_error_std"],
        "angle_error_mean_deg": stats["angle_error_mean"],
        "angle_error_std_deg": stats["angle_error_std"],
    }


def summarize(
    experiment_dirs: list[Path],
    assets_root: Path,
    tasks_filter: set[str] | None = None,
    skip_errors: bool = False,
    verbose_runs: bool = False,
) -> tuple[list[Row], list[Row], list[Row]]:
    by_model_task: dict[tuple[str, str], Stats] = defaultdict(empty_task_stats)
    by_model: dict[str, Stats] = defaultdict(empty_task_stats)
    by_model_background: dict[tuple[str, str], Stats] = defaultdict(empty_task_stats)
    by_model_objects: dict[tuple[str, int], Stats] = defaultdict(empty_task_stats)

    for experiment_dir in experiment_dirs:
        model = model_from_experiment(experiment_dir)
        for line_number, episode in iter_episode_records(experiment_dir):
            try:
                task = str(episode["task_name"])
                if tasks_filter is not None and task not in tasks_filter:
                    continue
                background = str(episode["background"])
                object_count = int(episode["num_tabletop_objects"])
                hdf5_path = episode_hdf5_path(experiment_dir, episode)
                goal_pos, goal_quat = goal_pose(task, assets_root)

                with h5py.File(hdf5_path, "r") as hdf5_file:
                    demo = hdf5_file["data"]["demo_0"]
                    result = run_result(
                        demo["ee_pose"]["position"],
                        demo["ee_pose"]["orientation"],
                        goal_pos,
                        goal_quat,
                    )
            except Exception as error:
                context = f"{experiment_dir / 'episode_results.jsonl'}:{line_number}"
                if skip_errors:
                    print(f"Skipping {context}: {error}")
                    continue
                raise RuntimeError(f"failed to process {context}") from error

            add_result(by_model_task[(model, task)], result)
            add_result(by_model[model], result)
            add_result(by_model_background[(model, background)], result)
            add_result(by_model_objects[(model, object_count)], result)

            if verbose_runs:
                status = "success" if result["successful"] else "failed"
                print(
                    f"{model} | {task} | bg={background} | objects={object_count} | "
                    f"{status} | pos_err={result['position_error']:.6f} | "
                    f"ang_err={result['angle_error']:.3f} deg"
                )

    task_rows: list[Row] = []
    background_rows: list[Row] = []
    object_rows: list[Row] = []
    for model, model_stats in sorted(by_model.items()):
        for (_, task), stats in sorted(
            item for item in by_model_task.items() if item[0][0] == model
        ):
            task_rows.append(make_row(model, "task", task, stats))
        task_rows.append(make_row(model, "task", "ALL_TASKS", model_stats))

        for (_, background), stats in sorted(
            item for item in by_model_background.items() if item[0][0] == model
        ):
            background_rows.append(make_row(model, "background", background, stats))
        background_rows.append(
            make_row(model, "background", "ALL_BACKGROUNDS", model_stats)
        )

        for (_, objects), stats in sorted(
            item for item in by_model_objects.items() if item[0][0] == model
        ):
            object_rows.append(make_row(model, "objects", objects, stats))
        object_rows.append(make_row(model, "objects", "ALL_OBJECT_COUNTS", model_stats))
    return task_rows, background_rows, object_rows


def print_table(title: str, rows: list[Row], group: str) -> None:
    print(f"\n{title}")
    print(
        f"{'Model':<20} {group.title():<32} {'Runs':>5} {'Succ':>5} {'SR':>7} "
        f"{'Pos Mean':>10} {'Pos Std':>10} {'Ang Mean':>10} {'Ang Std':>10}"
    )
    print("-" * 128)
    for row in rows:
        print(
            f"{str(row['model']):<20} {str(row[group]):<32} "
            f"{int(row['runs']):>5} {int(row['successes']):>5} "
            f"{float(row['success_rate']):>7.2%} "
            f"{float(row['position_error_mean']):>10.6f} "
            f"{float(row['position_error_std']):>10.6f} "
            f"{float(row['angle_error_mean_deg']):>10.3f} "
            f"{float(row['angle_error_std_deg']):>10.3f}"
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
        "--experiment",
        action="append",
        default=None,
        help="Experiment directory or glob relative to the hard-coded results root; repeatable.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
        help=f"Root containing wm_tasks/<Task>/status.json (default: {DEFAULT_ASSETS_ROOT})",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task name to include; repeatable. Overrides TASKS_FILTER when supplied.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optionally write task.csv, background.csv, and objects.csv here.",
    )
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--verbose-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dirs = discover_experiments(RESULTS_ROOT, args.experiment)
    if not experiment_dirs:
        raise SystemExit(f"No experiments with episode_results.jsonl found in {RESULTS_ROOT}")

    print(f"Found {len(experiment_dirs)} experiment(s)")
    tasks_filter = set(args.task) if args.task else (
        set(TASKS_FILTER) if TASKS_FILTER is not None else None
    )
    task_rows, background_rows, object_rows = summarize(
        experiment_dirs,
        args.assets_root,
        tasks_filter,
        args.skip_errors,
        args.verbose_runs,
    )

    print_table("Task-wise and aggregate metrics", task_rows, "task")
    print_table("All-task metrics by background", background_rows, "background")
    print_table("All-task metrics by number of objects", object_rows, "objects")

    if args.csv_dir is not None:
        write_csv(task_rows, args.csv_dir / "task.csv")
        write_csv(background_rows, args.csv_dir / "background.csv")
        write_csv(object_rows, args.csv_dir / "objects.csv")
        print(f"\nWrote CSV tables to {args.csv_dir}")


if __name__ == "__main__":
    main()
