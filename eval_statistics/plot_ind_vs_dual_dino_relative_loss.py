#!/usr/bin/env python3
"""Compare combined relative loss for dual and independent DINO.

For every selected task/run and episode step, this script flattens all CEM
iterations and candidate actions, then averages
``combined_relative_to_zero_action``. It plots the mean across task/run
trajectories with one population-standard-deviation band at each episode step
for dual-DINOv3 and independent-DINOv3.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


RESULTS_ROOT = Path(
    "/anvme/workspace/v106be10-valpa-robolab/.cache/output_angledpickup"
)
MODEL_ARCHIVES = {
    "dual_dinov3": RESULTS_ROOT / "dual_dinov3_angledpickup.zip",
    "ind_dinov3": RESULTS_ROOT / "ind_dinov3_angledpickup.zip",
}
MODEL_LABELS = {
    "dual_dinov3": "Dual-DINOv3",
    "ind_dinov3": "Ind-DINOv3",
}
DEFAULT_OUTPUT = Path(__file__).with_name(
    "ind_vs_dual_dino_combined_relative_loss.png"
)

# Edit these defaults or use --task, --run, and --num-steps.
TASKS_FILTER: list[str] | None = [
    "AngledPickupKetchupTask",
    # "AngledPickupLemonTask",
    # "AngledPickupLizardFigurineTask",
    "AngledPickupSoftScrubBottleTask",
]
RUNS_FILTER: list[int] | None = None
NUM_STEPS = 130
LOSS_NAME = "combined_relative_to_zero_action"
METRICS_PATTERN = re.compile(r"^metrics_(\d+)_0\.pt$")


@dataclass
class RunLoss:
    task: str
    run: int
    values: torch.Tensor


def normalize_task_name(task: str) -> str:
    return task[:-4] if task.endswith("Task") else task


def load_metrics(data: bytes) -> dict:
    try:
        return torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(io.BytesIO(data), map_location="cpu")


def find_metric_members(
    archive: zipfile.ZipFile,
    tasks: set[str] | None,
    runs: set[int] | None,
) -> list[tuple[str, int, str]]:
    matches: list[tuple[str, int, str]] = []
    for item in archive.infolist():
        if item.is_dir():
            continue
        path = PurePosixPath(item.filename)
        match = METRICS_PATTERN.fullmatch(path.name)
        if match is None or len(path.parts) < 3:
            continue
        task = normalize_task_name(path.parent.name)
        run = int(match.group(1))
        if tasks is not None and task not in tasks:
            continue
        if runs is not None and run not in runs:
            continue
        matches.append((task, run, item.filename))
    return sorted(matches)


def extract_run_loss(payload: dict, num_steps: int) -> torch.Tensor:
    names = list(payload.get("candidate_loss_names", []))
    try:
        loss_index = names.index(LOSS_NAME)
    except ValueError as error:
        raise KeyError(f"{LOSS_NAME!r} is absent from {names}") from error

    losses = payload.get("candidate_losses")
    if not isinstance(losses, torch.Tensor) or losses.ndim != 4:
        shape = None if losses is None else tuple(losses.shape)
        raise ValueError(
            "expected candidate_losses [step, CEM, candidate, metric], "
            f"found {shape}"
        )
    if losses.shape[1] < 1 or losses.shape[2] < 1:
        raise ValueError(f"empty CEM/candidate dimensions: {tuple(losses.shape)}")

    # Flatten every CEM iteration and candidate (normally 2 * 800), then take
    # their mean at each episode step.
    all_cem_candidates = losses[:num_steps, :, :, loss_index].float()
    return all_cem_candidates.flatten(1).mean(dim=-1)


def collect_model_runs(
    archive_path: Path,
    tasks: set[str] | None,
    runs: set[int] | None,
    matched_pairs: set[tuple[str, int]],
    num_steps: int,
    skip_errors: bool,
) -> list[RunLoss]:
    collected: list[RunLoss] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = find_metric_members(archive, tasks, runs)
        for task, run, member in members:
            if (task, run) not in matched_pairs:
                continue
            try:
                values = extract_run_loss(
                    load_metrics(archive.read(member)), num_steps
                )
                if not torch.isfinite(values).any():
                    raise ValueError("trajectory contains no finite loss values")
            except Exception as error:
                if skip_errors:
                    print(f"Skipping {member}: {error}", file=sys.stderr)
                    continue
                raise RuntimeError(f"failed to process {member}") from error
            collected.append(RunLoss(task, run, values.cpu()))
            finite_values = values[torch.isfinite(values)]
            print(
                f"  {task} run {run}: {values.numel()} steps, "
                f"mean={finite_values.mean().item():.6g}"
            )
    return collected


def aggregate_steps(
    runs: list[RunLoss],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum_steps = max(run.values.numel() for run in runs)
    stacked = torch.full((len(runs), maximum_steps), float("nan"))
    for index, run in enumerate(runs):
        stacked[index, : run.values.numel()] = run.values

    finite = torch.isfinite(stacked)
    counts = finite.sum(dim=0)
    safe_values = torch.where(finite, stacked, torch.zeros_like(stacked))
    means = safe_values.sum(dim=0) / counts.clamp_min(1)
    differences = torch.where(
        finite,
        stacked - means.unsqueeze(0),
        torch.zeros_like(stacked),
    )
    standard_deviations = (
        differences.square().sum(dim=0) / counts.clamp_min(1)
    ).sqrt()
    valid_steps = counts > 0
    return means[valid_steps], standard_deviations[valid_steps], counts[valid_steps]


def print_summary(results: dict[str, list[RunLoss]]) -> None:
    print("\nAll-CEM mean combined_relative_to_zero_action")
    print(f"{'Model':<14} {'Mean':>12} {'Std':>12} {'Runs':>8} {'Values':>10}")
    print("-" * 62)
    for model_name in MODEL_ARCHIVES:
        runs = results[model_name]
        values = torch.cat([run.values for run in runs])
        values = values[torch.isfinite(values)]
        print(
            f"{MODEL_LABELS[model_name]:<14} "
            f"{values.mean().item():>12.6g} "
            f"{values.std(unbiased=False).item():>12.6g} "
            f"{len(runs):>8} {values.numel():>10}"
        )


def plot_results(
    results: dict[str, list[RunLoss]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6.5))
    colors = {"dual_dinov3": "tab:blue", "ind_dinov3": "tab:orange"}
    for model_name in MODEL_ARCHIVES:
        means, standard_deviations, counts = aggregate_steps(results[model_name])
        steps = torch.arange(means.numel()).numpy()
        mean_values = means.numpy()
        std_values = standard_deviations.numpy()
        label = f"{MODEL_LABELS[model_name]} (n={int(counts.max())})"
        axis.plot(
            steps,
            mean_values,
            color=colors[model_name],
            linewidth=2,
            label=label,
        )
        axis.fill_between(
            steps,
            mean_values - std_values,
            mean_values + std_values,
            color=colors[model_name],
            alpha=0.2,
        )

    axis.set_title(
        "Independent vs dual DINO: mean across all CEM candidates"
    )
    axis.set_xlabel("Episode step")
    axis.set_ylabel("combined_relative_to_zero_action (relative L1)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dual-zip", type=Path, default=MODEL_ARCHIVES["dual_dinov3"]
    )
    parser.add_argument(
        "--ind-zip", type=Path, default=MODEL_ARCHIVES["ind_dinov3"]
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Pickup task to include; repeatable. Overrides TASKS_FILTER.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Use every task with metrics in both archives.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=int,
        default=None,
        help="Run index to include; repeatable. Overrides RUNS_FILTER.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Use every matching run (the default when RUNS_FILTER is None).",
    )
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching metric files without loading their tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise SystemExit("--num-steps must be positive")
    archive_paths = {
        "dual_dinov3": args.dual_zip.expanduser().resolve(),
        "ind_dinov3": args.ind_zip.expanduser().resolve(),
    }
    for model_name, path in archive_paths.items():
        if not path.is_file():
            raise SystemExit(f"{MODEL_LABELS[model_name]} archive not found: {path}")

    selected_tasks = None if args.all_tasks else (
        args.task if args.task is not None else TASKS_FILTER
    )
    tasks = (
        None
        if selected_tasks is None
        else {normalize_task_name(task) for task in selected_tasks}
    )
    selected_runs = None if args.all_runs else (
        args.run if args.run is not None else RUNS_FILTER
    )
    runs = None if selected_runs is None else set(selected_runs)

    members_by_model: dict[str, list[tuple[str, int, str]]] = {}
    for model_name, archive_path in archive_paths.items():
        with zipfile.ZipFile(archive_path, "r") as archive:
            members_by_model[model_name] = find_metric_members(
                archive, tasks, runs
            )
    pair_sets = {
        model_name: {(task, run) for task, run, _ in members}
        for model_name, members in members_by_model.items()
    }
    matched_pairs = set.intersection(*pair_sets.values())
    if not matched_pairs:
        raise SystemExit("No matched task/run metric files found in both archives")
    for model_name, pairs in pair_sets.items():
        unmatched = len(pairs - matched_pairs)
        if unmatched:
            print(
                f"Ignoring {unmatched} unmatched {MODEL_LABELS[model_name]} "
                "trajectory/trajectories"
            )

    if args.dry_run:
        for model_name, archive_path in archive_paths.items():
            print(f"{MODEL_LABELS[model_name]}: {archive_path}")
            for task, run, member in members_by_model[model_name]:
                if (task, run) in matched_pairs:
                    print(f"  {task} run {run}: {member}")
        return

    results: dict[str, list[RunLoss]] = {}
    for model_name, archive_path in archive_paths.items():
        print(f"\nLoading {MODEL_LABELS[model_name]}: {archive_path}")
        results[model_name] = collect_model_runs(
            archive_path,
            tasks,
            runs,
            matched_pairs,
            args.num_steps,
            args.skip_errors,
        )
        if not results[model_name]:
            raise SystemExit(f"No usable metrics found for {MODEL_LABELS[model_name]}")

    print_summary(results)
    output_path = args.output.expanduser().resolve()
    plot_results(results, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
