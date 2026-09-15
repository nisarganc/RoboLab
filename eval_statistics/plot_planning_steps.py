#!/usr/bin/env python3
"""Plot mean planning steps by model with standard-deviation error bars."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .eval_summary import (
        model_variant_from_zip,
        resolve_zip_paths,
        summarize,
    )
except ImportError:
    from eval_summary import (
        model_variant_from_zip,
        resolve_zip_paths,
        summarize,
    )


MODEL_ORDER = [
    "right_vjepa",
    "right_dinov3",
    "wrist_vjepa",
    "wrist_dinov3",
    "ind_vjepa",
    "ind_dinov3",
    "dual_vjepa",
    "dual_dinov3",
]

DEFAULT_ZIP_GLOB = str(
    Path(__file__).resolve().parents[2] / ".cache" / "output_reach" / "*.zip"
)

DEFAULT_TASKS = {
    "ReachCoffeePotTask",
    "ReachCoffeeCanTask",
    "ReachBananaTask",
    "ReachOrangeJuiceCartonTask",
    "ReachYogurtCupTask",
    "ReachAppleTask",
    "ReachOrangeTask",
    "ReachBagelTask",
    "ReachPitcherTask",
    "ReachCeramicMugTask",
}

MODEL_GROUPS = [
    ("Side-view", "right_vjepa", "right_dinov3", "#F1F7FC"),
    ("Wrist-view", "wrist_vjepa", "wrist_dinov3", "#F8F3FC"),
    ("Indp. dual-view", "ind_vjepa", "ind_dinov3", "#F1FAF5"),
    ("DUET", "dual_vjepa", "dual_dinov3", "#FFF7ED"),
]


def canonical_model_name(model: str) -> str:
    """Normalize the task suffix used by angled-reach archive names."""
    suffix = "_angledreach"
    return model[: -len(suffix)] if model.endswith(suffix) else model


def filter_model_zip_paths(zip_paths: list[Path]) -> list[Path]:
    """Keep only archives corresponding to the eight models in MODEL_ORDER."""
    requested_models = set(MODEL_ORDER)
    return [
        path
        for path in zip_paths
        if canonical_model_name(model_variant_from_zip(path)) in requested_models
    ]


def aggregate_model_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the model-level planning statistics produced by eval_summary."""
    model_rows = []
    for row in rows:
        model = canonical_model_name(str(row["model"]))
        if row["task"] == "ALL_TASKS" and model in MODEL_ORDER:
            normalized_row = dict(row)
            normalized_row["model"] = model
            model_rows.append(normalized_row)

    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    return sorted(
        model_rows,
        key=lambda row: (order.get(str(row["model"]), len(order)), str(row["model"])),
    )


def plot_planning_steps(
    model_rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    """Create a model bar chart with planning-step standard deviations."""
    if not model_rows:
        raise ValueError("no model-level planning results to plot")

    import matplotlib.pyplot as plt

    rows_by_model = {str(row["model"]): row for row in model_rows}
    missing_models = [model for model in MODEL_ORDER if model not in rows_by_model]
    if missing_models:
        raise ValueError(f"missing planning results for: {', '.join(missing_models)}")

    group_labels = [group[0] for group in MODEL_GROUPS]
    vjepa_means = [
        float(rows_by_model[group[1]]["planning_steps_mean"])
        for group in MODEL_GROUPS
    ]
    vjepa_stds = [
        float(rows_by_model[group[1]]["planning_steps_std"])
        for group in MODEL_GROUPS
    ]
    dinov3_means = [
        float(rows_by_model[group[2]]["planning_steps_mean"])
        for group in MODEL_GROUPS
    ]
    dinov3_stds = [
        float(rows_by_model[group[2]]["planning_steps_std"])
        for group in MODEL_GROUPS
    ]

    group_positions = list(range(len(MODEL_GROUPS)))
    bar_width = 0.34
    figure, axis = plt.subplots(figsize=(8.4, 4.8), dpi=200)

    for position, (_, _, _, background_color) in zip(group_positions, MODEL_GROUPS):
        axis.axvspan(
            position - 0.48,
            position + 0.48,
            color=background_color,
            zorder=0,
        )

    vjepa_bars = axis.bar(
        [position - bar_width / 2 for position in group_positions],
        vjepa_means,
        yerr=vjepa_stds,
        capsize=5,
        width=bar_width,
        label="VJEPA 2",
        color="#E07A5F",
        edgecolor="#7A3E2F",
        linewidth=0.6,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
        zorder=3,
    )
    dinov3_bars = axis.bar(
        [position + bar_width / 2 for position in group_positions],
        dinov3_means,
        yerr=dinov3_stds,
        capsize=5,
        width=bar_width,
        label="DINOv3",
        color="#4C78A8",
        edgecolor="#294A67",
        linewidth=0.6,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
        zorder=3,
    )

    # axis.set_title(title)
    # axis.set_xlabel("AC-Predictors")
    axis.set_ylabel("Planning steps", fontsize=15, fontweight="bold", labelpad=8)
    axis.set_xticks(
        group_positions,
        group_labels,
        fontsize=13,
        fontweight="bold",
    )
    upper_limit = max(
        max(mean + std for mean, std in zip(vjepa_means, vjepa_stds)),
        max(mean + std for mean, std in zip(dinov3_means, dinov3_stds)),
    )
    axis.set_ylim(0.0, max(1.0, upper_limit * 1.15))
    axis.tick_params(axis="y", labelsize=13)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontweight("bold")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.bar_label(
        vjepa_bars,
        labels=[f"{value:.1f}" for value in vjepa_means],
        padding=4,
        fontsize=12,
        fontweight="bold",
    )
    axis.bar_label(
        dinov3_bars,
        labels=[f"{value:.1f}" for value in dinov3_means],
        padding=4,
        fontsize=12,
        fontweight="bold",
    )
    axis.legend(
        frameon=False,
        ncols=2,
        loc="upper right",
        prop={"size": 13, "weight": "bold"},
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot planning-step means and standard deviations by model."
    )
    parser.add_argument(
        "--zip",
        dest="zip_patterns",
        action="append",
        default=None,
        help="Zip path or glob. Can be repeated.",
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
        help="Task name to include. Can be repeated. Default: the 10 reach tasks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("planning_steps_by_model.pdf"),
        help="Output image path. The extension selects the format.",
    )
    parser.add_argument(
        "--title",
        default="Planning Steps by Model",
        help="Chart title.",
    )
    parser.add_argument("--verbose-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_paths = filter_model_zip_paths(
        resolve_zip_paths(args.zip_patterns or [DEFAULT_ZIP_GLOB])
    )
    tasks_filter = set(args.tasks) if args.tasks else DEFAULT_TASKS
    rows = summarize(zip_paths, args.assets_root, tasks_filter, args.verbose_runs)
    model_rows = aggregate_model_rows(rows)

    plot_planning_steps(model_rows, args.output, args.title)

    for row in model_rows:
        print(
            f"{row['model']}: "
            f"{row['planning_steps_mean']:.2f} +/- "
            f"{row['planning_steps_std']:.2f} steps"
        )
    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()
