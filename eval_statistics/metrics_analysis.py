
# Take an episode from one of the episodes in ../../.cache/output_reach

#!/usr/bin/env python3
"""Inspect a saved VALPA run_metrics.pt and plot predicted-patch PCA."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", type=Path, help="Path to run_metrics.pt")
    parser.add_argument(
        "--output",
        type=Path,
        help="PCA image path (default: <metrics directory>/next_frame_pca.png)",
    )
    parser.add_argument(
        "--full-candidate-losses",
        action="store_true",
        help="Print every candidate loss instead of PyTorch's abbreviated view.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=12_000,
        help="Maximum scatter points per camera view (default: 12000).",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def describe_tensor(name: str, value: torch.Tensor | None) -> None:
    if value is None:
        print(f"\n{name}: None")
        return
    finite = value[torch.isfinite(value)]
    print(f"\n{name}: shape={tuple(value.shape)}")
    if finite.numel():
        print(
            "finite summary: "
            f"min={finite.min().item():.6g}, "
            f"mean={finite.mean().item():.6g}, "
            f"max={finite.max().item():.6g}"
        )
    # print(value)


def print_named_summary(
    values: torch.Tensor, names: list[str], heading: str
) -> None:
    flattened = values.reshape(-1, values.shape[-1])
    print(f"\n{heading} (over all run steps and CEM iterations):")
    print(f"{'name':42s} {'min':>12s} {'mean':>12s} {'std':>12s} {'max':>12s}")
    for index, name in enumerate(names):
        column = flattened[:, index]
        finite = column[torch.isfinite(column)]
        if not finite.numel():
            print(f"{name:42s} {'nan':>12s} {'nan':>12s} {'nan':>12s} {'nan':>12s}")
            continue
        print(
            f"{name:42s} "
            f"{finite.min().item():12.6g} "
            f"{finite.mean().item():12.6g} "
            f"{finite.std(unbiased=False).item():12.6g} "
            f"{finite.max().item():12.6g}"
        )


def flatten_patch_predictions(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [patch observations, embedding dimension] and a run-step label."""
    if frames.ndim < 2:
        raise ValueError(f"Expected a step and feature dimension, got {tuple(frames.shape)}")
    steps = frames.shape[0]
    flattened = frames.float().reshape(steps, -1, frames.shape[-1])
    labels = torch.arange(steps).repeat_interleave(flattened.shape[1])
    return flattened.reshape(-1, flattened.shape[-1]), labels


def subsample(
    points: torch.Tensor, labels: torch.Tensor, maximum: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if maximum <= 0 or points.shape[0] <= maximum:
        return points, labels
    indices = torch.linspace(0, points.shape[0] - 1, maximum).long()
    return points[indices], labels[indices]


def plot_joint_pca(
    predictions: dict[str, torch.Tensor], output: Path, max_points: int
) -> None:
    missing = {"side", "wrist"} - predictions.keys()
    if missing:
        raise KeyError(f"Missing patch-prediction view(s): {sorted(missing)}")

    side, side_steps = flatten_patch_predictions(predictions["side"])
    wrist, wrist_steps = flatten_patch_predictions(predictions["wrist"])
    if side.shape[1] != wrist.shape[1]:
        raise ValueError(
            f"Side/wrist embedding dimensions differ: {side.shape[1]} vs {wrist.shape[1]}"
        )

    combined = torch.cat((side, wrist), dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    centered = combined - mean
    _, singular_values, components = torch.pca_lowrank(
        centered, q=2, center=False, niter=4
    )
    projected = centered @ components[:, :2]
    side_pca = projected[: side.shape[0]]
    wrist_pca = projected[side.shape[0] :]

    total_variance = centered.var(dim=0, unbiased=False).sum()
    explained = singular_values[:2].square() / max(centered.shape[0] - 1, 1)
    explained_ratio = explained / total_variance.clamp_min(torch.finfo(explained.dtype).eps)

    side_pca, side_steps = subsample(side_pca, side_steps, max_points)
    wrist_pca, wrist_steps = subsample(wrist_pca, wrist_steps, max_points)
    number_of_steps = max(predictions["side"].shape[0], predictions["wrist"].shape[0])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    scatter = None
    for axis, title, points, labels in (
        (axes[0], "next_frame_side", side_pca, side_steps),
        (axes[1], "next_frame_wrist", wrist_pca, wrist_steps),
    ):
        scatter = axis.scatter(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            c=labels.numpy(),
            cmap="viridis",
            vmin=0,
            vmax=max(number_of_steps - 1, 1),
            s=5,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel(f"PC1 ({100 * explained_ratio[0].item():.2f}% variance)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(f"PC2 ({100 * explained_ratio[1].item():.2f}% variance)")
    figure.colorbar(scatter, ax=axes, label="Inference step", pad=0.02)
    figure.suptitle("Joint PCA of final-action predicted patch embeddings")
    figure.subplots_adjust(left=0.07, right=0.88, bottom=0.11, top=0.88, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(
        f"\nPCA: fitted jointly on {combined.shape[0]} patch embeddings "
        f"with dimension {combined.shape[1]}"
    )
    print(
        "Explained variance: "
        f"PC1={100 * explained_ratio[0].item():.3f}%, "
        f"PC2={100 * explained_ratio[1].item():.3f}%"
    )
    print(f"Saved PCA plot: {output}")


def main() -> None:
    args = parse_args()
    path = args.metrics_file.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else path.parent / "next_frame_pca.png"
    )
    payload = load_metrics(path)
    torch.set_printoptions(
        precision=6,
        linewidth=160,
        threshold=None if args.full_candidate_losses else 10_000,
    )

    print(f"Metrics file: {path}")
    print(f"Saved keys: {list(payload)}")

    # cem metrics: [run_step, cem_iteration, 6]
    cem_metrics = payload.get("cem_metrics")
    metric_names = payload.get("metric_names", [])
    describe_tensor("planned_action_sequences", payload.get("planned_action_sequences"))
    describe_tensor("cem_metrics", cem_metrics)
    print_named_summary(cem_metrics, metric_names, "CEM metric summary")

    # candidate losses: [run_step, cem_iteration, num_candidate_losses(5)]
    candidate_losses = payload.get("candidate_losses")
    loss_names = payload.get("candidate_loss_names", [])
    describe_tensor("candidate_losses", candidate_losses)
    print_named_summary(candidate_losses, loss_names, "Candidate-loss summary")


    # final action patch predictions: [run_step, cem_iteration, num_predictions]
    predictions = payload.get("final_action_patch_predictions", {})
    # describe_tensor("final_action_patch_predictions", predictions)
    for view, tensor in predictions.items():
        print(f"{view}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    plot_joint_pca(predictions, output, args.max_plot_points)


if __name__ == "__main__":
    main()
