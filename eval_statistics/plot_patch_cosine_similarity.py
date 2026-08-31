"""Plot temporal and predicted patch cosine-similarity and L1 maps.

For each dual-view model, this script takes one reference patch from the
encoder feature map of video frame ``t`` and computes its cosine similarity
against every patch in:

1. the encoder feature map at frame ``t``;
2. the encoder feature map at the recorded frame ``t + 1``; and
3. the predicted ``t + 1`` feature map obtained from frame ``t``, pose ``t``,
   and the recorded planned action ``t + 1``.

Video frame ``t`` and HDF5 pose ``t`` are aligned with planned action
``t + 1`` because the episode videos are written after ``env.step``.  The
selected reference patch is zero-based and row-major.
"""

from __future__ import annotations

import argparse
import gc
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_gt_temporal_l1 import (
    MODEL_SPECS,
    SOURCE_ZIP,
    TASK,
    extract_video,
    find_video_members,
    load_trajectory_conditioning,
    make_world_model,
    predict_from_input_patches,
    resolve_checkpoint,
    split_rgb_views,
)


# ---------------------------------------------------------------------------
# Easy-to-edit defaults. CLI options below override all of these values.
# Frames are zero-based and must describe one transition: t -> t + 1.
# Patch indices are zero-based and row-major. DINO and V-JEPA need separate
# defaults because their grids are 14x14 (196 patches) and 16x16 (256 patches).
# Change these two patch indices to patches containing ketchup in your frames.
# ---------------------------------------------------------------------------
RUN = 3
FRAME_T = 50
FRAME_T_PLUS_1 = 51
REFERENCE_VIEW = "side"  # "side" or "wrist"
PATCH_INDEX_BY_MODEL = {
    # Human position: 6th row, 8th column.
    "dual_dinov3": 77,  # zero-based (row 5, col 7), 14x14 grid
    # Same normalized image location on V-JEPA's finer patch grid.
    "dual_vjepa": 104,  # zero-based (row 6, col 8), 16x16 grid
}

MODELS = ("dual_dinov3", "dual_vjepa")
MODEL_TITLES = {
    "dual_dinov3": "Dual-DINOv3",
    "dual_vjepa": "Dual-V-JEPA",
}
DEFAULT_OUTPUT = Path(__file__).with_name("patch_cosine_similarity.png")
DEFAULT_L1_OUTPUT = Path(__file__).with_name("patch_l1_distance.png")


@dataclass
class SimilarityResult:
    model_name: str
    grid_size: int
    patch_index: int
    encoded_t: np.ndarray
    encoded_next: np.ndarray
    predicted_next: np.ndarray
    l1_encoded_t: np.ndarray
    l1_encoded_next: np.ndarray
    l1_predicted_next: np.ndarray


def read_selected_frames(
    archive: zipfile.ZipFile,
    member: str,
    frame_indices: tuple[int, ...],
) -> dict[int, dict[str, torch.Tensor]]:
    """Decode only the requested zero-based frames and split both views."""
    requested = set(frame_indices)
    frames: dict[int, dict[str, torch.Tensor]] = {}
    with tempfile.TemporaryDirectory(prefix="patch_cosine_") as temp_dir:
        video_path = Path(temp_dir) / "sensor.mp4"
        extract_video(archive, member, video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"could not open extracted video {member}")
        try:
            for index in range(max(requested) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise EOFError(
                        f"video ended before requested frame {index}: {member}"
                    )
                if index in requested:
                    frames[index] = split_rgb_views(frame)
        finally:
            capture.release()
    return frames


def square_grid_size(tokens: torch.Tensor, model_name: str) -> int:
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(
            f"{model_name} returned unexpected token shape {tuple(tokens.shape)}"
        )
    grid_size = math.isqrt(tokens.shape[1])
    if grid_size * grid_size != tokens.shape[1]:
        raise ValueError(
            f"{model_name} has {tokens.shape[1]} tokens, not a square patch grid"
        )
    return grid_size


def cosine_map(
    reference_patch: torch.Tensor,
    feature_map: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    similarities = F.cosine_similarity(
        feature_map[0], reference_patch.unsqueeze(0), dim=-1, eps=1e-8
    )
    
    return similarities.reshape(grid_size, grid_size).float().cpu().numpy()


def l1_map(
    reference_patch: torch.Tensor,
    feature_map: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    """Return the embedding-dimension L1 norm for every spatial patch."""
    distances = torch.linalg.vector_norm(
        feature_map[0] - reference_patch.unsqueeze(0), ord=1, dim=-1
    )
    return distances.reshape(grid_size, grid_size).float().cpu().numpy()


def analyze_model(
    model_name: str,
    world_model,
    frames: dict[int, dict[str, torch.Tensor]],
    conditioning,
    frame_t: int,
    frame_next: int,
    view: str,
    patch_index: int,
) -> SimilarityResult:
    """Encode the two frames and predict the one-step next feature map."""
    with torch.inference_mode():
        encoded_t = {
            name: world_model.encode(image)
            for name, image in frames[frame_t].items()
        }
        encoded_next = {
            name: world_model.encode(image)
            for name, image in frames[frame_next].items()
        }
        predicted_next = predict_from_input_patches(
            world_model,
            encoded_t,
            conditioning.planned_actions[frame_next],
            conditioning.poses[frame_t],
        )

    grid_size = square_grid_size(encoded_t[view], model_name)
    number_of_patches = grid_size * grid_size
    if not 0 <= patch_index < number_of_patches:
        raise ValueError(
            f"patch index {patch_index} is invalid for {model_name}; expected "
            f"0..{number_of_patches - 1} ({grid_size}x{grid_size} grid)"
        )
    for label, tokens in (
        ("encoded t+1", encoded_next[view]),
        ("predicted t+1", predicted_next[view]),
    ):
        if tokens.shape[:2] != encoded_t[view].shape[:2]:
            raise ValueError(
                f"{model_name} {label} shape {tuple(tokens.shape)} does not "
                f"match frame-t shape {tuple(encoded_t[view].shape)}"
            )

    reference_patch = encoded_t[view][0, patch_index]
    return SimilarityResult(
        model_name=model_name,
        grid_size=grid_size,
        patch_index=patch_index,
        encoded_t=cosine_map(reference_patch, encoded_t[view], grid_size),
        encoded_next=cosine_map(reference_patch, encoded_next[view], grid_size),
        predicted_next=cosine_map(reference_patch, predicted_next[view], grid_size),
        l1_encoded_t=l1_map(reference_patch, encoded_t[view], grid_size),
        l1_encoded_next=l1_map(reference_patch, encoded_next[view], grid_size),
        l1_predicted_next=l1_map(
            reference_patch, predicted_next[view], grid_size
        ),
    )


def plot_results(
    results: list[SimilarityResult],
    frame_t: int,
    frame_next: int,
    view: str,
    run: int,
    output_path: Path,
    metric: str,
) -> None:
    figure, axes = plt.subplots(
        len(results),
        3,
        figsize=(13.5, 4.2 * len(results)),
        constrained_layout=True,
        squeeze=False,
    )
    column_titles = (
        f"Encoder frame {frame_t}",
        f"Encoder frame {frame_next}",
        f"Predicted frame {frame_next}\n(from encoder frame {frame_t})",
    )
    if metric not in ("cosine", "l1"):
        raise ValueError(f"unsupported metric: {metric}")
    if metric == "cosine":
        cmap, vmin, vmax = "coolwarm", -1.0, 1.0
        colorbar_label = "Cosine similarity to reference patch at frame t"
    else:
        l1_values = [
            values
            for result in results
            for values in (
                result.l1_encoded_t,
                result.l1_encoded_next,
                result.l1_predicted_next,
            )
        ]
        # Use one L1 scale across both models and all three feature maps.
        cmap, vmin = "viridis", 0.0
        vmax = max(float(np.nanmax(values)) for values in l1_values)
        if vmax <= 0.0:
            vmax = 1.0
        colorbar_label = "L1 norm to reference patch at frame t"

    image = None
    for row, result in enumerate(results):
        maps = (
            (
                result.encoded_t,
                result.encoded_next,
                result.predicted_next,
            )
            if metric == "cosine"
            else (
                result.l1_encoded_t,
                result.l1_encoded_next,
                result.l1_predicted_next,
            )
        )
        patch_row, patch_col = divmod(result.patch_index, result.grid_size)
        for column, (axis, values) in enumerate(zip(axes[row], maps)):
            image = axis.imshow(
                values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="upper",
            )
            axis.set_title(column_titles[column])
            axis.set_xlabel("Patch column")
            axis.set_ylabel("Patch row")
            axis.set_xticks(range(result.grid_size))
            axis.set_yticks(range(result.grid_size))
            axis.tick_params(labelsize=7)
            if column == 0:
                axis.scatter(
                    [patch_col],
                    [patch_row],
                    marker="s",
                    s=120,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.8,
                    label=f"reference patch {result.patch_index}",
                )
                axis.legend(loc="lower right", fontsize=8)
            if column == 0:
                axis.text(
                    -0.22,
                    0.5,
                    f"{MODEL_TITLES[result.model_name]}\n"
                    f"{result.grid_size}x{result.grid_size} grid\n"
                    f"patch {result.patch_index} = ({patch_row}, {patch_col})",
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
        colorbar.set_label(colorbar_label)
    figure.suptitle(
        f"{metric.upper()} patch map | Run {run} | {view} view | "
        f"frame {frame_t} -> {frame_next}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=SOURCE_ZIP)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--run", type=int, default=RUN)
    parser.add_argument("--frame-t", type=int, default=FRAME_T)
    parser.add_argument("--frame-next", type=int, default=FRAME_T_PLUS_1)
    parser.add_argument("--view", choices=("side", "wrist"), default=REFERENCE_VIEW)
    parser.add_argument(
        "--patch-index",
        type=int,
        default=None,
        help="Use one raw zero-based patch index for both models.",
    )
    parser.add_argument(
        "--dino-patch-index",
        type=int,
        default=None,
        help="Override the dual-DINO reference patch only.",
    )
    parser.add_argument(
        "--vjepa-patch-index",
        type=int,
        default=None,
        help="Override the dual-V-JEPA reference patch only.",
    )
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    parser.add_argument("--vjepa-checkpoint", type=Path, default=None)
    parser.add_argument("--dual-dino-predictor", type=Path, default=None)
    parser.add_argument("--dual-vjepa-predictor", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--l1-output",
        type=Path,
        default=DEFAULT_L1_OUTPUT,
        help="Output path for the additional per-patch L1-distance plot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate archive/frame selection without loading model weights.",
    )
    return parser.parse_args()


def selected_patch_indices(args: argparse.Namespace) -> dict[str, int]:
    indices = dict(PATCH_INDEX_BY_MODEL)
    if args.patch_index is not None:
        indices = {model_name: args.patch_index for model_name in MODELS}
    if args.dino_patch_index is not None:
        indices["dual_dinov3"] = args.dino_patch_index
    if args.vjepa_patch_index is not None:
        indices["dual_vjepa"] = args.vjepa_patch_index
    return indices


def main() -> None:
    args = parse_args()
    if args.frame_t < 0 or args.frame_next < 0:
        raise SystemExit("frame indices must be nonnegative")
    if args.frame_next != args.frame_t + 1:
        raise SystemExit(
            "the one-step predictor requires --frame-next == --frame-t + 1"
        )
    source_zip = args.source_zip.expanduser().resolve()
    if not source_zip.is_file():
        raise SystemExit(f"source archive not found: {source_zip}")
    members = find_video_members(source_zip, args.task, {args.run})
    if args.run not in members:
        raise SystemExit(
            f"no sensor video matched task {args.task!r}, run {args.run}"
        )
    member = members[args.run]
    patch_indices = selected_patch_indices(args)
    print(f"Source archive: {source_zip}")
    print(f"Video: {member}")
    print(f"Transition: frame {args.frame_t} -> frame {args.frame_next}")
    print(f"Reference view: {args.view}")
    print(f"Patch indices: {patch_indices}")

    with zipfile.ZipFile(source_zip, "r") as archive:
        frames = read_selected_frames(
            archive, member, (args.frame_t, args.frame_next)
        )
        conditioning = load_trajectory_conditioning(
            archive, args.task, args.run
        )
    if args.frame_t >= conditioning.poses.shape[0]:
        raise SystemExit(
            f"frame t={args.frame_t} exceeds {conditioning.poses.shape[0]} poses"
        )
    if args.frame_next >= conditioning.planned_actions.shape[0]:
        raise SystemExit(
            f"action {args.frame_next} exceeds "
            f"{conditioning.planned_actions.shape[0]} planned actions"
        )
    if args.dry_run:
        print("Dry run succeeded; frames, pose, and action are available.")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    checkpoint_overrides = {
        "dual_dinov3": (args.dino_checkpoint, args.dual_dino_predictor),
        "dual_vjepa": (args.vjepa_checkpoint, args.dual_vjepa_predictor),
    }
    results: list[SimilarityResult] = []
    for model_name in MODELS:
        spec = MODEL_SPECS[model_name]
        encoder_override, predictor_override = checkpoint_overrides[model_name]
        try:
            encoder_checkpoint = resolve_checkpoint(
                encoder_override, spec.encoder_checkpoint
            )
            predictor_checkpoint = resolve_checkpoint(
                predictor_override, spec.predictor_checkpoints[0]
            )
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        print(f"Loading {spec.label} encoder: {encoder_checkpoint}")
        print(f"Loading {spec.label} predictor: {predictor_checkpoint}")
        world_model = make_world_model(
            model_name,
            device,
            encoder_checkpoint,
            (predictor_checkpoint,),
        )
        result = analyze_model(
            model_name,
            world_model,
            frames,
            conditioning,
            args.frame_t,
            args.frame_next,
            args.view,
            patch_indices[model_name],
        )
        results.append(result)
        row, column = divmod(result.patch_index, result.grid_size)
        print(
            f"{spec.label}: {result.grid_size}x{result.grid_size} grid, "
            f"reference patch {result.patch_index} = row {row}, col {column}"
        )
        del world_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = args.output.expanduser().resolve()
    plot_results(
        results,
        args.frame_t,
        args.frame_next,
        args.view,
        args.run,
        output_path,
        "cosine",
    )
    print(f"Saved plot: {output_path}")
    l1_output_path = args.l1_output.expanduser().resolve()
    plot_results(
        results,
        args.frame_t,
        args.frame_next,
        args.view,
        args.run,
        l1_output_path,
        "l1",
    )
    print(f"Saved plot: {l1_output_path}")


if __name__ == "__main__":
    main()
