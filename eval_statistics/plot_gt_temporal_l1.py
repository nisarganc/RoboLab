"""Plot GT temporal L1 and action-conditioned predictor change.

The same recorded angled-pickup sensor trajectories are passed independently
through the pretrained DINOv3 and V-JEPA encoders.  For each transition from
video frame ``t - 1`` to frame ``t``, all patch and embedding dimensions are
flattened and reduced with the scalar L1 function from ``mpc_utils_dual.py``.

Sensor MP4s contain the side view in the left half and wrist view in the right
half.  Side, wrist, and their summed dual-view L1 are plotted separately.  The
default selection is AngledPickupKetchup run 0, independently passed through
each encoder ten times; curves show the across-repeat mean and one population
standard deviation.

The second row feeds the *same* trajectory, poses, and planned action sequences
from the dual-DINO metrics file through the dual-DINO, dual-V-JEPA, and
independent-DINO predictors. It plots the scalar L1 between each predictor's
next-patch output and the encoded observation patches supplied as its input.
Because episode videos are written after ``env.step``, video frame ``t - 1``
and HDF5 pose ``t - 1`` are the inputs associated with planned action ``t``;
planned action 0 has no recorded pre-action frame and is intentionally skipped.
"""

from __future__ import annotations

import argparse
import gc
import io
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation

RESULTS_ROOT = Path(
    "/anvme/workspace/v106be10-valpa-robolab/.cache/output_angledpickup"
)
SOURCE_ZIP = RESULTS_ROOT / "dual_dinov3_angledpickup.zip"
DEFAULT_OUTPUT = Path(__file__).with_name("gt_temporal_l1_dino_vjepa.png")
REPO_ROOT = Path(__file__).resolve().parents[1]
VALPA_ROOT = REPO_ROOT / "valpa"
CHECKPOINT_ROOT = REPO_ROOT.parent / ".cache" / "checkpoints"
CONFIG_ROOT = VALPA_ROOT / "configs" / "inference" / "valpa-angledpickup"

# Edit these defaults or use the command-line options below.
TASK = "AngledPickupKetchupTask"
RUNS_FILTER: list[int] = [0]
NUM_STEPS = 60
REPEATS = 1

ENCODER_MODELS = ("dual_dinov3", "dual_vjepa")
PREDICTOR_MODELS = ("dual_dinov3", "dual_vjepa", "ind_dinov3")
ENCODER_LABELS = {
    "dual_dinov3": "DINOv3",
    "dual_vjepa": "VJEPA2",
}
PREDICTOR_LABELS = {
    "dual_dinov3": "DINOv3 Predictor",
    "dual_vjepa": "VJEPA2 Predictor",
    "ind_dinov3": "Independent DINOv Predictor",
}
VIEWS = ("side", "wrist")
LOSS_SERIES = (*VIEWS, "combined")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    config_name: str
    encoder_checkpoint: str
    predictor_checkpoints: tuple[str, ...]


MODEL_SPECS = {
    "dual_dinov3": ModelSpec(
        "Dual-DINOv3",
        "droid-224px-8f-dual.yaml",
        "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        ("shared_latent_dinov3.pt",),
    ),
    "dual_vjepa": ModelSpec(
        "Dual-V-JEPA",
        "droid-256px-8f-dual.yaml",
        "vitg.pt",
        ("shared_latent_vjepa.pt",),
    ),
    "ind_dinov3": ModelSpec(
        "Ind-DINOv3",
        "droid-224px-8f-ind.yaml",
        "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        ("dinov3_right.pt", "dinov3_wrist.pt"),
    ),
}


def normalize_task_name(task: str) -> str:
    return task[:-4] if task.endswith("Task") else task


@dataclass
class RunTemporalLoss:
    steps: torch.Tensor
    losses: dict[str, torch.Tensor]


@dataclass
class TrajectoryConditioning:
    planned_actions: torch.Tensor
    poses: torch.Tensor


def resolve_checkpoint(path: Path | None, default_name: str) -> Path:
    candidates = []
    if path is not None:
        candidates.append(path.expanduser())
    else:
        candidates.extend(
            (
                CHECKPOINT_ROOT / default_name,
                REPO_ROOT / "checkpoints" / default_name,
            )
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"checkpoint {default_name!r} not found; searched: {searched}"
    )


def make_world_model(
    model_name: str,
    device: torch.device,
    encoder_checkpoint: Path,
    predictor_checkpoints: tuple[Path, ...],
):
    """Build the exact pickup WorldModel while overriding stale config paths."""
    if str(VALPA_ROOT) not in sys.path:
        sys.path.insert(0, str(VALPA_ROOT))
    from inference.serve_policy_pickup import VALPADroidEEPolicy

    spec = MODEL_SPECS[model_name]
    config_path = CONFIG_ROOT / spec.config_name
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["device"] = str(device)
    if config["model"].get("use_dinov3_encoder", False):
        config["meta"]["pretrain_dinocheckpoint"] = str(encoder_checkpoint)
    else:
        config["meta"]["pretrain_checkpoint"] = str(encoder_checkpoint)
    config["model"]["predictor_checkpoint"] = (
        str(predictor_checkpoints[0])
        if len(predictor_checkpoints) == 1
        else [str(path) for path in predictor_checkpoints]
    )

    policy = VALPADroidEEPolicy.__new__(VALPADroidEEPolicy)
    policy.cfg = config
    policy._initialize_model()
    return policy.world_model


def find_run_member(
    archive: zipfile.ZipFile,
    task: str,
    run: int,
    kind: str,
) -> str:
    normalized_task = normalize_task_name(task)
    suffixes = {
        "metrics": f"/{normalized_task}/metrics_{run}_0.pt",
        "hdf5": f"/{normalized_task}Task/run_{run}.hdf5",
    }
    suffix = suffixes[kind]
    matches = [name for name in archive.namelist() if f"/{name}".endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {kind} member ending in {suffix}, found {len(matches)}"
        )
    return matches[0]


def load_trajectory_conditioning(
    archive: zipfile.ZipFile,
    task: str,
    run: int,
) -> TrajectoryConditioning:
    metrics_member = find_run_member(archive, task, run, "metrics")
    payload = torch.load(
        io.BytesIO(archive.read(metrics_member)),
        map_location="cpu",
        weights_only=True,
    )
    planned_actions = payload["planned_action_sequences"].float()
    if planned_actions.ndim != 3 or planned_actions.shape[-1] != 7:
        raise ValueError(
            f"unexpected planned actions shape {tuple(planned_actions.shape)}"
        )
    del payload

    hdf5_member = find_run_member(archive, task, run, "hdf5")
    with h5py.File(io.BytesIO(archive.read(hdf5_member)), "r") as trajectory:
        demo = trajectory["data/demo_0"]
        positions = demo["ee_pose/position"][:]
        quaternions_wxyz = demo["ee_pose/orientation"][:]
        gripper = demo["gripper_state"][:]
    rotations_xyz = Rotation.from_quat(
        quaternions_wxyz[:, [1, 2, 3, 0]]
    ).as_euler("xyz", degrees=False)
    poses = torch.as_tensor(
        np.concatenate((positions, rotations_xyz, gripper), axis=-1),
        dtype=torch.float32,
    )
    return TrajectoryConditioning(planned_actions, poses)


def find_video_members(
    archive_path: Path,
    task: str,
    runs_filter: set[int] | None,
) -> dict[int, str]:
    normalized_task = normalize_task_name(task)
    task_directory = f"/{normalized_task}Task/"
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.namelist()

    matches: dict[int, str] = {}
    for member in members:
        if task_directory not in f"/{member}":
            continue
        if not member.endswith(".mp4") or member.endswith("_viewport.mp4"):
            continue
        stem = Path(member).stem
        prefix = f"{normalized_task}_"
        if not stem.startswith(prefix):
            continue
        run_text = stem[len(prefix) :]
        if not run_text.isdigit():
            continue
        run = int(run_text)
        if runs_filter is not None and run not in runs_filter:
            continue
        if run in matches:
            raise ValueError(f"multiple sensor videos found for run {run}")
        matches[run] = member
    return dict(sorted(matches.items()))


def extract_video(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
) -> None:
    with archive.open(member, "r") as source:
        with destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def split_rgb_views(bgr_frame) -> dict[str, torch.Tensor]:
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    width = rgb_frame.shape[1]
    if width <= 0 or width % 2:
        raise ValueError(f"expected an even-width sensor frame, found {width}")
    midpoint = width // 2
    return {
        "side": torch.from_numpy(rgb_frame[:, :midpoint].copy()),
        "wrist": torch.from_numpy(rgb_frame[:, midpoint:].copy()),
    }


def analyze_video(
    archive: zipfile.ZipFile,
    member: str,
    run: int,
    encoder_label: str,
    world_model,
    num_steps: int,
    repeat: int,
    total_repeats: int,
) -> RunTemporalLoss:
    from inference.utils.mpc_utils_dual import l1

    with tempfile.TemporaryDirectory(prefix="gt_temporal_l1_") as temp_dir:
        video_path = Path(temp_dir) / "sensor.mp4"
        extract_video(archive, member, video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"could not open extracted video {member}")
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        steps_to_read = min(num_steps, video_frames)
        if steps_to_read < 2:
            capture.release()
            raise ValueError(
                f"need at least two frames, found {steps_to_read} in {member}"
            )

        transition_steps = torch.arange(1, steps_to_read, dtype=torch.int64)
        losses = {
            name: torch.empty(transition_steps.numel(), dtype=torch.float32)
            for name in LOSS_SERIES
        }
        print(
            f"Encoding {encoder_label} | run {run} | "
            f"repeat {repeat}/{total_repeats}: "
            f"{steps_to_read} frames, {transition_steps.numel()} transitions"
        )
        try:
            ok, first_frame = capture.read()
            if not ok:
                raise EOFError(f"could not decode frame 0 from {member}")
            with torch.inference_mode():
                previous = {
                    view: world_model.encode(frame)
                    for view, frame in split_rgb_views(first_frame).items()
                }
                for index in range(transition_steps.numel()):
                    ok, next_frame = capture.read()
                    if not ok:
                        raise EOFError(
                            f"video ended at frame {index + 1}: {member}"
                        )
                    current = {
                        view: world_model.encode(frame)
                        for view, frame in split_rgb_views(next_frame).items()
                    }
                    for view in VIEWS:
                        losses[view][index] = l1(
                            previous[view].flatten(1),
                            current[view].flatten(1),
                        ).item()
                    losses["combined"][index] = (
                        losses["side"][index] + losses["wrist"][index]
                    )
                    previous = current
                    if (index + 1) % 10 == 0 or (
                        index + 1 == transition_steps.numel()
                    ):
                        print(
                            f"  encoded {index + 1}/"
                            f"{transition_steps.numel()} transitions",
                            flush=True,
                        )
        finally:
            capture.release()
    return RunTemporalLoss(transition_steps, losses)


def predict_from_input_patches(
    world_model,
    inputs: dict[str, torch.Tensor],
    action: torch.Tensor,
    pose: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply one saved action using the wrapper's exact predictor semantics."""
    actions = action.to(world_model.device).unsqueeze(0)
    poses = pose.to(world_model.device).view(1, 1, 7)
    tokens = world_model.tokens_per_frame
    with torch.inference_mode():
        if world_model.inferred_mode == "dual":
            side, wrist = world_model.predictor["dual"](
                inputs["side"], inputs["wrist"], actions, poses
            )
        elif world_model.inferred_mode == "independent_dual":
            side = world_model.predictor["side"](
                inputs["side"], actions, poses
            )
            wrist = world_model.predictor["wrist"](
                inputs["wrist"], actions, poses
            )
        else:
            raise ValueError(
                f"analysis requires two views, got {world_model.inferred_mode}"
            )
        outputs = {"side": side[:, -tokens:], "wrist": wrist[:, -tokens:]}
        if world_model.normalize_reps:
            outputs = {
                view: F.layer_norm(value, (value.size(-1),))
                for view, value in outputs.items()
            }
    return outputs


def analyze_predictor_video(
    archive: zipfile.ZipFile,
    member: str,
    run: int,
    model_name: str,
    world_model,
    conditioning: TrajectoryConditioning,
    num_steps: int,
    repeat: int,
    total_repeats: int,
) -> RunTemporalLoss:
    from inference.utils.mpc_utils_dual import l1

    with tempfile.TemporaryDirectory(prefix="predictor_input_l1_") as temp_dir:
        video_path = Path(temp_dir) / "sensor.mp4"
        extract_video(archive, member, video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"could not open extracted video {member}")
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame i and recorded pose i are the observation used for action i+1.
        number_of_values = min(
            num_steps - 1,
            video_frames,
            conditioning.poses.shape[0],
            conditioning.planned_actions.shape[0] - 1,
        )
        if number_of_values < 1:
            capture.release()
            raise ValueError(f"no aligned predictor inputs available in {member}")
        steps = torch.arange(1, number_of_values + 1, dtype=torch.int64)
        losses = {
            name: torch.empty(number_of_values, dtype=torch.float32)
            for name in LOSS_SERIES
        }
        label = MODEL_SPECS[model_name].label
        print(
            f"Predicting {label} | run {run} | repeat {repeat}/{total_repeats}: "
            f"{number_of_values} aligned actions"
        )
        try:
            for index in range(number_of_values):
                ok, frame = capture.read()
                if not ok:
                    raise EOFError(f"video ended at frame {index}: {member}")
                input_patches = {
                    view: world_model.encode(image)
                    for view, image in split_rgb_views(frame).items()
                }
                action_step = index + 1
                predictions = predict_from_input_patches(
                    world_model,
                    input_patches,
                    conditioning.planned_actions[action_step],
                    conditioning.poses[index],
                )
                for view in VIEWS:
                    losses[view][index] = l1(
                        predictions[view].flatten(1),
                        input_patches[view].flatten(1),
                    ).item()
                losses["combined"][index] = (
                    losses["side"][index] + losses["wrist"][index]
                )
                if (index + 1) % 10 == 0 or index + 1 == number_of_values:
                    print(
                        f"  predicted {index + 1}/{number_of_values} actions",
                        flush=True,
                    )
        finally:
            capture.release()
    return RunTemporalLoss(steps, losses)


def aggregate_runs(
    runs: list[RunTemporalLoss],
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum_steps = max(int(run.steps.numel()) for run in runs)
    stacked = torch.full((len(runs), maximum_steps), float("nan"))
    for index, run in enumerate(runs):
        values = run.losses[loss_name]
        stacked[index, : values.numel()] = values

    finite = torch.isfinite(stacked)
    counts = finite.sum(dim=0)
    safe = torch.where(finite, stacked, torch.zeros_like(stacked))
    means = safe.sum(dim=0) / counts.clamp_min(1)
    differences = torch.where(
        finite,
        stacked - means.unsqueeze(0),
        torch.zeros_like(stacked),
    )
    standard_deviations = (
        differences.square().sum(dim=0) / counts.clamp_min(1)
    ).sqrt()
    return means, standard_deviations, counts


def print_summary(
    results: dict[str, list[RunTemporalLoss]],
    heading: str,
    labels: dict[str, str] | None = None,
) -> None:
    print(f"\n{heading}")
    print(
        f"{'Encoder':<10} {'Loss':<10} {'Mean':>12} "
        f"{'Std':>12} {'Values':>10}"
    )
    print("-" * 60)
    for model_name, runs in sorted(results.items()):
        label = (
            labels[model_name]
            if labels is not None
            else MODEL_SPECS[model_name].label
        )
        for loss_name in LOSS_SERIES:
            values = torch.cat([run.losses[loss_name] for run in runs])
            finite = values[torch.isfinite(values)]
            print(
                f"{label:<10} {loss_name:<10} "
                f"{finite.mean().item():>12.6g} "
                f"{finite.std(unbiased=False).item():>12.6g} "
                f"{finite.numel():>10}"
            )


def plot_comparison(
    temporal_results: dict[str, list[RunTemporalLoss]],
    predictor_results: dict[str, list[RunTemporalLoss]],
    encoder_models: tuple[str, ...],
    predictor_models: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(LOSS_SERIES),
        figsize=(18, 5.8),
        sharex=True,
    )
    encoder_colors = {
        "dual_dinov3": "tab:blue",
        "dual_vjepa": "tab:red",
    }
    predictor_colors = {
        "dual_dinov3": "tab:orange",
        "dual_vjepa": "tab:green",
        "ind_dinov3": "tab:green",
    }
    view_titles = {
        "side": "Side view",
        "wrist": "Wrist view",
        "combined": "Dual view",
    }
    for axis, loss_name in zip(axes, LOSS_SERIES):
        for model_name in encoder_models:
            runs = temporal_results[model_name]
            means, standard_deviations, _ = aggregate_runs(runs, loss_name)
            steps = torch.arange(1, means.numel() + 1).numpy()
            mean_values = means.numpy()
            std_values = standard_deviations.numpy()
            axis.plot(
                steps,
                mean_values,
                linewidth=2,
                color=encoder_colors[model_name],
                label=ENCODER_LABELS[model_name],
            )
            axis.fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                color=encoder_colors[model_name],
                alpha=0.18,
            )
        for model_name in predictor_models:
            runs = predictor_results[model_name]
            means, standard_deviations, _ = aggregate_runs(runs, loss_name)
            steps = torch.arange(1, means.numel() + 1).numpy()
            mean_values = means.numpy()
            std_values = standard_deviations.numpy()
            axis.plot(
                steps,
                mean_values,
                linewidth=1.8,
                linestyle="--",
                color=predictor_colors[model_name],
                label=PREDICTOR_LABELS[model_name],
            )
            axis.fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                color=predictor_colors[model_name],
                alpha=0.1,
            )
        axis.set_title(view_titles[loss_name])
        axis.set_xlabel("Episode transition step")
        axis.set_ylabel("L1 norm")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def suffixed_output(output_path: Path, suffix: str) -> Path:
    extension = output_path.suffix or ".png"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}_{suffix}{extension}")


def plot_results(
    temporal_results: dict[str, list[RunTemporalLoss]],
    predictor_results: dict[str, list[RunTemporalLoss]],
    output_path: Path,
) -> list[Path]:
    outputs = [
        output_path,
        suffixed_output(output_path, "encoders"),
        suffixed_output(output_path, "dino_encoder_vs_dual_predictor"),
        suffixed_output(output_path, "vjepa_encoder_vs_dual_predictor"),
    ]
    comparisons = (
        (ENCODER_MODELS, PREDICTOR_MODELS),
        (ENCODER_MODELS, ()),
        (("dual_dinov3",), ("dual_dinov3",)),
        (("dual_vjepa",), ("dual_vjepa",)),
    )
    for path, (encoder_models, predictor_models) in zip(outputs, comparisons):
        plot_comparison(
            temporal_results,
            predictor_results,
            encoder_models,
            predictor_models,
            path,
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=SOURCE_ZIP,
        help="Dual-DINO archive supplying the shared MP4, poses, and actions.",
    )
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--run",
        action="append",
        type=int,
        default=None,
        help="Run to include; repeatable. Overrides RUNS_FILTER.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Use every matching task video in the source archive.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=NUM_STEPS,
        help=f"Maximum frames per trajectory (default: {NUM_STEPS}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help=f"Independent encoder/predictor passes per video (default: {REPEATS}).",
    )
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    parser.add_argument("--vjepa-checkpoint", type=Path, default=None)
    parser.add_argument("--dual-dino-predictor", type=Path, default=None)
    parser.add_argument("--dual-vjepa-predictor", type=Path, default=None)
    parser.add_argument("--ind-dino-side-predictor", type=Path, default=None)
    parser.add_argument("--ind-dino-wrist-predictor", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected archive members without loading model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_steps < 2:
        raise SystemExit("--num-steps must be at least 2")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    source_zip = args.source_zip.expanduser().resolve()
    if not source_zip.is_file():
        raise SystemExit(f"Source archive not found: {source_zip}")

    selected_runs = None if args.all_runs else (
        args.run if args.run is not None else RUNS_FILTER
    )
    runs_filter = set(selected_runs) if selected_runs is not None else None
    video_members = find_video_members(source_zip, args.task, runs_filter)
    if not video_members:
        raise SystemExit("No sensor videos matched the task and run selection")
    print(f"Source archive: {source_zip}")
    print(f"Task: {normalize_task_name(args.task)}")
    print(f"Independent model passes per video: {args.repeats}")
    with zipfile.ZipFile(source_zip, "r") as archive:
        for run, member in video_members.items():
            print(f"  run {run}: {member}")
            print(f"    actions: {find_run_member(archive, args.task, run, 'metrics')}")
            print(f"    poses:   {find_run_member(archive, args.task, run, 'hdf5')}")
    if args.dry_run:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    encoder_overrides = {
        "dual_dinov3": args.dino_checkpoint,
        "dual_vjepa": args.vjepa_checkpoint,
        "ind_dinov3": args.dino_checkpoint,
    }
    predictor_overrides = {
        "dual_dinov3": (args.dual_dino_predictor,),
        "dual_vjepa": (args.dual_vjepa_predictor,),
        "ind_dinov3": (
            args.ind_dino_side_predictor,
            args.ind_dino_wrist_predictor,
        ),
    }
    conditioning_by_run = {}
    with zipfile.ZipFile(source_zip, "r") as archive:
        for run in video_members:
            print(f"Loading shared actions and poses for run {run}")
            conditioning_by_run[run] = load_trajectory_conditioning(
                archive, args.task, run
            )

    temporal_results: dict[str, list[RunTemporalLoss]] = defaultdict(list)
    predictor_results: dict[str, list[RunTemporalLoss]] = defaultdict(list)
    for model_name in PREDICTOR_MODELS:
        spec = MODEL_SPECS[model_name]
        try:
            encoder_checkpoint = resolve_checkpoint(
                encoder_overrides[model_name], spec.encoder_checkpoint
            )
            predictor_checkpoints = tuple(
                resolve_checkpoint(override, default_name)
                for override, default_name in zip(
                    predictor_overrides[model_name],
                    spec.predictor_checkpoints,
                )
            )
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        print(f"Loading {spec.label} encoder: {encoder_checkpoint}")
        for checkpoint in predictor_checkpoints:
            print(f"Loading {spec.label} predictor: {checkpoint}")
        world_model = make_world_model(
            model_name,
            device,
            encoder_checkpoint,
            predictor_checkpoints,
        )
        with zipfile.ZipFile(source_zip, "r") as archive:
            for run, member in video_members.items():
                for repeat in range(1, args.repeats + 1):
                    try:
                        if model_name in ENCODER_MODELS:
                            temporal_results[model_name].append(
                                analyze_video(
                                    archive,
                                    member,
                                    run,
                                    spec.label,
                                    world_model,
                                    args.num_steps,
                                    repeat,
                                    args.repeats,
                                )
                            )
                        predictor_results[model_name].append(
                            analyze_predictor_video(
                                archive,
                                member,
                                run,
                                model_name,
                                world_model,
                                conditioning_by_run[run],
                                args.num_steps,
                                repeat,
                                args.repeats,
                            )
                        )
                    except Exception as error:
                        if args.skip_errors:
                            print(
                                f"Skipping run {run}, repeat {repeat}: "
                                f"{error}",
                                file=sys.stderr,
                            )
                            continue
                        raise RuntimeError(
                            f"failed run {run}, repeat {repeat} with "
                            f"{spec.label}"
                        ) from error
        del world_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if any(not temporal_results[model_name] for model_name in ENCODER_MODELS):
        raise SystemExit("At least one encoder produced no results")
    if any(not predictor_results[model_name] for model_name in PREDICTOR_MODELS):
        raise SystemExit("At least one predictor produced no results")
    print_summary(
        temporal_results,
        "GT temporal L1 over all repeats and transition steps",
        ENCODER_LABELS,
    )
    print_summary(
        predictor_results,
        "Predicted-patch vs input-patch L1 over shared actions",
    )
    output_path = args.output.expanduser().resolve()
    output_paths = plot_results(
        temporal_results,
        predictor_results,
        output_path,
    )
    for path in output_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
