# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Goal image cache/generation helpers for VALPA world-model tasks.

This module keeps task goal-image capture out of the policy episode runner.
It can also be run directly inside an Isaac Lab Python session to precompute
goal images under ``assets/wm_tasks``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import torch


# TODO: This is a hack to make goal image generation work with mounted repo onto apptainer.
# REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path("/anvme/workspace/v106be10-valpa-robolab/RoboLab")
WM_GOAL_DIR = REPO_ROOT / "assets" / "wm_tasks"



def goal_image_paths(env_cfg) -> dict[str, Path]:
    external_key = env_cfg.goal.get("external_camera", "over_shoulder_right_camera")
    wrist_key = env_cfg.goal.get("wrist_camera", "wrist_cam")
    task_name = getattr(env_cfg, "_task_name", env_cfg.__class__.__name__)
    configured_root = getattr(env_cfg, "_goal_image_dir", None)
    root = Path(configured_root) if configured_root else WM_GOAL_DIR / task_name
    return {
        "external": root / f"{external_key}.png",
        "wrist": root / f"{wrist_key}.png",
        "status": root / "status.json",
    }


def _save_rgb_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = image.detach().cpu().numpy()
    if image_np.dtype != "uint8":
        image_np = image_np.clip(0, 255).astype("uint8")
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_np):
        raise OSError(f"Failed to write goal image: {path}")


def _compute_reach_goal_positions(env, target_object: str, z_offset: float) -> torch.Tensor:
    from robolab.core.world.world_state import get_world

    # target position
    world = get_world(env)
    corners, centroid = world.get_bbox(target_object, env_id=None)
    target_positions = centroid.clone()
    target_positions[:, 2] = corners[:, :, 2].max(dim=1).values + z_offset
    return target_positions + env.scene.env_origins


def _angled_status_path(task_name: str) -> Path:
    task_root = WM_GOAL_DIR / task_name
    for filename in ("status.json", "status_draft.json"):
        candidate = task_root / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No angled-reach status pose found for {task_name} in {task_root}.")


def drive_to_valpa_goal(env, env_cfg, obs: dict | None = None) -> dict:
    """Drive the robot to the task's configured goal pose and return latest obs."""
    from robolab.core.world.world_state import get_world

    mode = env_cfg.goal.get("mode")
    reached = False

    if obs is None:
        obs, _ = env.reset()

    target_object = env_cfg.goal["object"]
    max_steps = int(env_cfg.goal.get("drive_steps", 80))
    settle_steps = int(env_cfg.goal.get("settle_steps", 4))
    ik_action_scale = float(env_cfg.goal.get("ik_action_scale", 0.5))
    max_action = float(env_cfg.goal.get("max_action", 0.10))
    link_name = env_cfg.goal.get("link_name", "panda_link8")

    action_dim = 7
    actions = torch.zeros(env.num_envs, action_dim, device=env.device)

    if mode == "reach":
        z_offset = float(env_cfg.goal.get("z_offset", 0.15))
        target_positions = _compute_reach_goal_positions(env, target_object, z_offset)
        for _ in range(max_steps):
            gripper_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)
            pos_error = target_positions - gripper_pose[:, :3]
            pos_done = torch.linalg.norm(pos_error, dim=1).max().item() <= 0.001

            actions.zero_()
            actions[:, :3] = torch.clamp(pos_error, -max_action, max_action)

            if pos_done:
                reached = True
                last_dist = torch.linalg.norm(pos_error, dim=1).max().item()
                last_gripper_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)
                break
            obs, _, _, _, _ = env.step(actions)

    elif mode == "angled_reach":
        import isaaclab.utils.math as math_utils

        task_name = getattr(env_cfg, "_task_name", env_cfg.__class__.__name__)
        status_path = _angled_status_path(task_name)
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        target_pose = status.get("last_ee_pose")
        if not isinstance(target_pose, list) or len(target_pose) != 7:
            raise ValueError(f"{status_path} must contain a 7D 'last_ee_pose'.")

        target = torch.tensor(target_pose, dtype=torch.float32, device=env.device)
        target_pos = target[:3].unsqueeze(0).repeat(env.num_envs, 1)
        target_quat = target[3:7].unsqueeze(0).repeat(env.num_envs, 1)
        pos_tolerance = float(env_cfg.goal.get("drive_pos_tolerance", 0.01))
        angle_tolerance = float(env_cfg.goal.get("drive_angle_tolerance", 0.08))
        last_angle_dist = math.inf

        for _ in range(max(1, max_steps)):
            gripper_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)
            pos_error = target_pos - gripper_pose[:, :3]

            current_quat = torch.nn.functional.normalize(gripper_pose[:, 3:7], dim=-1)
            desired_quat = torch.nn.functional.normalize(target_quat, dim=-1)
            desired_quat = torch.where(
                (current_quat * desired_quat).sum(dim=-1, keepdim=True) < 0.0,
                -desired_quat,
                desired_quat,
            )
            delta_quat = math_utils.quat_mul(
                desired_quat,
                math_utils.quat_conjugate(current_quat),
            )
            rot_error = math_utils.axis_angle_from_quat(delta_quat)

            last_dist = torch.linalg.norm(pos_error, dim=1).max().item()
            last_angle_dist = torch.linalg.norm(rot_error, dim=1).max().item()
            if last_dist <= pos_tolerance and last_angle_dist <= angle_tolerance:
                reached = True
                break

            actions.zero_()
            actions[:, :3] = torch.clamp(pos_error, -max_action, max_action)
            actions[:, 3:6] = torch.clamp(rot_error, -max_action, max_action)
            obs, _, _, _, _ = env.step(actions)

        print(
            f"[RoboLab] Angled goal drive: reached={reached}, "
            f"pos_err={last_dist:.4f}, angle_err={last_angle_dist:.4f}",
            flush=True,
        )
    else:
        raise ValueError(f"Unsupported goal mode: {mode}")

    actions.zero_()
    for _ in range(settle_steps):
        obs, _, _, _, _ = env.step(actions)

    last_dist = torch.linalg.norm(pos_error, dim=1).max().item()
    last_gripper_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)
    return obs, reached, last_dist, last_gripper_pose


def generate_goal_images(env, env_cfg, obs: dict | None = None, *, overwrite: bool = False):
    """Generate and cache one canonical pair of goal images for a WM task."""
    
    paths = goal_image_paths(env_cfg)
    if not overwrite and all(path.exists() for path in paths.values()):
        return

    task_name = getattr(env_cfg, "_task_name")
    print(f"\033[96m[RoboLab] Generating goal images for {task_name}\033[0m")
    goal_obs, reached, last_dist, last_gripper_pose = drive_to_valpa_goal(env, env_cfg, obs=obs)
    if not reached:
        raise RuntimeError(
            f"Robot did not reach the configured goal pose for {task_name}; "
            f"last position error was {last_dist:.4f} m."
        )

    # save images
    external_key = env_cfg.goal.get("external_camera", "over_shoulder_right_camera")
    wrist_key = env_cfg.goal.get("wrist_camera", "wrist_cam")
    _save_rgb_image(goal_obs["image_obs"][external_key][0], paths["external"])
    _save_rgb_image(goal_obs["image_obs"][wrist_key][0], paths["wrist"])
    
    # save status in json
    ee_pose = last_gripper_pose.detach().cpu()
    if ee_pose.ndim > 1:
        ee_pose = ee_pose[0]
    ee_pose = [float(x) for x in ee_pose.tolist()]
    status_payload = {
        "reached": bool(reached),
        "last_distance": float(last_dist),
        "last_ee_pose": ee_pose,
    }

    with open(paths["status"], "w") as f:
        json.dump(status_payload, f, indent=2)

    
    return


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Generate cached goal images for WM tasks.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--task", required=True, help="Task name to generate, e.g. ReachBananaTask.")
    parser.add_argument("--task-dirs", nargs="+", default=["wm_tasks"], help="Task folders to register.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--instruction-type", "--instruction_type", default="default")
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid_ee.auto_env_registrations import auto_register_droid_ee_envs

        auto_register_droid_ee_envs(task_dirs=args_cli.task_dirs, task=[args_cli.task])
        task_envs = get_envs(task=[args_cli.task])
        if not task_envs:
            raise ValueError(f"Task '{args_cli.task}' was not registered.")

        env, env_cfg = create_env(
            task_envs[0],
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy="valpa",
        )
        generate_goal_images(env, env_cfg)
        env.close()

    finally:
        simulation_app.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
