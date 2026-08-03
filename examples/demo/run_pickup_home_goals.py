# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive to a manually captured pickup pose, grasp, and return home.

The script expects the first goal capture to already exist under:

    assets/wm_tasks/<TaskName>/over_shoulder_right_camera_1.png
    assets/wm_tasks/<TaskName>/wrist_cam_1.png
    assets/wm_tasks/<TaskName>/status_draft.json

Goal assets are always loaded from and saved to ``assets/wm_tasks/<TaskName>/``.
``status_draft.json`` must contain ``last_ee_pose``. The script drives the
end-effector to that pose, closes the gripper, saves goal images with suffix
``_2`` and writes ``last_ee_pose_2`` back into ``status_draft.json``. It then
drives back to the reset/home end-effector pose while keeping the gripper
closed, saves goal images with suffix ``_3``, and writes ``last_ee_pose_3``
back into ``status_draft.json``.
"""

import argparse
import json
import math
from pathlib import Path

import cv2  # Must import this before isaaclab. Do not remove.
from isaaclab.app import AppLauncher

DEFAULT_KIT_ARGS = "--/app/livestream/publicEndpointAddress=172.29.5.11  --/app/livestream/port=49100"

parser = argparse.ArgumentParser(description="Replay pickup goals for a DROID WM task.")
parser.add_argument("--task", required=True, help="Task name, e.g. AngledPickupDrillTask.")
parser.add_argument(
    "--task-dirs",
    nargs="+",
    default=["wm_tasks/angledpickup", "wm_tasks/angledreach"],
    help="Python task source folders under robolab/tasks; goal assets are inferred from --task.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--status-in", default="status_draft.json", help="Input status file in the task goal folder.")
parser.add_argument("--status-out", default="status_draft.json", help="Output status file in the task goal folder.")
parser.add_argument("--pose-key", default="last_ee_pose", help="Pose key to drive to from the input status file.")
parser.add_argument("--drive-steps", type=int, default=140, help="Maximum steps for each automated pose drive.")
parser.add_argument("--drive-pos-tol", type=float, default=0.01, help="Position tolerance in meters.")
parser.add_argument("--drive-angle-tol", type=float, default=0.08, help="Orientation tolerance in radians.")
parser.add_argument(
    "--max-drive-action",
    type=float,
    default=0.10,
    help="Maximum absolute xyz/rotation action component during automated drives.",
)
parser.add_argument("--gripper-steps", type=int, default=160, help="Number of close-gripper env steps.")
parser.add_argument("--settle-steps", type=int, default=2, help="Settling steps after grasp and after home drive.")
parser.add_argument("--force-livestream", action="store_true", help="Force WebRTC livestream mode.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# Goal assets are always resolved from assets/wm_tasks/<TaskName>/ via goal_image_paths().
# If an asset path is accidentally passed as --task-dirs, keep registration usable.
if any(str(task_dir).startswith("assets/wm_tasks") for task_dir in args_cli.task_dirs):
    print(
        "Ignoring asset-style --task-dirs; goal assets are inferred from --task. "
        "Using WM Python task source folders instead.",
        flush=True,
    )
    args_cli.task_dirs = ["wm_tasks/angledpickup", "wm_tasks/angledreach"]

if args_cli.force_livestream:
    args_cli.livestream = 2
    args_cli.kit_args = DEFAULT_KIT_ARGS

args_cli.enable_cameras = True
args_cli.activate_contact_sensors = True
args_cli.save_videos = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.utils.math as math_utils  # noqa: E402

from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.tasks.wm_tasks.goal_images import _save_rgb_image, goal_image_paths  # noqa: E402


def _zero_action(env):
    return torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)


def _task_goal_root(env_cfg) -> Path:
    return goal_image_paths(env_cfg)["status"].parent


def _status_path(env_cfg, filename: str) -> Path:
    return _task_goal_root(env_cfg) / filename


def _load_status(env_cfg) -> dict:
    path = _status_path(env_cfg, args_cli.status_in)
    if not path.exists():
        raise FileNotFoundError(f"Missing input status file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if args_cli.pose_key not in payload:
        raise KeyError(f"{path} does not contain {args_cli.pose_key!r}")
    return payload


def _read_output_status(env_cfg, fallback: dict) -> dict:
    path = _status_path(env_cfg, args_cli.status_out)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = dict(fallback)
    return payload


def _write_output_status(env_cfg, payload: dict) -> None:
    path = _status_path(env_cfg, args_cli.status_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _current_ee_pose(env, env_cfg) -> torch.Tensor:
    link_name = env_cfg.goal.get("link_name", "panda_link8")
    return get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)[0]


def _pose_to_list(pose: torch.Tensor) -> list[float]:
    return [float(x) for x in pose.detach().cpu().tolist()]


def _gripper_joint_state(env) -> tuple[list[float], list[float]]:
    robot = env.scene["robot"]
    joint_idx = robot.data.joint_names.index("finger_joint")
    actual = robot.data.joint_pos[:, joint_idx].detach().cpu().tolist()
    target = robot.data.joint_pos_target[:, joint_idx].detach().cpu().tolist()
    return [float(x) for x in actual], [float(x) for x in target]


def _print_gripper_joint_state(env, label: str) -> None:
    actual, target = _gripper_joint_state(env)
    print(f"{label}: finger_joint actual={actual}, target={target}", flush=True)


def _quat_error_axis_angle(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    current_quat = torch.nn.functional.normalize(current_quat, dim=-1)
    target_quat = torch.nn.functional.normalize(target_quat, dim=-1)
    target_quat = torch.where(
        (current_quat * target_quat).sum(dim=-1, keepdim=True) < 0.0,
        -target_quat,
        target_quat,
    )
    delta_quat = math_utils.quat_mul(target_quat, math_utils.quat_conjugate(current_quat))
    return math_utils.axis_angle_from_quat(delta_quat)


def _drive_to_pose(env, env_cfg, obs, target_pose, *, gripper_action: float, label: str):
    link_name = env_cfg.goal.get("link_name", "panda_link8")
    target = torch.tensor(target_pose, dtype=torch.float32, device=env.device)
    if target.numel() != 7:
        raise ValueError(f"Expected 7D pose for {label}, got {target.numel()} values")

    target_pos = target[:3].unsqueeze(0).repeat(env.num_envs, 1)
    target_quat = target[3:7].unsqueeze(0).repeat(env.num_envs, 1)
    reached = False
    last_pos_err = math.inf
    last_angle_err = math.inf

    for _ in range(max(1, args_cli.drive_steps)):
        current_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)
        pos_error = target_pos - current_pose[:, :3]
        rot_error = _quat_error_axis_angle(current_pose[:, 3:7], target_quat)

        last_pos_err = torch.linalg.norm(pos_error, dim=1).max().item()
        last_angle_err = torch.linalg.norm(rot_error, dim=1).max().item()
        if last_pos_err <= args_cli.drive_pos_tol and last_angle_err <= args_cli.drive_angle_tol:
            reached = True
            break

        action = _zero_action(env)
        action[:, :3] = torch.clamp(pos_error, -args_cli.max_drive_action, args_cli.max_drive_action)
        action[:, 3:6] = torch.clamp(rot_error, -args_cli.max_drive_action, args_cli.max_drive_action)
        action[:, 6] = gripper_action
        obs, _, _, _, _ = env.step(action)

    status = "reached" if reached else "stopped"
    print(f"drive {status} {label}: pos_err={last_pos_err:.4f}, angle_err={last_angle_err:.4f}", flush=True)
    return obs


def _close_gripper(env, obs):
    _print_gripper_joint_state(env, "before close")
    action = _zero_action(env)
    action[:, 6] = 1.0
    for _ in range(max(1, args_cli.gripper_steps)):
        obs, _, _, _, _ = env.step(action)
    _print_gripper_joint_state(env, "after close")
    return obs


def _settle(env, obs, *, gripper_action: float):
    action = _zero_action(env)
    action[:, 6] = gripper_action
    for _ in range(max(0, args_cli.settle_steps)):
        obs, _, _, _, _ = env.step(action)
    return obs


def _save_goal(env, env_cfg, obs, status_payload: dict, suffix: int, label: str):
    paths = goal_image_paths(env_cfg)
    external_key = env_cfg.goal.get("external_camera", "over_shoulder_right_camera")
    wrist_key = env_cfg.goal.get("wrist_camera", "wrist_cam")

    external_path = paths["external"].with_name(f"{paths['external'].stem}_{suffix}{paths['external'].suffix}")
    wrist_path = paths["wrist"].with_name(f"{paths['wrist'].stem}_{suffix}{paths['wrist'].suffix}")

    _save_rgb_image(obs["image_obs"][external_key][0], external_path)
    _save_rgb_image(obs["image_obs"][wrist_key][0], wrist_path)

    ee_pose = _pose_to_list(_current_ee_pose(env, env_cfg))
    gripper_actual, gripper_target = _gripper_joint_state(env)
    status_payload["reached"] = True
    status_payload["manual_capture"] = True
    status_payload["automated_pickup_home"] = True
    status_payload["last_distance"] = 0.0
    status_payload[f"last_ee_pose_{suffix}"] = ee_pose
    status_payload[f"gripper_joint_pos_{suffix}"] = gripper_actual
    status_payload[f"gripper_joint_target_{suffix}"] = gripper_target
    _write_output_status(env_cfg, status_payload)

    print(f"saved {label}: {external_path.name}, {wrist_path.name}, last_ee_pose_{suffix}", flush=True)
    return obs


def main():
    from robolab.core.environments.factory import get_envs
    from robolab.registrations.droid_ee.auto_env_registrations_angled import auto_register_droid_ee_envs

    env = None
    try:
        auto_register_droid_ee_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)
        task_envs = get_envs(task=args_cli.task)
        if not task_envs:
            raise RuntimeError(f"No registered env found for task {args_cli.task!r}")

        env_cfg = parse_env_cfg(
            task_envs[0],
            device=args_cli.device,
            seed=0,
            num_envs=args_cli.num_envs,
            use_fabric=True,
        )
        # This is a goal-capture trajectory, not an episode evaluation. Pickup
        # success and the normal episode time limit would otherwise terminate
        # and auto-reset the environment partway through the return-home drive.
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None

        env, env_cfg = create_env(
            env_cfg,
            policy="a",
        )
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(_zero_action(env))

        input_status = _load_status(env_cfg)
        status_payload = _read_output_status(env_cfg, input_status)
        status_payload["last_ee_pose"] = input_status[args_cli.pose_key]
        _write_output_status(env_cfg, status_payload)

        home_pose = _pose_to_list(_current_ee_pose(env, env_cfg))
        print(f"loaded {args_cli.pose_key} from {_status_path(env_cfg, args_cli.status_in)}", flush=True)

        obs = _drive_to_pose(
            env,
            env_cfg,
            obs,
            input_status[args_cli.pose_key],
            gripper_action=0.0,
            label="goal 1 pickup pose",
        )
        # obs = _save_goal(env, env_cfg, obs, status_payload, 1, "goal 1 home")
        # exit()

        obs = _close_gripper(env, obs)
        obs = _settle(env, obs, gripper_action=1.0)
        obs = _save_goal(env, env_cfg, obs, status_payload, 2, "goal 2 grasp")

        obs = _drive_to_pose(
            env,
            env_cfg,
            obs,
            home_pose,
            gripper_action=1.0,
            label="goal 3 home with object",
        )
        obs = _settle(env, obs, gripper_action=1.0)
        obs = _save_goal(env, env_cfg, obs, status_payload, 3, "goal 3 home")

    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
