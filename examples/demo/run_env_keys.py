# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard-driven manual goal capture for a generated DROID environment.

This applies small relative end-effector jog commands from stdin and overwrites
the cached goal files after each command:

    assets/wm_tasks/<TaskName>/over_shoulder_right_camera.png
    assets/wm_tasks/<TaskName>/wrist_cam.png
    assets/wm_tasks/<TaskName>/status_draft.json

Usage:
    $ python examples/demo/run_env_keys.py --headless

    Capture a different angled task:
    $ python examples/demo/run_env_keys.py --headless --task AngledReachShelfForkTask

Press Ctrl+C or q when the saved pose/images look good.
"""

import argparse
import json
import re

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

DEFAULT_KIT_ARGS = "--/app/livestream/publicEndpointAddress=172.29.5.11  --/app/livestream/port=49100"

parser = argparse.ArgumentParser(description="Keyboard EE jogger for manual DROID goal capture.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn.",
)
parser.add_argument(
    "--task",
    type=str,
    default="AngledReachDrillTask",
    help="Registered angled task to open.",
)
parser.add_argument(
    "--task-dirs",
    nargs="+",
    default=["wm_tasks/angledreach"],
    help="Task folders passed to the angled EE registrar.",
)
parser.add_argument(
    "--pos-step",
    type=float,
    default=0.15,
    help="Translation jog size in meters before the IK action scale is applied.",
)
parser.add_argument(
    "--rot-step",
    type=float,
    default=0.75,
    help="Angle-axis rotation jog size in radians before the IK action scale is applied.",
)
parser.add_argument(
    "--repeat-steps",
    type=int,
    default=1,
    help="Number of env steps to repeat each nonzero key command.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=0,
    help="Zero-action steps after each key command before saving images.",
)
parser.add_argument(
    "--gripper-steps",
    type=int,
    default=12,
    help="Number of env steps to repeat each gripper open/close command.",
)
parser.add_argument(
    "--pickup-lift-steps",
    type=int,
    default=4,
    help="Number of upward env steps for the v pickup macro.",
)
parser.add_argument(
    "--pickup-lift-action",
    type=float,
    default=0.15,
    help="Raw +z action value for each upward step in the v pickup macro.",
)
parser.add_argument(
    "--force-livestream",
    action="store_true",
    help="Force WebRTC livestream mode in addition to terminal control.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.activate_contact_sensors = True
args_cli.save_videos = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from robolab.core.environments.runtime import create_env  # noqa
from robolab.core.world.world_state import get_world
from robolab.tasks.wm_tasks.goal_images import _save_rgb_image, goal_image_paths


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

SIMPLE_HELP = """
Type keys, then Enter. Repeated letters repeat the move.
  w/s: +x/-x    a/d: +y/-y    r/f: +z/-z
  i/k: +rx/-rx  j/l: +ry/-ry  u/o: +rz/-rz  (angle-axis)
  c: close and keep squeezing    b: open gripper
  v: close gripper, lift upward, and keep squeezing
  .: save       x: reset      q: quit
"""


def _simple_zero_action(env):
    return torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)


def _simple_action(env, key: str, gripper_hold: float = 0.0):
    action = _simple_zero_action(env)
    action[:, 6] = gripper_hold
    commands = {
        "w": (0, args_cli.pos_step),
        "s": (0, -args_cli.pos_step),
        "a": (1, args_cli.pos_step),
        "d": (1, -args_cli.pos_step),
        "r": (2, args_cli.pos_step),
        "f": (2, -args_cli.pos_step),
        "i": (3, args_cli.rot_step),
        "k": (3, -args_cli.rot_step),
        "j": (4, args_cli.rot_step),
        "l": (4, -args_cli.rot_step),
        "u": (5, args_cli.rot_step),
        "o": (5, -args_cli.rot_step),
    }
    if key not in commands:
        return None
    idx, value = commands[key]
    action[:, idx] = value
    return action


def _apply_gripper(env, obs, value: float, label: str):
    action = _simple_zero_action(env)
    action[:, 6] = value
    for _ in range(max(1, args_cli.gripper_steps)):
        obs, _, _, _, _ = env.step(action)
    print(label, flush=True)
    return obs


def _pickup_lift(env, obs):
    obs = _apply_gripper(env, obs, 1.0, "closed gripper")
    action = _simple_zero_action(env)
    action[:, 2] = args_cli.pickup_lift_action
    action[:, 6] = 1.0
    for _ in range(max(1, args_cli.pickup_lift_steps)):
        obs, _, _, _, _ = env.step(action)
    print("lifted with gripper closing", flush=True)
    return obs


def _clean_keys(keys: str) -> str:
    keys = ANSI_ESCAPE_RE.sub("", keys)
    return "".join(key for key in keys.lower() if key in "wsadrfijkluocvb.xqh")


def _write_status(paths, ee_pose):
    status_path = paths["status"].with_name("status_draft.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "reached": True,
                "manual_capture": True,
                "last_distance": 0.0,
                "last_ee_pose": ee_pose,
            },
            handle,
            indent=2,
        )


def _simple_save(env, env_cfg, obs, label: str):
    paths = goal_image_paths(env_cfg)
    external_key = env_cfg.goal.get("external_camera", "over_shoulder_right_camera")
    wrist_key = env_cfg.goal.get("wrist_camera", "wrist_cam")
    image_obs = obs["image_obs"]

    _save_rgb_image(image_obs[external_key][0], paths["external"])
    _save_rgb_image(image_obs[wrist_key][0], paths["wrist"])

    link_name = env_cfg.goal.get("link_name", "panda_link8")
    ee_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)[0]
    ee_pose = [float(x) for x in ee_pose.detach().cpu().tolist()]
    _write_status(paths, ee_pose)
    print(f"saved {label}: {paths['status'].parent}", flush=True)
    return obs


def main():
    from robolab.core.environments.factory import get_envs
    # TODO: change here
    # from robolab.registrations.droid_ee.auto_env_registrations_angled import auto_register_droid_ee_envs
    from robolab.registrations.droid_ee.auto_env_registrations import auto_register_droid_ee_envs

    env = None
    try:
        auto_register_droid_ee_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)
        task_envs = get_envs(task=args_cli.task)
        if not task_envs:
            raise RuntimeError(f"No registered env found for task {args_cli.task!r}")

        env, env_cfg = create_env(
            task_envs[0],
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            policy="valpa",
        )
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(_simple_zero_action(env))
        print(SIMPLE_HELP, flush=True)
        _simple_save(env, env_cfg, obs, "initial")

        gripper_hold = 0.0

        while True:
            keys = _clean_keys(input("key> ").strip())
            if not keys:
                continue
            if "q" in keys:
                break
            if "h" in keys:
                print(SIMPLE_HELP, flush=True)
                continue
            if "x" in keys:
                obs, _ = env.reset()
                obs, _, _, _, _ = env.step(_simple_zero_action(env))
                gripper_hold = 0.0
                _simple_save(env, env_cfg, obs, "reset")
                continue
            if keys == ".":
                _simple_save(env, env_cfg, obs, "current")
                continue

            applied = 0
            for key in keys:
                if key == ".":
                    continue
                if key == "c":
                    obs = _apply_gripper(env, obs, 1.0, "closed gripper")
                    gripper_hold = 1.0
                    applied += 1
                    continue
                if key == "b":
                    obs = _apply_gripper(env, obs, -1.0, "opened gripper")
                    gripper_hold = 0.0
                    applied += 1
                    continue
                if key == "v":
                    obs = _pickup_lift(env, obs)
                    gripper_hold = 1.0
                    applied += 1
                    continue

                action = _simple_action(env, key, gripper_hold=gripper_hold)
                if action is None:
                    print(f"unknown key {key!r}; skipping", flush=True)
                    continue

                for _ in range(max(1, args_cli.repeat_steps)):
                    obs, _, _, _, _ = env.step(action)
                applied += 1

            if applied == 0:
                continue
            settle_action = _simple_zero_action(env)
            settle_action[:, 6] = gripper_hold
            for _ in range(max(0, args_cli.settle_steps)):
                obs, _, _, _, _ = env.step(settle_action)
            _simple_save(env, env_cfg, obs, keys)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Closing environment.", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
