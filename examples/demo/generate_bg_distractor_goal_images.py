# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Generate goal images under assets/wm_tasks/bg_distractors for every scene variant."""

import argparse
import os
import sys
import traceback

import cv2  # Must import this before isaaclab. Do not remove.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-dirs", nargs="+", default=["wm_tasks/bg_distractor"])
parser.add_argument("--task", nargs="+", default=None)
parser.add_argument("--num-backgrounds", "--num_backgrounds", type=int, default=5)
parser.add_argument("--background-seed", "--background_seed", type=int, default=1)
parser.add_argument("--backgrounds", nargs="+", default=None)
parser.add_argument("--object-counts", "--object_counts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--instruction-type", "--instruction_type", default="default")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.livestream = 0
args_cli.enable_cameras = True
args_cli.activate_contact_sensors = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.factory import get_envs_by_tag  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid_ee.auto_env_registrations_angled_bg_variations import (  # noqa: E402
    auto_register_droid_ee_envs_bg_variations,
)
from robolab.tasks.wm_tasks.goal_images import generate_goal_images, goal_image_paths  # noqa: E402


def main():
    auto_register_droid_ee_envs_bg_variations(
        task_dirs=args_cli.task_dirs,
        tasks=args_cli.task,
        backgrounds=args_cli.backgrounds,
        num_backgrounds=args_cli.num_backgrounds,
        background_seed=args_cli.background_seed,
        object_counts=args_cli.object_counts,
    )
    task_envs = sorted(get_envs_by_tag("angled_reach_background_variations"))
    if not task_envs:
        raise RuntimeError("No bg_distractor environments were registered.")

    print(f"[RoboLab] Preparing goal images for {len(task_envs)} variants.", flush=True)
    generated = 0
    skipped = 0
    for index, task_env in enumerate(task_envs, start=1):
        env = None
        env_cfg = parse_env_cfg(
            task_env,
            device=args_cli.device,
            seed=0,
            num_envs=1,
            use_fabric=True,
        )
        paths = goal_image_paths(env_cfg)
        if not args_cli.overwrite and all(path.exists() for path in paths.values()):
            print(f"[{index}/{len(task_envs)}] Goal images exist, skipping {task_env}", flush=True)
            skipped += 1
            continue

        print(f"[{index}/{len(task_envs)}] Generating {task_env}", flush=True)
        output_dir = os.path.join(PACKAGE_DIR, "output", "bg_distractor_goal_generation", task_env)
        os.makedirs(output_dir, exist_ok=True)
        set_output_dir(output_dir)

        # Goal-pose replay is a capture trajectory, not an evaluated episode.
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        try:
            env, env_cfg = create_env(
                env_cfg,
                instruction_type=args_cli.instruction_type,
                policy="valpa",
            )
            generate_goal_images(env, env_cfg, overwrite=args_cli.overwrite)
            generated += 1
        finally:
            if env is not None:
                env.close()

    print(
        f"[RoboLab] Goal generation complete: generated={generated}, skipped={skipped}, "
        f"total={len(task_envs)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[RoboLab] Goal generation failed: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
