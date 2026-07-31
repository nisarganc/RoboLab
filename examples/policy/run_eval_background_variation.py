# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run policy evaluation across a TASK × BACKGROUND matrix.

This script registers every (task, background) combination as a separate env
(via auto_register_droid_envs_bg_variations) and evaluates each one. Use it to
measure robustness of the SAME task across MANY backgrounds.

For "give each task a different random background in a single benchmark pass"
(per-task random, NOT a matrix sweep), use `run_eval.py --randomize-background`
instead. See docs/background.md → "Choosing a Background Strategy".

Usage:
    $ python run_eval_background_variation.py --headless
    $ python run_eval_background_variation.py --task BananaInBowlTableTask --headless
    $ python run_eval_background_variation.py --angled-reach --headless

Output:
    Results are saved to: output/<output_folder_name>/
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import re
import traceback
import sys
from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None,
                       help="List of tasks to evaluate on ")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks to evaluate on ")
parser.add_argument("--task-dirs", nargs='+', default=None,
                       help="List of task directories to evaluate on")
parser.add_argument("--policy", choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "valpa", "molmo", "openvla", "openvla_oft"], default=None,
                       help="Action-prediction backend (default: valpa for --angled-reach, otherwise pi05)")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                       help="Number of sequential runs per task (default: 1). Total episodes = num_runs * num_envs. Prefer increasing --num_envs for more episodes. Only increase --num-runs if you run out of GPU memory with the desired num_envs.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                       help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                       help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                       help="Output folder name under /robolab/output.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                       help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                       help="Debug output (default: False)")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                       help="Remote host for policy server (default: localhost)")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                       help="Remote port for policy server (default: 8000)")
parser.add_argument("--angled-reach", "--angled_reach", action="store_true",
                    help="Use the droid-EE angled-task registrar and default to all angled-reach tasks.")
parser.add_argument("--num-backgrounds", "--num_backgrounds", type=int, default=5,
                    help="Number of backgrounds sampled for --angled-reach (default: 5).")
parser.add_argument("--background-seed", "--background_seed", type=int, default=1,
                    help="Seed for deterministic background sampling (default: 1).")
parser.add_argument("--backgrounds", nargs='+', default=None,
                    help="Explicit background filenames or paths; overrides --num-backgrounds.")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                    help="Which instruction variant to use (default: default).")
parser.add_argument("--video-mode", "--video_mode", choices=["all", "viewport", "sensor", "none"], default="sensor",
                    help="Which videos to save (default: sensor).")
args_cli, _= parser.parse_known_args()
if args_cli.task_dirs is None:
    args_cli.task_dirs = ["wm_tasks/bg_distractor"] if args_cli.angled_reach else DEFAULT_TASK_SUBFOLDERS
if args_cli.policy is None:
    args_cli.policy = "valpa" if args_cli.angled_reach else "pi05"
args_cli.livestream = 0
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir # noqa
from robolab.core.environments.runtime import create_env # noqa
from robolab.eval import create_client, run_episode, summarize_run # noqa
from robolab.core.environments.factory import get_envs_by_tag # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete # noqa
from robolab.core.logging.results import init_experiment, summarize_experiment_results # noqa
import robolab.constants # noqa

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

if args_cli.angled_reach:
    from robolab.registrations.droid_ee.auto_env_registrations_angled_bg_variations import auto_register_droid_ee_envs_bg_variations # noqa
    auto_register_droid_ee_envs_bg_variations(
        task_dirs=args_cli.task_dirs,
        tasks=args_cli.task,
        backgrounds=args_cli.backgrounds,
        num_backgrounds=args_cli.num_backgrounds,
        background_seed=args_cli.background_seed,
    )
    variation_tag = "angled_reach_background_variations"
else:
    from robolab.registrations.droid_jointpos.auto_env_registrations_bg_variations import auto_register_droid_envs_bg_variations # noqa
    auto_register_droid_envs_bg_variations(
        task_dirs=args_cli.task_dirs,
        tasks=args_cli.task,
        backgrounds=args_cli.backgrounds,
    )
    variation_tag = "background_variations"


def main():
    """Main function."""
    if args_cli.output_folder_name is None:
        if args_cli.angled_reach and args_cli.policy == "valpa":
            from robolab_policy_client.valpa import VALPADroidEEClient
            metadata_client = VALPADroidEEClient(
                remote_host=args_cli.remote_host,
                remote_port=args_cli.remote_port,
            )
            model_name = metadata_client.metadata()["modelname"]
            metadata_client.close()
            background_count = len(args_cli.backgrounds) if args_cli.backgrounds else args_cli.num_backgrounds
            args_cli.output_folder_name = (
                f"{model_name}_angledreach_bg{background_count}_objects1to5_seed{args_cli.background_seed}"
            )
        else:
            args_cli.output_folder_name = get_timestamp() + f"_{args_cli.policy}_background_variation"

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    task_envs = get_envs_by_tag(variation_tag)

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print(f"Output directory: {output_dir}")
    print(f"Running {len(task_envs)} background variation environments, {total_episodes} episodes each")

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:

        bg_name = task_env.split("_bg_")[-1] if "_bg_" in task_env else "default"
        variant_task_name = task_env.split("_bg_")[0]
        object_count_match = re.search(r"Objects([1-5])Task$", variant_task_name)
        object_count = int(object_count_match.group(1)) if object_count_match else None
        task_name = re.sub(r"Objects[1-5]Task$", "Task", variant_task_name)
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(
            task_env,
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy=args_cli.policy,
        )

        client = create_client(
            args_cli.policy,
            remote_host=args_cli.remote_host,
            remote_port=args_cli.remote_port,
        )

        for run_idx in range(num_runs):

            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id) for ep_id in run_episode_ids):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                continue

            run_name = task_env + f"_{run_idx}"
            print(f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' (run {run_idx}, {num_envs} envs)\033[0m")

            env_results, msgs, timing = run_episode(env=env,
                        env_cfg=env_cfg,
                        episode=run_idx,
                        client=client,
                        save_videos=args_cli.save_videos,
                        video_mode=args_cli.video_mode,
                        headless=args_cli.headless)

            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                timing=timing,
                env=env,
                env_cfg=env_cfg,
                num_envs=num_envs,
                run_idx=run_idx,
                run_name=run_name,
                task_env=task_env,
                scene_output_dir=scene_output_dir,
                policy=args_cli.policy,
                instruction_type=args_cli.instruction_type,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
                task_name=task_name,
                extra_fields={
                    "background": bg_name,
                    "num_tabletop_objects": object_count,
                    "num_distractors": object_count - 1 if object_count is not None else None,
                },
            )

            env.reset_eval_state()

        if hasattr(client, "close"):
            client.close()
        env.close()

    summarize_experiment_results(episode_results, show_timing=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
