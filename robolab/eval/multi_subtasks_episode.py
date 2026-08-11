# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Multi-goal episode runner for staged WM pickup tasks.

This runner is intended for tasks whose cached goal assets are named with
stage suffixes, e.g. ``over_shoulder_right_camera_1.png`` for angled reach,
``*_2.png`` for grasp, and ``*_3.png`` for lifted/home-with-object. Goal-image
stages advance strictly from the configured step budgets.
"""

import logging
import os
import re
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.eval.base_client import InferenceClient
from robolab.eval.episode import TimingStats, _load_rgb_image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WM_GOAL_DIR = REPO_ROOT / "assets" / "wm_tasks"


def _stage_goal_paths(env_cfg, stage: int) -> tuple[Path, Path]:
    suffix = stage + 1
    external_key = env_cfg.goal.get("external_camera", "over_shoulder_right_camera")
    wrist_key = env_cfg.goal.get("wrist_camera", "wrist_cam")
    task_name = getattr(env_cfg, "_task_name", env_cfg.__class__.__name__)
    root = WM_GOAL_DIR / task_name
    return root / f"{external_key}_{suffix}.png", root / f"{wrist_key}_{suffix}.png"


def set_client_goal_images_for_stage(
    client,
    env,
    env_cfg,
    instruction: str,
    stage: int,
    *,
    run_idx: int | None = None,
    env_ids=None,
):
    """Set stage-specific goal images on the client and clear local chunks."""
    external_path, wrist_path = _stage_goal_paths(env_cfg, stage)
    external_goal = _load_rgb_image(external_path)
    wrist_goal = _load_rgb_image(wrist_path)

    if env_ids is None:
        env_ids = range(env.num_envs)

    for env_id in env_ids:
        client.set_goal_images(
            external_goal,
            wrist_goal,
            env_id=int(env_id),
            instruction=instruction,
            run_idx=run_idx,
        )
        # Keep the new goal from inheriting an old open-loop chunk. Avoid the
        # VALPA client's server-side reset here; set_goal_images already updates
        # server goal state.
        InferenceClient.reset(client, env_id=int(env_id))


def _stage_from_step(env_cfg, step: int) -> int:
    """Return the goal stage using only the configured cumulative step budgets."""
    angled_steps = int(env_cfg.angledreach_steps)
    grasp_steps = int(env_cfg.grasp_steps)

    if step >= angled_steps + grasp_steps:
        return 2
    if step >= angled_steps:
        return 1
    return 0


def _update_goal_stages(env, env_cfg, stages: list[int], step: int) -> list[int]:
    """Advance each active environment to the stage selected by ``step``."""
    changed_envs: list[int] = []
    target_stage = _stage_from_step(env_cfg, step)

    for env_id in range(env.num_envs):
        if env._frozen_envs[env_id]:
            continue
        if target_stage > stages[env_id]:
            stages[env_id] = target_stage
            changed_envs.append(env_id)

    return changed_envs


def run_multi_subtasks_episode(env, env_cfg, episode, client: InferenceClient, *, headless=False, save_videos=True, video_mode="all"):
    """Run an episode while swapping WM goal images across pickup subtasks."""
    timer = TimingStats()

    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = getattr(env_cfg, "episode_steps", None)
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    instruction = env_cfg.instruction
    action_dim = getattr(getattr(env, "action_manager", None), "total_action_dim", None) or env.action_space.shape[-1]

    subtask_status = []
    clients = [client] * env.num_envs
    stages = [0 for _ in range(env.num_envs)]

    if hasattr(client, "reset"):
        client.reset()
    if hasattr(client, "set_goal_images"):
        set_client_goal_images_for_stage(client, env, env_cfg, instruction, 0, run_idx=episode)

    if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                video_writers_obs.append(VideoWriter(os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4"), video_fps))
            if save_viewport:
                video_writers_viewport.append(VideoWriter(os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4"), video_fps))

    import omni.kit.app
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    actual_steps = 0
    try:
        for step in tqdm(range(max_steps)):
            while not timeline.is_playing():
                kit_app.update()

            timer.start("policy_inference")
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            for env_id in env.active_env_ids:
                ret = clients[env_id].infer(obs, instruction, env_id=env_id)
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if VISUALIZE:
                get_world(env).visualize()

            timer.start("env_step")
            obs, reward, term, trunc, info = env.step(actions)
            timer.stop("env_step")

            per_env_infos = get_all_env_subtask_infos(env)
            subtask_status.append(per_env_infos)

            if hasattr(client, "set_goal_images"):
                changed_envs = _update_goal_stages(env, env_cfg, stages, step + 1)
                for env_id in changed_envs:
                    print(f"Env {env_id} advancing to stage {stages[env_id]} at step {step+1}")
                    set_client_goal_images_for_stage(
                        client,
                        env,
                        env_cfg,
                        instruction,
                        stages[env_id],
                        run_idx=episode,
                        env_ids=[env_id],
                    )

            if save_videos:
                timer.start("video_write")
                for env_id in range(env.num_envs):
                    if env._frozen_envs[env_id]:
                        continue
                    if save_sensor:
                        frame_obs = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        video_writers_obs[env_id].write(frame_obs)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        video_writers_viewport[env_id].write(frame_vp)
                timer.stop("video_write")

            actual_steps += 1
            if env.all_terminated:
                break
    finally:
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                logger.exception("Failed to release video writer")
        try:
            client.reset()
        except Exception:
            logger.exception("Failed to reset client after episode")

    if env.recorder_manager is not None and hasattr(env.recorder_manager, "export_episodes"):
        if env.active_env_ids:
            try:
                env.recorder_manager.export_episodes(env_ids=env.active_env_ids)
                if hasattr(env.recorder_manager, "clear"):
                    env.recorder_manager.clear(env_ids=env.active_env_ids)
            except Exception:
                logger.exception("Failed to export recordings for timed-out envs")

    timing = timer.to_dict(actual_steps)
    return env.get_env_results(), subtask_status, timing