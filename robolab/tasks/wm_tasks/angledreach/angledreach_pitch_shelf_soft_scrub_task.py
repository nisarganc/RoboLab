# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import ASSET_DIR
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import angled_reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


TARGET = "soft_scrub"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachPitchShelfSoftScrubTask" / "status.json"


@configclass
class AngledReachPitchShelfSoftScrubTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.08, "angle_tolerance": 0.15, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachPitchShelfSoftScrubTask(Task):
    contact_object_list = ["table", "soft_scrub"]
    scene = import_scene("angledreach_pitch_soft_scrub_single_shelf_right.usda", contact_object_list)
    terminations = AngledReachPitchShelfSoftScrubTerminations
    instruction = {
        "default": "AngledReachPitchShelfSoftScrub",
        "vague": "Reach the Soft Scrub bottle body using a -55 degree pitch",
        "specific": (
            "Approach the upper body of the Soft Scrub bottle on the "
            "upper-right corner of the shelf, tilted toward the robot base, adding a -55 degree pitch about the y-axis near the end, "
            "stopping short of contact so the gripper can advance to a body grasp, without roll, yaw, or grasping"
        ),
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_pitch", "-ry", "goal"]
    goal = {
        "mode": "angled_reach",
        "capture_grasp_point": [0.6500070957023854, -0.2431180591493606, 0.25521183427693317],
        'capture_grasp_local': [0.005, 0, 0.02],
        "capture_pitch_degrees": -55,
        "capture_pregrasp_offset": [-0.04, 0.0, 0.09],
        "capture_pad_offset": [0, 0, 0.215],
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_pitch_shelf_soft_scrub",
            conditions={
                TARGET: [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.08,
                            angle_tolerance=0.15,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    )
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
