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


TARGET = "ceramic_mug"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachRollMugHandleTask" / "status.json"


@configclass
class AngledReachRollMugHandleTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.08, "angle_tolerance": 0.15, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachRollMugHandleTask(Task):
    contact_object_list = ["table", "ceramic_mug"]
    scene = import_scene("angledreach_roll_tilted_mug.usda", contact_object_list)
    terminations = AngledReachRollMugHandleTerminations
    instruction = {
        "default": "AngledReachRollMugHandle",
        "vague": "Reach the tilted ceramic mug handle with a negative roll",
        "specific": (
            "Approach the tilted ceramic mug handle. Add -30 degrees of wrist roll "
            "about the x-axis near the end so the open fingers match the object's tilt, "
            "without pitch, yaw, or closing the gripper."
        ),
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_roll", "-rx", "goal"]
    goal = {
        "mode": "angled_reach",
        "capture_grasp_point": [0.502, 0.125, 0.2133013],
        "capture_grasp_local": [0, -0.058, 0.05],
        "capture_roll_degrees": -30,
        "capture_pad_offset": [0, 0, 0.145],
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_roll_mug_handle",
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
