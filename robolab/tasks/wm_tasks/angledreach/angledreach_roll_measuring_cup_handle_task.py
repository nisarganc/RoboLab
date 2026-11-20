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


TARGET = "measuring_cup"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachRollMeasuringCupHandleTask" / "status.json"


@configclass
class AngledReachRollMeasuringCupHandleTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.08, "angle_tolerance": 0.15, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachRollMeasuringCupHandleTask(Task):
    contact_object_list = ["table", "measuring_cup"]
    scene = import_scene("angledreach_roll_tilted_measuring_cup.usda", contact_object_list)
    terminations = AngledReachRollMeasuringCupHandleTerminations
    instruction = {
        "default": "AngledReachRollMeasuringCupHandle",
        "vague": "Reach the thick straight round handle of the tilted measuring cup with a negative roll",
        "specific": (
            "Approach the thick straight cylindrical handle of the tilted measuring cup. "
            "Add -40 degrees of wrist roll "
            "about the x-axis near the end so the open fingers straddle the cylindrical handle and match the cup's tilt, "
            "without pitch, yaw, or closing the gripper."
        ),
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_roll", "-rx", "goal"]
    goal = {
        "mode": "angled_reach",
        "capture_grasp_point": [0.483, -0.1503582, 0.2114907],
        "capture_grasp_local": [-0.067, 0, 0.015],
        "capture_roll_degrees": -40,
        "capture_pad_offset": [0, 0, 0.18],
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_roll_measuring_cup_handle",
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
