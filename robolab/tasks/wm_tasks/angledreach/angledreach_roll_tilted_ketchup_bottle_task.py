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


TARGET = "ketchup_bottle"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachRollTiltedKetchupBottleTask" / "status.json"


@configclass
class AngledReachRollTiltedKetchupBottleTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.08, "angle_tolerance": 0.15, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachRollTiltedKetchupBottleTask(Task):
    contact_object_list = ["table", "ketchup_bottle"]
    scene = import_scene("angledreach_roll_tilted_ketchup_bottle.usda", contact_object_list)
    terminations = AngledReachRollTiltedKetchupBottleTerminations
    instruction = {
        "default": "AngledReachRollTiltedKetchupBottle",
        "vague": "Reach the tilted ketchup bottle neck with a positive roll",
        "specific": (
            "Approach the exposed cap and upper neck of the ketchup bottle in its inclined cradle. "
            "Add +30 degrees of wrist roll about the x-axis near the end so the gripper follows "
            "the bottle's lean and the open fingers straddle its neck, without pitch, yaw, or closing."
        ),
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_roll", "+rx", "goal"]
    goal = {
        "mode": "angled_reach",
        "capture_grasp_point": [0.55, 0.09400000000000001, 0.20457270707278985],
        "capture_grasp_local": [0, 0, 0.072],
        "capture_roll_degrees": 30,
        "capture_pad_offset": [0, 0, 0.18],
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_roll_tilted_ketchup_bottle",
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
