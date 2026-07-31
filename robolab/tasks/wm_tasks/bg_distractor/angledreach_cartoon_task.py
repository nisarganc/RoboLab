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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachCartoonTask" / "status.json"

@configclass
class AngledReachCartoonTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
                "pos_tolerance": 0.04, 
                "angle_tolerance": 0.09, 
                "status_path": STATUS_PATH
            },
    )


@dataclass
class AngledReachCartoonTask(Task):
    contact_object_list = [
        "table",
        "alphabet_soup_can",
        "milk_carton",
        "orange_juice_carton",
        "smartphone",
        "mug",
        "mayonnaise_bottle",
        "ketchup_bottle",
        "cubebox_a02",
    ]
    scene = import_scene("cartons_on_box_orange_center.usda", contact_object_list)
    terminations = AngledReachCartoonTerminations
    instruction = {
        "default": "AngledReachCartoon",
        "vague": "Reach the orange juice carton near the center of the packing table with a yawed approach",
        "specific": "Move the robot gripper above the orange juice carton lying on the table with a yawed approach, such that the gripper is aligned with the longer side of the carton. The orange juice carton is located near the center of the packing table.",
    }
    episode_steps: int = 75
    attributes = ["angled_reach", "dominant_yaw", "+rz", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "orange_juice_carton",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_cartoon",
            conditions={
                "orange_juice_carton": [
                    (partial(angled_reach_object, 
                            pos_tolerance=0.04, 
                            angle_tolerance=0.09, 
                            status_path=STATUS_PATH),
                    1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
