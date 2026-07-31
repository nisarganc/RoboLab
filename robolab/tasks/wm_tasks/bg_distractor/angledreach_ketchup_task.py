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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachKetchupTask" / "status.json"

@configclass
class AngledReachKetchupTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
                "pos_tolerance": 0.04, 
                "angle_tolerance": 0.09, 
                "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachKetchupTask(Task):
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
    scene = import_scene("cartons_on_box.usda", contact_object_list)
    terminations = AngledReachKetchupTerminations
    instruction = {
        "default": "AngledReachKetchup",
        "vague": "Reach the ketchup bottle in the center of the packing table with a yawed approach",
        "specific": "Move the robot gripper to the ketchup bottle near the center of the table, yawed to face the bottle from the side of the surrounding cartons and box, without grasping it",
    }
    episode_steps: int = 75
    attributes = ["angled_reach", "dominant_yaw", "+rz", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "ketchup_bottle",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_ketchup",
            conditions={
                "ketchup_bottle": [
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
