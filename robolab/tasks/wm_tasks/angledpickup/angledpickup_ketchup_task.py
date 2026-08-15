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
from robolab.core.task.conditionals import object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupKetchupTask" / "status.json"


@configclass
class AngledPickupKetchupTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={
            "object": "ketchup_bottle",
            "surface": "table",
            "distance": 0.2058982690388593,
            "status_path": STATUS_PATH,
            "angle_tolerance": 0.09,
        },
    )


@dataclass
class AngledPickupKetchupTask(Task):
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
    scene = import_scene("angledpickup_ketchup_high_friction.usda", contact_object_list)
    terminations = AngledPickupKetchupTerminations
    instruction = {
        "default": "AngledPickupKetchup",
        "vague": "Reach the ketchup bottle with a yawed wrist, grasp it, and lift it up",
        "specific": "Move the robot gripper to the ketchup bottle with the wrist yawed to face the bottle from the side, grasp the bottle, and lift it off the table",
    }
    episode_steps: int = 165
    angledreach_steps: int = 70
    grasp_steps: int = 10
    pickup_steps: int = 90

    attributes = ["angled_reach", "pickup", "grasp", "lift", "dominant_yaw", "+rz", "goal"]
    goal = {
        "mode": "angled_pickup",
        "object": "ketchup_bottle",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_ketchup",
            conditions={
                "ketchup_bottle": [
                    (
                        partial(
                            object_picked_up,
                            object="ketchup_bottle",
                            surface="table",
                            distance=0.16,
                        ),
                        1.0,
                    ),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
