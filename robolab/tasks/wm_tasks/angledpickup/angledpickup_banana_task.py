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
from robolab.core.task.conditionals import angled_reach_object, object_grabbed, object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupBananaTask" / "status.json"


@configclass
class AngledPickupBananaTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={
            "object": "banana",
            "surface": "table",
            "distance": 0.34510199220528137,
            "status_path": STATUS_PATH,
            "angle_tolerance": 0.09,
        },
    )


@dataclass
class AngledPickupBananaTask(Task):
    contact_object_list = ["table", "bowl", "banana"]
    scene = import_scene("banana_bowl.usda", contact_object_list) #import_scene("angledpickup_banana_high_friction.usda", contact_object_list)
    terminations = AngledPickupBananaTerminations
    instruction = {
        "default": "AngledPickupBanana",
        "vague": "Approach the banana from an angle, grasp it, and lift it",
        "specific": "Move the robot gripper above the banana beside the bowl with a positive yaw rotation so the fingers follow its long axis, grasp it, and lift it at least 16 cm from the table",
    }
    episode_steps: int = 165
    angledreach_steps: int = 70
    grasp_steps: int = 10
    pickup_steps: int = 90
    attributes = [
        "angled_reach",
        "pickup",
        "grasp",
        "lift",
        "dominant_yaw",
        "+rz",
        "goal",
    ]
    goal = {
        "mode": "angled_pickup",
        "object": "banana",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_banana",
            conditions={
                "banana": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.04,
                            angle_tolerance=0.09,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    ),
                    (partial(object_grabbed, object="banana"), 1.0),
                    (
                        partial(
                            object_picked_up,
                            object="banana",
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
