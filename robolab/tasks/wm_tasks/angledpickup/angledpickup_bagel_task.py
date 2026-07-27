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


STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupBagelTask" / "status.json"


@configclass
class AngledPickupBagelTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={
            "object": "bagel_00",
            "surface": "table",
            "distance": 0.16,
        },
    )


@dataclass
class AngledPickupBagelTask(Task):
    contact_object_list = [
        "table",
        "bowl",
        "plate_large",
        "banana",
        "bagel_00",
        "bagel_06",
    ]
    scene = import_scene("bagel_plate_banana_bowl.usda", contact_object_list)
    terminations = AngledPickupBagelTerminations
    instruction = {
        "default": "AngledPickupBagel",
        "vague": "Approach the small bagel from an angle, grasp it, and lift it",
        "specific": "Move the robot gripper above the lightweight bagel with a negative yaw rotation so the fingers align across its thin section, grasp it, and lift it at least 16 cm from the table",
    }
    episode_steps: int = 160
    angledreach_steps: int = 75
    grasp_steps: int = 10
    pickup_steps: int = 75
    attributes = [
        "angled_reach",
        "pickup",
        "grasp",
        "lift",
        "size",
        "dominant_yaw",
        "-rz",
        "goal",
    ]
    goal = {
        "mode": "angled_pickup",
        "object": "bagel_00",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_bagel",
            conditions={
                "bagel_00": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.05,
                            angle_tolerance=0.1745,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    ),
                    (partial(object_grabbed, object="bagel_00"), 1.0),
                    (
                        partial(
                            object_picked_up,
                            object="bagel_00",
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
