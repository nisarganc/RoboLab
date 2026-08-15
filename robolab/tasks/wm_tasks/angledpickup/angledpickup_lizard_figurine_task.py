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


TARGET = "lizard_figurine"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupLizardFigurineTask" / "status.json"


@configclass
class AngledPickupLizardFigurineTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": TARGET, "surface": "table", "distance": 0.16},
    )


@dataclass
class AngledPickupLizardFigurineTask(Task):
    contact_object_list = ["table","lizard_figurine","rubiks_cube","birdhouse","yellow_block"]
    scene = import_scene("angledpickup_lizard_distractors.usda", contact_object_list)
    terminations = AngledPickupLizardFigurineTerminations
    instruction = {
        "default": "AngledPickupLizardFigurine",
        "vague": "Approach the lizard figurine at an angle, grasp its body, and lift it",
        "specific": (
            "Yaw the gripper clockwise to follow the lizard figurine's body, grasp around "
            "its torso without catching the legs or tail, and lift it at least 16 cm"
        ),
    }
    episode_steps: int = 165
    angledreach_steps: int = 70
    grasp_steps: int = 10
    pickup_steps: int = 90
    attributes = ["angled_reach", "pickup", "grasp", "lift", "dominant_yaw", "-rz", "goal"]
    goal = {
        "mode": "angled_pickup",
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_lizard_figurine",
            conditions={
                TARGET: [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.04,
                            angle_tolerance=0.09,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    ),
                    (partial(object_grabbed, object=TARGET), 1.0),
                    (
                        partial(
                            object_picked_up,
                            object=TARGET,
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
