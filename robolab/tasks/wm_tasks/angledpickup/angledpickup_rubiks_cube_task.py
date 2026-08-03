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


TARGET = "rubiks_cube"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupRubiksCubeTask" / "status.json"


@configclass
class AngledPickupRubiksCubeTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": TARGET, "surface": "table", "distance": 0.16},
    )


@dataclass
class AngledPickupRubiksCubeTask(Task):
    contact_object_list = ["rubiks_cube", "bowl", "table"]
    scene = import_scene("rubiks_cube_bowl.usda", contact_object_list)
    terminations = AngledPickupRubiksCubeTerminations
    instruction = {
        "default": "AngledPickupRubiksCube",
        "vague": "Pick up the multicolored cube",
        "specific": (
            "Approach the multicolored cube near the center of the table with a yawed "
            "wrist, grasp two opposite faces, and lift it at least 16 cm"
        ),
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
        "cuboid",
        "high_contrast",
        "reachable",
        "goal",
    ]
    goal = {
        "mode": "angled_pickup",
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_rubiks_cube",
            conditions={
                TARGET: [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.05,
                            angle_tolerance=0.1745,
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
