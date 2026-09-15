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
from robolab.core.task.conditionals import reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "ReachBagelTask" / "status.json"


@configclass
class ReachBagelTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=reach_object,
        params={"tolerance": 0.04, "status_path": STATUS_PATH},
    )


@dataclass
class ReachBagelTask(Task):
    contact_object_list = ["table", "bowl", "banana", "bagel_07", "coffee_can", "banana_01", "yogurt_cup", "coffee_pot", "ceramic_mug", "pitcher", "fork_big", "spoon_big", "apple_01", "orange2", "milk_carton", "orange_juice_carton", "bagel_01", "bagel_02", "plate_small", "plate_large"]
    scene = import_scene("breakfast_table.usda", contact_object_list)
    terminations = ReachBagelTerminations
    instruction = {
        "default": "ReachBagel",
        "vague": "Reach the bagel",
        "specific": "Move the robot gripper to a position just above the bagel without grasping it",
    }
    episode_steps: int = 100
    attributes = ["reach", "goal"]
    goal = {
        "mode": "reach",
        "object": "bagel_07",
        "z_offset": 0.17,
        "drive_steps": 50,
        "settle_steps": 4,
        "external_camera": "over_shoulder_left_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="reach_above_bagel",
            conditions={
                "bagel_07": [
                    (partial(reach_object, tolerance=0.04, status_path=STATUS_PATH), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
