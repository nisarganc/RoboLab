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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachDrillTask" / "status.json"


@configclass
class AngledReachDrillTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
            "pos_tolerance": 0.04,
            "angle_tolerance": 0.09,
            "status_path": STATUS_PATH,
        },
    )


@dataclass
class AngledReachDrillTask(Task):
    contact_object_list = [
        "table",
        "cordless_drill",
        "left_bin",
        "right_bin",
    ]
    scene = import_scene("tools_container2.usda", contact_object_list)
    terminations = AngledReachDrillTerminations
    instruction = {
        "default": "AngledReachDrill",
        "vague": "Reach the cordless drill near the center of the tool table with a with a yawed wrist",
        "specific": "Move the robot gripper above the cordless drill near the center of the table with the wrist yawed so the fingers align vertically with the drill handle, without grasping it",
    }
    episode_steps: int = 75
    attributes = ["angled_reach", "dominant_yaw", "-rz", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "cordless_drill",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_drill",
            conditions={
                "cordless_drill": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.04,
                            angle_tolerance=0.09,
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
