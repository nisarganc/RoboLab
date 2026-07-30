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


STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachMarker2Task" / "status.json"


@configclass
class AngledReachMarker2Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
            "pos_tolerance": 0.05,
            "angle_tolerance": 0.09,
            "status_path": STATUS_PATH,
        },
    )


@dataclass
class AngledReachMarker2Task(Task):
    contact_object_list = [
        "table",
        "mug",
        "bowl",
        "mustard",
        "dry_erase_marker",
    ]
    scene = import_scene("bin_mug_mustard_marker_bowl2.usda", contact_object_list)
    terminations = AngledReachMarker2Terminations
    instruction = {
        "default": "AngledReachMarker2",
        "vague": "Approach the small marker from an angle",
        "specific": "Move the robot gripper above the lightweight dry-erase marker with a positive yaw rotation so the fingers align with its narrow barrel",
    }
    episode_steps: int = 75
    attributes = [
        "angled_reach",
        "size",
        "dominant_yaw",
        "+rz",
        "goal",
    ]
    goal = {
        "mode": "angled_reach",
        "object": "dry_erase_marker",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_marker",
            conditions={
                "dry_erase_marker": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.05,
                            angle_tolerance=0.09,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    ),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
