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


TARGET = "mustard"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachPitchShelfMustardTask" / "status.json"


@configclass
class AngledReachPitchShelfMustardTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.08, "angle_tolerance": 0.15, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachPitchShelfMustardTask(Task):
    contact_object_list = ["table", "mustard"]
    scene = import_scene("angledreach_pitch_mustard_single_shelf_left.usda", contact_object_list)
    terminations = AngledReachPitchShelfMustardTerminations
    instruction = {
        "default": "AngledReachPitchShelfMustard",
        "vague": "Reach the mustard bottle on the left shelf using only a negative pitch",
        "specific": (
            "Move the gripper toward the mustard bottle on top of the left-shifted single-level "
            "shelf with the wrist pitched downward about its y-axis toward the upper bottle body, "
            "stopping short of contact so the gripper can advance to a body grasp, without roll, yaw, or grasping"
        ),
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_pitch", "-ry", "goal"]
    goal = {
        "mode": "angled_reach",
        "capture_grasp_point": [0.62, 0.15, 0.2605],
        'capture_grasp_local': [0, 0, 0.02],
        "capture_pitch_degrees": -50,
        "capture_pad_offset": [0, 0, 0.215],
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_pitch_shelf_mustard",
            conditions={
                TARGET: [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.08,
                            angle_tolerance=0.15,
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
