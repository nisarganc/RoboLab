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


TARGET = "utilityjug_a02"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachPitchSidewaysJugTask" / "status.json"


@configclass
class AngledReachPitchSidewaysJugTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.08, "angle_tolerance": 0.15, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachPitchSidewaysJugTask(Task):
    contact_object_list = [
        "table",
        "whitepackerbottle_a01",
        "whitepackerbottle_a02",
        "utilityjug_a01",
        "utilityjug_a02",
        "plasticpail_a02",
        "whitepackerbottle_a03",
    ]
    scene = import_scene("angledreach_pitch_sideways_jug_colored.usda", contact_object_list)
    terminations = AngledReachPitchSidewaysJugTerminations
    instruction = {
        "default": "AngledReachPitchSidewaysJug",
        "vague": "Reach the exposed handle of the sideways utility jug with a positive pitch",
        "specific": (
            "Move the open gripper toward the exposed handle of the centered dark burnt-orange utility jug lying "
            "on its side. Add positive pitch about the y-axis at the end so the fingers "
            "are ready to grasp the handle, with a slight -10 degree yaw for alignment, "
            "without roll or closing the gripper."
        ),
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_pitch", "+ry", "goal"]
    goal = {
        "mode": "angled_reach",
        "capture_grasp_point": [0.54, 0.135, 0.142],
        "capture_pitch_degrees": 40,
        "capture_yaw_degrees": -10,
        "capture_pad_offset": [0, 0, 0.145],
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_pitch_sideways_jug",
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
