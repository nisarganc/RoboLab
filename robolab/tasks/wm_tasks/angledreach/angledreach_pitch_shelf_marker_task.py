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


TARGET = "dry_erase_marker"
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachPitchShelfMarkerTask" / "status.json"


@configclass
class AngledReachPitchShelfMarkerTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.08, "angle_tolerance": 0.15, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachPitchShelfMarkerTask(Task):
    contact_object_list = ["table", "dry_erase_marker"]
    scene = import_scene("angledreach_pitch_marker_single_shelf_center.usda", contact_object_list)
    terminations = AngledReachPitchShelfMarkerTerminations
    instruction = {
        "default": "AngledReachPitchShelfMarker",
        "vague": "Reach the inclined marker with a negative pitch",
        "specific": (
            "Approach the center of the dry-erase marker resting on the inclined supports "
            "on the centered shelf. Finish with a negative wrist pitch about the y-axis "
            "with the open fingers still short of the barrel, ready to move closer, without roll, yaw, "
            "or closing the gripper."
        ),
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_pitch", "-ry", "goal"]
    goal = {
        "mode": "angled_reach",
        "capture_grasp_point": [0.62, 0, 0.23],
        "capture_pitch_degrees": -30,
        "capture_pad_offset": [0, 0, 0.205],
        "object": TARGET,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_pitch_shelf_marker",
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
