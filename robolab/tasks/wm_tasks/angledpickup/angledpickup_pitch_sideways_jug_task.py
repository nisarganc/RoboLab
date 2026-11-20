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
from robolab.core.task.conditionals import object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

TARGET = 'utilityjug_a02'
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupPitchSidewaysJugTask" / "status.json"


@configclass
class AngledPickupPitchSidewaysJugTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": TARGET, "surface": "table", "distance": 0.05,
                "status_path": STATUS_PATH, "angle_tolerance": 0.09},
    )


@dataclass
class AngledPickupPitchSidewaysJugTask(Task):
    contact_object_list = ['table', 'whitepackerbottle_a01', 'whitepackerbottle_a02', 'utilityjug_a01', 'utilityjug_a02', 'plasticpail_a02', 'whitepackerbottle_a03']
    scene = import_scene('angledpickup_pitch_sideways_jug_colored.usda', contact_object_list)
    terminations = AngledPickupPitchSidewaysJugTerminations
    instruction = {
        "default": "AngledPickupPitchSidewaysJug",
        "vague": "Approach the sideways orange jug's exposed handle with +40 degrees of pitch, grasp it, and lift it",
        "specific": (
            "Approach the sideways orange jug's exposed handle with +40 degrees of wrist pitch near the end. "
            "Move closer with the fingers open, close around the grasp point, lift it clear of its support, "
            "and carry it to the exact reset/home end-effector position and orientation."
        ),
    }
    episode_steps: int = 180
    angledreach_steps: int = 80
    grasp_steps: int = 30
    pickup_steps: int = 70
    attributes = ['angled_reach', 'pickup', 'grasp', 'lift', 'dominant_pitch', '+ry', 'goal']
    goal = {
        'mode': 'angled_pickup',
        'capture_pregrasp_distance': 0.025,
        'capture_grasp_point': [0.54, 0.143, 0.112],
        'capture_pitch_degrees': 40,
        'capture_yaw_degrees': -10,
        'capture_pad_offset': [0, 0, 0.16],
        'object': 'utilityjug_a02',
        'external_camera': 'over_shoulder_right_camera',
        'wrist_camera': 'wrist_cam',
        'reach_task': 'AngledReachPitchSidewaysJugTask',
        'capture_lift_height': 0.24,
        'capture_home_height_offset': 0.0,
        'capture_home_rotation': 'reset',
        'capture_gripper_steps': 100,
        'pickup_clearance': 0.05,
        'capture_reference_reach_pose': 'AngledReachPitchSidewaysJugTask',
    }
    subtasks = [
        Subtask(
            name="angled_pickup_pitch_sideways_jug",
            conditions={TARGET: [(partial(object_picked_up, object=TARGET, surface="table",
                                          distance=0.05, status_path=STATUS_PATH,
                                          angle_tolerance=0.09), 1.0)]},
            logical="all",
            score=1.0,
        )
    ]
