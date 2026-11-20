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

TARGET = 'ketchup_bottle'
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupRollTiltedKetchupBottleTask" / "status.json"


@configclass
class AngledPickupRollTiltedKetchupBottleTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": TARGET, "surface": "table", "distance": 0.05,
                "status_path": STATUS_PATH, "angle_tolerance": 0.09},
    )


@dataclass
class AngledPickupRollTiltedKetchupBottleTask(Task):
    contact_object_list = ['table', 'ketchup_bottle']
    scene = import_scene('angledpickup_roll_tilted_ketchup_bottle.usda', contact_object_list)
    terminations = AngledPickupRollTiltedKetchupBottleTerminations
    instruction = {
        "default": "AngledPickupRollTiltedKetchupBottle",
        "vague": "Approach the tilted ketchup bottle's neck with +30 degrees of roll, grasp it, and lift it",
        "specific": (
            "Approach the tilted ketchup bottle's neck with +30 degrees of wrist roll near the end. "
            "Move closer with the fingers open, close around the grasp point, lift it clear of its support, "
            "and carry it to the exact reset/home end-effector position and orientation."
        ),
    }
    episode_steps: int = 180
    angledreach_steps: int = 80
    grasp_steps: int = 30
    pickup_steps: int = 70
    attributes = ['angled_reach', 'pickup', 'grasp', 'lift', 'dominant_roll', '+rx', 'goal']
    goal = {
        'mode': 'angled_pickup',
        'capture_pregrasp_distance': 0.04,
        'capture_grasp_point': [0.55, 0.09400000000000001, 0.20457270707278985],
        'capture_grasp_local': [0, 0, 0.072],
        'capture_roll_degrees': 30,
        'capture_pad_offset': [0, 0, 0.1725],
        'object': 'ketchup_bottle',
        'external_camera': 'over_shoulder_right_camera',
        'wrist_camera': 'wrist_cam',
        'reach_task': 'AngledReachRollTiltedKetchupBottleTask',
        'capture_lift_height': 0.16,
        'capture_home_height_offset': 0.0,
        'capture_home_rotation': 'reset',
        'capture_gripper_steps': 10,
        'capture_closed_rotation_gain': 2.0,
        'capture_closed_rotation_limit': 0.2,
        'capture_motion_steps': 40,
        'capture_motion_settle_steps': 5,
        'capture_home_hold_steps': 10,
        'pickup_clearance': 0.05,
    }
    subtasks = [
        Subtask(
            name="angled_pickup_roll_tilted_ketchup_bottle",
            conditions={TARGET: [(partial(object_picked_up, object=TARGET, surface="table",
                                          distance=0.05, status_path=STATUS_PATH,
                                          angle_tolerance=0.09), 1.0)]},
            logical="all",
            score=1.0,
        )
    ]
