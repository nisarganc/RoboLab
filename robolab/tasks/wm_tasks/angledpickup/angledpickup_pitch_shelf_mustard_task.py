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
from robolab.tasks.wm_tasks.angledpickup.bottle_grasp_events import BottleBodyGraspEvents

TARGET = 'mustard'
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupPitchShelfMustardTask" / "status.json"


@configclass
class AngledPickupPitchShelfMustardTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": TARGET, "surface": "table", "distance": 0.10,
                "status_path": STATUS_PATH, "angle_tolerance": 0.09},
    )


@dataclass
class AngledPickupPitchShelfMustardTask(Task):
    contact_object_list = ['table', 'mustard']
    scene = import_scene('angledpickup_pitch_mustard_single_shelf_left.usda', contact_object_list)
    terminations = AngledPickupPitchShelfMustardTerminations
    events = BottleBodyGraspEvents
    instruction = {
        "default": "AngledPickupPitchShelfMustard",
        "vague": "Approach the mustard bottle body with -50 degrees of pitch, grasp it, and lift it",
        "specific": (
            "Approach the mustard bottle body with -50 degrees of wrist pitch near the end. "
            "Move closer with the fingers open, close around the grasp point, lift it clear of its support, "
            "and carry it to the exact reset/home end-effector position and orientation."
        ),
    }
    episode_steps: int = 180
    angledreach_steps: int = 80
    grasp_steps: int = 30
    pickup_steps: int = 70
    attributes = ['angled_reach', 'pickup', 'grasp', 'lift', 'dominant_pitch', '-ry', 'goal']
    goal = {
        'mode': 'angled_pickup',
        'capture_pregrasp_distance': 0.045,
        'capture_grasp_point': [0.62, 0.15, 0.2605],
        'capture_grasp_local': [0, 0, 0.02],
        'capture_pitch_degrees': -50,
        'capture_pad_offset': [0, 0, 0.17],
        'object': 'mustard',
        'external_camera': 'over_shoulder_right_camera',
        'wrist_camera': 'wrist_cam',
        'reach_task': 'AngledReachPitchShelfMustardTask',
        'capture_lift_height': 0.16,
        'capture_home_height_offset': 0.0,
        'capture_home_rotation': 'reset',
        'capture_home_controller': 'joint_position',
        'capture_gripper_steps': 100,
        'pickup_clearance': 0.1,
        'capture_reference_reach_pose': 'AngledReachPitchShelfMustardTask',
    }
    subtasks = [
        Subtask(
            name="angled_pickup_pitch_shelf_mustard",
            conditions={TARGET: [(partial(object_picked_up, object=TARGET, surface="table",
                                          distance=0.10, status_path=STATUS_PATH,
                                          angle_tolerance=0.09), 1.0)]},
            logical="all",
            score=1.0,
        )
    ]
