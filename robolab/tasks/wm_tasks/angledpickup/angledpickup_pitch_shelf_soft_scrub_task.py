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

TARGET = 'soft_scrub'
STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupPitchShelfSoftScrubTask" / "status.json"


@configclass
class AngledPickupPitchShelfSoftScrubTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": TARGET, "surface": "table", "distance": 0.10,
                "status_path": STATUS_PATH, "angle_tolerance": 0.09},
    )


@dataclass
class AngledPickupPitchShelfSoftScrubTask(Task):
    contact_object_list = ['table', 'soft_scrub']
    scene = import_scene('angledpickup_pitch_soft_scrub_single_shelf_right.usda', contact_object_list)
    terminations = AngledPickupPitchShelfSoftScrubTerminations
    events = BottleBodyGraspEvents
    instruction = {
        "default": "AngledPickupPitchShelfSoftScrub",
        "vague": "Approach the Soft Scrub bottle body with -55 degrees of pitch, grasp it, and lift it",
        "specific": (
            "Approach the Soft Scrub bottle body with -55 degrees of wrist pitch near the end. "
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
        'capture_pregrasp_distance': 0.06,
        'capture_grasp_point': [0.6500070957023854, -0.2431180591493606, 0.25521183427693317],
        'capture_grasp_local': [0.005, 0, 0.02],
        'capture_pitch_degrees': -55,
        'capture_pregrasp_offset': [-0.04, 0.0, 0.09],
        'capture_pad_offset': [0, 0, 0.155],
        'object': 'soft_scrub',
        'external_camera': 'over_shoulder_right_camera',
        'wrist_camera': 'wrist_cam',
        'reach_task': 'AngledReachPitchShelfSoftScrubTask',
        'capture_lift_height': 0.30,
        'capture_home_height_offset': 0.0,
        'capture_home_rotation': 'reset',
        'capture_home_controller': 'joint_position',
        'capture_home_joint_steps': 180,
        'capture_return_via_raised_home': True,
        'capture_min_safe_height': 0.72,
        'capture_min_return_ee_height': 0.45,
        'capture_home_pose_correction_steps': 40,
        'capture_max_object_slip_m': 0.025,
        'capture_max_object_slip_rad': 0.26,
        'capture_gripper_steps': 100,
        'pickup_clearance': 0.1,
        'capture_reference_reach_pose': 'AngledReachPitchShelfSoftScrubTask',
    }
    subtasks = [
        Subtask(
            name="angled_pickup_pitch_shelf_soft_scrub",
            conditions={TARGET: [(partial(object_picked_up, object=TARGET, surface="table",
                                          distance=0.10, status_path=STATUS_PATH,
                                          angle_tolerance=0.09), 1.0)]},
            logical="all",
            score=1.0,
        )
    ]
