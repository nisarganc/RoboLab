# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass, noise

from robolab.constants import ROBOTS_DIR

_WRIST_CAM = TiledCameraCfg(
    # Deliberately named wrist_cam (not wrist_camera) to avoid collision with the
    # wrist_camera prim baked into the robot USD, which has different intrinsics.
    # We spawn our own sensor here with policy-calibrated focal_length 2.8 to match
    # pi05 / DreamZero training.
    prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
    ),
)


@configclass
class DroidCfg:
    """Cfg class that adds robot articulation to scene configurations."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path= os.path.join(ROBOTS_DIR, "franka_robotiq_2f_85_flattened.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0, 0, 0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * np.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * np.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * np.pi,
                "panda_joint7": 0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                # stiffness=None,
                # damping=None,
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                # stiffness=None,
                # damping=None,
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=None,
                damping=None,
                # effort_limit=150.0,
                velocity_limit=5.0, #2.175,
                # stiffness=1000.0,
                # damping=40.0,
            ),
        },
    )

    wrist_cam = _WRIST_CAM

    # NOTE: FrameTransformer disabled - using articulation body state for EE pose instead (faster)
    # frames = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/robot/panda_link0",
    #     debug_vis=False,
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link",
    #             name="end_effector",
    #             offset=OffsetCfg(
    #                 pos=[0.0, 0.0, 0.0],
    #             ),
    #         ),
    #     ],
    # )


@configclass
class WristCameraCfg:
    """Introspection wrapper so the wrist camera can be passed to generate_image_obs_from_cameras.
    The scene's wrist_cam is still sourced from DroidCfg; this wrapper only exposes the name.
    """
    wrist_cam = _WRIST_CAM

########################################################
# Contact gripper
########################################################

# IsaacLab ContactSensor requires exactly one prim per env for filter_prim_paths_expr
# (force_matrix_w) to work. .*_inner_finger matches 2 bodies (left + right) per env,
# breaking filtered contact detection. Use one finger only.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/left_inner_finger"}

########################################################
# Definitions
########################################################


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """ Returns gripper position as 0 for open and 1 for closed.
    """
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


def ee_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Returns the end effector position (x, y, z) in the world frame."""
    robot = env.scene[asset_cfg.name]
    # Get the body index for the end effector link
    ee_body_name = "panda_link8"  # Panda end effector link
    body_idx = robot.data.body_names.index(ee_body_name)
    # Return position (shape: [num_envs, 3])
    return robot.data.body_pos_w[:, body_idx, :]


def ee_quat(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Returns the end effector orientation as quaternion (w, x, y, z) in the world frame."""
    robot = env.scene[asset_cfg.name]
    # Get the body index for the end effector link
    ee_body_name = "panda_link8"  # Panda end effector link
    body_idx = robot.data.body_names.index(ee_body_name)
    # Return quaternion (shape: [num_envs, 4])
    return robot.data.body_quat_w[:, body_idx, :]

########################################################
# Actions
########################################################

class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction


class StepRelativeJointPositionAction(RelativeJointPositionAction):
    """Apply a relative joint delta once per env step and hold it across decimation."""

    def __init__(self, cfg: RelativeJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reference_joint_pos = torch.zeros_like(self.raw_actions)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._reference_joint_pos[:] = self._asset.data.joint_pos[:, self._joint_ids]

    def apply_actions(self):
        joint_targets = self._reference_joint_pos + self.processed_actions
        # For Droid IK gripper control, snap post-delta state to binary open/close.
        # `finger_joint` spans [0, pi/4], so 0.5 in normalized space is pi/8.
        half_closed = np.pi / 8
        fully_closed = np.pi / 4
        joint_targets = torch.where(
            joint_targets > half_closed,
            torch.full_like(joint_targets, fully_closed),
            torch.zeros_like(joint_targets),
        )
        self._asset.set_joint_position_target(joint_targets, joint_ids=self._joint_ids)


@configclass
class StepRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    class_type = StepRelativeJointPositionAction


@configclass
class DroidJointPositionActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

@configclass
class DroidIKActionCfg:
    """
    End-effector pose control for the Franka + Robotiq DROID setup.

    Notes:
        - Relative pose mode expects 6 values for the arm action:
          (dx, dy, dz, droll, dpitch, dyaw)
        - The gripper action is also relative and expects 1 value:
          (dg), interpreted as a delta on the finger joint.
    """

    body = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_link8",
        controller=DifferentialIKControllerCfg(command_type="pose", 
                                                use_relative_mode=True, 
                                                ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )
    
    # finger_joint = BinaryJointPositionZeroToOneActionCfg(
    #     asset_name="robot",
    #     joint_names=["finger_joint"],
    #     open_command_expr = {"finger_joint": 0.0},
    #     close_command_expr={"finger_joint": np.pi / 4},
    # )
    finger_joint = StepRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        scale=np.pi / 4,
    )

########################################################
# Observations
########################################################
@configclass
class ProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(
        func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
    )
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False # must include
        self.concatenate_terms = False # must include
