# SPDX-License-Identifier: CC-BY-NC-4.0

"""Gentle physical gripper settings for broad plastic bottle body grasps."""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from robolab.core.environments.base import BaseEventCfg


def set_bottle_grasp_effort(env, env_ids, effort_limit=5.0):
    # A broad body stops the fingers early. Limit pinch torque so it does not
    # overpower the arm's wrist while the gripper remains commanded closed.
    robot = env.scene['robot']
    joint_ids, _ = robot.find_joints('finger_joint')
    robot.write_joint_effort_limit_to_sim(effort_limit, joint_ids=joint_ids, env_ids=env_ids)


@configclass
class BottleBodyGraspEvents(BaseEventCfg):
    bottle_grasp_effort = EventTerm(
        func=set_bottle_grasp_effort, mode='startup', params={'effort_limit': 5.0})
