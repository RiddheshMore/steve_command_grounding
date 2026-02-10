from __future__ import annotations
import time
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None

from .global_parameters import *
from .basic_perception import get_rgb_picture # Simplified
from .basic_movement import *
from scipy.spatial.transform import Rotation
from .coordinates import Pose2D, Pose3D, from_a_to_b_distanced, pose_distanced, get_door_opening_poses
from .importer import PointCloud
from .recursive_config import Config
from .time import convert_time

from .stretch_package.stretch_movement.move_body import BaseController
from .stretch_package.stretch_movement.move_to_pose import JointPoseController
# from .stretch_package.stretch_movement.move_to_position import JointPositionController
from .stretch_package.stretch_movement.stow_arm import StowArmController
from .stretch_package.stretch_state.frame_transformer import FrameTransformer

def move_body_distanced(node: BaseController, end_pose: Pose2D, distance: float, sleep: bool = True) -> None:
    sleep_multiplier = 1 if sleep else 0
    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
    current_dir = np.arctan2(odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) * 2.0
    start_pose = Pose2D(current_pos, current_dir)
    destination_pose = from_a_to_b_distanced(start_pose, end_pose, distance)
    move_body(node, destination_pose)
    time.sleep(1 * sleep_multiplier)

def pull_drawer(pose_node: JointPoseController):
    set_gripper(pose_node, 0.4)
    pull_pose_start = {'wrist_extension': (0.5, 40.0)}
    pose_node.send_joint_pose(pull_pose_start)
    spin_until_complete(pose_node)
    set_gripper(pose_node, -0.01)
    pull_pose_end = {'wrist_extension': 0.05}
    pose_node.send_joint_pose(pull_pose_end)
    spin_until_complete(pose_node)
    time.sleep(2.0)

def push(pose_node: JointPoseController, height: float) -> None:
    push_pose = {'wrist_extension': 0.0, 'joint_lift': height, 'joint_wrist_pitch': 0.0}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    push_pose = {'wrist_extension': (0.49, 50.0)}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    time.sleep(1.0)
    push_pose = {'wrist_extension': 0.1}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)

def move_in_front_of(
    stow_node: StowArmController, base_node: BaseController, head_node: any, pose_node: JointPoseController, 
    body_pose: Pose3D, target_center: Pose3D, yaw: float, pitch: float, roll: float, lift: float, stow: bool = True, grasp: bool = False
) -> None:
    if stow:
        stow_arm(stow_node)
    move_body(base_node, body_pose.to_dimension(2))
    turn_body(pose_node, target_center.to_dimension(2), grasp=grasp)
    if grasp:
        look_ahead(pose_node)
        # Simplified unstow
        unstow_pos = {'joint_lift': target_center.coordinates[2]-lift, 'joint_wrist_yaw': yaw, 'joint_wrist_pitch': pitch}
        pose_node.send_joint_pose(unstow_pos)
        spin_until_complete(pose_node)
    else:
        # move_head(head_node, target_center, tilt_bool=True)
        time.sleep(1)
