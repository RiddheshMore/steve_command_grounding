from __future__ import annotations
import time
import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .stretch_package.stretch_movement.move_body import BaseController
from .stretch_package.stretch_movement.move_to_pose import JointPoseController
# from .stretch_package.stretch_movement.move_to_position import JointPositionController
from .stretch_package.stretch_movement.move_head import HeadJointController
from .stretch_package.stretch_movement.stow_arm import StowArmController
from .stretch_package.stretch_state.jointstate_subscriber import JointStateSubscriber
from .stretch_package.stretch_state.odom_subscriber import OdomSubscriber
from .coordinates import Pose2D, Pose3D
from .global_parameters import *

def spin_until_complete(node: Node) -> None:
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node)
    node.done = False

def get_odom() -> Odometry:
    odom_node = OdomSubscriber()
    spin_until_complete(odom_node)
    odom = odom_node.odom
    odom_node.destroy_node()
    return odom

def get_joint_states() -> JointState:
    joint_state_node = JointStateSubscriber()
    spin_until_complete(joint_state_node)
    joint_state = joint_state_node.jointstate
    joint_state_node.destroy_node()
    return joint_state

def move_body(node: BaseController, pose: Pose2D) -> bool:
    goal_pos = np.array([pose.coordinates[0], pose.coordinates[1]])
    node.send_goal(round(float(pose.coordinates[0]), 3),round(float(pose.coordinates[1]), 3),round(float(pose.direction()[0]), 3), round(float(pose.direction()[1]), 3))
    spin_until_complete(node)
    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])   
    if np.allclose(current_pos, goal_pos, atol=POS_TOL):
        return True
    return False

def turn_body(node: JointPoseController, pose: Pose2D, grasp: bool= True) -> None:
    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
    current_dir_quat = np.array([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
    # Simple yaw extraction
    current_dir = np.arctan2(current_dir_quat[2], current_dir_quat[3]) * 2.0
    goal_pos = pose.coordinates
    goal_dir = np.arctan2(goal_pos[1]-current_pos[1], goal_pos[0]-current_pos[0])
    if grasp:
        turn_dir = goal_dir - current_dir + np.pi/2.0
    else:
        turn_dir = goal_dir - current_dir
    norm_turn_dir = (turn_dir + np.pi) % (2*np.pi) - np.pi
    turn_value = {'rotate_mobile_base': norm_turn_dir}
    node.send_joint_pose(turn_value)
    spin_until_complete(node)

def stow_arm(node: StowArmController) -> None:
    node.send_stow_request()
    # Note: spin_until_complete might need adaptation for service future
    # Using simplified wait for now
    time.sleep(2)

def set_gripper(node: JointPoseController, gripper_open: bool | float) -> None:
    fraction = float(gripper_open) * 0.22 
    gripper_pos = {'gripper_aperture': fraction}
    node.send_joint_pose(gripper_pos)
    spin_until_complete(node)

def move_head(node: HeadJointController, pose: Pose3D, z_fix: float = 0.0, tilt_bool: bool = True) -> None:
    pos = np.array([round(pose.coordinates[0], 3),round(pose.coordinates[1], 3), round(pose.coordinates[2]+z_fix, 3)])
    node.send_joint_pose(pos, tilt_bool=tilt_bool)
    spin_until_complete(node)

def look_ahead(node: JointPoseController) -> None:
    ahead_pos = {'joint_head_pan': -np.pi/2, 'joint_head_tilt': 0.0}
    node.send_joint_pose(ahead_pos)
    spin_until_complete(node)
