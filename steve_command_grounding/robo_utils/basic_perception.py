"""
All things video and imaging.
"""

from __future__ import annotations
import cv2
import numpy as np
from rclpy.node import Node

from .stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from .stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
# from .stretch_package.stretch_images.depth_image_subscriber import DepthImageSubscriber
from .stretch_package.stretch_state.frame_transformer import FrameTransformer
from .stretch_package.stretch_movement.move_to_pose import JointPoseController

from .basic_movement import set_gripper, spin_until_complete
from .importer import PointCloud, Vector3dVector 
from .recursive_config import Config

def get_rgb_picture(source_node: Node, joint_node: JointPoseController, topic: str, gripper: bool = False, save_block: bool = False, vis_block: bool = False) -> np.ndarray:
    if gripper:
        set_gripper(joint_node, True)
    image_node = source_node(topic, not gripper, save_block, vis_block)
    # Simple blocking wait if node doesn't have its own loop
    while rclpy.ok() and image_node.cv_image is None:
        rclpy.spin_once(image_node, timeout_sec=0.1)
    image = image_node.cv_image
    image_node.destroy_node()
    return image

def get_depth_picture(source_node: Node, pose_node: JointPoseController, topic: str, gripper: bool = False, save_block: bool = False, vis_block: bool = False) -> np.ndarray:
    if gripper:
        set_gripper(pose_node, True)
    image_node = source_node(topic, not gripper, save_block, vis_block)
    while rclpy.ok() and image_node.cv_image is None:
        rclpy.spin_once(image_node, timeout_sec=0.1)
    image = image_node.cv_image
    image_node.destroy_node()
    return image
