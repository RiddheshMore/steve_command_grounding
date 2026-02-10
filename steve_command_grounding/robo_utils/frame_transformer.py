"""
FrameTransformer for handling coordinate transformations.
"""

from __future__ import annotations
import typing
import numpy as np
from .coordinates import Pose2D, Pose3D

AnyPose = typing.Union[Pose2D, Pose3D]

class FrameTransformer:
    def __init__(self):
        self.frames_tform_vision = {}

    def transform(self, start_frame: str, end_frame: str, start_pose: AnyPose) -> AnyPose:
        # Simplified for now as Steve might use different TF mechanism
        # In a real scenario, this would interface with tf2_ros
        return start_pose 

    def get_tf_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        # Placeholder for matrix transformation
        return np.eye(4)
