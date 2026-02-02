#!/usr/bin/env python3
import math
import numpy as np
from geometry_msgs.msg import PoseStamped

class SpatialPlanner:
  """
  Handles spatial calculations for robot positioning.
  """
  def __init__(self, standoff_dist=0.8, max_offset=2.0):
    self.standoff_dist = standoff_dist
    self.max_offset = max_offset  # Maximum distance from centroid

  def calculate_standoff_pose(self, target_centroid, target_dimensions=None, robot_pose=None):
    """
    Calculate a position to stand near the target.
    Uses the SMALLER horizontal dimension to estimate approach distance,
    as the larger dimension often represents length of long furniture.
    
    Returns: (x, y, yaw)
    """
    tx, ty, tz = target_centroid
    
    # Direction defaults to line from origin to target
    # But if robot_pose is provided, use line from robot to target
    ref_x, ref_y = (0.0, 0.0)
    if robot_pose:
      ref_x, ref_y = robot_pose[0], robot_pose[1]

    dx = tx - ref_x
    dy = ty - ref_y
    dist_to_ref = math.sqrt(dx**2 + dy**2)

    if dist_to_ref < 0.1:
      # If target is too close to reference, just stand back along X
      return (tx - self.standoff_dist, ty, 0.0)
        
    ux = dx / dist_to_ref
    uy = dy / dist_to_ref
    
    if target_dimensions:
      # Use the SMALLER of the two horizontal dimensions (x, y) as approach radius
      # This is because furniture often has one long side and one short side
      # We want to approach from the short side
      horiz_dims = [target_dimensions[0], target_dimensions[1]]
      radius = min(horiz_dims) / 2.0
      # Cap the radius to avoid huge offsets from bad scene graph data
      radius = min(radius, 1.5)
      total_dist = radius + self.standoff_dist
    else:
      total_dist = self.standoff_dist
    
    # Cap total distance to reasonable max
    total_dist = min(total_dist, self.max_offset)
        
    robot_x = tx - ux * total_dist
    robot_y = ty - uy * total_dist
    
    # Face the target
    yaw = math.atan2(uy, ux)
    
    return (robot_x, robot_y, yaw)

  def create_pose_stamped(self, x, y, yaw, frame_id='map'):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    
    qz = math.sin(yaw / 2)
    qw = math.cos(yaw / 2)
    
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    
    return pose
