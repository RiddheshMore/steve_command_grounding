#!/usr/bin/env python3
import math
import numpy as np
from geometry_msgs.msg import PoseStamped

class SpatialPlanner:
  """
  Handles spatial calculations for robot positioning.
  """
  def __init__(self, standoff_dist=0.8, max_offset=2.5):
    self.standoff_dist = standoff_dist
    self.max_offset = max_offset  # Maximum distance from centroid
    
    # Camera constants (from stretch-compose)
    self.H_FOV = 42.0  # horizontal fov
    self.V_FOV = 69.0  # vertical fov

  def calculate_fov_standoff(self, dimensions):
    """
    Calculate standoff distance to capture the entire furniture surface using FOV math.
    """
    if not dimensions or len(dimensions) < 3:
      return self.standoff_dist
      
    # dimensions: (width, depth, height)
    # distance = (size + padding) / (2 * tan(fov/2)) + depth/2
    padding = 0.1
    dist_w = (dimensions[0] + padding) / (2 * math.tan(math.radians(self.H_FOV / 2.0))) + dimensions[1] / 2.0
    dist_h = (dimensions[2] + padding) / (2 * math.tan(math.radians(self.V_FOV / 2.0))) + dimensions[1] / 2.0
    # Final distance from center = fov_dist
    total_fov_dist = max(dist_w, dist_h)
    
    return max(self.standoff_dist, total_fov_dist)

  def calculate_standoff_pose(self, target_centroid, target_dimensions=None, robot_pose=None, front_normal=None):
    """
    Calculate a position to stand square to the target.
    """
    tx, ty, tz = target_centroid
    
    # 1. Determine distance
    # Use 1.3m as the default "perfect" standoff seen in the user's example
    # but fall back to FOV-based if dimensions are provided
    dist = 1.3
    if target_dimensions:
      dist = self.calculate_fov_standoff(target_dimensions)
    
    # 2. Determine Normal Orientation
    # Default fallback: Assume furniture is aligned with Y axis if no normal provided
    ux, uy = (0.0, 1.0)
    
    if front_normal:
      nx, ny, _ = front_normal
      mag_n = math.sqrt(nx**2 + ny**2)
      if mag_n > 0.01:
        ux, uy = nx / mag_n, ny / mag_n
    elif robot_pose:
      # Use odometry to guess which side the front is
      rx, ry = robot_pose[0], robot_pose[1]
      dx, dy = rx - tx, ry - ty
      mag = math.sqrt(dx**2 + dy**2)
      if mag > 0.1:
        ux, uy = dx/mag, dy/mag

    # 3. Calculate Position
    robot_x = tx + ux * dist
    robot_y = ty + uy * dist
    
    # 4. Face target (Square to normal)
    # The user wants the robot "square" to the furniture, 
    # so we face directly opposite to the front normal.
    yaw = math.atan2(-uy, -ux)
    
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
