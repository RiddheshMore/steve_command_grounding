#!/usr/bin/env python3
from steve_command_grounding.navigation import Navigation
from steve_command_grounding.perception import Perception
from steve_command_grounding.manipulation import Manipulation

class Robot:
  """
  Main Robot class that aggregates navigation, perception, and manipulation.
  """
  def __init__(self, node, scene_graph_path, map_yaml_path=None):
    self.node = node
    self.navigation = Navigation(node)
    self.perception = Perception(node, scene_graph_path, map_yaml_path=map_yaml_path)
    self.manipulation = Manipulation(node)
    
    self.node.get_logger().info("Robot class initialized")

  def wait_for_services(self, timeout_sec=10.0):
    """Wait for all necessary action servers and services."""
    self.node.get_logger().info("Waiting for robot services...")
    self.navigation.wait_for_server(timeout_sec=timeout_sec)
    self.node.get_logger().info("All robot services ready")

  def get_current_pose(self):
    """Returns (x, y) if available, else None."""
    return self.navigation.get_current_pose()
