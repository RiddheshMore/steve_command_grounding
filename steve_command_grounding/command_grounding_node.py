#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import os
import threading
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_share_directory

from steve_command_grounding.robot import Robot
from steve_command_grounding.nlp_parser import NLPParser
from steve_command_grounding.semantic_reasoner import SemanticReasoner
from steve_command_grounding.spatial_planner import SpatialPlanner

class CommandGroundingNode(Node):
  """
  Command Grounding Node (Robot Class Architecture)
  Orchestrates the translation of natural language commands to robot actions.
  """
  def __init__(self):
    super().__init__('command_grounding_node')
    
    # Parameters
    self.declare_parameter('scene_graph_path', '/home/ritz/steve_ros2_ws/generated_graph')
    scene_graph_path = self.get_parameter('scene_graph_path').value
    
    # Get map path from steve_simulation package
    try:
      neo_sim_dir = get_package_share_directory('steve_simulation')
      map_yaml_path = os.path.join(neo_sim_dir, 'maps', 'small_house.yaml')
    except Exception as e:
      self.get_logger().warn(f"Could not find steve_simulation package: {e}")
      map_yaml_path = None
    
    # Initialize Robot (Aggregates Nav/Perc/Manip)
    self.robot = Robot(self, scene_graph_path, map_yaml_path=map_yaml_path)
    
    # Grounding Components
    self.parser = NLPParser()
    self.reasoner = SemanticReasoner()
    self.planner = SpatialPlanner(standoff_dist=1.0)
    
    # State management
    self.current_goal_thread = None
    self.callback_group = MutuallyExclusiveCallbackGroup()
    
    # Subscribers
    self.command_sub = self.create_subscription(
      String, 
      '/command', 
      self.command_callback, 
      10,
      callback_group=self.callback_group
    )
    
    self.get_logger().info("Steve Command Grounding Node Ready")

  def command_callback(self, msg):
    command_text = msg.data
    self.get_logger().info(f"Received command: {command_text}")
    
    # Cancel previous goal if any
    if self.current_goal_thread and self.current_goal_thread.is_alive():
      self.get_logger().info("Cancelling current goal to process new command")
      self.robot.navigation.cancel_current_goal()
      # We don't necessarily need to join here, as Navigation handles one goal at a time
    
    # Process in a new thread to avoid blocking the callback
    self.current_goal_thread = threading.Thread(
      target=self._process_command,
      args=(command_text,)
    )
    self.current_goal_thread.start()

  def _process_command(self, command_text):
    parsed = self.parser.parse(command_text)
    target_obj = parsed.get('object')
    action = parsed.get('action', 'none')
    
    if not target_obj or target_obj == 'none':
      self.get_logger().warn("Could not understand target object")
      return
      
    self.get_logger().info(f"Action: {action}, Target: {target_obj}")
    
    # Determine if this is a "go to location" command vs "find object" command
    is_navigation_command = action in ['go to', 'navigate to']

    # I. Fast Path: Check memory (supports multiple matches sequentially)
    matches = self.robot.perception.find_objects(target_obj)
    
    if matches:
      self.get_logger().info(f"Fast Path: found {len(matches)} matches for '{target_obj}' in memory")
      for i, found_data in enumerate(matches):
        furniture_name = found_data.get('label', 'unknown')
        self.get_logger().info(f"Fast Path match {i+1}/{len(matches)}: {furniture_name} (id: {found_data.get('id')}, centroid: {found_data.get('centroid')})")
        
        success = self._navigate_to_furniture(found_data)
        
        if success:
          if is_navigation_command:
            # For "go to" commands, we're done once we reach the location
            self.get_logger().info(f"Successfully reached {furniture_name}!")
            return
          else:
            # For "find" commands, we need to search for object at this location
            self.get_logger().info(f"Reached {furniture_name}. Object not found (simulated), checking next match...")
        else:
          self.get_logger().warn(f"Failed to reach {furniture_name}. Moving to next match...")
      
      if is_navigation_command:
        self.get_logger().warn(f"Could not reach any '{target_obj}' location")
        return
        
      self.get_logger().info(f"Memory matches for '{target_obj}' exhausted. Checking Slow Path proposals...")
    
    # II. Slow Path: Ask LLM for proposals (only for "find object" commands)
    if not is_navigation_command:
      self.get_logger().info(f"Object not in memory. Asking LLM for locations...")
      proposals = self.reasoner.suggest_locations(
        target_obj, 
        self.robot.perception.get_available_furniture()
      )
      
      if not proposals:
        self.get_logger().error(f"No location proposals found for {target_obj}")
        return

      # Sequential Checking Loop
      for i, proposal in enumerate(proposals):
        furniture_id = proposal['furniture_id']
        furniture_name = proposal['furniture_name']
        
        self.get_logger().info(f"LLM proposal {i+1}/{len(proposals)}: {furniture_name} (id: {furniture_id})")
        
        target_furniture = self.robot.perception.get_available_furniture().get(furniture_id)
        if target_furniture:
          success = self._navigate_to_furniture(target_furniture)
          
          if success:
            # Simulate Detection:
            self.get_logger().info(f"Reached {furniture_name}. Object not found, checking next proposal...")
          else:
            self.get_logger().warn(f"Failed to reach {furniture_name}. Moving to next proposal...")
        else:
          self.get_logger().warn(f"Furniture {furniture_id} not found in scene data")

  def _navigate_to_furniture(self, target_data):
    """Wait and return result of navigation."""
    centroid = target_data.get('centroid')
    dimensions = target_data.get('dimensions')
    
    self.get_logger().info(f"Target data: centroid={centroid}, dimensions={dimensions}")
    
    if not centroid or centroid == [0, 0, 0]:
      self.get_logger().error(f"Target data missing or invalid centroid: {centroid}")
      return False
        
    robot_pose = self.robot.get_current_pose()
    self.get_logger().info(f"Robot current pose: {robot_pose}")
    
    x, y, yaw = self.planner.calculate_standoff_pose(centroid, dimensions, robot_pose=robot_pose)
    self.get_logger().info(f"Calculated standoff pose: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
    
    # Using blocking wait provided by Nav class logic
    res = self.robot.navigation.go_to_pose(x, y, yaw, wait=True)
    return True if res is not False else False

def main(args=None):
  rclpy.init(args=args)
  node = CommandGroundingNode()
  
  # Use MultiThreadedExecutor to allow concurrent execution of callbacks
  executor = rclpy.executors.MultiThreadedExecutor()
  executor.add_node(node)
  
  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    try:
      rclpy.shutdown()
    except Exception:
      pass  # Already shut down by signal handler

if __name__ == '__main__':
  main()
