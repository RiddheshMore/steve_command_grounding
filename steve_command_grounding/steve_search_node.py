#!/usr/bin/env python3
"""
Steve Search Pipeline - Simplified Stretch-Compose Workflow

Workflow:
1. Parse natural language command → extract target object
2. Spatial Reasoning: Check scene graph for known object location
3. Navigate to location and verify with SAM3
4. Semantic Reasoning: If not found, ask DeepSeek for likely locations
5. Visit proposals sequentially, verify each with SAM3

No manipulation, no drawer opening - assumes objects are visible.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import threading
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_share_directory

from steve_command_grounding.robot import Robot
from steve_command_grounding.nlp_parser import NLPParser
from steve_command_grounding.semantic_reasoner import SemanticReasoner
from steve_command_grounding.spatial_planner import SpatialPlanner
from steve_command_grounding.search_execution import execute_search, SearchResult


class SteveSearchNode(Node):
    """
    Steve Search Node - Object Search and Localization
    
    Implements simplified Stretch-Compose workflow:
    - Spatial reasoning (scene graph lookup)
    - Semantic reasoning (DeepSeek LLM)
    - Visual verification (SAM3)
    
    No manipulation, no drawer interaction.
    """
    
    def __init__(self):
        super().__init__('steve_search_node')
        
        # Parameters
        self.declare_parameter('scene_graph_path', '/home/ritz/steve_ros2_ws/maps/ground_truth_graph')
        self.declare_parameter('sam3_host', '127.0.0.1')
        self.declare_parameter('sam3_port', 5005)
        self.declare_parameter('use_pan_tilt', True)  # Use pan-tilt camera for search
        self.declare_parameter('use_wrist', True)     # Use wrist camera for close-up
        
        scene_graph_path = self.get_parameter('scene_graph_path').value
        sam3_host = self.get_parameter('sam3_host').value
        sam3_port = self.get_parameter('sam3_port').value
        self.use_pan_tilt = self.get_parameter('use_pan_tilt').value
        
        # Get map path
        try:
            neo_sim_dir = get_package_share_directory('steve_simulation')
            map_yaml_path = os.path.join(neo_sim_dir, 'maps', 'steve_house.yaml')
            if not os.path.exists(map_yaml_path):
                map_yaml_path = os.path.join(neo_sim_dir, 'maps', 'small_house.yaml')
        except Exception as e:
            self.get_logger().warn(f"Could not find steve_simulation package: {e}")
            map_yaml_path = None
        
        # Initialize Robot with SAM3 support
        self.robot = Robot(
            self, 
            scene_graph_path, 
            map_yaml_path=map_yaml_path,
            sam3_host=sam3_host,
            sam3_port=sam3_port
        )
        
        # Grounding Components
        self.parser = NLPParser()
        self.reasoner = SemanticReasoner()
        self.planner = SpatialPlanner(standoff_dist=0.7)
        
        # State
        self.current_search_thread = None
        self.callback_group = MutuallyExclusiveCallbackGroup()
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, 
            '/command', 
            self.command_callback, 
            10,
            callback_group=self.callback_group
        )
        
        # Publishers
        self.result_pub = self.create_publisher(String, '/search_result', 10)
        
        self.get_logger().info("="*50)
        self.get_logger().info("  Steve Search Node Ready")
        self.get_logger().info("  - Scene graph: " + scene_graph_path)
        self.get_logger().info("  - SAM3 server: " + f"{sam3_host}:{sam3_port}")
        self.get_logger().info("="*50)

    def command_callback(self, msg):
        """Handle incoming search commands."""
        command_text = msg.data
        self.get_logger().info(f"Received command: {command_text}")
        
        # Cancel previous search
        if self.current_search_thread and self.current_search_thread.is_alive():
            self.get_logger().info("Cancelling previous search...")
            self.robot.navigation.cancel_current_goal()
        
        # Start new search
        self.current_search_thread = threading.Thread(
            target=self._execute_search,
            args=(command_text,)
        )
        self.current_search_thread.start()

    def _execute_search(self, command_text: str):
        """
        Execute the search pipeline (stretch-compose style).
        Delegates to search_execution.execute_search.
        """
        parsed = self.parser.parse(command_text)
        target_obj = parsed.get("object")
        if not target_obj or target_obj == "none":
            self.get_logger().error("Could not understand target object")
            self._publish_result(SearchResult(
                success=False,
                object_name="unknown",
                reasoning_path="parse_error",
            ))
            return

        self.get_logger().info(f"Searching for: {target_obj}")

        result = execute_search(
            self.robot,
            self.parser,
            self.reasoner,
            self.planner,
            target_obj,
            no_proposals=3,
            logger=self.get_logger(),
        )

        if result.success:
            self.robot.navigation.cancel_current_goal()
        self._publish_result(result)

    def _publish_result(self, result: SearchResult):
        """Publish search result."""
        msg = String()
        
        if result.success:
            furniture_info = f" at {result.found_at_furniture}" if result.found_at_furniture else ""
            msg.data = f"OBJECT FOUND: {result.object_name}{furniture_info} (path={result.reasoning_path})"
            self.get_logger().info(f"✅ {msg.data}")
        else:
            msg.data = f"OBJECT NOT FOUND: {result.object_name} (searched via {result.reasoning_path})"
            self.get_logger().info(f"❌ {msg.data}")
        
        self.result_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SteveSearchNode()
    
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
            pass


if __name__ == '__main__':
    main()
