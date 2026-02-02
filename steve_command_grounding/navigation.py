import math
import rclpy
import threading
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

def yaw_to_quaternion(yaw: float):
  """Convert yaw (rad) to geometry_msgs/Quaternion."""
  half = yaw * 0.5
  cz = math.cos(half)
  sz = math.sin(half)
  return {
    "x": 0.0,
    "y": 0.0,
    "z": sz,
    "w": cz,
  }

class Navigation:
  """
  Simple wrapper to drive a robot using Nav2 NavigateToPose.
  """

  def __init__(self, node):
    self.node = node
    self._action_client = ActionClient(
      node,
      NavigateToPose,
      "navigate_to_pose"
    )
    self._goal_handle = None
    self._nav_event = threading.Event()
    self._current_map_pose = None
    self._send_in_progress = False
    self._last_result = None
    self._goal_accepted = False  # Track if goal was actually accepted
    
    # Subscribe to amcl_pose for tracking (uses PoseWithCovarianceStamped)
    self._pose_sub = node.create_subscription(
      PoseWithCovarianceStamped,
      '/amcl_pose',
      self._pose_callback,
      10
    )
    
    # Debug publisher for testing
    self._debug_pub = node.create_publisher(PoseStamped, "/navigation_goal", 10)

  def _pose_callback(self, msg):
    self._current_map_pose = msg

  def get_current_pose(self):
    if self._current_map_pose:
      p = self._current_map_pose.pose.pose.position  # Note: pose.pose for PoseWithCovarianceStamped
      return (p.x, p.y)
    return None

  def wait_for_server(self, timeout_sec: float = 10.0) -> bool:
    """Wait for Nav2 action server."""
    return self._action_client.wait_for_server(timeout_sec=timeout_sec)

  def go_to_pose(self, x: float, y: float, yaw: float, frame_id: str = "map", wait: bool = True):
    """Send a navigation goal. Optionally block until result."""
    if self._send_in_progress:
      self.node.get_logger().warn("A goal is already in progress. Ignoring new goal.")
      return None

    if not self._action_client.server_is_ready():
      self.node.get_logger().error("navigate_to_pose server not ready")
      return None

    goal_msg = NavigateToPose.Goal()
    goal_msg.pose = self.build_pose(x, y, yaw, frame_id)
    self._debug_pub.publish(goal_msg.pose)

    self.node.get_logger().info(f"Sending goal to x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
    self._send_in_progress = True
    self._goal_accepted = False  # Reset acceptance flag
    self._nav_event.clear()
    self._last_result = None

    send_future = self._action_client.send_goal_async(
      goal_msg, 
      feedback_callback=self._feedback_callback
    )
    
    if wait:
      send_future.add_done_callback(self._goal_response_callback)
      # Wait for the event to be set in _result_callback or _goal_response_callback (if rejected)
      # Timeout of 120s as a safety measure
      finished = self._nav_event.wait(timeout=120.0)
      if not finished:
        self.node.get_logger().error("Navigation timed out")
        self._send_in_progress = False
      return self._last_result
    else:
      send_future.add_done_callback(self._goal_response_callback)
      return None

  def build_pose(self, x: float, y: float, yaw: float, frame_id: str = "map") -> PoseStamped:
    pose = PoseStamped()
    pose.header.stamp = self.node.get_clock().now().to_msg()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    q = yaw_to_quaternion(yaw)
    pose.pose.orientation.x = q["x"]
    pose.pose.orientation.y = q["y"]
    pose.pose.orientation.z = q["z"]
    pose.pose.orientation.w = q["w"]
    return pose

  def _goal_response_callback(self, future):
    goal_handle = future.result()
    if not goal_handle.accepted:
      self.node.get_logger().error("Goal rejected")
      self._send_in_progress = False
      self._nav_event.set()
      return

    self.node.get_logger().info("Goal accepted")
    self._goal_handle = goal_handle
    self._goal_accepted = True  # Now feedback can check distance
    result_future = goal_handle.get_result_async()
    result_future.add_done_callback(self._result_callback)

  def _result_callback(self, future):
    self._last_result = future.result().result
    self._send_in_progress = False
    self._goal_accepted = False
    status = future.result().status
    if status == GoalStatus.STATUS_SUCCEEDED:
        self.node.get_logger().info("Navigation succeeded")
    else:
        self.node.get_logger().info(f"Navigation ended with status: {status}")
    self._nav_event.set()

  def _feedback_callback(self, feedback_msg):
    # Only check distance if goal has been accepted and we're actively navigating
    if not self._goal_accepted:
      return
      
    fb = feedback_msg.feedback
    dist = getattr(fb, 'distance_remaining', 1000.0)
    
    # Log progress occasionally
    if dist > 0.5:
      self.node.get_logger().info(f"Distance remaining: {dist:.2f}m", throttle_duration_sec=2.0)
    elif dist > 0.0:
      self.node.get_logger().info(f"Distance remaining: {dist:.2f}m (approaching goal)")
    
    # Only consider early success if distance is very small but not zero
    # (zero often means navigation hasn't computed distance yet)
    if 0.0 < dist < 0.15:
        self.node.get_logger().info(f"Robot is very close to goal (dist={dist:.3f}m). Considering success.")
        self._last_result = None
        self._send_in_progress = False
        self._goal_accepted = False
        self._nav_event.set()

  def cancel_current_goal(self):
    """Cancel the current navigation goal if any."""
    if self._goal_handle is None or not self._send_in_progress:
      return
    
    self.node.get_logger().info("Requesting goal cancellation...")
    self._goal_handle.cancel_goal_async()
    # Wait for result callback to clear _send_in_progress
    self._nav_event.wait(timeout=5.0)
