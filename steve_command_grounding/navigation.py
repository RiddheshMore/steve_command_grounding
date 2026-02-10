import math
import rclpy
import threading
import tf2_ros
from rclpy.duration import Duration
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
    
    # TF Support
    self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
    
    self._action_client = ActionClient(
      node,
      NavigateToPose,
      "navigate_to_pose"
    )
    # Wait for server briefly to avoid immediate failure on startup
    self.node.get_logger().info("Waiting for navigate_to_pose action server...")
    self._action_client.wait_for_server(timeout_sec=5.0)
    self._goal_handle = None
    self._nav_event = threading.Event()
    self._current_map_pose = None
    self._goal_yaw = 0.0
    self._send_in_progress = False
    self._last_result = None
    self._goal_accepted = False  # Track if goal was actually accepted
    
    # Stability tracking for convergence check
    self._last_dist = 1000.0
    self._distance_remaining = 1000.0
    self._stable_count = 0
    self._goal_x = 0.0
    self._goal_y = 0.0
    
    # Subscribe to amcl_pose for tracking (uses PoseWithCovarianceStamped)
    self._pose_sub = node.create_subscription(
      PoseWithCovarianceStamped,
      '/amcl_pose',
      self._pose_callback,
      10
    )
    
    # Debug publisher for testing
    self._debug_pub = node.create_publisher(PoseStamped, "/navigation_goal", 10)

  @property
  def distance_remaining(self):
      return self._distance_remaining

  @property
  def goal_handle(self):
      return self._goal_handle

  def _pose_callback(self, msg):
    self._current_map_pose = msg

  def get_current_pose(self):
    if self._current_map_pose:
      p = self._current_map_pose.pose.pose.position  # Note: pose.pose for PoseWithCovarianceStamped
      return (p.x, p.y)
    return None

  def get_current_pose_full(self):
    """Returns (x, y, yaw) if available, else None. Uses TF2 with AMCL fallback."""
    # 1. Try TF2 (Most robust, buffered)
    try:
        now = rclpy.time.Time()
        trans = self.tf_buffer.lookup_transform('map', 'base_link', now, timeout=Duration(seconds=0.1))
        p = trans.transform.translation
        o = trans.transform.rotation
        siny_cosp = 2 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (p.x, p.y, yaw)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        # self.node.get_logger().debug(f"TF lookup failed: {e}")
        pass

    # 2. Fallback to /amcl_pose subscription
    if self._current_map_pose:
      p = self._current_map_pose.pose.pose.position
      o = self._current_map_pose.pose.pose.orientation
      # Basic yaw extraction from quaternion (assuming 2D navigation)
      siny_cosp = 2 * (o.w * o.z + o.x * o.y)
      cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
      yaw = math.atan2(siny_cosp, cosy_cosp)
      return (p.x, p.y, yaw)
    return None

  def is_done(self):
    """Check if the current navigation goal is finished."""
    return not self._send_in_progress

  def get_result(self):
    """Returns the last navigation success/failure status."""
    return self._last_result

  def wait_for_server(self, timeout_sec: float = 10.0) -> bool:
    """Wait for Nav2 action server."""
    return self._action_client.wait_for_server(timeout_sec=timeout_sec)

  def go_to_pose(self, x: float, y: float, yaw: float, frame_id: str = "map", wait: bool = True):
    """
    Send a navigation goal. Optionally block until result.
    """
    if self._send_in_progress:
      self.node.get_logger().warn("A goal is already in progress, cancel or wait")
      return None

    if not self._action_client.server_is_ready():
      self.node.get_logger().error("navigate_to_pose server not ready")
      return None

    goal_msg = NavigateToPose.Goal()
    goal_msg.pose = self.build_pose(x, y, yaw, frame_id)
    self._debug_pub.publish(goal_msg.pose)

    self.node.get_logger().info(f"Sending goal to x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} rad in {frame_id}")
    self._goal_x = x
    self._goal_y = y
    self._goal_yaw = yaw
    self._send_in_progress = True
    self._goal_accepted = False
    self._last_dist = 1000.0
    self._stable_count = 0
    self._nav_event.clear()
    self._last_result = None

    send_future = self._action_client.send_goal_async(
      goal_msg, 
      feedback_callback=self._feedback_callback
    )
    
    if not wait:
      send_future.add_done_callback(self._goal_response_callback)
      return None

    # Blocking mode
    def _result_ready_cb(goal_handle_future):
      goal_handle = goal_handle_future.result()
      if not goal_handle.accepted:
        self.node.get_logger().error("Goal was rejected by server")
        self._send_in_progress = False
        self._nav_event.set()
        return
      self._goal_handle = goal_handle
      self._goal_accepted = True
      result_future = goal_handle.get_result_async()
      result_future.add_done_callback(self._result_callback)

    send_future.add_done_callback(_result_ready_cb)
    
    # Wait for the event to be set in _result_callback or _result_ready_cb (if rejected)
    finished = self._nav_event.wait(timeout=120.0)
    if not finished:
      self.node.get_logger().error("Navigation timed out")
      self._send_in_progress = False
      
    return self._last_result

  def build_pose(self, x: float, y: float, yaw: float, frame_id: str = "map") -> PoseStamped:
    pose = PoseStamped()
    pose.header.stamp = self.node.get_clock().now().to_msg()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0
    q = yaw_to_quaternion(yaw)
    pose.pose.orientation.x = float(q["x"])
    pose.pose.orientation.y = float(q["y"])
    pose.pose.orientation.z = float(q["z"])
    pose.pose.orientation.w = float(q["w"])
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
    self._goal_accepted = True
    result_future = goal_handle.get_result_async()
    result_future.add_done_callback(self._result_callback)

  def _result_callback(self, future):
    result = future.result().result
    status = future.result().status
    self.node.get_logger().info(f"Result received with status={status}")
    
    # If we already set _last_result to True (via stall/threshold check), keep it
    if self._last_result is not True:
        self._last_result = (status == GoalStatus.STATUS_SUCCEEDED)
        
    self._send_in_progress = False
    self._goal_accepted = False
    self._nav_event.set()

  def _feedback_callback(self, feedback_msg):
    if not self._goal_accepted:
      return
      
    fb = feedback_msg.feedback
    dist = getattr(fb, 'distance_remaining', 1000.0)
    self._distance_remaining = dist
    
    # Smart Convergence Logic
    curr = self.get_current_pose_full()
    if not curr: return
    
    cx, cy, cyaw = curr
    yaw_err = abs(math.atan2(math.sin(cyaw - self._goal_yaw), math.cos(cyaw - self._goal_yaw)))
    manual_dist = math.sqrt((cx - self._goal_x)**2 + (cy - self._goal_y)**2)
    effective_dist = max(dist, manual_dist) if dist < 0.05 else dist

    is_at_goal = False
    if effective_dist < 0.12 and yaw_err < 0.1:
        is_at_goal = True
    elif effective_dist < 0.35 and yaw_err < 0.2:
        if abs(effective_dist - self._last_dist) < 0.005:
            self._stable_count += 1
        else:
            self._stable_count = 0
            
        if self._stable_count >= 10: # ~2 seconds
            is_at_goal = True

    self._last_dist = effective_dist

    if is_at_goal:
        self.node.get_logger().info(f"Convergence met. Resulting True.")
        self._last_result = True
        if self._goal_handle:
             self._goal_handle.cancel_goal_async()
        self._nav_event.set()

  def cancel_current_goal(self):
    """Cancel the current navigation goal if any."""
    if self._goal_handle is None or not self._send_in_progress:
      return
    
    self.node.get_logger().info("Requesting goal cancellation...")
    self._goal_handle.cancel_goal_async()
    self._nav_event.wait(timeout=5.0)
