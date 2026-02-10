#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class PanTiltController:
    """
    Controls the pan-tilt unit of the robot to look at specific targets.
    """
    def __init__(self, node: Node):
        self.node = node
        self.publisher = self.node.create_publisher(
            JointTrajectory,
            '/pan_tilt_controller/joint_trajectory',
            10
        )
        
        # Joint names from pan_tilt_controllers.yaml
        # Base motor (PAN) is pan_tilt_tilt_motor_joint
        # Second motor (TILT) is pan_tilt_pan_motor_joint
        self.joint_names = ['pan_tilt_tilt_motor_joint', 'pan_tilt_pan_motor_joint']
        
        # Camera height/offset from URDF (approximate centroids)
        self.camera_height = 1.6  # approx meters from base_link
        self.camera_x_offset = -0.15  # meters from base_link
        
        self.node.get_logger().info("PanTiltController initialized")

    def look_at(self, target_xyz, robot_pose=None):
        """
        Calculates and sends joint commands to look at a 3D point.
        target_xyz: (x, y, z) in map frame
        robot_pose: (x, y, yaw) in map frame
        """
        tx, ty, tz = target_xyz
        
        if robot_pose is None:
            self.node.get_logger().warn("Robot pose not provided, cannot calculate precise pan-tilt")
            return
            
        rx, ry, ryaw = robot_pose
        
        # 1. Transform target to robot base frame
        # (Actually, we want it relative to the pan-tilt base)
        # Pan-tilt base is at (-0.15, 0) relative to base_link
        
        # Robot's world position of pan-tilt base
        pt_base_x = rx + self.camera_x_offset * math.cos(ryaw)
        pt_base_y = ry + self.camera_x_offset * math.sin(ryaw)
        
        dx = tx - pt_base_x
        dy = ty - pt_base_y
        dz = tz - self.camera_height
        
        dist_horiz = math.sqrt(dx**2 + dy**2)
        
        # 2. Calculate angles
        # Pan relative to map
        target_pan_world = math.atan2(dy, dx)
        
        # The pan-tilt tower is mounted forward facing
        pan_angle = target_pan_world - ryaw
        
        # Normalize pan_angle to [-pi, pi]
        while pan_angle > math.pi: pan_angle -= 2 * math.pi
        while pan_angle < -math.pi: pan_angle += 2 * math.pi
        
        # Tilt angle (negative is looking down)
        tilt_angle = math.atan2(dz, dist_horiz)
        
        self.node.get_logger().info(f"Looking at target {target_xyz} (Pan: {pan_angle:.2f}, Tilt: {tilt_angle:.2f})")
        
        self.send_command(pan_angle, tilt_angle)

    def send_command(self, pan, tilt, duration_sec=1.5):
        """Send joint trajectory command."""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(pan), float(tilt)]
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        
        msg.points = [point]
        self.publisher.publish(msg)

    def reset(self, duration_sec=2.0):
        """Reset camera to forward-looking position."""
        self.send_command(0.0, -0.2, duration_sec=duration_sec)
