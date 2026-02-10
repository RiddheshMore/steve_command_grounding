#!/usr/bin/env python3
"""
Camera Perception Module for Steve Robot
Subscribes to dual camera topics and provides SAM3-based object detection.

Cameras:
  - Pan-tilt camera (L515): /pan_tilt_camera/color/image_raw
  - Wrist camera (D405): /wrist_camera/color/image_raw
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs

from steve_command_grounding.sam3_interface import SAM3Client, SegmentationMask, visualize_masks
from steve_command_grounding.robo_utils.point_clouds import pointcloud2_to_o3d


@dataclass
class CameraFrame:
    """Holds latest camera data."""
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    point_cloud: Optional[PointCloud2] = None
    camera_info: Optional[CameraInfo] = None
    timestamp: Optional[float] = None
    header: Optional[any] = None


class CameraPerception:
    """
    Camera-based perception with SAM3 integration.
    
    Subscribes to:
      - /pan_tilt_camera/color/image_raw
      - /pan_tilt_camera/aligned_depth_to_color/image_raw
      - /wrist_camera/color/image_raw
      - /wrist_camera/aligned_depth_to_color/image_raw
    
    Usage:
        perception = CameraPerception(node)
        
        # Get current frame
        frame = perception.get_frame("pan_tilt")
        
        # Detect all objects in view
        masks = perception.detect_objects("pan_tilt")
        
        # Segment object at point (clicked by user or from detection)
        masks = perception.segment_at_point("wrist", x=320, y=240)
    """
    
    CAMERAS = {
        "pan_tilt": {
            "color_topic": "/pan_tilt_camera/pan_tilt_camera/image_raw",
            "depth_topic": "/pan_tilt_camera/pan_tilt_camera/depth/image_raw",
            "info_topic": "/pan_tilt_camera/pan_tilt_camera/camera_info",
            "points_topic": "/pan_tilt_camera/pan_tilt_camera/points"
        },
        "wrist": {
            "color_topic": "/wrist_camera/wrist_camera/image_raw",
            "depth_topic": "/wrist_camera/wrist_camera/depth/image_raw",
            "info_topic": "/wrist_camera/wrist_camera/camera_info",
            "points_topic": "/wrist_camera/wrist_camera/points"
        }
    }
    
    def __init__(self, node: Node, sam3_host: str = "127.0.0.1", sam3_port: int = 5005, tf_buffer: Optional[tf2_ros.Buffer] = None):
        self.node = node
        self.bridge = CvBridge()
        self.tf_buffer = tf_buffer
        
        # Camera data storage
        self.frames: Dict[str, CameraFrame] = {
            cam: CameraFrame() for cam in self.CAMERAS
        }
        self._locks = {cam: threading.Lock() for cam in self.CAMERAS}
        
        # SAM3 client
        self.sam3 = SAM3Client(host=sam3_host, port=sam3_port)
        self._sam3_available = self.sam3.is_available()
        if not self._sam3_available:
            self.node.get_logger().warn("SAM3 server not available - detection disabled")
        
        # Create subscribers
        self._create_subscribers()
        
        # Publisher for visualizations
        self.vis_pub = self.node.create_publisher(Image, '/sam3/visualization', 10)
        
        self.node.get_logger().info("CameraPerception initialized with shared TF buffer")
    
    def _create_subscribers(self):
        """Create ROS2 subscribers for all cameras."""
        for cam_name, topics in self.CAMERAS.items():
            # Color image
            self.node.create_subscription(
                Image,
                topics["color_topic"],
                lambda msg, c=cam_name: self._color_callback(msg, c),
                10
            )
            
            # Depth image
            self.node.create_subscription(
                Image,
                topics["depth_topic"],
                lambda msg, c=cam_name: self._depth_callback(msg, c),
                10
            )
            
            # Camera info
            self.node.create_subscription(
                CameraInfo,
                topics["info_topic"],
                lambda msg, c=cam_name: self._info_callback(msg, c),
                10
            )

            # Point Cloud
            self.node.create_subscription(
                PointCloud2,
                topics["points_topic"],
                lambda msg, c=cam_name: self._points_callback(msg, c),
                10
            )
            
            self.node.get_logger().info(f"Subscribed to {cam_name} camera topics")
    
    def _color_callback(self, msg: Image, camera: str):
        """Process color image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._locks[camera]:
                self.frames[camera].rgb = cv_image
                self.frames[camera].timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.frames[camera].header = msg.header
        except Exception as e:
            self.node.get_logger().error(f"Color callback error ({camera}): {e}")
    
    def _depth_callback(self, msg: Image, camera: str):
        """Process depth image."""
        try:
            # Depth is typically 16UC1 (mm) or 32FC1 (meters)
            if msg.encoding == "16UC1":
                cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            
            with self._locks[camera]:
                self.frames[camera].depth = cv_image
                self.frames[camera].header = msg.header
        except Exception as e:
            self.node.get_logger().error(f"Depth callback error ({camera}): {e}")
    
    def _info_callback(self, msg: CameraInfo, camera: str):
        """Store camera info."""
        with self._locks[camera]:
            self.frames[camera].camera_info = msg
            self.frames[camera].header = msg.header

    def _points_callback(self, msg: PointCloud2, camera: str):
        """Store PointCloud2."""
        with self._locks[camera]:
            self.frames[camera].point_cloud = msg
            self.frames[camera].header = msg.header
    
    def get_frame(self, camera: str = "pan_tilt") -> Optional[CameraFrame]:
        """Get latest frame from specified camera."""
        if camera not in self.frames:
            self.node.get_logger().error(f"Unknown camera: {camera}")
            return None
        
        with self._locks[camera]:
            frame = self.frames[camera]
            if frame.rgb is None:
                return None
            # Return a copy to avoid threading issues
            return CameraFrame(
                rgb=frame.rgb.copy() if frame.rgb is not None else None,
                depth=frame.depth.copy() if frame.depth is not None else None,
                point_cloud=frame.point_cloud, # No copy for now (heavy)
                camera_info=frame.camera_info,
                timestamp=frame.timestamp,
                header=frame.header
            )

    def get_latest_pcd(self, camera: str = "pan_tilt"):
        """Returns Open3D PointCloud from latest ROS PointCloud2."""
        frame = self.get_frame(camera)
        if frame and frame.point_cloud:
            return pointcloud2_to_o3d(frame.point_cloud)
        return None
    
    def detect_objects(
        self, 
        camera: str = "pan_tilt",
        min_area: int = 500,
        max_masks: int = 30
    ) -> List[SegmentationMask]:
        """
        Detect all objects in camera view using SAM3.
        
        Args:
            camera: Which camera to use ("pan_tilt" or "wrist")
            min_area: Minimum object area in pixels
            max_masks: Maximum number of objects to detect
            
        Returns:
            List of SegmentationMask objects
        """
        if not self._sam3_available:
            self.node.get_logger().warn("SAM3 not available")
            return []
        
        frame = self.get_frame(camera)
        if frame is None or frame.rgb is None:
            self.node.get_logger().warn(f"No image from {camera} camera")
            return []
        
        self.node.get_logger().info(f"Detecting objects in {camera} camera...")
        masks = self.sam3.detect_all(frame.rgb, min_area=min_area, max_masks=max_masks)
        self.node.get_logger().info(f"Detected {len(masks)} objects")
        
        # Publish visualization
        self.publish_visualization(camera, masks)
        
        return masks
    
    def find_object(
        self, 
        camera: str = "pan_tilt",
        prompt: str = "object",
        min_area: int = 500
    ) -> List[SegmentationMask]:
        """
        Find specific object using text prompt.
        
        Args:
            camera: Which camera to use
            prompt: Text prompt for searching
            min_area: Minimum area filter
            
        Returns:
            List of SegmentationMask objects matching the prompt
        """
        if not self._sam3_available:
            return []
            
        frame = self.get_frame(camera)
        if frame is None or frame.rgb is None:
            return []
            
        self.node.get_logger().info(f"Searching for '{prompt}' in {camera} camera...")
        self.node.get_logger().info(f"Image stats: shape={frame.rgb.shape}, dtype={frame.rgb.dtype}, min={frame.rgb.min()}, max={frame.rgb.max()}")
        masks = self.sam3.segment_text(frame.rgb, prompt)
        
        # Filter by area
        masks = [m for m in masks if m.area >= min_area]
        self.node.get_logger().info(f"Found {len(masks)} candidate objects for '{prompt}'")
        
        # Publish visualization
        self.publish_visualization(camera, masks)
        
        return masks

    def publish_visualization(
        self,
        camera: str,
        masks: List[SegmentationMask]
    ):
        """Publish SAM3 masks to the visualization topic."""
        frame = self.get_frame(camera)
        if frame is None or frame.rgb is None:
            return
            
        vis_image = visualize_masks(frame.rgb, masks)
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            vis_msg.header.stamp = self.node.get_clock().now().to_msg()
            self.vis_pub.publish(vis_msg)
        except Exception as e:
            self.node.get_logger().error(f"Failed to publish visualization: {e}")
    
    def segment_at_point(
        self, 
        camera: str = "pan_tilt",
        x: int = None,
        y: int = None,
        multimask: bool = False
    ) -> List[SegmentationMask]:
        """
        Segment object at specific point in camera view.
        
        Args:
            camera: Which camera to use
            x, y: Pixel coordinates (default: center of image)
            multimask: Return multiple mask proposals
            
        Returns:
            List of SegmentationMask objects
        """
        if not self._sam3_available:
            return []
        
        frame = self.get_frame(camera)
        if frame is None or frame.rgb is None:
            return []
        
        # Default to center if not specified
        if x is None:
            x = frame.rgb.shape[1] // 2
        if y is None:
            y = frame.rgb.shape[0] // 2
        
        self.node.get_logger().info(f"Segmenting at point ({x}, {y}) in {camera} camera")
        return self.sam3.segment_point(frame.rgb, points=[[x, y]], multimask=multimask)
    
    def segment_with_box(
        self, 
        camera: str = "pan_tilt",
        box: Tuple[int, int, int, int] = None
    ) -> List[SegmentationMask]:
        """
        Segment object within bounding box.
        
        Args:
            camera: Which camera to use
            box: (x1, y1, x2, y2) bounding box
            
        Returns:
            List of SegmentationMask objects
        """
        if not self._sam3_available or box is None:
            return []
        
        frame = self.get_frame(camera)
        if frame is None or frame.rgb is None:
            return []
        
        return self.sam3.segment_box(frame.rgb, box)
    
    def pixel_to_3d(
        self, 
        camera: str,
        x: int, 
        y: int
    ) -> Optional[Tuple[float, float, float]]:
        """
        Convert pixel coordinates to 3D point using depth.
        
        Args:
            camera: Which camera
            x, y: Pixel coordinates
            
        Returns:
            (X, Y, Z) in camera frame, or None if depth unavailable
        """
        frame = self.get_frame(camera)
        if frame is None or frame.depth is None or frame.camera_info is None:
            return None
        
        # Get depth value
        if 0 <= y < frame.depth.shape[0] and 0 <= x < frame.depth.shape[1]:
            depth = frame.depth[y, x]
            
            # Convert to meters if needed
            if frame.depth.dtype == np.uint16:
                depth = depth / 1000.0  # mm to meters
            
            if depth <= 0 or depth > 10.0:  # Invalid depth
                return None
            
            # Camera intrinsics
            fx = frame.camera_info.k[0]
            fy = frame.camera_info.k[4]
            cx = frame.camera_info.k[2]
            cy = frame.camera_info.k[5]
            
            # Back-project to 3D
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            return (X, Y, Z)
        
        return None
    
    def transform_to_map(self, camera_point: Tuple[float, float, float], header=None) -> Optional[Tuple[float, float, float]]:
        """
        Transform a 3D point from camera frame to map frame using TF2.
        """
        if not self.tf_buffer or not header:
            return None
            
        try:
            # Create a PoseStamped for the point
            ps = PoseStamped()
            ps.header = header
            ps.pose.position.x = float(camera_point[0])
            ps.pose.position.y = float(camera_point[1])
            ps.pose.position.z = float(camera_point[2])
            ps.pose.orientation.w = 1.0
            
            # 1. Try exact timestamp
            ps_map = self.tf_buffer.transform(ps, "map", timeout=Duration(seconds=0.1))
            return (ps_map.pose.position.x, ps_map.pose.position.y, ps_map.pose.position.z)
        except Exception:
            try:
                # 2. Fallback to latest available transform
                ps.header.stamp = rclpy.time.Time().to_msg()
                ps_map = self.tf_buffer.transform(ps, "map", timeout=Duration(seconds=0.1))
                return (ps_map.pose.position.x, ps_map.pose.position.y, ps_map.pose.position.z)
            except Exception as e:
                self.node.get_logger().warn(f"TF transform failed (including latest fallback): {e}")
                return None

    def transform_pcd_to_map(self, pcd: o3d.geometry.PointCloud, header: Optional[any] = None) -> Optional[o3d.geometry.PointCloud]:
        """Transform an entire point cloud to the map frame using TF2."""
        if not self.tf_buffer or not header:
            self.node.get_logger().warn("Cannot transform PCD without TF buffer and header")
            return None
            
        try:
            # 1. Try exact timestamp
            trans = self.tf_buffer.lookup_transform("map", header.frame_id, header.stamp, timeout=Duration(seconds=0.2))
        except Exception:
            try:
                # 2. Fallback to latest available transform (time 0)
                trans = self.tf_buffer.lookup_transform("map", header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.2))
                # self.node.get_logger().debug("PCD transform fell back to latest available TF")
            except Exception as e:
                self.node.get_logger().error(f"PCD TF transform failed (exact and latest): {e}")
                return None
            
        try:
            # Convert to 4x4 matrix
            T = np.eye(4)
            t = trans.transform.translation
            q = trans.transform.rotation
            T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[:3, 3] = [t.x, t.y, t.z]
            
            # Apply to pcd
            return pcd.transform(T)
        except Exception as e:
            self.node.get_logger().error(f"PCD Matrix application failed: {e}")
            return None

    def get_object_3d_position(
        self,
        camera: str,
        mask: SegmentationMask,
        robot_pose: Optional[Tuple[float, float, float]] = None
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get 3D centroid of segmented object.
        """
        frame = self.get_frame(camera)
        if not frame: return None

        cx, cy = mask.centroid
        cam_p = self.pixel_to_3d(camera, cx, cy)
        
        if cam_p:
            # Use TF2 if buffer available and we have a frame
            header = frame.header
                
            map_p = self.transform_to_map(cam_p, header=header)
            if map_p:
                return map_p
                
            # If robot_pose was provided, it implies we want MAP frame.
            # Falling back to local cam_p would cause the robot to navigate to (X_cam, Y_cam) near 0,0.
            if robot_pose is not None:
                self.node.get_logger().warn("TF transform failed for object position. Discarding local coordinates to prevent navigation drift.")
                return None

        return cam_p

    def visualize_detections(
        self,
        camera: str,
        masks: List[SegmentationMask],
        save_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Create visualization of detected objects.
        
        Args:
            camera: Which camera
            masks: List of SegmentationMask objects
            save_path: Optional path to save image
            
        Returns:
            Visualization image
        """
        frame = self.get_frame(camera)
        if frame is None or frame.rgb is None:
            return None
        
        vis_image = visualize_masks(frame.rgb, masks)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            self.node.get_logger().info(f"Saved visualization to {save_path}")
        
        return vis_image
