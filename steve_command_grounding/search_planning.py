#!/usr/bin/env python3
"""
Search Planning - Stretch-Compose style planning for Steve.

Mirrors searchnet_planning logic for:
- Furniture standoff (distance from FOV and furniture dimensions)
- Front normal for body orientation
- No drawers / no manipulation.

Uses scene graph (graph.json, furniture.json) and SpatialPlanner.
"""

import math
import numpy as np
from typing import List, Optional, Tuple

from steve_command_grounding.spatial_planner import SpatialPlanner
from steve_command_grounding.robo_utils.point_clouds import body_planning_front, get_shelf_front_normal, crop_point_cloud


# Camera FOV (degrees)
H_FOV = 42.0
V_FOV = 69.0


def get_distance_to_shelf(
    furniture_centroid: List[float],
    furniture_dimensions: List[float],
    furniture_name: str = "",
) -> float:
    """
    Distance the robot should stand from furniture to capture the whole surface (FOV).
    """
    if not furniture_dimensions or len(furniture_dimensions) < 3:
        return 1.3

    w, d, h = furniture_dimensions[0], furniture_dimensions[1], furniture_dimensions[2]
    
    # Refined math from stretch-compose: padding 0.05 -> 0.1, max 1.5m
    circle_radius_width = (w + 0.1) / (2 * math.tan(math.radians(H_FOV / 2))) + d / 2
    circle_radius_height = (h + 0.1) / (2 * math.tan(math.radians(V_FOV / 2))) + d / 2
    
    if furniture_name and furniture_name.lower() in ["armchair", "couch", "sofa"]:
        circle_radius_width = (w - 0.05) / (2 * math.tan(math.radians(H_FOV / 2)))
        circle_radius_height = (h - 0.05) / (2 * math.tan(math.radians(V_FOV / 2)))

    # Cap distance at 1.2m for small houses, but allow minimum of 0.8m
    final_dist = max(circle_radius_width, circle_radius_height)
    return max(min(final_dist, 1.2), 0.8)


def plan_furniture_search(
    furniture_id: str,
    furniture_label: str,
    centroid: List[float],
    dimensions: List[float],
    robot_pose: Optional[Tuple[float, float]] = None,
    front_normal: Optional[Tuple[float, float, float]] = None,
    pose_matrix: Optional[List] = None,
    standoff_dist: float = 0.8,
    env_pcd: Optional[any] = None,
) -> Tuple[float, float, float, Tuple[float, float, float]]:
    """
    Plan where the robot should stand to view a piece of furniture.
    Integrates O3D-based collision check if env_pcd is provided.
    """
    # Determine distance
    dist = get_distance_to_shelf(centroid, dimensions, furniture_label)
    dist = max(standoff_dist, dist, 0.8)

    # 1. Dynamically derive front normal from PCD if possible (Perceptual approach)
    if env_pcd:
        try:
            furn_pcd = crop_point_cloud(env_pcd, centroid, dimensions)
            if len(furn_pcd.points) > 50:
                nx, ny, nz = get_shelf_front_normal(furn_pcd, furniture_label)
                front_normal = (float(nx), float(ny), float(nz))
                print(f"Dynamic normal for {furniture_label}: {front_normal}")
        except Exception as e:
            print(f"Dynamic normal estimation failed: {e}")

    # Fallback to explicit front normal or pose matrix
    if not front_normal and pose_matrix:
        from steve_command_grounding.search_planning import get_shelf_front_normal_from_pose
        front_normal = get_shelf_front_normal_from_pose(pose_matrix)

    if not front_normal:
        # Fallback logic: face target from robot's current pose
        if robot_pose:
            rx, ry = robot_pose[0], robot_pose[1]
            tx, ty = centroid[0], centroid[1]
            # Vector from target to robot (robot stands on this normal)
            nx, ny = rx - tx, ry - ty
            mag = math.sqrt(nx**2 + ny**2)
            if mag > 0.01:
                front_normal = (nx/mag, ny/mag, 0.0)
                print(f"Calculated approach normal for {furniture_label} from robot pose: {front_normal}")
        
    if not front_normal:
        # Final static fallback
        front_normal = (0.0, 1.0, 0.0)

    # If we have a point cloud, use collision-aware planning
    if env_pcd:
        try:
            print(f"DEBUG: Planning for {furniture_label} with target={centroid}, dist={dist}, normal={front_normal}")
            pose_3d = body_planning_front(
                env_pcd,
                target=np.array(centroid),
                furniture_normal=np.array(front_normal),
                min_target_distance=dist,
                max_target_distance=dist + 0.2,
                min_obstacle_distance=0.4
            )
            gx, gy = pose_3d.coordinates[0], pose_3d.coordinates[1]
            dx, dy = centroid[0] - gx, centroid[1] - gy
            calc_yaw = math.atan2(dy, dx)
            print(f"DEBUG: Goal=({gx:.3f}, {gy:.3f}), Object=({centroid[0]:.3f}, {centroid[1]:.3f}), PlanDist={math.sqrt(dx*dx+dy*dy):.3f}")
            print(f"DEBUG: PoseYaw={pose_3d.get_yaw():.3f}, CalcYaw={calc_yaw:.3f}")
            return (gx, gy, pose_3d.get_yaw(), front_normal)
        except Exception as e:
            print(f"Collision-aware planning failed: {e}. Falling back to geometric.")

    # Geometric fallback (SpatialPlanner)
    planner = SpatialPlanner(standoff_dist=dist)
    x, y, yaw = planner.calculate_standoff_pose(
        centroid,
        target_dimensions=dimensions,
        robot_pose=robot_pose,
        front_normal=front_normal,
    )

    return (x, y, yaw, front_normal)


def plan_object_search(
    object_centroid: List[float],
    front_normal: Tuple[float, float, float],
    env_pcd: Optional[any] = None,
    standoff_dist: float = 0.45,
    furniture_info: Optional[dict] = None,
) -> Tuple[float, float, float]:
    """
    Plan standoff position in front of a detected object.
    Moves closer (e.g. 0.45m) for confirmation or handover.
    Optionally shifts the target to the furniture edge for better stability.
    """
    target_pos = np.array(object_centroid)
    nx, ny, nz = front_normal
    f_normal_np = np.array([nx, ny, nz])

    # 1. Furniture-Edge Shift (from stretch-compose)
    # This prevents 'robot-inside-table' errors by projecting to the surface edge
    if furniture_info:
        f_centroid = furniture_info.get("centroid")
        f_dims = furniture_info.get("dimensions")
        if f_centroid and f_dims:
            f_centroid_np = np.array(f_centroid)
            # Distance of object along the normal relative to furniture center
            obj_dist_along_normal = np.dot(target_pos - f_centroid_np, f_normal_np)
            # Distance of edge along normal (half-depth)
            edge_dist_along_normal = f_dims[1] / 2.0
            
            # Small padding to ensure we are truly in front
            if edge_dist_along_normal < 0.3:
                edge_dist_along_normal += 0.08
                
            # Shift target to the functional edge
            target_pos = target_pos + (edge_dist_along_normal - obj_dist_along_normal) * f_normal_np
    
    # 2. Collision-Aware Planning
    if env_pcd:
        try:
            pose_3d = body_planning_front(
                env_pcd,
                target=np.array(target_pos),
                furniture_normal=f_normal_np,
                min_target_distance=standoff_dist,
                max_target_distance=standoff_dist + 0.1,
                min_obstacle_distance=0.3
            )
            return (pose_3d.coordinates[0], pose_3d.coordinates[1], pose_3d.get_yaw())
        except Exception as e:
            print(f"Object standoff planning failed: {e}")

    # 3. Geometric Fallback
    tx, ty, tz = target_pos
    mag = math.sqrt(nx * nx + ny * ny)
    if mag < 1e-6:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = nx / mag, ny / mag

    robot_x = tx + ux * standoff_dist
    robot_y = ty + uy * standoff_dist
    yaw = math.atan2(-uy, -ux)
    return (robot_x, robot_y, yaw)

