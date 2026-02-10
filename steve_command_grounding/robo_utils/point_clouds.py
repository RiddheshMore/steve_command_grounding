from __future__ import annotations

import copy
import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Optional

try:
    import open3d as o3d
except ImportError:
    o3d = None

from .coordinates import Pose3D, get_circle_points, pose_distanced
from .importer import PointCloud
from .time import convert_time

# Note: Many of these functions depend on Open3D.
# Ported from stretch-compose with relative imports.

def pointcloud2_to_o3d(ros_pc2):
    """
    Converts a sensor_msgs.msg.PointCloud2 to an open3d.geometry.PointCloud.
    """
    if o3d is None: return None
    
    # Extract points (x, y, z)
    fmt = ros_pc2.point_step
    data = np.frombuffer(ros_pc2.data, dtype=np.uint8)
    
    # Reshape to (H*W, point_step)
    data = data.reshape(-1, fmt)
    
    # Assuming standard float32 x,y,z at start of each step
    points = data[:, :12].view(np.float32).reshape(-1, 3)
    
    # Filter out NaNs if any
    points = points[~np.isnan(points).any(axis=1)]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def add_coordinate_system(
    cloud: PointCloud, color: tuple[int, int, int], ground_coordinate: np.ndarray = None,
    e1: np.ndarray = None, e2: np.ndarray = None, e3: np.ndarray = None, e_relative_to_ground: bool = True, size: int = 1
) -> PointCloud:
    if o3d is None: return cloud
    nrx, nry, nrz = 40 * size, 20 * size, 5 * size
    if ground_coordinate is None: ground_coordinate = np.asarray([0, 0, 0])
    if e1 is None: e1 = np.asarray([1, 0, 0])
    if e2 is None: e2 = np.asarray([0, 1, 0])
    if e3 is None: e3 = np.asarray([0, 0, 1])

    if not e_relative_to_ground:
        e1 = e1 - ground_coordinate
        e2 = e2 - ground_coordinate
        e3 = e3 - ground_coordinate

    e1 = e1 * size
    e2 = e2 * size
    e3 = e3 * size

    x_vector = np.linspace(0, 1, nrx).reshape((nrx, 1))
    full_x_vector = x_vector * np.tile(e1, (nrx, 1))
    y_vector = np.linspace(0, 1, nry).reshape((nry, 1))
    full_y_vector = y_vector * np.tile(e2, (nry, 1))
    z_vector = np.linspace(0, 1, nrz).reshape((nrz, 1))
    full_z_vector = z_vector * np.tile(e3, (nrz, 1))

    full_vector = np.vstack([full_x_vector, full_y_vector, full_z_vector])
    ground_coordinate = np.tile(ground_coordinate, (full_vector.shape[0], 1))
    full_vector = full_vector + ground_coordinate

    color_vector = np.asarray(color)
    color_vector = np.tile(color_vector, (full_vector.shape[0], 1))
    
    points = np.asarray(cloud.points)
    new_points = np.vstack([points, full_vector])
    cloud.points = o3d.utility.Vector3dVector(new_points)

    colors = np.asarray(cloud.colors)
    new_colors = np.vstack([colors, color_vector])
    cloud.colors = o3d.utility.Vector3dVector(new_colors)
    return cloud


def body_planning(
    env_cloud: PointCloud,
    target_pose: Pose3D,
    resolution: int = 16,
    nr_circles: int = 3,
    floor_height_thresh: float = -0.1,
    body_height: float = 1.20,
    min_distance: float = 0.75,
    max_distance: float = 1,
    lambda_distance: float = 0.5,
    n_best: int = 1,
    vis_block: bool = False,
) -> list[tuple[Pose3D, float]]:
    if o3d is None: return []
    target = target_pose.as_ndarray()
    
    points = np.asarray(env_cloud.points)
    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)
    points_bool = points[:, 2] > floor_height_thresh
    index = np.where(points_bool)[0]
    pc_no_ground = env_cloud.select_by_index(index)

    circle_points = get_circle_points(
        resolution=resolution,
        nr_circles=nr_circles,
        start_radius=min_distance,
        end_radius=max_distance,
        return_cartesian=True,
    )
    target_at_body_height = target.copy()
    target_at_body_height[-1] = body_height
    target_at_body_height = target_at_body_height.reshape((1, 1, 3))
    circle_points = circle_points + target_at_body_height
    
    circle_points_bool = (min_points+0.2 <= circle_points) & (circle_points <= max_points-0.2)
    circle_points_bool = np.all(circle_points_bool, axis=2)
    filtered_circle_points = circle_points[circle_points_bool].reshape((-1, 3))

    if len(filtered_circle_points) == 0:
        return []

    # transform point cloud to mesh to calculate SDF from
    try:
        ball_sizes = (0.02, 0.011, 0.005)
        ball_sizes = o3d.utility.DoubleVector(ball_sizes)
        mesh_no_ground = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pc_no_ground, radii=ball_sizes)
        mesh_no_ground_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_no_ground)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_no_ground_legacy)
    except Exception as e:
        print(f"Mesh generation / Raycasting scene creation failed: {e}")
        return []

    ray_directions = filtered_circle_points - target
    rays_starts = np.tile(target, (ray_directions.shape[0], 1))
    rays = np.concatenate([rays_starts, ray_directions], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    response = scene.cast_rays(rays_tensor)
    direct_connection_bool = response["t_hit"].numpy() > 1.0 # No collision between target and point
    filtered_circle_points = filtered_circle_points[direct_connection_bool]
    
    if len(filtered_circle_points) == 0:
        return []

    circle_tensors = o3d.core.Tensor(filtered_circle_points, dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_signed_distance(circle_tensors).numpy()
    
    target_distances = filtered_circle_points - target.reshape((1, 3))
    target_distances = np.linalg.norm(target_distances, ord=2, axis=-1)
    
    scores = distances - lambda_distance * target_distances
    flat_indices = np.argsort(-scores.flatten())
    top_n_indices = flat_indices[:n_best]
    top_n_coordinates = filtered_circle_points[top_n_indices]
    top_n_scores = scores[top_n_indices]

    poses = []
    for score, coord in zip(top_n_scores, top_n_coordinates):
        pose = Pose3D(coord)
        pose.set_rot_from_direction(target - coord)
        poses.append((pose, score))
    return poses


def body_planning_front(
    env_cloud: PointCloud,
    target: np.ndarray,
    furniture_normal: np.ndarray,
    floor_height_thresh: float = 0,
    body_height: float = 0.45,
    min_target_distance: float = 0.75,
    max_target_distance: float = 1,
    min_obstacle_distance: float = 0.5,
    n: int = 4,
    vis_block: bool = False
) -> Pose3D:
    if o3d is None: return Pose3D(target + furniture_normal * min_target_distance)
    start_time = time.time_ns()
    
    points = np.asarray(env_cloud.points)
    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)
    points_bool = points[:, 2] > floor_height_thresh
    index = np.where(points_bool)[0]
    pc_no_ground = env_cloud.select_by_index(index)

    circle_points = get_circle_points(
        resolution=32,
        nr_circles=2,
        start_radius=min_target_distance,
        end_radius=max_target_distance,
        return_cartesian=True,
    )
    target_at_body_height = target.copy()
    target_at_body_height[-1] = body_height
    target_at_body_height = target_at_body_height.reshape((1, 1, 3))
    circle_points = circle_points + target_at_body_height
    
    print(f"DEBUG: Circle points generated: start_radius={min_target_distance}, end_radius={max_target_distance}")
    circle_points_bool = (min_points <= circle_points) & (circle_points <= max_points)
    circle_points_bool = np.all(circle_points_bool, axis=2)
    filtered_circle_points = circle_points[circle_points_bool].reshape((-1, 3))
    print(f"DEBUG: Filtered circle points: {len(filtered_circle_points)} / {circle_points.size//3} (Bounds: {min_points[:2]} to {max_points[:2]})")

    if len(filtered_circle_points) == 0:
         print("DEBUG: No circle points in bounds! Falling back to geometric point.")
         return Pose3D(target + furniture_normal * min_target_distance)

    # transform point cloud to mesh to calculate SDF from
    try:
        pc_no_ground_down = pc_no_ground.voxel_down_sample(voxel_size=0.01)
        pc_no_ground_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        mesh_no_ground, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc_no_ground_down, depth=6)
        mesh_no_ground_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_no_ground)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_no_ground_legacy)
    except Exception as e:
        print(f"Collision mesh generation failed: {e}")
        return Pose3D(target + furniture_normal * min_target_distance)

    ray_directions = filtered_circle_points - target
    rays_starts = np.tile(target, (ray_directions.shape[0], 1))
    rays = np.concatenate([rays_starts, ray_directions], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    response = scene.cast_rays(rays_tensor)
    direct_connection_bool = response["t_hit"].numpy() > 1.0
    filtered_circle_points = filtered_circle_points[direct_connection_bool]
    
    if len(filtered_circle_points) == 0:
        return Pose3D(target + furniture_normal * min_target_distance)

    circle_tensors = o3d.core.Tensor(filtered_circle_points, dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_signed_distance(circle_tensors).numpy()
    
    valid_mask = np.abs(distances) > min_obstacle_distance
    if not np.any(valid_mask):
        # Fallback to point with max distance from obstacles if none perfectly valid
        valid_points_num = filtered_circle_points
    else:
        valid_points_num = filtered_circle_points[valid_mask]

    directions = valid_points_num - target
    directions_2d = directions[:, :2]
    furniture_normal_2d = furniture_normal[:2]
    furniture_normal_norm = furniture_normal_2d / (np.linalg.norm(furniture_normal_2d) + 1e-6)
    cosine_angles = np.dot(directions_2d, furniture_normal_norm)
    most_aligned_point_index = np.argmax(cosine_angles)
    selected_coordinates = valid_points_num[most_aligned_point_index]

    pose = Pose3D(selected_coordinates)
    # Important: Face target in XY only for base navigation
    face_vector = target - selected_coordinates
    face_vector[2] = 0 # Ignore Z for orientation
    pose.set_rot_from_direction(face_vector)
    
    end_time = time.time_ns()
    minutes, seconds = convert_time(end_time - start_time)
    print(f"DEBUG: Body planning selected goal={selected_coordinates[:2]}, dist={np.linalg.norm(target[:2]-selected_coordinates[:2]):.3f}")
    print(f"Body planning RUNTIME: {minutes}min {seconds}s")
    
    return pose


def crop_point_cloud(pcd: PointCloud, centroid: list[float], dimensions: list[float], padding: float = 0.2) -> PointCloud:
    if o3d is None: return pcd
    min_bound = np.array(centroid) - (np.array(dimensions) / 2.0 + padding)
    max_bound = np.array(centroid) + (np.array(dimensions) / 2.0 + padding)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(bbox)


def get_shelf_front_normal(furniture_pcd: PointCloud, furniture_name: str = "") -> np.ndarray:
    if o3d is None: return np.array([0, 1, 0])
    
    try:
        obb = furniture_pcd.get_minimal_oriented_bounding_box()
        rotation = obb.R
        extents = obb.extent
        center = obb.center
        
        z_alignment = np.abs(rotation.T @ np.array([0, 0, 1]))
        vertical_axis = np.argsort(z_alignment)[-2] # Second to last (verticality ranking)
        
        R_new = np.zeros((3, 3))
        vertical_direction = np.array([0, 0, np.sign(rotation[2, vertical_axis])])
        R_new[:, vertical_axis] = vertical_direction
        
        other_axes = [i for i in range(3) if i != vertical_axis]
        horizontal_components = [np.linalg.norm(rotation[:2, i]) for i in other_axes]
        front_axis = other_axes[np.argmax(horizontal_components)]
        
        horizontal_dir = rotation[:2, front_axis] / np.linalg.norm(rotation[:2, front_axis])
        R_new[:2, front_axis] = horizontal_dir
        R_new[2, front_axis] = 0
        
        third_axis = [i for i in range(3) if i != vertical_axis and i != front_axis][0]
        R_new[:, third_axis] = np.cross(R_new[:, front_axis], R_new[:, vertical_axis])
        
        points = np.asarray(furniture_pcd.points)
        points_centered = points - center
        points_rotated = points_centered @ R_new
        min_vals = np.min(points_rotated, axis=0)
        max_vals = np.max(points_rotated, axis=0)
        
        extents_new = max_vals - min_vals
        center_new = center + R_new @ ((min_vals + max_vals) / 2)
        
        vertical_faces = []
        for axis in range(3):
            for direction in [1, -1]:
                if furniture_name and furniture_name.lower() in ["armchair", "couch", "sofa"]:
                    normal = R_new[:, axis] * direction
                    if np.linalg.norm(center_new) < np.linalg.norm(center_new + normal):
                        normal = -normal
                else:
                    normal = rotation[:, axis] * direction
                    if np.linalg.norm(center) < np.linalg.norm(center + normal):
                        normal = -normal
                
                if abs(normal[2]) < 0.1: # Horizontal-ish normal = vertical face
                    dim1, dim2 = (axis + 1) % 3, (axis + 2) % 3
                    if furniture_name and furniture_name.lower() in ["armchair", "couch", "sofa"]:
                        area = extents_new[dim1] * extents_new[dim2]
                    else:
                        area = extents[dim1] * extents[dim2]
                    vertical_faces.append({'normal': normal, 'area': area})
        
        if not vertical_faces:
             return np.array([0, 1, 0])
        
        if furniture_name and furniture_name.lower() in ["armchair", "couch", "sofa"]:
            front = min(vertical_faces, key=lambda x: x['area'])
        else:
            front = max(vertical_faces, key=lambda x: x['area'])
            
        return front['normal']
    except Exception as e:
        print(f"Normal estimation failed: {e}")
        return np.array([0, 1, 0])


def icp(
    pcd1: PointCloud, pcd2: PointCloud, threshold: float = 0.2, trans_init: np.ndarray | None = None, 
    max_iteration: int = 1000, point_to_point: bool = False,
) -> np.ndarray:
    if o3d is None: return np.eye(4)
    if trans_init is None: trans_init = np.eye(4)
    if point_to_point:
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold, init=trans_init,
        estimation_method=method, criteria=criteria,
    )
    return reg_p2p.transformation
