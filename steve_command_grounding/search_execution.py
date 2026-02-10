#!/usr/bin/env python3
"""
Search Execution Pipeline - Stretch-Compose style for Steve.

Single pipeline: no manipulation, no drawers.
A) Object in scene graph → navigate to furniture → verify with SAM3.
B) Else → LLM proposals → visit each → verify with SAM3.

Uses: Robot (Nav, Perception, Camera/SAM3), NLPParser, SemanticReasoner,
SearchPlanning, and SAM3 in Docker.
"""
import math
import time
import os
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import List, Optional, Tuple

from steve_command_grounding.robot import Robot
from steve_command_grounding.nlp_parser import NLPParser
from steve_command_grounding.semantic_reasoner import SemanticReasoner
from steve_command_grounding.spatial_planner import SpatialPlanner
from steve_command_grounding.search_planning import plan_furniture_search, plan_object_search
from steve_command_grounding.sam3_interface import SegmentationMask
from steve_command_grounding.robo_utils.point_clouds import pointcloud2_to_o3d, crop_point_cloud, get_shelf_front_normal


@dataclass
class SearchResult:
    """Result of object search (mirrors steve_search_node.SearchResult)."""
    success: bool
    object_name: str
    found_at_furniture: Optional[str] = None
    position_3d: Optional[Tuple[float, float, float]] = None
    mask: Optional[SegmentationMask] = None
    reasoning_path: str = ""


def execute_search(
    robot: Robot,
    parser: NLPParser,
    reasoner: SemanticReasoner,
    planner: SpatialPlanner,
    target_obj: str,
    no_proposals: int = 3,
    logger=None,
) -> SearchResult:
    """
    Run the full search pipeline (stretch-compose logic, no manipulation/drawers).

    Pipeline:
    1. Spatial: if target in scene graph → navigate to each match → SAM3 verify.
    2. Semantic: LLM proposals → navigate to each → SAM3 verify.
    3. Return first success or exhausted result.

    Args:
        robot: Robot (navigation, perception, camera/SAM3).
        parser: NLPParser (used if target_obj came from raw command; can be unused).
        reasoner: SemanticReasoner for LLM/fallback proposals.
        planner: SpatialPlanner for standoff (also used inside planning).
        target_obj: Object name to find (e.g. "water bottle").
        no_proposals: Max LLM proposals to try.
        logger: Optional rclpy logger.

    Returns:
        SearchResult with success, object_name, found_at_furniture, position_3d, reasoning_path.
    """
    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # ---- PCD Capture (Real-time) ----
    log("Capturing live Point Cloud for planning...")
    frame = robot.camera.get_frame("pan_tilt")
    env_pcd = None
    if frame and frame.point_cloud:
        raw_pcd = pointcloud2_to_o3d(frame.point_cloud)
        # Transform PCD to MAP frame for consistent planning using robust TF2
        env_pcd = robot.camera.transform_pcd_to_map(raw_pcd, frame.header)
        
        # Verify if transform succeeded
        if env_pcd and len(env_pcd.points) > 0:
            log(f"Captured and robustly transformed PCD to map frame ({len(env_pcd.points)} points).")
        else:
            log("Warning: Transformed PCD is empty or invalid. Falling back to geometric planning.")
            env_pcd = None
    else:
        log("Warning: Live PCD not available. Falling back to geometric planning.")

    # ---- Stage 0: Initial View Search (Check if already visible) ----
    log("=" * 40)
    log("STAGE 0: Initial View Search")
    log("=" * 40)
    curr_full = robot.navigation.get_current_pose_full()
    found, mask, position_3d = _verify_object_with_sam3(robot, target_obj, logger, robot_pose=curr_full)
    if found:
        log(f"Object '{target_obj}' found immediately in current view!")
        
        return SearchResult(
            success=True,
            object_name=target_obj,
            found_at_furniture="current_view",
            position_3d=position_3d,
            mask=mask,
            reasoning_path="initial_view",
        )

    # ---- Stage 1: Spatial reasoning (object in scene graph) ----
    log("=" * 40)
    log("STAGE 1: Spatial Reasoning")
    log("=" * 40)
    
    matches = robot.perception.find_objects(target_obj)
    if matches:
        log(f"Found {len(matches)} matches in scene graph. Navigating with stall protection...")
        for i, match in enumerate(matches):
            furniture_name = match.get("label", "unknown")
            log(f"Checking spatial match {i+1}/{len(matches)}: {furniture_name}")

            result = _navigate_and_verify(
                robot, planner, target_obj, match, "spatial", logger, env_pcd=env_pcd
            )
            if result.success:
                return result
    else:
        log(f"'{target_obj}' not in scene graph")

    # ---- Stage 2: Semantic reasoning (LLM proposals) ----
    log("=" * 40)
    log("STAGE 2: Semantic Reasoning (LLM)")
    log("=" * 40)

    proposals = reasoner.suggest_locations(
        target_obj,
        robot.perception.get_available_furniture(),
    )
    proposals = _geometric_filter(target_obj, proposals, robot, logger)
    proposals = proposals[:no_proposals]

    if not proposals:
        log(f"No location proposals for '{target_obj}'")
        return SearchResult(
            success=False,
            object_name=target_obj,
            reasoning_path="no_proposals",
        )

    log(f"Checking {len(proposals)} proposals")
    for i, proposal in enumerate(proposals):
        furniture_id = proposal["furniture_id"]
        furniture_name = proposal["furniture_name"]
        log(f"Proposal {i+1}/{len(proposals)}: {furniture_name}")

        target_furniture = robot.perception.get_available_furniture().get(furniture_id)
        if target_furniture:
            result = _navigate_and_verify(
                robot, planner, target_obj, target_furniture, "semantic", logger, env_pcd=env_pcd
            )
            if result.success:
                return result
        else:
            log(f"Furniture {furniture_id} not in scene data")

    log(f"'{target_obj}' not found in any location")
    return SearchResult(
        success=False,
        object_name=target_obj,
        reasoning_path="exhausted",
    )


def _navigate_and_verify(
    robot: Robot,
    planner: SpatialPlanner,
    target_obj: str,
    furniture_data: dict,
    reasoning_path: str,
    logger=None,
    env_pcd=None,
) -> SearchResult:
    """Navigate to furniture and verify with SAM3. No manipulation."""


    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    furniture_name = furniture_data.get("label", "unknown")
    furniture_id = furniture_data.get("id", "")
    centroid = furniture_data.get("centroid")
    dimensions = furniture_data.get("dimensions")

    if not centroid or centroid == [0, 0, 0]:
        log(f"Invalid centroid for {furniture_name}")
        return SearchResult(
            success=False,
            object_name=target_obj,
            found_at_furniture=furniture_name,
            reasoning_path=reasoning_path,
        )


    # Front normal from furniture context or pose
    front_normal = None
    
    # 1. PERCEPTUAL: If we have a PCD, try to get the real surface normal
    if env_pcd:
        try:
            # Crop to furniture area
            furn_pcd = crop_point_cloud(env_pcd, centroid, dimensions)
            if len(furn_pcd.points) > 50:
                nx, ny, nz = get_shelf_front_normal(furn_pcd, furniture_name)
                front_normal = (float(nx), float(ny), float(nz))
                log(f"Extracted surface normal for {furniture_name}: {front_normal}")
        except Exception as e:
            log(f"Perceptual normal extraction failed: {e}")

    # 2. GEOMETRIC FALLBACK: Pose or Scene Graph relative position
    if not front_normal:
        furniture = furniture_data.get("furniture")
        if furniture:
            f_centroid = furniture.get("centroid")
            if f_centroid and centroid:
                nx = centroid[0] - f_centroid[0]
                ny = centroid[1] - f_centroid[1]
                if (nx ** 2 + ny ** 2) > 0.01 ** 2:
                    front_normal = (nx, ny, 0.0)
        
        if not front_normal:
            pose_matrix = furniture_data.get("pose")
            if pose_matrix and len(pose_matrix) >= 3:
                try:
                    # In this scene graph, furniture front is usually along local Y (index [][1])
                    front_normal = (pose_matrix[0][1], pose_matrix[1][1], 0.0)
                except (IndexError, TypeError):
                    pass

    robot_pose = robot.get_current_pose()
    x, y, yaw, front_normal = plan_furniture_search(
        furniture_id,
        furniture_name,
        centroid,
        dimensions or [1.0, 1.0, 1.0],
        robot_pose=robot_pose,
        front_normal=front_normal,
        pose_matrix=furniture_data.get("pose"),
        standoff_dist=planner.standoff_dist,
        env_pcd=env_pcd
    )

    # STANDOFF SAFETY CHECK: If off-map, try flipping or shrinking
    if not robot.perception.is_navigable(x, y):
        log(f"WARNING: Preferred standoff ({x:.2f}, {y:.2f}) is off-map. Attempting Safety Flip...")
        
        # 1. Try flipping the side (180 degrees)
        if front_normal:
            flipped_normal = (-front_normal[0], -front_normal[1], 0.0)
            fx, fy, fyaw, _ = plan_furniture_search(
                furniture_id, furniture_name, centroid, dimensions, 
                robot_pose=robot_pose, front_normal=flipped_normal, standoff_dist=planner.standoff_dist
            )
            if robot.perception.is_navigable(fx, fy):
                log(f"Safety Flip SUCCESS: New standoff at ({fx:.2f}, {fy:.2f})")
                x, y, yaw = fx, fy, fyaw
            else:
                # 2. Try shrinking the distance (0.6m instead of 0.8m+)
                log("Flipped side also off-map. Shrinking standoff...")
                sx, sy, syaw, _ = plan_furniture_search(
                    furniture_id, furniture_name, centroid, dimensions,
                    robot_pose=robot_pose, front_normal=front_normal, standoff_dist=0.5
                )
                x, y, yaw = sx, sy, syaw

    log(f"Navigating to {furniture_name} at ({x:.2f}, {y:.2f})")
    
    # Final pre-check
    if not robot.perception.is_navigable(x, y):
        log(f"CRITICAL WARNING: No navigable standoff found for {furniture_name}. Proceeding with best guess.")

    # 1. NAVIGATE TO GOAL (Non-blocking)
    robot.navigation.go_to_pose(x, y, yaw, wait=False)

    found = False
    mask = None
    position_3d = None
    
    # Stall Detection variables
    start_time = time.time()
    last_stall_check_time = time.time()
    last_dist = 1000.0
    stall_start_time = None
    
    log(f"Monitoring approach to {furniture_name} (Stall recovery: 30s)...")

    # 2. MONITORING LOOP
    while not robot.navigation.is_done():
        curr_dist = robot.navigation.distance_remaining
        curr_time = time.time()
        
        # Check for stall: Movement < 0.05m
        if abs(curr_dist - last_dist) < 0.05:
            if stall_start_time is None:
                stall_start_time = curr_time
            
            stall_duration = curr_time - stall_start_time
            if stall_duration >= 15.0:
                log(f"STALL DETECTED: Robot stationary for {stall_duration:.1f}s. Triggering early verification.")
                curr_full = robot.navigation.get_current_pose_full()
                found, mask, position_3d = _verify_object_with_sam3(robot, target_obj, logger, robot_pose=curr_full)
                if found:
                    log(f"EARLY SUCCESS: '{target_obj}' detected during stall! Cancelling navigation.")
                    robot.navigation.cancel_current_goal()
                    break
                else:
                    log("Stall verification failed. Continuing to wait or finalize.")
                    # Reset stall timer after a failed verification to avoid hammering SAM3
                    stall_start_time = curr_time 
        else:
            # Robot is moving
            stall_start_time = None
            last_dist = curr_dist
            
        # Timeout safety (2 minutes)
        if (curr_time - start_time) > 120.0:
            log("Navigation timeout (2m) reached during approach.")
            robot.navigation.cancel_current_goal()
            break
            
        time.sleep(2.0)

    # 3. FINAL CHECK IF NOT FOUND DURING STALL
    if not found:
        nav_success = robot.navigation.get_result()
        if nav_success:
            log(f"Reached {furniture_name}, doing final verification...")
            curr_full = robot.navigation.get_current_pose_full()
            found, mask, position_3d = _verify_object_with_sam3(robot, target_obj, logger, robot_pose=curr_full)
        else:
            log(f"Navigation to {furniture_name} failed or cancelled (found={found})")
            if not found: # If we didn't find it during a stall breaker either
                return SearchResult(
                    success=False,
                    object_name=target_obj,
                    found_at_furniture=furniture_name,
                    reasoning_path=reasoning_path,
                )

    if found:
        log(f"SUCCESS: Found '{target_obj}' at {furniture_name}!")
        
        return SearchResult(
            success=True,
            object_name=target_obj,
            found_at_furniture=furniture_name,
            position_3d=position_3d,
            mask=mask,
            reasoning_path=reasoning_path,
        )

    log(f"'{target_obj}' not visible at {furniture_name}")
    return SearchResult(
        success=False,
        object_name=target_obj,
        found_at_furniture=furniture_name,
        reasoning_path=reasoning_path,
    )



def _verify_object_with_sam3(
    robot: Robot, target_obj: str, logger=None, robot_pose=None
) -> Tuple[bool, Optional[SegmentationMask], Optional[Tuple[float, float, float]]]:
    """Use SAM3 to detect target. Returns (found, mask, position_3d in MAP frame)."""
    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    camera = "pan_tilt"
    if not robot.camera.sam3.is_available():
        log("SAM3 not available, skipping visual verification")
        return (True, None, None)

    prompts = [target_obj]
    synonyms = {
        "beer can": ["can", "beer", "drink", "tin"],
        "coke can": ["can", "coke", "drink", "soda"],
        "water bottle": ["bottle", "water"],
        "potted plant": ["plant", "pot", "flower"],
        "tennis ball": ["ball", "tennis"],
    }
    if target_obj.lower() in synonyms:
        prompts.extend(synonyms[target_obj.lower()])
    elif " " in target_obj:
        prompts.extend(target_obj.split(" "))
    masks = []
    for prompt in prompts:
        attempt_masks = robot.camera.find_object(camera, prompt, min_area=100)
        if attempt_masks:
            best_score = max(m.score for m in attempt_masks)
            log(f"Found {len(attempt_masks)} candidate objects for '{prompt}' (best score: {best_score:.2f})")
            for m in attempt_masks:
                m.label = prompt
            masks.extend(attempt_masks)
            if best_score > 0.8:
                break

    if not masks:
        return (False, None, None)

    best_mask = None
    for mask in masks:
        if mask.score > 0.5 and mask.area > 100:
            if best_mask is None or mask.score > best_mask.score:
                best_mask = mask

    if best_mask:
        # Returns MAP frame position if robot_pose provided
        position_3d = robot.camera.get_object_3d_position(camera, best_mask, robot_pose=robot_pose)
        robot.camera.visualize_detections(
            camera, [best_mask],
            save_path=f"/tmp/sam3_detection_{target_obj.replace(' ', '_')}.png",
        )
        return (True, best_mask, position_3d)
    return (False, None, None)


def _geometric_filter(
    target_obj: str,
    proposals: List[dict],
    robot: Robot,
    logger=None,
) -> List[dict]:
    """Filter proposals by object/furniture size (geometric reasoning)."""
    size_map = {
        "beer can": (0.07, 0.07, 0.12),
        "coke can": (0.07, 0.07, 0.12),
        "pot": (0.25, 0.25, 0.15),
        "saucepan": (0.20, 0.20, 0.10),
        "bottle": (0.08, 0.08, 0.25),
        "football": (0.22, 0.22, 0.22),
        "tennis ball": (0.07, 0.07, 0.07),
    }
    obj_size = size_map.get(target_obj.lower(), (0.1, 0.1, 0.1))
    filtered = []
    for p in proposals:
        fid = p["furniture_id"]
        furniture = robot.perception.get_available_furniture().get(fid)
        if not furniture:
            filtered.append(p)
            continue
        f_dims = furniture.get("dimensions", (1.0, 1.0, 1.0))
        if any(f_dim < obj_dim for f_dim, obj_dim in zip(f_dims, obj_size)):
            if logger:
                logger.info(f"Geometrically excluding {p['furniture_name']} for {target_obj}")
            continue
        filtered.append(p)
    return filtered if filtered else proposals
