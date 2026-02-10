#!/usr/bin/env python3
"""
Steve Search Pipeline Launch File (Stretch-Compose style)

Launches full simulation pipeline for Gazebo + Nav2 + Steve Search:
1. steve_simulation: Gazebo + mmo_700 (pan-tilt + wrist cameras)
2. steve_navigation: localization_simulation (AMCL + Nav2)
3. Steve Search Node (SAM3 verification, no manipulation/drawers)

Usage:
    ros2 launch steve_command_grounding steve_search.launch.py
    ros2 launch steve_command_grounding steve_search.launch.py world:=small_house

Then:
    ros2 topic pub /command std_msgs/String "data: 'find the water bottle'"
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="steve_house",
        description="Gazebo world: steve_house, small_house, neo_workshop, neo_track1",
    )
    map_arg = DeclareLaunchArgument(
        "map",
        default_value="",
        description="Full path to map yaml (optional; auto-detected from world if empty)",
    )
    sam3_host_arg = DeclareLaunchArgument(
        "sam3_host",
        default_value="127.0.0.1",
        description="SAM3 Docker server hostname",
    )
    sam3_port_arg = DeclareLaunchArgument(
        "sam3_port",
        default_value="5005",
        description="SAM3 Docker server port",
    )
    scene_graph_arg = DeclareLaunchArgument(
        "scene_graph_path",
        default_value="/home/ritz/steve_ros2_ws/maps/generated_graph",
        description="Path to scene graph (graph.json, furniture.json)",
    )

    ld.add_action(world_arg)
    ld.add_action(map_arg)
    ld.add_action(sam3_host_arg)
    ld.add_action(sam3_port_arg)
    ld.add_action(scene_graph_arg)

    # Full stack: simulation + localization (AMCL) + navigation (steve_navigation)
    try:
        nav_bringup_dir = get_package_share_directory("steve_navigation")
        full_stack = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav_bringup_dir, "launch", "localization_simulation.launch.py")
            ),
            launch_arguments={
                "world": LaunchConfiguration("world"),
                "map": LaunchConfiguration("map"),
            }.items(),
        )
        ld.add_action(full_stack)
    except Exception as e:
        print(f"Warning: Could not find steve_navigation: {e}")

    # Steve Search Node (delayed so Nav2 and AMCL are up)
    search_node = Node(
        package="steve_command_grounding",
        executable="steve_search_node",
        name="steve_search_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "sam3_host": LaunchConfiguration("sam3_host"),
            "sam3_port": LaunchConfiguration("sam3_port"),
            "scene_graph_path": LaunchConfiguration("scene_graph_path"),
        }],
    )
    delayed_search = TimerAction(period=25.0, actions=[search_node])
    ld.add_action(delayed_search)

    return ld
