import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
  ld = LaunchDescription()

  # 1. Bring up simulation + localization + navigation
  # This uses the localization_simulation launch from steve_navigation
  sim_nav_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
      os.path.join(get_package_share_directory('steve_navigation'), 'launch', 'localization_simulation.launch.py')
    ),
    launch_arguments={
      'world': 'small_house',
      'use_sim_time': 'true'
    }.items()
  )

  # 2. RViz for visualization (optional - can be launched separately if there are library issues)
  # Uncomment if RViz works on your system:
  # rviz_config = os.path.join(
  #   get_package_share_directory('steve_navigation'), 
  #   'rviz', 
  #   'single_robot.rviz'
  # )
  # 
  # rviz_node = Node(
  #   package='rviz2',
  #   executable='rviz2',
  #   name='rviz2',
  #   arguments=['-d', rviz_config],
  #   parameters=[{'use_sim_time': True}],
  #   output='screen'
  # )

  # 3. Command Grounding Node
  grounding_node = Node(
    package='steve_command_grounding',
    executable='command_grounding_node',
    name='command_grounding_node',
    output='screen',
    parameters=[{'use_sim_time': True}]
  )

  ld.add_action(sim_nav_launch)
  # ld.add_action(rviz_node)  # Uncomment if RViz works
  ld.add_action(grounding_node)

  return ld
