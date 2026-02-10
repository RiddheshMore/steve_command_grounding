# Separate launch commands for Steve Search pipeline

Run these in **separate terminals**. Use the same world/map name (e.g. `steve_house` or `small_house`) everywhere.

---

## 0. One-time setup (each new shell)

```bash
cd /home/ritz/steve_ros2_ws
source install/setup.bash
```

---

## Terminal 1 – Simulation (Gazebo + robot)

```bash
ros2 launch neo_simulation2 simulation.launch.py \
  my_robot:=mmo_700 \
  world:=steve_house \
  arm_type:=ur5e \
  include_pan_tilt:=true \
  include_wrist_camera:=true \
  use_rviz:=false \
  enable_teleop:=false
```

Optional: use `world:=small_house` if you prefer that map.

---

## Terminal 2 – Localization (AMCL + map server)

Set the map to match the world (e.g. `steve_house` or `small_house`). Replace `steve_house` if you used another world.

```bash
# If your workspace is installed (e.g. under /home/ritz/steve_ros2_ws/install)
export NEO_SIM_SHARE=$(ros2 pkg prefix neo_simulation2)/share/neo_simulation2
export NEO_NAV_SHARE=$(ros2 pkg prefix neo_nav2_bringup)/share/neo_nav2_bringup

ros2 launch neo_nav2_bringup localization_amcl.launch.py \
  map:=${NEO_SIM_SHARE}/maps/steve_house.yaml \
  use_sim_time:=true \
  params_file:=${NEO_NAV_SHARE}/config/localization.yaml
```

If `ros2 pkg prefix` is not available, use the full path:

```bash
ros2 launch neo_nav2_bringup localization_amcl.launch.py \
  map:=/home/ritz/steve_ros2_ws/install/neo_simulation2/share/neo_simulation2/maps/steve_house.yaml \
  use_sim_time:=true \
  params_file:=/home/ritz/steve_ros2_ws/install/neo_nav2_bringup/share/neo_nav2_bringup/config/localization.yaml
```

---

## Terminal 3 – Navigation (Nav2)

```bash
export NEO_NAV_SHARE=$(ros2 pkg prefix neo_nav2_bringup)/share/neo_nav2_bringup

ros2 launch neo_nav2_bringup navigation_neo.launch.py \
  use_sim_time:=true \
  params_file:=${NEO_NAV_SHARE}/config/navigation.yaml \
  use_rviz:=True
```

Or with full path:

```bash
ros2 launch neo_nav2_bringup navigation_neo.launch.py \
  use_sim_time:=true \
  params_file:=/home/ritz/steve_ros2_ws/install/neo_nav2_bringup/share/neo_nav2_bringup/config/navigation.yaml \
  use_rviz:=True
```

---

## Terminal 4 – Steve Search node

Start after simulation, localization, and navigation are running (e.g. wait ~15–20 s after Terminal 3).

```bash
ros2 run steve_command_grounding steve_search_node \
  --ros-args \
  -p use_sim_time:=true \
  -p scene_graph_path:=/home/ritz/steve_ros2_ws/maps/generated_graph \
  -p sam3_host:=127.0.0.1 \
  -p sam3_port:=5005
```

---

## Terminal 5 (optional) – Send a command

```bash
ros2 topic pub -1 /command std_msgs/msg/String "data: 'find the water bottle'"
```

---

## Order and timing

1. **Terminal 1** – Start first; wait until Gazebo and the robot are up.
2. **Terminal 2** – Start after Terminal 1; wait until map_server and AMCL are running.
3. **Terminal 3** – Start after Terminal 2; wait until Nav2 is ready (you can use “2D Pose Estimate” in RViz to set initial pose).
4. **Terminal 4** – Start Steve Search node.
5. **Terminal 5** – Publish a command when you want to run a search.

---

## SAM3 Docker

Ensure the SAM3 server is running (e.g. in Docker) and reachable at `sam3_host:sam3_port` (default `127.0.0.1:5005`). If it runs on another host/port, set `sam3_host` and `sam3_port` in Terminal 4.

---

## Sanity check (optional)

Before launching the node, you can run:

```bash
ros2 run steve_command_grounding search_testing --scene_graph /home/ritz/steve_ros2_ws/maps/generated_graph
```

Use `--no_nav` if Nav2 is not running yet.
