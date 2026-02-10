# Steve Command Grounding

Open-vocabulary object search and localization for the Steve robot (MMO-700).

## Overview

This package implements a simplified **Stretch-Compose** workflow:

1. **Command Parsing**: Natural language → target object extraction
2. **Spatial Reasoning**: Check scene graph for known object locations
3. **Semantic Reasoning**: Ask DeepSeek LLM for likely locations
4. **Visual Verification**: SAM3 object detection and segmentation

**Current scope**: Search and localize objects (no manipulation, no drawer opening)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Natural Language Command                                   │
│  "Find the water bottle"                                    │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│  NLP Parser                                                 │
│  → action: "find", object: "water bottle"                   │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  SPATIAL REASONING        │   │  SEMANTIC REASONING       │
│  (Scene Graph Lookup)     │   │  (DeepSeek LLM)           │
│  Fast path: known objects │   │  Slow path: novel objects │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Navigation                                                 │
│  Go to proposed furniture location                          │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│  VISUAL VERIFICATION (SAM3)                                 │
│  Pan-tilt camera → SAM3 → Object detection/segmentation     │
│  Wrist camera → Close-up verification (optional)            │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Result: Object found at [furniture] with 3D position       │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. SAM3 Docker Server

Start the SAM3 server before launching the search node:

```bash
cd ~/steve_ros2_ws/src/steve_perception/docker/sam3
chmod +x run_sam3.sh
./run_sam3.sh

# Verify it's running
curl http://localhost:5005/health
```

### 2. Scene Graph

Generate or provide a scene graph in `/home/ritz/steve_ros2_ws/generated_graph/`:
- `graph.json` - Scene graph with furniture and objects
- `furniture.json` - Furniture metadata

### 3. Build the Package

```bash
cd ~/steve_ros2_ws
colcon build --packages-select steve_command_grounding
source install/setup.bash
```

## Usage

### Quick Start (Simulation)

**Terminal 1**: Start SAM3 server
```bash
cd ~/steve_ros2_ws/src/steve_perception/docker/sam3
docker-compose up
```

**Terminal 2**: Launch simulation + search node
```bash
source ~/steve_ros2_ws/install/setup.bash
ros2 launch steve_command_grounding steve_search.launch.py
```

**Terminal 3**: Send search commands
```bash
# Find an object
ros2 topic pub /command std_msgs/String "data: 'find the water bottle'" -1

# Go to a location
ros2 topic pub /command std_msgs/String "data: 'go to the kitchen table'" -1

# Listen for results
ros2 topic echo /search_result
```

### Manual Launch

```bash
# 1. Start simulation with cameras
ros2 launch steve_simulation simulation.launch.py \
    my_robot:=mmo_700 \
    world:=steve_house \
    arm_type:=ur5e \
    include_pan_tilt:=true \
    include_wrist_camera:=true

# 2. Start navigation
ros2 launch steve_navigation navigation.launch.py use_sim_time:=true

# 3. Start search node
ros2 run steve_command_grounding steve_search_node
```

## Camera Topics

| Camera | RGB Topic | Depth Topic |
|--------|-----------|-------------|
| Pan-Tilt (L515) | `/pan_tilt_camera/color/image_raw` | `/pan_tilt_camera/aligned_depth_to_color/image_raw` |
| Wrist (D405) | `/wrist_camera/color/image_raw` | `/wrist_camera/aligned_depth_to_color/image_raw` |

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/command` | `std_msgs/String` | Input: natural language commands |
| `/search_result` | `std_msgs/String` | Output: search results |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scene_graph_path` | `steve_ros2_ws/generated_graph` | Path to scene graph |
| `sam3_host` | `127.0.0.1` | SAM3 server hostname |
| `sam3_port` | `5005` | SAM3 server port |


## SAM3 API

The SAM3 Docker server provides:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /segment_point` | Segment at point coordinates |
| `POST /segment_box` | Segment within bounding box |
| `POST /detect_all` | Automatic mask generation |
| `POST /segment_with_detection` | Refine YOLO detections with SAM |


## Contributors & Acknowledgments

This project is developed at the **Humanoid Robots Lab (HRL)**, University of Bonn.

- **Riddhesh More** - [GitHub](https://github.com/RiddheshMore) | [Email](mailto:riddheshmore311@gmail.com)

### Acknowledgements
- **Rohit Menon** - For mentorship and technical guidance on Neobotix platforms.
- **Prof. Maren Bennewitz** - Head of the Humanoid Robots Lab, University of Bonn.

---

**License**: All custom code is released under the MIT License. Third-party packages retain their original licenses.