# Development Plan - GA-Based ROS Local Planner

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Building](#building)
5. [VS Code Development with Docker](#vs-code-development-with-docker)
6. [Integration with Navigation Stack](#integration-with-navigation-stack)
7. [Running the Planner](#running-the-planner)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Host Machine Requirements

**Software:**
- Miniconda3 (installed)
- Docker (installed)
- Git
- 8GB+ RAM recommended
- 20GB+ free disk space

**Hardware:**
- CPU: Multi-core recommended for parallel GA training
- GPU: Optional, can accelerate NN training

### Docker Requirements

Docker must be installed and the user must have permission to run Docker commands:

```bash
# Verify Docker installation
docker --version

# Test Docker (should run without sudo)
docker run hello-world
```

---

## Installation

### Step 1: Clone or Navigate to Project

```bash
cd /home/ANT.AMAZON.COM/basancht/plan_ga
```

### Step 2: Setup Python Environment (Host)

```bash
# Create conda environment from spec
conda env create -f environment.yml

# Activate environment
conda activate plan_ga

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import onnx; print(f'ONNX: {onnx.__version__}')"
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
```

### Step 3: Build Docker Containers

#### ROS1 Noetic Container

```bash
cd docker/ros1
./build.sh

# This will:
# - Pull ros:noetic-ros-core base image
# - Install build tools and dependencies
# - Install nav-core, costmap-2d, stage-ros
# - Download and install ONNX Runtime 1.16.3
# - Create catkin workspace

# Expected build time: 10-15 minutes (first time)
```

#### ROS2 Humble Container

```bash
cd docker/ros2
./build.sh

# This will:
# - Pull ros:humble-ros-core base image
# - Install build tools and colcon
# - Install nav2-core, nav2-costmap-2d
# - Download and install ONNX Runtime 1.16.3
# - Create ros2 workspace

# Expected build time: 10-15 minutes (first time)
```

#### Verify Docker Images

```bash
docker images | grep plan_ga

# Should show:
# plan_ga_ros1    latest    ...
# plan_ga_ros2    latest    ...
```

---

## Configuration

### Planner Parameters

Edit the configuration files based on your robot specifications:

#### ROS1 Configuration

**File**: `samples/configs/planner_params_ros1.yaml`

```yaml
PlanGAROS1Plugin:
  # Model path (inside container, mounted from host)
  model_path: "/models/planner_policy.onnx"

  # Planning parameters
  num_control_steps: 20
  control_frequency: 10.0  # Hz
  time_horizon: 2.0        # seconds

  # Velocity limits (customize for your robot)
  max_v_x: 1.0     # m/s
  min_v_x: -0.5    # m/s
  max_v_y: 0.5     # m/s (set to 0.0 for differential drive)
  min_v_y: -0.5    # m/s (set to 0.0 for differential drive)
  max_omega: 1.0   # rad/s
  min_omega: -1.0  # rad/s

  # Costmap
  costmap_window_size: 50  # pixels (50x50 grid)

  # Safety
  lethal_cost_threshold: 253
  enable_collision_check: true

  # Goal tolerance
  xy_goal_tolerance: 0.1   # meters
  yaw_goal_tolerance: 0.1  # radians
```

#### ROS2 Configuration

**File**: `samples/configs/planner_params_ros2.yaml`

```yaml
controller_server:
  ros__parameters:
    controller_frequency: 10.0

    PlanGAROS2Plugin:
      plugin: "plan_ga_ros2::PlanGAROS2Plugin"

      # Model path
      model_path: "/models/planner_policy.onnx"

      # Planning parameters
      num_control_steps: 20
      control_frequency: 10.0
      time_horizon: 2.0

      # Velocity limits
      max_v_x: 1.0
      min_v_x: -0.5
      max_v_y: 0.5
      min_v_y: -0.5
      max_omega: 1.0
      min_omega: -1.0

      # Costmap
      costmap_window_size: 50

      # Safety
      lethal_cost_threshold: 253
      enable_collision_check: true
```

### Robot Footprint

Define your robot's footprint polygon in the costmap configuration:

```yaml
# For rectangular robot (0.4m x 0.4m)
footprint: [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]

# For circular robot (0.3m radius)
# Approximate with octagon
footprint: [[0.3, 0.0], [0.212, 0.212], [0.0, 0.3], [-0.212, 0.212],
            [-0.3, 0.0], [-0.212, -0.212], [0.0, -0.3], [0.212, -0.212]]
```

---

## Building

### ROS1 Package

```bash
# Launch ROS1 container
cd docker/ros1
./run.sh

# Inside container:
cd /catkin_ws
source /opt/ros/noetic/setup.bash

# Build packages
catkin_make

# If build succeeds:
source devel/setup.bash
```

### ROS2 Package

```bash
# Launch ROS2 container
cd docker/ros2
./run.sh

# Inside container:
cd /ros2_ws
source /opt/ros/humble/setup.bash

# Build packages
colcon build --symlink-install

# If build succeeds:
source install/setup.bash
```

### Verify Plugin Registration

#### ROS1

```bash
# Inside ROS1 container after building:
rospack plugins --attrib=plugin nav_core

# Should list plan_ga_ros1
```

#### ROS2

```bash
# Inside ROS2 container after building:
ros2 pkg list | grep plan_ga

# Should show:
# plan_ga_ros2
```

---

## VS Code Development with Docker

For a better development experience, you can attach VS Code to the running Docker containers to edit, build, and debug code directly.

### Prerequisites

Install the **Dev Containers** extension in VS Code:
- Extension ID: `ms-vscode-remote.remote-containers`
- Install from Extensions marketplace or command palette

### Method 1: Attach to Running Container (Recommended)

#### Step 1: Start the Container

```bash
# Start ROS1 container (runs in background)
cd docker/ros1
./run.sh

# OR start ROS2 container (runs in background)
cd docker/ros2
./run.sh
```

The container will start in detached mode and keep running in the background.

#### Step 2: Attach VS Code

1. Open VS Code
2. Press `F1` or `Ctrl+Shift+P` to open command palette
3. Type: **"Dev Containers: Attach to Running Container..."**
4. Select `plan_ga_ros1` or `plan_ga_ros2` from the list
5. VS Code will open a new window attached to the container

#### Step 3: Open Workspace in Container

In the attached VS Code window:
1. Click **File → Open Folder**
2. Navigate to:
   - ROS1: `/catkin_ws/src/plan_ga`
   - ROS2: `/ros2_ws/src/plan_ga`
3. Click **OK**

You can now edit files directly in the container!

### Method 2: Use devcontainer.json (Advanced)

Create `.devcontainer/devcontainer.json` in project root:

```json
{
  "name": "ROS1 Plan GA Development",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "ros1",
  "workspaceFolder": "/catkin_ws/src/plan_ga",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-vscode.cpptools",
        "ms-vscode.cmake-tools",
        "twxs.cmake",
        "ms-python.python"
      ],
      "settings": {
        "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
        "files.associations": {
          "*.h": "cpp",
          "*.hpp": "cpp"
        }
      }
    }
  },
  "postCreateCommand": "source /opt/ros/noetic/setup.bash"
}
```

Then use: **"Dev Containers: Reopen in Container"**

### Recommended VS Code Extensions (Install in Container)

Once attached to the container, install these extensions:

```bash
# C/C++ development
- ms-vscode.cpptools (C/C++ IntelliSense)
- ms-vscode.cmake-tools (CMake integration)
- twxs.cmake (CMake syntax)

# ROS development
- ms-iot.vscode-ros (ROS support)

# General
- eamodio.gitlens (Git integration)
```

### Building from VS Code

#### Terminal in VS Code

Open integrated terminal (`Ctrl+` backtick):

```bash
# ROS1 build
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make

# ROS2 build
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

#### Tasks Configuration

Create `.vscode/tasks.json` in workspace:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "catkin_make",
      "type": "shell",
      "command": "cd /catkin_ws && source /opt/ros/noetic/setup.bash && catkin_make",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always"
      },
      "problemMatcher": ["$gcc"]
    }
  ]
}
```

Then use `Ctrl+Shift+B` to build.

### IntelliSense Configuration

Create `.vscode/c_cpp_properties.json`:

```json
{
  "configurations": [
    {
      "name": "ROS1",
      "includePath": [
        "${workspaceFolder}/**",
        "/opt/ros/noetic/include/**",
        "/opt/onnxruntime-linux-x64-1.16.3/include/**",
        "/catkin_ws/devel/include/**"
      ],
      "defines": [],
      "compilerPath": "/usr/bin/g++",
      "cStandard": "c11",
      "cppStandard": "c++17",
      "intelliSenseMode": "linux-gcc-x64"
    }
  ],
  "version": 4
}
```

### Container Management

**Starting Containers:**
```bash
cd docker/ros1  # or docker/ros2
./run.sh
```
- Creates and starts container in background
- If container already exists, just starts it
- Containers persist until explicitly removed

**Attaching to Containers:**
```bash
# Via terminal
docker exec -it plan_ga_ros1 bash
docker exec -it plan_ga_ros2 bash

# Via VS Code
# Use "Dev Containers: Attach to Running Container"
```

**Stopping Containers:**
```bash
cd docker/ros1  # or docker/ros2
./stop.sh
```
- Stops the container but preserves it (including build artifacts)
- Can be restarted later with `./run.sh`

**Removing Containers:**
```bash
cd docker/ros1  # or docker/ros2
./remove.sh
```
- Completely removes the container
- Build artifacts will be lost
- Next `./run.sh` creates a fresh container

**Checking Container Status:**
```bash
# List all containers
docker ps -a | grep plan_ga

# Check if running
docker ps | grep plan_ga
```

### Tips

**Container Persistence:**
- Containers run in background and survive terminal closes
- Source code changes persist (volume mounted)
- Build artifacts persist until container is removed
- VS Code can attach/detach without affecting container

**File Editing:**
- Changes made in VS Code are immediately reflected on host (volume mounted)
- No need to copy files back and forth

**Multiple Windows:**
- You can attach multiple VS Code windows to the same container
- Useful for editing different packages simultaneously

**Git Integration:**
- Git operations work normally in VS Code
- Commits are made as if on the host machine

**Rebuilding:**
- After code changes, rebuild using the integrated terminal
- Use `--symlink-install` (colcon) or regular catkin_make

---

## Integration with Navigation Stack

### ROS1 (move_base)

#### 1. Create move_base Configuration

**File**: `samples/configs/move_base_params.yaml`

```yaml
base_local_planner: "plan_ga_ros1/PlanGAROS1Plugin"

# Global costmap
global_costmap:
  global_frame: map
  robot_base_frame: base_link
  update_frequency: 5.0
  publish_frequency: 2.0
  static_map: true

# Local costmap
local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 10.0
  publish_frequency: 10.0
  static_map: false
  rolling_window: true
  width: 3.0
  height: 3.0
  resolution: 0.05

  # Layers
  plugins:
    - {name: static_layer, type: "costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "costmap_2d::InflationLayer"}

  inflation_layer:
    inflation_radius: 0.5
    cost_scaling_factor: 10.0
```

#### 2. Create Launch File

**File**: `samples/launch/ros1/test_planner.launch`

```xml
<launch>
  <!-- Stage simulator (optional for testing) -->
  <node pkg="stage_ros" type="stageros" name="stageros"
        args="$(find plan_ga_ros1)/worlds/simple_corridor.world"/>

  <!-- Move base -->
  <node pkg="move_base" type="move_base" name="move_base" output="screen">
    <rosparam file="$(find plan_ga_ros1)/config/costmap_common.yaml" command="load" ns="global_costmap"/>
    <rosparam file="$(find plan_ga_ros1)/config/costmap_common.yaml" command="load" ns="local_costmap"/>
    <rosparam file="$(find plan_ga_ros1)/config/planner_params.yaml" command="load"/>

    <param name="base_local_planner" value="plan_ga_ros1/PlanGAROS1Plugin"/>
  </node>

  <!-- Rviz for visualization -->
  <node pkg="rviz" type="rviz" name="rviz"
        args="-d $(find plan_ga_ros1)/rviz/navigation.rviz"/>
</launch>
```

### ROS2 (Nav2)

#### 1. Create Nav2 Configuration

**File**: `samples/configs/nav2_params.yaml`

```yaml
bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: map
    robot_base_frame: base_link

controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 10.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001

    # Use our GA-based controller
    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "plan_ga_ros2::PlanGAROS2Plugin"
      # Plugin parameters from planner_params_ros2.yaml

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 10.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05

      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 10.0
        inflation_radius: 0.5
```

#### 2. Create Launch File

**File**: `samples/launch/ros2/test_planner.launch.py`

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('plan_ga_ros2')

    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[nav2_params]
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_params]
        ),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_params]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_dir, 'rviz', 'navigation.rviz')]
        )
    ])
```

---

## Running the Planner

### Prerequisites

1. **Trained ONNX model must exist**: `models/planner_policy.onnx`
   - If not trained yet, see [Training Plan](training_plan.md)

2. **Robot simulation or hardware running**

### ROS1 Execution

```bash
# Terminal 1: Launch ROS1 container
cd docker/ros1
./run.sh

# Inside container:
source /catkin_ws/devel/setup.bash
roslaunch plan_ga_ros1 test_planner.launch

# Terminal 2: Send navigation goal (on host or in another container)
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '{
  header: {frame_id: "map"},
  pose: {
    position: {x: 2.0, y: 2.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

### ROS2 Execution

```bash
# Terminal 1: Launch ROS2 container
cd docker/ros2
./run.sh

# Inside container:
source /ros2_ws/install/setup.bash
ros2 launch plan_ga_ros2 test_planner.launch.py

# Terminal 2: Send navigation goal
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped '{
  header: {frame_id: "map"},
  pose: {
    position: {x: 2.0, y: 2.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

### Monitoring

#### Topics to Monitor

**ROS1:**
```bash
# Velocity commands
rostopic echo /cmd_vel

# Local plan
rostopic echo /move_base/PlanGAROS1Plugin/local_plan

# Costmap
rostopic echo /move_base/local_costmap/costmap
```

**ROS2:**
```bash
# Velocity commands
ros2 topic echo /cmd_vel

# Local plan
ros2 topic echo /local_plan

# Costmap
ros2 topic echo /local_costmap/costmap
```

#### Performance Metrics

```bash
# ROS1: Check planning frequency
rostopic hz /cmd_vel

# ROS2: Check planning frequency
ros2 topic hz /cmd_vel

# Target: 10-20 Hz
```

---

## Troubleshooting

### Issue: ONNX Model Not Found

**Symptoms:**
```
ERROR: Failed to load ONNX model: /models/planner_policy.onnx
```

**Solution:**
1. Verify model exists on host: `ls models/planner_policy.onnx`
2. Check Docker volume mount in `docker/ros*/run.sh`
3. Inside container, verify: `ls /models/planner_policy.onnx`

### Issue: Plugin Not Found

**Symptoms:**
```
ERROR: Failed to load plugin plan_ga_ros1::PlanGAROS1Plugin
```

**Solution (ROS1):**
```bash
# Rebuild package
cd /catkin_ws
catkin_make clean
catkin_make

# Verify plugin XML
cat src/plan_ga/plan_ga_ros1/plan_ga_plugin.xml

# Re-source
source devel/setup.bash
```

**Solution (ROS2):**
```bash
# Rebuild package
cd /ros2_ws
colcon build --packages-select plan_ga_ros2

# Verify plugin registration
ros2 pkg list | grep plan_ga
```

### Issue: Slow Planning Frequency (<10 Hz)

**Diagnosis:**
```bash
# Profile ONNX inference time
# Add timing logs in onnx_inference.cpp
```

**Solutions:**
1. **Reduce model complexity**: Retrain with smaller network
2. **Enable ONNX optimizations**: Check session options in code
3. **Use GPU inference**: Install ONNX Runtime GPU version
4. **Reduce control steps**: Change `num_control_steps` from 20 to 10

### Issue: Robot Collides with Obstacles

**Diagnosis:**
- Check costmap values: Are obstacles properly inflated?
- Check footprint: Is it accurate?
- Review fitness function weights in training

**Solutions:**
1. **Increase collision penalty**: Retrain with higher `collision` weight
2. **Check costmap parameters**: Verify `inflation_radius` and `cost_scaling_factor`
3. **Adjust safety threshold**: Increase `lethal_cost_threshold` margin

### Issue: Robot Does Not Reach Goal

**Diagnosis:**
- Check goal tolerance settings
- Monitor distance to goal over time
- Review NN training convergence

**Solutions:**
1. **Adjust goal tolerance**: Increase `xy_goal_tolerance` and `yaw_goal_tolerance`
2. **Retrain with goal-focused fitness**: Increase `goal_distance` weight
3. **Check for local minima**: Add exploration noise in training

### Issue: Docker Container Won't Start

**Symptoms:**
```
Error response from daemon: driver failed
```

**Solution:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check disk space
df -h

# Clean up old containers
docker system prune
```

### Issue: Build Fails - ONNX Runtime Not Found

**Symptoms:**
```
CMake Error: Could not find ONNX Runtime
```

**Solution:**
```bash
# Inside container, verify ONNX Runtime installation
ls /opt/onnxruntime-linux-x64-1.16.3

# Check environment variable
echo $ONNXRUNTIME_ROOT

# If missing, rebuild Docker image
exit  # Exit container
cd docker/ros1  # or ros2
./build.sh
```

### Getting Help

1. **Check logs**: Look for ERROR/WARN messages in console output
2. **Enable debug logging**: Set `ROSCONSOLE_MIN_SEVERITY=DEBUG` (ROS1) or adjust logger level (ROS2)
3. **Visualize in RViz**: Check costmap, local plan, and robot footprint
4. **Test with DWA**: Compare behavior with default DWA planner
5. **Profile performance**: Use `ros2 run` with `--ros-args --log-level debug`

---

## Best Practices

1. **Always test in simulation first** (Stage/Gazebo) before deploying to real robot
2. **Monitor planning frequency**: Should consistently achieve 10-20 Hz
3. **Validate ONNX model**: Run `training/utils/export_onnx.py` verification
4. **Version control models**: Tag ONNX models with training parameters
5. **Keep Docker images updated**: Rebuild periodically for security updates
6. **Use rosbags for debugging**: Record failed navigation attempts for analysis

---

## Performance Tuning

### Inference Optimization

1. **ONNX Runtime Configuration**:
   ```cpp
   session_options->SetGraphOptimizationLevel(ORT_ENABLE_ALL);
   session_options->SetIntraOpNumThreads(1);  // Experiment with thread count
   ```

2. **Model Quantization** (if latency critical):
   ```python
   # In training/utils/export_onnx.py
   # Convert to int8 quantization for faster inference
   ```

3. **Reduce Planning Horizon**:
   - Decrease `num_control_steps` from 20 to 10-15
   - Shorter horizon = faster inference

### Memory Optimization

1. **Reuse Tensors**: Preallocate input/output tensors
2. **Reduce Costmap Size**: Use 40x40 instead of 50x50 if acceptable
3. **Limit History**: Don't accumulate trajectory history unnecessarily

---

## Next Steps

After successful deployment:

1. **Collect Performance Data**: Record success rate, planning frequency, path quality
2. **Compare with Baseline**: Run side-by-side comparison with DWA planner
3. **Iterative Improvement**: Retrain with failure cases
4. **Transfer to Real Robot**: Test on physical hardware
5. **Fine-tune Parameters**: Adjust velocity limits, goal tolerance for your robot

For training the model, refer to [Training Plan](training_plan.md).
