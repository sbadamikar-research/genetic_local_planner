# Deployment

## Prerequisites

- ✅ Built ROS plugin (see [building.md](building.md))
- ✅ Trained ONNX model at `models/planner_policy.onnx` (see [training.md](training.md))
- ✅ Running ROS container

## ROS1 Integration (move_base)

### 1. Copy Configuration

```bash
# On host
cp samples/configs/planner_params_ros1.yaml /path/to/your/robot/config/
```

### 2. Update move_base Configuration

Edit your move_base launch file or create a new one:

```xml
<!-- my_navigation.launch -->
<launch>
  <!-- Load planner parameters -->
  <rosparam file="$(find your_robot)/config/planner_params_ros1.yaml" command="load" />

  <!-- move_base node -->
  <node pkg="move_base" type="move_base" name="move_base" output="screen">
    <!-- Use plan_ga as local planner -->
    <param name="base_local_planner" value="plan_ga_ros1/PlanGAROS1Plugin"/>

    <!-- Other move_base parameters -->
    <rosparam file="$(find your_robot)/config/costmap_common.yaml" command="load" ns="global_costmap"/>
    <rosparam file="$(find your_robot)/config/costmap_common.yaml" command="load" ns="local_costmap"/>
    <rosparam file="$(find your_robot)/config/global_costmap.yaml" command="load"/>
    <rosparam file="$(find your_robot)/config/local_costmap.yaml" command="load"/>
  </node>
</launch>
```

### 3. Launch Navigation

Inside the ROS1 container:

```bash
source /catkin_ws/devel/setup.bash
roslaunch your_robot my_navigation.launch
```

### 4. Send Navigation Goals

```bash
# Using rviz
rosrun rviz rviz

# Or programmatically
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped ...
```

## ROS2 Integration (Nav2)

### 1. Copy Configuration

```bash
# On host
cp samples/configs/planner_params_ros2.yaml /path/to/your/robot/config/
```

### 2. Update Nav2 Configuration

Edit your Nav2 parameter file (e.g., `nav2_params.yaml`):

```yaml
controller_server:
  ros__parameters:
    controller_frequency: 10.0

    # Use plan_ga controller
    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "plan_ga_ros2::PlanGAROS2Plugin"
      # Parameters from planner_params_ros2.yaml will be loaded
```

### 3. Load Parameters

Update your launch file to load plan_ga parameters:

```python
# navigation_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('your_robot')

    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    planner_params = os.path.join(pkg_share, 'config', 'planner_params_ros2.yaml')

    return LaunchDescription([
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            parameters=[nav2_params, planner_params],
            output='screen'
        ),
        # Other Nav2 nodes...
    ])
```

### 4. Launch Navigation

Inside the ROS2 container:

```bash
source /ros2_ws/install/setup.bash
ros2 launch your_robot navigation_launch.py
```

### 5. Send Navigation Goals

```bash
# Using rviz2
ros2 run rviz2 rviz2

# Or using Nav2 CLI
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose ...
```

## Verification

### Check Plugin Loading

**ROS1**:
```bash
rosrun rqt_console rqt_console
# Look for: "Using plugin plan_ga_ros1/PlanGAROS1Plugin"
```

**ROS2**:
```bash
ros2 topic echo /controller_server/transition_event
# Check for successful activation
```

### Monitor Planner Output

**ROS1**:
```bash
# View local plan
rostopic echo /local_plan

# Monitor planner status
rostopic echo /move_base/status
```

**ROS2**:
```bash
# View local plan
ros2 topic echo /local_plan

# Monitor controller status
ros2 topic echo /controller_server/transition_event
```

### Visualize in RViz

Add these displays:
- **LocalCostmap**: Subscribed to `/move_base/local_costmap/costmap` (ROS1) or `/local_costmap/costmap` (ROS2)
- **Local Plan**: Path display on `/local_plan`
- **Robot Footprint**: Polygon display
- **Goal**: PoseStamped display

## Testing

### Simple Navigation Test

**ROS1**:
```bash
# Terminal 1: Launch navigation
roslaunch your_robot my_navigation.launch

# Terminal 2: Send goal
rosrun rviz rviz
# Use "2D Nav Goal" tool
```

**ROS2**:
```bash
# Terminal 1: Launch navigation
ros2 launch your_robot navigation_launch.py

# Terminal 2: Send goal
ros2 run rviz2 rviz2
# Use "Nav2 Goal" tool
```

### Stage Simulator Integration

For testing without a physical robot:

**ROS1**:
```bash
# Install Stage
sudo apt-get install ros-noetic-stage-ros

# Launch with Stage
roslaunch stage_ros willow_garage.launch
roslaunch your_robot my_navigation.launch
```

**ROS2**:
```bash
# Install Stage (if available for ROS2)
# Or use Gazebo instead

# Launch simulator
ros2 launch your_robot simulation.launch.py
ros2 launch your_robot navigation_launch.py
```

## Common Integration Issues

### Plugin Not Found

**Error**: `Failed to load plugin plan_ga_ros1/PlanGAROS1Plugin`

**Solution**:
```bash
# Verify plugin registration
rospack plugins --attrib=plugin nav_core | grep plan_ga  # ROS1
ros2 plugin list nav2_core::Controller | grep PlanGA     # ROS2

# Re-source workspace
source /catkin_ws/devel/setup.bash  # ROS1
source /ros2_ws/install/setup.bash  # ROS2
```

### Model File Not Found

**Error**: `Could not load ONNX model: /models/planner_policy.onnx`

**Solution**:
```bash
# Check file exists
ls -lh /models/planner_policy.onnx

# Verify path in config
grep model_path samples/configs/planner_params_ros*.yaml

# Update path if needed
# Edit planner_params_ros*.yaml: model_path: "/models/planner_policy.onnx"
```

### Planner Not Computing Path

**Symptoms**: Robot doesn't move, no local plan published

**Debug**:
```bash
# ROS1
rosnode info /move_base
rostopic hz /local_plan

# ROS2
ros2 node info /controller_server
ros2 topic hz /local_plan

# Check logs for errors
# ROS1: rqt_console
# ROS2: ros2 topic echo /rosout
```

**Common Causes**:
- ONNX model not loaded
- Costmap not publishing
- Goal tolerance too strict
- Invalid velocity limits

## Performance Monitoring

### Check Planning Frequency

**ROS1**:
```bash
rostopic hz /local_plan
# Should be ~10 Hz (configurable)
```

**ROS2**:
```bash
ros2 topic hz /local_plan
# Should be ~10 Hz (configurable)
```

### Monitor CPU Usage

```bash
# Inside container
top -p $(pgrep move_base)  # ROS1
top -p $(pgrep controller_server)  # ROS2
```

Expected CPU usage: 10-30% per planning cycle

### Check Inference Time

Enable debug mode in config:

```yaml
PlanGAROS1Plugin:  # or PlanGAROS2Plugin
  debug_mode: true
```

Look for log messages:
```
[INFO] ONNX inference time: 15.3 ms
[INFO] Total planning time: 23.7 ms
```

Target: <50 ms for 10 Hz operation

## Next Steps

- **Tune parameters**: See [configuration.md](configuration.md)
- **Troubleshooting**: See [troubleshooting.md](troubleshooting.md)
- **Performance optimization**: Adjust planning frequency and control steps
