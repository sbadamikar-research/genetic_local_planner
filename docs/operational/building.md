# Building

## Overview

The planner consists of three packages:
- `plan_ga_planner`: Core library (ROS-agnostic)
- `plan_ga_ros1`: ROS1 Noetic plugin
- `plan_ga_ros2`: ROS2 Humble plugin

All builds happen inside Docker containers.

## ROS1 Build

### Start Container

```bash
cd docker/ros1
./run.sh
```

### Attach to Container

**Option A: Terminal**
```bash
docker exec -it plan_ga_ros1 bash
```

**Option B: VS Code**
```
F1 → "Dev Containers: Attach to Running Container" → plan_ga_ros1
```

### Build

```bash
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
```

### Verify

```bash
source devel/setup.bash

# Check package
rospack find plan_ga_ros1

# Check plugin registration
rospack plugins --attrib=plugin nav_core | grep plan_ga
```

Expected output:
```
plan_ga_ros1 /catkin_ws/src/plan_ga/plan_ga_ros1/plan_ga_plugin.xml
```

### Troubleshooting

**Error: `Could not find ONNX Runtime`**
```bash
# Check if ONNX Runtime is installed
ls /usr/local/lib/libonnxruntime*
```

**Error: `Package 'plan_ga_planner' not found`**
```bash
# Ensure both packages are being built
catkin_make --force-cmake
```

## ROS2 Build

### Start Container

```bash
cd docker/ros2
./run.sh
```

### Attach to Container

```bash
docker exec -it plan_ga_ros2 bash
```

### Build

```bash
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### Verify

```bash
source install/setup.bash

# Check packages
ros2 pkg list | grep plan_ga

# Check plugin registration
ros2 plugin list nav2_core::Controller | grep PlanGA
```

Expected output:
```
plan_ga_ros2
  PlanGAROS2Plugin (nav2_core::Controller)
```

### Troubleshooting

**Error: `Package 'plan_ga_planner' not found`**
```bash
# Clean and rebuild
rm -rf build/ install/ log/
colcon build --symlink-install
```

**Error: `setuptools deprecation warning`**
- Safe to ignore; does not affect functionality

## Clean Builds

### ROS1
```bash
cd /catkin_ws
rm -rf build/ devel/
catkin_make
```

### ROS2
```bash
cd /ros2_ws
rm -rf build/ install/ log/
colcon build --symlink-install
```

## Build Options

### Debug Build (ROS1)
```bash
catkin_make -DCMAKE_BUILD_TYPE=Debug
```

### Debug Build (ROS2)
```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug
```

### Specific Package Only
```bash
# ROS1
catkin_make --pkg plan_ga_ros1

# ROS2
colcon build --packages-select plan_ga_ros2
```

## Next Steps

- **Docker workflow**: See [docker.md](docker.md)
- **Deploy planner**: See [deployment.md](deployment.md)
- **Train model**: See [training.md](training.md)
