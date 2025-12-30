# Troubleshooting

## Build Issues

### ONNX Runtime Not Found

**Symptoms**:
```
CMake Error: Could not find ONNX Runtime
```

**Solution**:
```bash
# Check installation
ls /usr/local/lib/libonnxruntime*

# If missing, rebuild Docker image
cd docker/ros1  # or ros2
./remove.sh
./build.sh
```

### Package Not Found

**Symptoms**:
```
Could not find package 'plan_ga_planner'
```

**Solution**:
```bash
# Ensure correct workspace
cd /catkin_ws  # or /ros2_ws

# Clean build
rm -rf build/ devel/  # ROS1
rm -rf build/ install/ log/  # ROS2

# Rebuild
catkin_make  # ROS1
colcon build --symlink-install  # ROS2
```

### Include Errors

**Symptoms**:
```
fatal error: plan_ga_planner/types.h: No such file or directory
```

**Solution**:
```bash
# Verify directory structure
ls src/plan_ga/plan_ga_planner/include/plan_ga_planner/

# Check CMakeLists.txt include directories
grep include_directories src/plan_ga/plan_ga_ros1/CMakeLists.txt

# Force reconfigure
catkin_make --force-cmake  # ROS1
colcon build --cmake-force-configure  # ROS2
```

## Runtime Issues

### Plugin Not Loading

**Symptoms**:
```
Failed to load plugin plan_ga_ros1/PlanGAROS1Plugin
```

**Diagnosis**:
```bash
# Check plugin registration
rospack plugins --attrib=plugin nav_core | grep plan_ga  # ROS1
ros2 plugin list nav2_core::Controller | grep PlanGA     # ROS2

# Verify library exists
ls /catkin_ws/devel/lib/libplan_ga_ros1_plugin.so  # ROS1
ls /ros2_ws/install/plan_ga_ros2/lib/libplan_ga_ros2_plugin.so  # ROS2
```

**Solution**:
```bash
# Re-source workspace
source /catkin_ws/devel/setup.bash  # ROS1
source /ros2_ws/install/setup.bash  # ROS2

# Verify plugin XML
cat src/plan_ga/plan_ga_ros1/plan_ga_plugin.xml

# Rebuild if needed
catkin_make  # ROS1
colcon build --symlink-install  # ROS2
```

### ONNX Model Not Found

**Symptoms**:
```
Could not load ONNX model: /models/planner_policy.onnx
```

**Diagnosis**:
```bash
# Check file exists
ls -lh /models/planner_policy.onnx

# Verify mount
docker inspect plan_ga_ros1 | grep -A5 Mounts
```

**Solution**:
```bash
# Train model if missing (on host)
conda activate plan_ga
python training/train_nn.py \
  --data models/checkpoints/ga_trajectories.pkl \
  --config training/config/nn_config.yaml \
  --output models/planner_policy.onnx

# Verify from container
docker exec plan_ga_ros1 ls -lh /models/planner_policy.onnx

# Update config if path is wrong
grep model_path samples/configs/planner_params_ros*.yaml
```

### Planner Not Computing Paths

**Symptoms**: Robot doesn't move, no local plan published

**Diagnosis**:
```bash
# Check if planner is being called
# ROS1
rostopic hz /move_base/goal
rostopic hz /local_plan

# ROS2
ros2 topic hz /goal_pose
ros2 topic hz /local_plan

# Check for errors
# ROS1: rosrun rqt_console rqt_console
# ROS2: ros2 topic echo /rosout
```

**Common Causes & Solutions**:

1. **Goal not received**:
   ```bash
   # Verify goal topic
   rostopic list | grep goal  # ROS1
   ros2 topic list | grep goal  # ROS2
   ```

2. **Costmap not publishing**:
   ```bash
   rostopic hz /move_base/local_costmap/costmap  # ROS1
   ros2 topic hz /local_costmap/costmap  # ROS2
   ```

3. **Model inference failing**:
   - Enable debug mode in config: `debug_mode: true`
   - Check logs for ONNX errors

4. **Invalid velocity limits**:
   - Verify limits in config match robot capabilities
   - Check for `max < min` errors

### Goal Not Reached

**Symptoms**: Planner stops before reaching goal

**Diagnosis**:
```bash
# Check goal tolerance
rosparam get /move_base/PlanGAROS1Plugin/xy_goal_tolerance  # ROS1
ros2 param get /controller_server xy_goal_tolerance  # ROS2
```

**Solution**:
```yaml
# Increase tolerance in config
xy_goal_tolerance: 0.15  # was 0.1
yaw_goal_tolerance: 0.2  # was 0.1
```

### Collision with Obstacles

**Symptoms**: Planner generates trajectories that hit obstacles

**Diagnosis**:
- Check if model was trained with sufficient obstacle diversity
- Verify costmap is correct in rviz
- Enable collision checking: `enable_collision_check: true`

**Solution**:

1. **Retrain with more scenarios**:
   ```bash
   # Increase obstacle density in ga_config.yaml
   costmap:
     num_obstacles_min: 5  # was 3
     num_obstacles_max: 12  # was 8

   # Increase collision weight
   fitness_weights:
     collision: 15.0  # was 10.0
   ```

2. **Adjust costmap inflation**:
   - Increase inflation radius in local costmap config
   - Verify robot footprint is correct

### Poor Performance (Low Hz)

**Symptoms**: Planning frequency below target (e.g., 5 Hz instead of 10 Hz)

**Diagnosis**:
```bash
# Check actual frequency
rostopic hz /local_plan  # ROS1
ros2 topic hz /local_plan  # ROS2

# Enable timing logs
# Set debug_mode: true in config
# Look for "ONNX inference time" in logs
```

**Solution**:

1. **Reduce model complexity** (requires retraining):
   ```yaml
   # In nn_config.yaml
   model:
     hidden_dim: 128  # was 256
     cnn:
       channels: [1, 16, 32, 64]  # was [1, 32, 64, 128]
   ```

2. **Reduce control steps**:
   ```yaml
   # In runtime config
   num_control_steps: 10  # was 20
   time_horizon: 1.0  # was 2.0
   ```

3. **Check CPU usage**:
   ```bash
   top -p $(pgrep move_base)  # ROS1
   top -p $(pgrep controller_server)  # ROS2
   ```

## Training Issues

### GA Training Slow

**Symptoms**: Training takes hours per 100 scenarios

**Diagnosis**:
```bash
# Check CPU usage
top

# Check number of workers
grep num_workers training/config/ga_config.yaml
```

**Solution**:
```bash
# Increase workers (match CPU cores)
python training/train_ga.py --num_workers 16 ...

# Reduce generations per scenario
# Edit ga_config.yaml:
ga:
  num_generations: 50  # was 100
```

### GA Poor Fitness

**Symptoms**: Best fitness values remain poor

**Diagnosis**:
- Check fitness component breakdown in logs
- Visualize best trajectories (requires matplotlib)

**Solution**:

1. **Adjust fitness weights**:
   ```yaml
   fitness_weights:
     goal_distance: 2.0  # Emphasize goal reaching
     collision: 5.0      # Reduce collision penalty
   ```

2. **Increase exploration**:
   ```yaml
   ga:
     population_size: 150  # was 100
     mutation_rate: 0.2    # was 0.1
   ```

3. **Simplify scenarios**:
   ```yaml
   costmap:
     num_obstacles_max: 5  # was 8
   scenarios:
     goal_distance_max: 2.0  # was 3.0
   ```

### NN Training Poor Validation Loss

**Symptoms**: Validation loss plateaus at high value

**Diagnosis**:
```bash
# Check dataset quality
python -c "
import pickle
data = pickle.load(open('models/checkpoints/ga_trajectories.pkl', 'rb'))
fitnesses = [d['fitness'] for d in data]
print(f'Mean fitness: {sum(fitnesses)/len(fitnesses):.2f}')
print(f'Min fitness: {min(fitnesses):.2f}')
"
```

**Solution**:

1. **Filter low-quality trajectories**:
   ```yaml
   # In nn_config.yaml
   training:
     filter_percentile: 50  # Use only top 50% (was 25)
   ```

2. **Collect more training data**:
   ```bash
   # Train GA with more scenarios
   python training/train_ga.py --num_scenarios 2000 ...
   ```

3. **Simplify model**:
   ```yaml
   model:
     hidden_dim: 128  # was 256
     policy_head:
       hidden_dims: [128, 128]  # was [256, 256]
   ```

4. **Augment data**:
   ```yaml
   augmentation:
     rotate: true
     add_noise: true
     noise_sigma: 5.0
   ```

### ONNX Export Fails

**Symptoms**:
```
RuntimeError: ONNX export failed
```

**Diagnosis**:
```bash
# Check PyTorch/ONNX versions
python -c "import torch; print(torch.__version__)"
python -c "import onnx; print(onnx.__version__)"
```

**Solution**:

1. **Update libraries**:
   ```bash
   pip install --upgrade torch onnx onnxruntime
   ```

2. **Use compatible opset**:
   ```yaml
   # In nn_config.yaml
   onnx:
     opset_version: 14  # Try 13, 14, or 15
   ```

3. **Simplify model**:
   - Remove unsupported operations
   - Check model for dynamic shapes

## Docker Issues

### Container Won't Start

**Symptoms**:
```
Error response from daemon: Conflict. The container name ... is already in use
```

**Solution**:
```bash
# Stop and remove existing container
./stop.sh
./remove.sh

# Start fresh
./run.sh
```

### DNS Resolution Fails

**Symptoms**:
```
Could not resolve 'snapshots.ros.org'
```

**Solution**:
```bash
# Update Docker daemon config
sudo nano /etc/docker/daemon.json

# Add:
{
  "dns": ["8.8.8.8", "1.1.1.1"]
}

# Restart Docker
sudo systemctl restart docker

# Rebuild image
./build.sh
```

### Volume Mount Issues

**Symptoms**: Code changes not reflected in container

**Diagnosis**:
```bash
# Check mounts
docker inspect plan_ga_ros1 | grep -A10 Mounts

# Verify from inside container
docker exec plan_ga_ros1 ls /catkin_ws/src/plan_ga/
```

**Solution**:
```bash
# Remove and recreate container
./remove.sh
./run.sh
```

## Debug Strategies

### Enable Verbose Logging

**Runtime**:
```yaml
debug_mode: true
```

**ROS1 Launch File**:
```xml
<node pkg="move_base" type="move_base" name="move_base" output="screen">
  <param name="base_local_planner" value="plan_ga_ros1/PlanGAROS1Plugin"/>
</node>
```

**ROS2 Launch File**:
```python
Node(
    package='nav2_controller',
    executable='controller_server',
    output='screen',
    arguments=['--ros-args', '--log-level', 'debug']
)
```

### Visualize Internal State

Enable costmap visualization:
```yaml
publish_cost_cloud: true
```

View in rviz:
- Add PointCloud2 display
- Subscribe to `/plan_ga/costmap_cloud`

### Test ONNX Model Independently

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('/models/planner_policy.onnx')

# Create test inputs
costmap = np.random.rand(1, 1, 50, 50).astype(np.float32)
robot_state = np.zeros((1, 9), dtype=np.float32)
goal = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
metadata = np.array([[0.05, 0.8]], dtype=np.float32)

# Run inference
try:
    outputs = session.run(
        None,
        {
            'costmap_input': costmap,
            'robot_state_input': robot_state,
            'goal_relative_input': goal,
            'costmap_metadata_input': metadata
        }
    )
    print(f"Success! Output shape: {outputs[0].shape}")
except Exception as e:
    print(f"Error: {e}")
```

## Getting Help

### Collect Debug Information

```bash
# System info
uname -a
docker --version
cat /etc/os-release

# ROS info
rosversion -d  # ROS1
echo $ROS_DISTRO  # ROS2

# Package info
rospack find plan_ga_ros1  # ROS1
ros2 pkg prefix plan_ga_ros2  # ROS2

# Recent logs
# ROS1: ~/.ros/log/latest/
# ROS2: ~/.ros/log/
```

### Check Documentation

- **Setup**: [setup.md](setup.md)
- **Building**: [building.md](building.md)
- **Training**: [training.md](training.md)
- **Configuration**: [configuration.md](configuration.md)
- **Learning Course**: [../learn/](../learn/)

### Common Error Messages Reference

| Error | File | Solution |
|-------|------|----------|
| `Plugin not found` | deployment.md | Re-source workspace, check plugin registration |
| `ONNX model not found` | training.md | Train model, check mount paths |
| `Build failed` | building.md | Clean build, check dependencies |
| `Poor fitness` | configuration.md | Adjust weights, increase exploration |
| `Low planning frequency` | configuration.md | Reduce control steps, simplify model |
