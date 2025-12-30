# Module 06: ROS Fundamentals & Navigation Stack

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Understand ROS communication patterns (topics, services, actions)
- ✅ Learn the navigation stack architecture (move_base, Nav2)
- ✅ Understand costmaps (layers, inflation, configuration)
- ✅ Master TF/TF2 for coordinate transformations
- ✅ Understand pluginlib and plugin discovery
- ✅ Configure and test the plan_ga planner plugin
- ✅ Debug common plugin loading issues
- ✅ Visualize navigation data with RViz

## Why Learn ROS?

ROS (Robot Operating System) is the de facto standard for robotics software:
- **Industry adoption**: Used by Boston Dynamics, Tesla, NASA, etc.
- **Modularity**: Components communicate via well-defined interfaces
- **Reusability**: Don't reinvent the wheel—use existing packages
- **Simulation**: Test before deploying on hardware

**For this project:** Our C++ planner integrates seamlessly with move_base (ROS1) or Nav2 (ROS2), getting free localization, mapping, and path planning.

---

## Key Concepts

### ROS Communication Patterns

ROS provides three main communication mechanisms:

```
┌─────────────────────────────────────────────────────────────────┐
│                  ROS COMMUNICATION PATTERNS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TOPICS (Pub/Sub) - Continuous data streams                 │
│     ┌──────────┐  /cmd_vel   ┌─────────┐  /odom  ┌──────────┐ │
│     │ Planner  │ ──────────> │  Robot  │ ──────> │ Odometry │ │
│     └──────────┘  Twist      └─────────┘  Pose   └──────────┘ │
│     • Asynchronous, one-to-many                                 │
│     • Best for sensor data, control commands                    │
│                                                                  │
│  2. SERVICES (Request/Response) - One-time queries              │
│     ┌──────────┐   "Clear costmap"    ┌──────────┐             │
│     │ Client   │ ──────────────────> │ Service  │             │
│     │          │ <────────────────── │          │             │
│     └──────────┘   "Done"             └──────────┘             │
│     • Synchronous, blocking                                     │
│     • Best for configuration, one-time operations               │
│                                                                  │
│  3. ACTIONS (Goal-based) - Long-running tasks with feedback     │
│     ┌──────────┐   "Go to (x,y)"     ┌──────────┐             │
│     │ Client   │ ──────────────────> │ Server   │             │
│     │          │ <────────────────── │          │             │
│     └──────────┘   "50% there..."     └──────────┘             │
│     • Asynchronous, cancelable                                  │
│     • Best for navigation goals, manipulation                   │
└─────────────────────────────────────────────────────────────────┘
```

### Navigation Stack Architecture (ROS1 move_base)

```
┌────────────────────────────────────────────────────────────────┐
│                     MOVE_BASE ARCHITECTURE                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User/Mission Planner                                          │
│        │                                                        │
│        ↓ /move_base/goal (action)                             │
│  ┌─────────────────┐                                          │
│  │   move_base     │  Master controller                        │
│  └─────────────────┘                                          │
│         │                                                       │
│         ├──> Global Planner (e.g., Dijkstra, A*)             │
│         │    • Creates overall path from start → goal         │
│         │    • Uses global costmap                            │
│         │                                                       │
│         └──> Local Planner (OUR PLUGIN!)                      │
│              • plan_ga_ros1/PlanGAROS1Plugin                  │
│              • Generates velocity commands                     │
│              • Avoids dynamic obstacles                        │
│              • Uses local costmap                             │
│                                                                 │
│  ┌─────────────────────────────────────────────┐              │
│  │         COSTMAPS (2D Occupancy Grid)        │              │
│  ├─────────────────────────────────────────────┤              │
│  │                                              │              │
│  │  Global Costmap:                            │              │
│  │  • Large area (entire map)                  │              │
│  │  • Static map layer (walls, furniture)      │              │
│  │  • Inflation layer (safety buffer)          │              │
│  │                                              │              │
│  │  Local Costmap:                             │              │
│  │  • Small area around robot (e.g., 10×10m)   │              │
│  │  • Obstacle layer (sensor data)             │              │
│  │  • Inflation layer                          │              │
│  │  • Rolling window (follows robot)           │              │
│  │                                              │              │
│  │  Cost values:                                │              │
│  │  0     = Free space                         │              │
│  │  1-252 = Inflation (closer = higher cost)   │              │
│  │  253   = Inscribed (robot barely fits)      │              │
│  │  254   = Lethal (collision!)                │              │
│  │  255   = Unknown (no sensor data)           │              │
│  └─────────────────────────────────────────────┘              │
│                                                                 │
│  Output: /cmd_vel (geometry_msgs/Twist)                        │
│         • linear.x (forward/backward)                          │
│         • linear.y (left/right, holonomic only)               │
│         • angular.z (rotation)                                 │
└────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Our planner is a plugin! move_base calls our `computeVelocityCommands()` method at 10-20 Hz, expecting velocity commands back.

### TF (Transform) System

ROS uses **TF** to manage coordinate frames:

```
              /map (world frame, fixed)
                 │
                 ↓ (where is robot in map?)
              /odom (odometry, drifts slowly)
                 │
                 ↓ (wheel encoders, IMU)
              /base_link (robot center)
                 │
                 ├─> /base_footprint (ground projection)
                 ├─> /laser (sensor location)
                 └─> /camera (sensor location)

Example transformation:
"Where is the laser point in the map frame?"
map → odom → base_link → laser
```

**Why it matters:** Our planner needs to transform goals from `/map` to `/base_link` (robot frame) for the neural network.

### Pluginlib: How Plugins Work

```
1. Package declares plugin in package.xml:
   <export>
     <nav_core plugin="${prefix}/plan_ga_plugin.xml" />
   </export>

2. Plugin descriptor (plan_ga_plugin.xml):
   <class name="plan_ga_ros1/PlanGAROS1Plugin"
          type="plan_ga_ros1::PlanGAROS1Plugin"
          base_class_type="nav_core::BaseLocalPlanner">

3. Runtime discovery:
   $ rospack plugins --attrib=plugin nav_core
   → Lists all available local planners

4. move_base loads plugin by name:
   base_local_planner: "plan_ga_ros1/PlanGAROS1Plugin"
   → Calls initialize(), setPlan(), computeVelocityCommands()
```

**Plugin interface (nav_core::BaseLocalPlanner):**
- `initialize()` - Setup (called once)
- `setPlan()` - Receive global path
- `computeVelocityCommands()` - Return velocities (called at control frequency)
- `isGoalReached()` - Check if done

---

## Hands-On Exercises

### Exercise 1: Explore ROS Topics and Nodes (30 min)

Launch a minimal ROS system and inspect its graph:

```bash
# Start ROS1 core
roscore &

# In another terminal, list running nodes
rosnode list
# Should show: /rosout (master logger)

# Launch turtlesim (simple 2D robot simulator)
rosrun turtlesim turtlesim_node &

# List topics
rostopic list
# /turtle1/cmd_vel, /turtle1/pose, etc.

# Echo a topic (see live data)
rostopic echo /turtle1/pose
# x, y, theta values update in real-time

# Get topic info
rostopic info /turtle1/cmd_vel
# Type: geometry_msgs/Twist
# Publishers: None
# Subscribers: /turtlesim

# Publish a command (move turtle)
rostopic pub /turtle1/cmd_vel geometry_msgs/Twist "linear:
  x: 1.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.5"

# Check message structure
rosmsg show geometry_msgs/Twist
```

**Questions:**
1. What happens when you change `linear.x` to 2.0?
2. What's the difference between a publisher and subscriber?
3. How would you stop the turtle?

### Exercise 2: Visualize with RViz (45 min)

RViz is ROS's 3D visualization tool:

```bash
# Launch RViz
rosrun rviz rviz

# In RViz GUI:
# 1. Set Fixed Frame to "map"
# 2. Add → By display type → Map → /map
# 3. Add → By display type → Path → /global_plan
# 4. Add → By display type → Path → /local_plan
# 5. Add → By topic → /costmap → Costmap

# Launch a pre-configured RViz
# (after building our project)
rviz -d /catkin_ws/src/plan_ga/samples/configs/navigation.rviz
```

**Task:** Visualize different elements:
- Costmap colors: Free (green) → Inflated (yellow) → Lethal (red)
- Global plan: Blue line from start to goal
- Local plan: Red line (our planner's trajectory)
- Robot footprint: Polygon showing robot shape

**Questions:**
1. Why is the local plan shorter than global plan?
2. What do inflated costs represent?
3. How does the costmap update when obstacles move?

### Exercise 3: Understand Costmap Configuration (1 hour)

Costmaps are critical for navigation. Let's dissect the configuration:

**File: `move_base/costmap_common_params.yaml`**

```yaml
# Robot configuration
robot_radius: 0.3  # meters (for circular robots)
# OR use footprint for non-circular:
footprint: [[-0.4, -0.3], [-0.4, 0.3], [0.4, 0.3], [0.4, -0.3]]

# Obstacle layer: sensor data → obstacles
obstacle_layer:
  enabled: true
  observation_sources: laser_scan
  laser_scan:
    topic: /scan
    data_type: LaserScan
    clearing: true     # Can clear obstacles
    marking: true      # Can add obstacles
    min_obstacle_height: 0.0
    max_obstacle_height: 2.0

# Inflation layer: grow obstacles for safety
inflation_layer:
  enabled: true
  inflation_radius: 0.5  # meters (safety buffer)
  cost_scaling_factor: 10.0  # decay rate

# Static layer: pre-built map (walls, furniture)
static_layer:
  enabled: true
  map_topic: /map
```

**Inflation formula:**
```
cost = 253 * e^(-decay * distance)

Where:
  decay = cost_scaling_factor / inflation_radius

Examples:
  At distance 0.0m: cost = 253 (inscribed)
  At distance 0.25m: cost = ~160 (high)
  At distance 0.5m: cost = ~40 (low)
  Beyond 0.5m: cost = 0 (free)
```

**Task:** Modify inflation parameters and observe effects:

```bash
# Open config
nano /catkin_ws/src/navigation/move_base/costmap_common_params.yaml

# Try:
# 1. inflation_radius: 0.3 (tight safety margin)
# 2. inflation_radius: 1.0 (wide safety margin)
# 3. cost_scaling_factor: 3.0 (slower decay)
# 4. cost_scaling_factor: 20.0 (faster decay)

# Restart move_base
rosnode kill /move_base
roslaunch your_robot move_base.launch
```

**Questions:**
1. What happens if inflation_radius is too small?
2. Why separate static and obstacle layers?
3. How does our planner use costmap data?

### Exercise 4: Debug TF Transformations (1 hour)

TF is crucial for coordinate conversions:

```bash
# View TF tree
rosrun tf view_frames
# Generates frames.pdf showing hierarchy

# Check specific transform
rosrun tf tf_echo /map /base_link
# Shows translation and rotation over time

# Lookup transform from command line
rostopic echo /tf
# Shows all published transforms

# Visualize in RViz
# Add → TF → Enable all frames
# Shows coordinate axes for each frame
```

**Common TF errors:**

```
Error: "Could not transform from map to base_link"
Causes:
1. Missing transform publisher
2. Frame names mismatch
3. Time synchronization issues
```

**Debug workflow:**
```bash
# 1. Check available frames
rosrun tf tf_monitor
# Lists all frames and publication rates

# 2. Find missing transform
rosrun tf view_frames
# Look for breaks in tree

# 3. Check transform latency
rosrun tf tf_echo /map /base_link
# Should update at >10 Hz
```

**Task:** Understand our planner's TF usage

File: `src/plan_ga_ros1/src/plan_ga_ros1_plugin.cpp:150`

```cpp
// Get robot pose in map frame
geometry_msgs::PoseStamped robot_pose;
if (!costmap_ros_->getRobotPose(robot_pose)) {
    ROS_ERROR("Cannot get robot pose");
    return false;
}

// Transform goal from map → base_link
geometry_msgs::PoseStamped goal_in_base_link;
try {
    tf_->transform(goal_map_frame, goal_in_base_link, "base_link");
} catch (tf2::TransformException& ex) {
    ROS_ERROR("TF error: %s", ex.what());
    return false;
}
```

**Questions:**
1. Why transform goal to robot frame?
2. What happens if TF lookup fails?
3. How often does TF update?

### Exercise 5: Configure and Test plan_ga Plugin (1.5 hours)

Time to integrate our planner with move_base!

**Step 1: Verify plugin is discoverable**

```bash
# Inside ROS1 container
cd /catkin_ws
source devel/setup.bash

# Check plugin registration
rospack plugins --attrib=plugin nav_core | grep plan_ga

# Expected output:
# plan_ga_ros1 /catkin_ws/src/plan_ga/plan_ga_ros1/plan_ga_plugin.xml

# If not found, check:
# 1. package.xml has <export> tag
# 2. plan_ga_plugin.xml exists
# 3. catkin_make completed successfully
```

**Step 2: Create move_base configuration**

File: `move_base_config.yaml`

```yaml
base_local_planner: "plan_ga_ros1/PlanGAROS1Plugin"

PlanGAROS1Plugin:
  model_path: "/models/planner_policy.onnx"

  # Control parameters
  control_frequency: 10.0
  num_control_steps: 20
  time_horizon: 2.0

  # Velocity limits (adjust for your robot!)
  max_v_x: 0.5      # Conservative for testing
  max_omega: 0.5

  # Goal tolerance
  xy_goal_tolerance: 0.2
  yaw_goal_tolerance: 0.2

  # Debugging
  debug_mode: true
  publish_local_plan: true
```

**Step 3: Launch move_base**

```bash
# Minimal launch file
<launch>
  <!-- Load parameters -->
  <rosparam file="move_base_config.yaml" command="load" />

  <!-- Launch move_base -->
  <node pkg="move_base" type="move_base" name="move_base" output="screen">
    <rosparam file="costmap_common_params.yaml" command="load" ns="global_costmap" />
    <rosparam file="costmap_common_params.yaml" command="load" ns="local_costmap" />
    <rosparam file="global_costmap_params.yaml" command="load" />
    <rosparam file="local_costmap_params.yaml" command="load" />
  </node>
</launch>

# Launch it
roslaunch your_package move_base.launch
```

**Step 4: Send a navigation goal**

```bash
# Method 1: RViz
# 1. Open RViz
# 2. Click "2D Nav Goal" button
# 3. Click and drag on map

# Method 2: Command line
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "
header:
  frame_id: 'map'
pose:
  position:
    x: 2.0
    y: 1.0
    z: 0.0
  orientation:
    w: 1.0"

# Watch for planner output
rostopic echo /cmd_vel
```

**Expected behavior:**
1. move_base receives goal
2. Global planner creates path
3. Our plugin is called: `computeVelocityCommands()`
4. ONNX inference runs
5. Velocity commands published to /cmd_vel
6. Robot moves!

**Questions:**
1. What happens if ONNX model is missing?
2. How fast does the planner run? (check logs for timing)
3. What if robot gets stuck?

### Exercise 6: Debug Plugin Loading Errors (1 hour)

Common issues and solutions:

**Error 1: Plugin not found**
```
[FATAL] Failed to load plugin plan_ga_ros1/PlanGAROS1Plugin
[ERROR] Could not load library: libplan_ga_ros1_plugin.so
```

**Debug steps:**
```bash
# 1. Check library exists
ls /catkin_ws/devel/lib/libplan_ga_ros1_plugin.so
# Should exist after catkin_make

# 2. Check plugin XML
cat /catkin_ws/src/plan_ga/plan_ga_ros1/plan_ga_plugin.xml
# Verify path matches library name

# 3. Check package.xml export
grep -A5 export /catkin_ws/src/plan_ga/plan_ga_ros1/package.xml
# Should have: <nav_core plugin="${prefix}/plan_ga_plugin.xml" />

# 4. Re-source workspace
source /catkin_ws/devel/setup.bash
```

**Error 2: ONNX Runtime crash**
```
terminate called after throwing an instance of 'Ort::Exception'
```

**Debug steps:**
```bash
# 1. Check model exists
ls -lh /models/planner_policy.onnx
# Should be ~1-5 MB

# 2. Verify ONNX Runtime installed
ldconfig -p | grep onnxruntime
# Should list libonnxruntime.so

# 3. Test model loading
python3 << EOF
import onnxruntime as ort
sess = ort.InferenceSession('/models/planner_policy.onnx')
print("Model loaded OK")
EOF

# 4. Check file permissions
chmod 644 /models/planner_policy.onnx
```

**Error 3: Velocity limits violated**
```
[WARN] Velocity command exceeds limits: v_x=1.5 > max_v_x=1.0
```

**Solution:** Adjust parameters in `planner_params_ros1.yaml`

**Error 4: Costmap access failure**
```
[ERROR] Failed to get costmap data
```

**Debug:**
```bash
# Check costmap is publishing
rostopic echo /move_base/local_costmap/costmap
# Should stream data

# Check for TF errors
rostopic echo /tf_static
```

---

## Code Walkthrough

### Plugin Registration and Discovery

**File:** `src/plan_ga_ros1/plan_ga_plugin.xml`

```xml
<library path="lib/libplan_ga_ros1_plugin">
  <class name="plan_ga_ros1/PlanGAROS1Plugin"
         type="plan_ga_ros1::PlanGAROS1Plugin"
         base_class_type="nav_core::BaseLocalPlanner">
    <description>
      GA-based local planner using neural network policy.
    </description>
  </class>
</library>
```

**Key elements:**
- `library path`: Name of .so file (without lib prefix or .so suffix)
- `class name`: How users reference it (`"plan_ga_ros1/PlanGAROS1Plugin"`)
- `type`: Actual C++ class name
- `base_class_type`: Interface it implements

**Macro that makes it work:** `src/plan_ga_ros1/src/plan_ga_ros1_plugin.cpp:7`

```cpp
PLUGINLIB_EXPORT_CLASS(plan_ga_ros1::PlanGAROS1Plugin,
                       nav_core::BaseLocalPlanner)
```

This macro generates boilerplate code for pluginlib to instantiate our class.

### Plugin Initialization Flow

**File:** `src/plan_ga_ros1/src/plan_ga_ros1_plugin.cpp:22-83`

```cpp
void PlanGAROS1Plugin::initialize(std::string name, tf2_ros::Buffer* tf,
                                   costmap_2d::Costmap2DROS* costmap_ros) {
    // 1. Save references
    name_ = name;
    tf_ = tf;
    costmap_ros_ = costmap_ros;

    // 2. Create parameter namespace
    private_nh_ = ros::NodeHandle("~/" + name);
    // Now can load: ~/PlanGAROS1Plugin/model_path

    // 3. Load parameters from ROS parameter server
    plan_ga_planner::PlannerConfig config;
    private_nh_.param("model_path", config.model_path,
                     std::string("/models/planner_policy.onnx"));
    // ... load all parameters ...

    // 4. Get robot footprint from costmap
    std::vector<geometry_msgs::Point> footprint = costmap_ros_->getRobotFootprint();
    for (const auto& point : footprint) {
        config.footprint.push_back({{point.x, point.y}});
    }

    // 5. Initialize core planner (load ONNX model)
    planner_core_ = std::make_unique<plan_ga_planner::PlannerCore>(config.model_path);
    if (!planner_core_->initialize(config)) {
        ROS_ERROR("Failed to initialize planner core");
        return;
    }

    // 6. Setup visualization publisher
    if (config.publish_local_plan) {
        plan_pub_ = private_nh_.advertise<nav_msgs::Path>("local_plan", 1);
    }

    initialized_ = true;
}
```

**Trace the flow:**
1. `move_base` loads plugin via pluginlib
2. Calls `initialize()` with TF and costmap references
3. Plugin loads parameters from `~/PlanGAROS1Plugin/...` namespace
4. Creates `PlannerCore` which loads ONNX model
5. Sets up ROS publishers/subscribers

### Velocity Command Generation

**File:** `src/plan_ga_ros1/src/plan_ga_ros1_plugin.cpp:98-180`

```cpp
bool PlanGAROS1Plugin::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
    // Called at control frequency (e.g., 10 Hz)

    // 1. Get current robot pose (map frame)
    geometry_msgs::PoseStamped robot_pose;
    if (!costmap_ros_->getRobotPose(robot_pose)) {
        return false;
    }

    // 2. Get current costmap
    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();

    // 3. Transform goal to robot frame
    geometry_msgs::PoseStamped goal_robot_frame;
    tf_->transform(global_plan_.back(), goal_robot_frame, "base_link");

    // 4. Convert to internal types
    plan_ga_planner::Pose current_pose = convertPose(robot_pose);
    plan_ga_planner::Pose goal_pose = convertPose(goal_robot_frame);
    plan_ga_planner::Costmap internal_costmap = convertCostmap(costmap);

    // 5. Call core planner (ONNX inference happens here!)
    plan_ga_planner::Velocity velocity = planner_core_->computeVelocity(
        current_pose,
        current_velocity_,
        goal_pose,
        internal_costmap
    );

    // 6. Convert back to ROS message
    cmd_vel.linear.x = velocity.v_x;
    cmd_vel.linear.y = velocity.v_y;
    cmd_vel.angular.z = velocity.omega;

    return true;
}
```

**Performance note:** This runs at 10-20 Hz. ONNX inference must be <50ms!

---

## Quiz

1. **Which communication pattern is best for sensor data?**
   a) Services (request/response)
   b) Topics (publish/subscribe)
   c) Actions (goal-based)
   d) Parameters

2. **What does the inflation layer do?**
   a) Grow obstacles for safety margin
   b) Compress the map
   c) Increase planning speed
   d) Smooth trajectories

3. **Why use TF instead of hardcoded transforms?**
   a) TF is faster
   b) Automatically handles time-varying transformations
   c) Required by ROS
   d) Easier to debug

4. **What does pluginlib enable?**
   a) Faster compilation
   b) Runtime plugin discovery and loading
   c) Python/C++ interop
   d) GPU acceleration

5. **What happens if computeVelocityCommands() returns false?**
   a) Robot stops immediately
   b) move_base retries or aborts goal
   c) Uses previous velocity
   d) Crash

<details>
<summary><b>Show Answers</b></summary>

1. b) Topics (continuous data streaming)
2. a) Grow obstacles for safety margin
3. b) Automatically handles time-varying transformations (robot moves!)
4. b) Runtime plugin discovery and loading (swap planners without recompiling)
5. b) move_base retries or aborts goal (depends on recovery behaviors)
</details>

---

## ✅ Checklist

- [ ] Understand ROS topics, services, and actions
- [ ] Can visualize navigation stack in RViz
- [ ] Understand costmap layers and inflation
- [ ] Can debug TF transformation issues
- [ ] Successfully loaded plan_ga plugin in move_base
- [ ] Sent navigation goals and observed planner behavior
- [ ] Debugged at least one plugin loading error
- [ ] Quiz score 80%+

---

## 📚 Resources

- [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) (official)
- [Navigation Stack Overview](http://wiki.ros.org/navigation)
- [move_base Documentation](http://wiki.ros.org/move_base)
- [Costmap 2D](http://wiki.ros.org/costmap_2d) (layers, inflation)
- [TF2 Tutorials](http://wiki.ros.org/tf2/Tutorials)
- [pluginlib](http://wiki.ros.org/pluginlib) (plugin system)
- [RViz User Guide](http://wiki.ros.org/rviz/UserGuide)

---

## 🎉 Next Steps

You now understand ROS and can integrate C++ planners with the navigation stack!

Next, learn how Docker simplifies development with containerized environments.

**→ [Continue to Module 07: Docker & Dev Containers](../07-docker-containers/)**
