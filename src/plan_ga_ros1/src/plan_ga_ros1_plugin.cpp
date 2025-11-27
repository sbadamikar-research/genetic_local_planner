#include "plan_ga_ros1/plan_ga_ros1_plugin.h"
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// Register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(plan_ga_ros1::PlanGAROS1Plugin, nav_core::BaseLocalPlanner)

namespace plan_ga_ros1 {

PlanGAROS1Plugin::PlanGAROS1Plugin()
    : initialized_(false),
      tf_(nullptr),
      costmap_ros_(nullptr),
      control_index_(0),
      have_prev_velocity_(false) {
    prev_velocity_.fill(0.0);
}

PlanGAROS1Plugin::~PlanGAROS1Plugin() {}

void PlanGAROS1Plugin::initialize(std::string name, tf2_ros::Buffer* tf,
                                   costmap_2d::Costmap2DROS* costmap_ros) {
    if (initialized_) {
        ROS_WARN("[PlanGAROS1Plugin] Already initialized");
        return;
    }

    name_ = name;
    tf_ = tf;
    costmap_ros_ = costmap_ros;
    private_nh_ = ros::NodeHandle("~/" + name);

    // Load parameters
    loadParameters_();

    // Create planner core
    plan_ga_planner::PlannerConfig config;
    private_nh_.param("model_path", config.model_path, config.model_path);
    private_nh_.param("num_control_steps", config.num_control_steps, config.num_control_steps);
    private_nh_.param("control_frequency", config.control_frequency, config.control_frequency);
    private_nh_.param("time_horizon", config.time_horizon, config.time_horizon);
    private_nh_.param("max_v_x", config.max_v_x, config.max_v_x);
    private_nh_.param("min_v_x", config.min_v_x, config.min_v_x);
    private_nh_.param("max_v_y", config.max_v_y, config.max_v_y);
    private_nh_.param("min_v_y", config.min_v_y, config.min_v_y);
    private_nh_.param("max_omega", config.max_omega, config.max_omega);
    private_nh_.param("min_omega", config.min_omega, config.min_omega);
    private_nh_.param("costmap_window_size", config.costmap_window_size, config.costmap_window_size);
    private_nh_.param("lethal_cost_threshold", reinterpret_cast<int&>(config.lethal_cost_threshold), static_cast<int>(config.lethal_cost_threshold));
    private_nh_.param("enable_collision_check", config.enable_collision_check, config.enable_collision_check);
    private_nh_.param("xy_goal_tolerance", config.xy_goal_tolerance, config.xy_goal_tolerance);
    private_nh_.param("yaw_goal_tolerance", config.yaw_goal_tolerance, config.yaw_goal_tolerance);
    private_nh_.param("debug_mode", config.debug_mode, config.debug_mode);
    private_nh_.param("publish_local_plan", config.publish_local_plan, config.publish_local_plan);

    // Load footprint from costmap_ros
    std::vector<geometry_msgs::Point> footprint = costmap_ros_->getRobotFootprint();
    config.footprint.clear();
    for (const auto& point : footprint) {
        config.footprint.push_back({{point.x, point.y}});
    }

    // Initialize planner core
    try {
        planner_core_ = std::make_unique<plan_ga_planner::PlannerCore>(config.model_path);
        if (!planner_core_->initialize(config)) {
            ROS_ERROR("[PlanGAROS1Plugin] Failed to initialize planner core");
            return;
        }
    } catch (const std::exception& e) {
        ROS_ERROR("[PlanGAROS1Plugin] Exception during initialization: %s", e.what());
        return;
    }

    // Setup publisher for local plan visualization
    if (config.publish_local_plan) {
        plan_pub_ = private_nh_.advertise<nav_msgs::Path>("local_plan", 1);
    }

    initialized_ = true;
    ROS_INFO("[PlanGAROS1Plugin] Initialized successfully");
}

bool PlanGAROS1Plugin::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) {
    if (!initialized_) {
        ROS_ERROR("[PlanGAROS1Plugin] Not initialized");
        return false;
    }

    global_plan_ = plan;
    control_index_ = 0;

    ROS_INFO("[PlanGAROS1Plugin] Received global plan with %zu poses", plan.size());
    return true;
}

bool PlanGAROS1Plugin::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
    if (!initialized_) {
        ROS_ERROR("[PlanGAROS1Plugin] Not initialized");
        return false;
    }

    if (global_plan_.empty()) {
        ROS_WARN("[PlanGAROS1Plugin] No global plan set");
        cmd_vel.linear.x = 0.0;
        cmd_vel.linear.y = 0.0;
        cmd_vel.angular.z = 0.0;
        return false;
    }

    // Get current state
    plan_ga_planner::Pose current_pose;
    if (!getCurrentPose_(current_pose)) {
        ROS_ERROR("[PlanGAROS1Plugin] Failed to get current pose");
        return false;
    }

    plan_ga_planner::Velocity current_velocity;
    if (!getCurrentVelocity_(current_velocity)) {
        ROS_WARN("[PlanGAROS1Plugin] Failed to get current velocity, using zero");
        current_velocity.fill(0.0);
    }

    plan_ga_planner::Acceleration current_acceleration;
    if (!getCurrentAcceleration_(current_acceleration)) {
        current_acceleration.fill(0.0);
    }

    // Get goal from global plan (last pose)
    plan_ga_planner::Pose goal_pose;
    goal_pose[0] = global_plan_.back().pose.position.x;
    goal_pose[1] = global_plan_.back().pose.position.y;
    goal_pose[2] = tf2::getYaw(global_plan_.back().pose.orientation);

    // Get costmap
    plan_ga_planner::Costmap costmap;
    if (!convertCostmap_(costmap)) {
        ROS_ERROR("[PlanGAROS1Plugin] Failed to convert costmap");
        return false;
    }

    // Compute control sequence
    plan_ga_planner::ControlSequence control_sequence;
    if (!planner_core_->computeControlSequence(
            costmap, current_pose, current_velocity, current_acceleration,
            goal_pose, control_sequence)) {
        ROS_ERROR("[PlanGAROS1Plugin] Failed to compute control sequence");
        return false;
    }

    if (control_sequence.empty()) {
        ROS_ERROR("[PlanGAROS1Plugin] Empty control sequence");
        return false;
    }

    // Use first control command
    const auto& cmd = control_sequence[0];
    cmd_vel.linear.x = cmd.v_x;
    cmd_vel.linear.y = cmd.v_y;
    cmd_vel.angular.z = cmd.omega;

    // Publish local plan for visualization
    if (plan_pub_.getNumSubscribers() > 0) {
        plan_ga_planner::Trajectory trajectory;
        planner_core_->computeTrajectory(current_pose, current_velocity, control_sequence, trajectory);
        publishPlan_(trajectory);
    }

    // Store current control sequence
    current_control_sequence_ = control_sequence;

    return true;
}

bool PlanGAROS1Plugin::isGoalReached() {
    if (!initialized_) {
        return false;
    }

    plan_ga_planner::Pose current_pose;
    if (!getCurrentPose_(current_pose)) {
        return false;
    }

    if (global_plan_.empty()) {
        return false;
    }

    plan_ga_planner::Pose goal_pose;
    goal_pose[0] = global_plan_.back().pose.position.x;
    goal_pose[1] = global_plan_.back().pose.position.y;
    goal_pose[2] = tf2::getYaw(global_plan_.back().pose.orientation);

    return planner_core_->isGoalReached(current_pose, goal_pose);
}

bool PlanGAROS1Plugin::convertCostmap_(plan_ga_planner::Costmap& costmap) {
    costmap_2d::Costmap2D* ros_costmap = costmap_ros_->getCostmap();

    costmap.width = ros_costmap->getSizeInCellsX();
    costmap.height = ros_costmap->getSizeInCellsY();
    costmap.resolution = ros_costmap->getResolution();
    costmap.origin_x = ros_costmap->getOriginX();
    costmap.origin_y = ros_costmap->getOriginY();

    // Get inflation parameters from parameter server
    private_nh_.param("inflation_decay", costmap.inflation_decay, 0.8);

    // Copy costmap data
    unsigned char* ros_data = ros_costmap->getCharMap();
    size_t size = costmap.width * costmap.height;
    costmap.data.assign(ros_data, ros_data + size);

    return true;
}

bool PlanGAROS1Plugin::getCurrentPose_(plan_ga_planner::Pose& pose) {
    geometry_msgs::PoseStamped robot_pose;
    if (!costmap_ros_->getRobotPose(robot_pose)) {
        return false;
    }

    pose[0] = robot_pose.pose.position.x;
    pose[1] = robot_pose.pose.position.y;
    pose[2] = tf2::getYaw(robot_pose.pose.orientation);

    return true;
}

bool PlanGAROS1Plugin::getCurrentVelocity_(plan_ga_planner::Velocity& velocity) {
    // In ROS1, velocity typically comes from odometry topic
    // For simplicity, using last computed velocity or zero
    // In a real implementation, should subscribe to odom topic

    if (have_prev_velocity_) {
        velocity = prev_velocity_;
        return true;
    }

    velocity.fill(0.0);
    return true;
}

bool PlanGAROS1Plugin::getCurrentAcceleration_(plan_ga_planner::Acceleration& acceleration) {
    // Estimate acceleration from velocity change
    plan_ga_planner::Velocity current_velocity;
    if (!getCurrentVelocity_(current_velocity)) {
        acceleration.fill(0.0);
        return false;
    }

    ros::Time current_time = ros::Time::now();

    if (have_prev_velocity_) {
        double dt = (current_time - prev_time_).toSec();
        if (dt > 0.0) {
            acceleration[0] = (current_velocity[0] - prev_velocity_[0]) / dt;
            acceleration[1] = (current_velocity[1] - prev_velocity_[1]) / dt;
            acceleration[2] = (current_velocity[2] - prev_velocity_[2]) / dt;
        } else {
            acceleration.fill(0.0);
        }
    } else {
        acceleration.fill(0.0);
    }

    // Update history
    prev_velocity_ = current_velocity;
    prev_time_ = current_time;
    have_prev_velocity_ = true;

    return true;
}

void PlanGAROS1Plugin::publishPlan_(const plan_ga_planner::Trajectory& trajectory) {
    nav_msgs::Path path_msg;
    path_msg.header.frame_id = costmap_ros_->getGlobalFrameID();
    path_msg.header.stamp = ros::Time::now();

    for (const auto& point : trajectory) {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = path_msg.header;
        pose_stamped.pose.position.x = point.pose[0];
        pose_stamped.pose.position.y = point.pose[1];
        pose_stamped.pose.position.z = 0.0;

        tf2::Quaternion quat;
        quat.setRPY(0, 0, point.pose[2]);
        pose_stamped.pose.orientation = tf2::toMsg(quat);

        path_msg.poses.push_back(pose_stamped);
    }

    plan_pub_.publish(path_msg);
}

void PlanGAROS1Plugin::loadParameters_() {
    // Parameters are loaded in initialize()
    ROS_INFO("[PlanGAROS1Plugin] Loading parameters from namespace: %s", private_nh_.getNamespace().c_str());
}

}  // namespace plan_ga_ros1
