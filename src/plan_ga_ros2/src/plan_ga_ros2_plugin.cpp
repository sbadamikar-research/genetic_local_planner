#include "plan_ga_ros2/plan_ga_ros2_plugin.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"

PLUGINLIB_EXPORT_CLASS(plan_ga_ros2::PlanGAROS2Plugin, nav2_core::Controller)

namespace plan_ga_ros2 {

void PlanGAROS2Plugin::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
    auto node = parent.lock();
    if (!node) {
        throw std::runtime_error("Unable to lock node!");
    }

    plugin_name_ = name;
    tf_ = tf;
    costmap_ros_ = costmap_ros;
    node_ = parent;
    logger_ = node->get_logger();
    clock_ = node->get_clock();

    RCLCPP_INFO(logger_, "Configuring PlanGAROS2Plugin: %s", plugin_name_.c_str());

    // Load parameters
    loadParameters_();

    // Create planner core
    try {
        planner_core_ = std::make_unique<plan_ga_planner::PlannerCore>(config_.model_path);
        if (!planner_core_->initialize(config_)) {
            throw std::runtime_error("Failed to initialize planner core");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Failed to create planner core: %s", e.what());
        throw;
    }

    // Create local plan publisher
    local_plan_pub_ = node->create_publisher<nav_msgs::msg::Path>(
        plugin_name_ + "/local_plan", 1);

    initialized_ = true;
    active_ = false;
    speed_limit_ = 0.0;
    speed_limit_percentage_ = false;

    // Initialize velocity history
    last_velocity_ = {0.0, 0.0, 0.0};
    last_time_ = clock_->now();

    RCLCPP_INFO(logger_, "PlanGAROS2Plugin configured successfully");
}

void PlanGAROS2Plugin::activate()
{
    RCLCPP_INFO(logger_, "Activating PlanGAROS2Plugin");
    local_plan_pub_->on_activate();
    active_ = true;
}

void PlanGAROS2Plugin::deactivate()
{
    RCLCPP_INFO(logger_, "Deactivating PlanGAROS2Plugin");
    local_plan_pub_->on_deactivate();
    active_ = false;
}

void PlanGAROS2Plugin::cleanup()
{
    RCLCPP_INFO(logger_, "Cleaning up PlanGAROS2Plugin");
    planner_core_.reset();
    local_plan_pub_.reset();
    initialized_ = false;
    active_ = false;
}

void PlanGAROS2Plugin::setPlan(const nav_msgs::msg::Path& path)
{
    if (!initialized_) {
        RCLCPP_ERROR(logger_, "Plugin not initialized!");
        return;
    }

    RCLCPP_INFO(logger_, "Received new global plan with %zu poses", path.poses.size());
    global_plan_ = path;
}

geometry_msgs::msg::TwistStamped PlanGAROS2Plugin::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped& pose,
    const geometry_msgs::msg::Twist& velocity,
    nav2_core::GoalChecker* /*goal_checker*/)
{
    if (!initialized_) {
        throw std::runtime_error("Plugin not initialized!");
    }

    if (!active_) {
        throw std::runtime_error("Plugin not active!");
    }

    // Convert ROS types to planner types
    plan_ga_planner::Pose current_pose;
    convertPose_(pose, current_pose);

    plan_ga_planner::Velocity current_velocity;
    convertVelocity_(velocity, current_velocity);

    // Estimate acceleration
    plan_ga_planner::Acceleration current_acceleration;
    double dt = (clock_->now() - last_time_).seconds();
    if (dt > 0.0) {
        estimateAcceleration_(current_velocity, dt, current_acceleration);
    } else {
        current_acceleration = {0.0, 0.0, 0.0};
    }

    // Get goal pose from global plan
    plan_ga_planner::Pose goal_pose;
    if (!getGoalPose_(goal_pose)) {
        RCLCPP_WARN(logger_, "No valid goal pose available");
        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.stamp = clock_->now();
        cmd_vel.header.frame_id = costmap_ros_->getBaseFrameID();
        return cmd_vel;
    }

    // Check if goal is reached
    if (planner_core_->isGoalReached(current_pose, goal_pose)) {
        RCLCPP_INFO(logger_, "Goal reached!");
        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.stamp = clock_->now();
        cmd_vel.header.frame_id = costmap_ros_->getBaseFrameID();
        return cmd_vel;
    }

    // Convert costmap
    plan_ga_planner::Costmap costmap;
    if (!convertCostmap_(costmap)) {
        RCLCPP_ERROR(logger_, "Failed to convert costmap");
        throw std::runtime_error("Costmap conversion failed");
    }

    // Compute control sequence
    plan_ga_planner::ControlSequence control_sequence;
    if (!planner_core_->computeControlSequence(
            costmap, current_pose, current_velocity,
            current_acceleration, goal_pose, control_sequence))
    {
        RCLCPP_WARN(logger_, "Failed to compute control sequence");
        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.stamp = clock_->now();
        cmd_vel.header.frame_id = costmap_ros_->getBaseFrameID();
        return cmd_vel;
    }

    // Generate and publish trajectory for visualization
    plan_ga_planner::Trajectory trajectory;
    planner_core_->computeTrajectory(
        current_pose, current_velocity, control_sequence, trajectory);
    publishPlan_(trajectory);

    // Extract first command
    if (control_sequence.empty()) {
        RCLCPP_WARN(logger_, "Empty control sequence");
        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.stamp = clock_->now();
        cmd_vel.header.frame_id = costmap_ros_->getBaseFrameID();
        return cmd_vel;
    }

    const auto& first_command = control_sequence[0];

    // Apply speed limit if set
    double v_x = first_command.v_x;
    double v_y = first_command.v_y;
    double omega = first_command.omega;

    if (speed_limit_ > 0.0) {
        double speed = std::sqrt(v_x * v_x + v_y * v_y);
        double max_speed = speed_limit_percentage_ ?
            speed_limit_ * config_.max_v_x : speed_limit_;

        if (speed > max_speed) {
            double scale = max_speed / speed;
            v_x *= scale;
            v_y *= scale;
        }
    }

    // Create velocity command
    geometry_msgs::msg::TwistStamped cmd_vel;
    cmd_vel.header.stamp = clock_->now();
    cmd_vel.header.frame_id = costmap_ros_->getBaseFrameID();
    cmd_vel.twist.linear.x = v_x;
    cmd_vel.twist.linear.y = v_y;
    cmd_vel.twist.angular.z = omega;

    // Update velocity history
    last_velocity_ = current_velocity;
    last_time_ = clock_->now();

    return cmd_vel;
}

void PlanGAROS2Plugin::setSpeedLimit(const double& speed_limit, const bool& percentage)
{
    speed_limit_ = speed_limit;
    speed_limit_percentage_ = percentage;
    RCLCPP_INFO(logger_, "Speed limit set to %.2f %s",
                speed_limit, percentage ? "%" : "m/s");
}

void PlanGAROS2Plugin::loadParameters_()
{
    auto node = node_.lock();
    if (!node) {
        throw std::runtime_error("Unable to lock node!");
    }

    // Model path
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".model_path",
        rclcpp::ParameterValue(std::string("/models/planner_policy.onnx")));
    node->get_parameter(plugin_name_ + ".model_path", config_.model_path);

    // Control parameters
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".num_control_steps",
        rclcpp::ParameterValue(20));
    node->get_parameter(plugin_name_ + ".num_control_steps", config_.num_control_steps);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".control_frequency",
        rclcpp::ParameterValue(10.0));
    node->get_parameter(plugin_name_ + ".control_frequency", config_.control_frequency);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".time_horizon",
        rclcpp::ParameterValue(2.0));
    node->get_parameter(plugin_name_ + ".time_horizon", config_.time_horizon);

    // Velocity limits
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".max_v_x",
        rclcpp::ParameterValue(1.0));
    node->get_parameter(plugin_name_ + ".max_v_x", config_.max_v_x);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".min_v_x",
        rclcpp::ParameterValue(-0.5));
    node->get_parameter(plugin_name_ + ".min_v_x", config_.min_v_x);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".max_v_y",
        rclcpp::ParameterValue(0.5));
    node->get_parameter(plugin_name_ + ".max_v_y", config_.max_v_y);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".max_omega",
        rclcpp::ParameterValue(2.0));
    node->get_parameter(plugin_name_ + ".max_omega", config_.max_omega);

    // Acceleration limits
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".max_acc_x",
        rclcpp::ParameterValue(2.5));
    node->get_parameter(plugin_name_ + ".max_acc_x", config_.max_acc_x);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".max_acc_y",
        rclcpp::ParameterValue(2.5));
    node->get_parameter(plugin_name_ + ".max_acc_y", config_.max_acc_y);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".max_acc_theta",
        rclcpp::ParameterValue(3.2));
    node->get_parameter(plugin_name_ + ".max_acc_theta", config_.max_acc_theta);

    // Costmap parameters
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".costmap_window_size",
        rclcpp::ParameterValue(50));
    node->get_parameter(plugin_name_ + ".costmap_window_size", config_.costmap_window_size);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".lethal_cost_threshold",
        rclcpp::ParameterValue(253));
    int lethal_threshold;
    node->get_parameter(plugin_name_ + ".lethal_cost_threshold", lethal_threshold);
    config_.lethal_cost_threshold = static_cast<uint8_t>(lethal_threshold);

    // Goal tolerance
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".xy_goal_tolerance",
        rclcpp::ParameterValue(0.1));
    node->get_parameter(plugin_name_ + ".xy_goal_tolerance", config_.xy_goal_tolerance);

    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".yaw_goal_tolerance",
        rclcpp::ParameterValue(0.1));
    node->get_parameter(plugin_name_ + ".yaw_goal_tolerance", config_.yaw_goal_tolerance);

    // Collision checking
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".enable_collision_checking",
        rclcpp::ParameterValue(true));
    node->get_parameter(plugin_name_ + ".enable_collision_checking",
                       config_.enable_collision_checking);

    // Footprint (simplified - assume square robot)
    nav2_util::declare_parameter_if_not_declared(
        node, plugin_name_ + ".robot_radius",
        rclcpp::ParameterValue(0.3));
    double robot_radius;
    node->get_parameter(plugin_name_ + ".robot_radius", robot_radius);

    // Create square footprint
    config_.footprint = {
        {robot_radius, robot_radius},
        {-robot_radius, robot_radius},
        {-robot_radius, -robot_radius},
        {robot_radius, -robot_radius}
    };

    RCLCPP_INFO(logger_, "Loaded parameters:");
    RCLCPP_INFO(logger_, "  model_path: %s", config_.model_path.c_str());
    RCLCPP_INFO(logger_, "  num_control_steps: %d", config_.num_control_steps);
    RCLCPP_INFO(logger_, "  control_frequency: %.1f Hz", config_.control_frequency);
    RCLCPP_INFO(logger_, "  max_v_x: %.2f m/s", config_.max_v_x);
}

bool PlanGAROS2Plugin::convertCostmap_(plan_ga_planner::Costmap& costmap)
{
    auto* costmap_2d = costmap_ros_->getCostmap();
    if (!costmap_2d) {
        RCLCPP_ERROR(logger_, "Costmap is null");
        return false;
    }

    // Copy costmap data
    unsigned int size_x = costmap_2d->getSizeInCellsX();
    unsigned int size_y = costmap_2d->getSizeInCellsY();
    unsigned char* data = costmap_2d->getCharMap();

    costmap.width = static_cast<int>(size_x);
    costmap.height = static_cast<int>(size_y);
    costmap.resolution = costmap_2d->getResolution();
    costmap.origin_x = costmap_2d->getOriginX();
    costmap.origin_y = costmap_2d->getOriginY();

    // Note: ROS2 costmap may use different cost semantics
    // nav2_costmap_2d uses: FREE_SPACE=0, INSCRIBED=253, LETHAL=254, NO_INFORMATION=255
    costmap.inflation_decay = 0.5;  // Default value

    // Copy data
    costmap.data.resize(size_x * size_y);
    std::copy(data, data + size_x * size_y, costmap.data.begin());

    return true;
}

void PlanGAROS2Plugin::convertPose_(
    const geometry_msgs::msg::PoseStamped& pose_stamped,
    plan_ga_planner::Pose& pose)
{
    pose[0] = pose_stamped.pose.position.x;
    pose[1] = pose_stamped.pose.position.y;
    pose[2] = tf2::getYaw(pose_stamped.pose.orientation);
}

void PlanGAROS2Plugin::convertVelocity_(
    const geometry_msgs::msg::Twist& twist,
    plan_ga_planner::Velocity& velocity)
{
    velocity[0] = twist.linear.x;
    velocity[1] = twist.linear.y;
    velocity[2] = twist.angular.z;
}

void PlanGAROS2Plugin::estimateAcceleration_(
    const plan_ga_planner::Velocity& current_velocity,
    double dt,
    plan_ga_planner::Acceleration& acceleration)
{
    if (dt <= 0.0) {
        acceleration = {0.0, 0.0, 0.0};
        return;
    }

    acceleration[0] = (current_velocity[0] - last_velocity_[0]) / dt;
    acceleration[1] = (current_velocity[1] - last_velocity_[1]) / dt;
    acceleration[2] = (current_velocity[2] - last_velocity_[2]) / dt;

    // Clamp to limits
    acceleration[0] = std::clamp(acceleration[0], -config_.max_acc_x, config_.max_acc_x);
    acceleration[1] = std::clamp(acceleration[1], -config_.max_acc_y, config_.max_acc_y);
    acceleration[2] = std::clamp(acceleration[2], -config_.max_acc_theta, config_.max_acc_theta);
}

void PlanGAROS2Plugin::publishPlan_(const plan_ga_planner::Trajectory& trajectory)
{
    if (!local_plan_pub_->is_activated()) {
        return;
    }

    nav_msgs::msg::Path path;
    path.header.stamp = clock_->now();
    path.header.frame_id = costmap_ros_->getGlobalFrameID();

    for (const auto& point : trajectory) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = path.header;
        pose.pose.position.x = point.pose[0];
        pose.pose.position.y = point.pose[1];
        pose.pose.position.z = 0.0;

        // Convert yaw to quaternion
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, point.pose[2]);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();

        path.poses.push_back(pose);
    }

    local_plan_pub_->publish(path);
}

bool PlanGAROS2Plugin::getGoalPose_(plan_ga_planner::Pose& goal_pose)
{
    if (global_plan_.poses.empty()) {
        return false;
    }

    // Use last pose in global plan as goal
    const auto& goal_stamped = global_plan_.poses.back();
    convertPose_(goal_stamped, goal_pose);

    return true;
}

}  // namespace plan_ga_ros2
