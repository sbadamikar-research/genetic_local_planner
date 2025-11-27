#ifndef PLAN_GA_ROS2_PLUGIN_HPP
#define PLAN_GA_ROS2_PLUGIN_HPP

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "tf2_ros/buffer.h"

#include "plan_ga_planner/planner_core.h"
#include "plan_ga_planner/types.h"

namespace plan_ga_ros2 {

/**
 * @class PlanGAROS2Plugin
 * @brief ROS2 Nav2 controller plugin using GA-trained neural network policy
 *
 * Implements nav2_core::Controller interface for integration with Nav2 stack.
 * Uses ONNX Runtime for real-time inference of learned navigation policies.
 */
class PlanGAROS2Plugin : public nav2_core::Controller {
public:
    PlanGAROS2Plugin() = default;
    ~PlanGAROS2Plugin() override = default;

    /**
     * @brief Configure the controller
     * @param parent Lifecycle node pointer
     * @param name Name of the controller
     * @param tf TF buffer pointer
     * @param costmap_ros Costmap ROS wrapper
     */
    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

    /**
     * @brief Activate the controller (lifecycle)
     */
    void activate() override;

    /**
     * @brief Deactivate the controller (lifecycle)
     */
    void deactivate() override;

    /**
     * @brief Cleanup the controller (lifecycle)
     */
    void cleanup() override;

    /**
     * @brief Set the global plan
     * @param path Global plan to follow
     */
    void setPlan(const nav_msgs::msg::Path& path) override;

    /**
     * @brief Compute velocity commands
     * @param pose Current robot pose
     * @param velocity Current robot velocity
     * @param goal_checker Goal checker for determining goal reached
     * @return TwistStamped velocity command
     */
    geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped& pose,
        const geometry_msgs::msg::Twist& velocity,
        nav2_core::GoalChecker* goal_checker) override;

    /**
     * @brief Set the maximum speed for the controller
     * @param speed_limit Maximum speed
     * @param percentage Whether speed_limit is a percentage
     */
    void setSpeedLimit(const double& speed_limit, const bool& percentage) override;

private:
    /**
     * @brief Load parameters from ROS2 parameter server
     */
    void loadParameters_();

    /**
     * @brief Convert ROS2 costmap to planner format
     * @param costmap Output costmap in planner format
     * @return True if successful
     */
    bool convertCostmap_(plan_ga_planner::Costmap& costmap);

    /**
     * @brief Convert ROS2 pose to planner pose
     * @param pose_stamped ROS2 PoseStamped
     * @param pose Output planner pose [x, y, theta]
     */
    void convertPose_(
        const geometry_msgs::msg::PoseStamped& pose_stamped,
        plan_ga_planner::Pose& pose);

    /**
     * @brief Convert ROS2 twist to planner velocity
     * @param twist ROS2 Twist
     * @param velocity Output planner velocity [v_x, v_y, omega]
     */
    void convertVelocity_(
        const geometry_msgs::msg::Twist& twist,
        plan_ga_planner::Velocity& velocity);

    /**
     * @brief Estimate acceleration from velocity history
     * @param current_velocity Current velocity
     * @param dt Time since last update
     * @param acceleration Output acceleration
     */
    void estimateAcceleration_(
        const plan_ga_planner::Velocity& current_velocity,
        double dt,
        plan_ga_planner::Acceleration& acceleration);

    /**
     * @brief Publish local plan for visualization
     * @param trajectory Planned trajectory
     */
    void publishPlan_(const plan_ga_planner::Trajectory& trajectory);

    /**
     * @brief Get goal pose from global plan
     * @param goal_pose Output goal pose
     * @return True if goal found
     */
    bool getGoalPose_(plan_ga_planner::Pose& goal_pose);

    // Core planner (ROS-agnostic)
    std::unique_ptr<plan_ga_planner::PlannerCore> planner_core_;

    // ROS2 interfaces
    rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    rclcpp::Logger logger_{rclcpp::get_logger("PlanGAROS2Plugin")};
    rclcpp::Clock::SharedPtr clock_;

    // Publishers
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr local_plan_pub_;

    // Plugin state
    std::string plugin_name_;
    bool initialized_;
    bool active_;

    // Global plan
    nav_msgs::msg::Path global_plan_;

    // Configuration parameters
    plan_ga_planner::PlannerConfig config_;

    // Velocity history for acceleration estimation
    plan_ga_planner::Velocity last_velocity_;
    rclcpp::Time last_time_;

    // Speed limit
    double speed_limit_;
    bool speed_limit_percentage_;
};

}  // namespace plan_ga_ros2

#endif  // PLAN_GA_ROS2_PLUGIN_HPP
