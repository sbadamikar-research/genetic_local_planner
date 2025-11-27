#ifndef PLAN_GA_ROS1_PLUGIN_H
#define PLAN_GA_ROS1_PLUGIN_H

#include <nav_core/base_local_planner.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include <memory>
#include <vector>
#include "plan_ga_planner/planner_core.h"

namespace plan_ga_ros1 {

/**
 * @brief ROS1 Noetic plugin wrapper implementing nav_core::BaseLocalPlanner
 *
 * Integrates PlannerCore with ROS1 navigation stack.
 */
class PlanGAROS1Plugin : public nav_core::BaseLocalPlanner {
public:
    /**
     * @brief Constructor
     */
    PlanGAROS1Plugin();

    /**
     * @brief Destructor
     */
    ~PlanGAROS1Plugin() override;

    /**
     * @brief Initialize the plugin
     *
     * @param name The name of this planner
     * @param tf A pointer to a transform listener
     * @param costmap_ros The cost map to use for planning
     */
    void initialize(std::string name, tf2_ros::Buffer* tf,
                   costmap_2d::Costmap2DROS* costmap_ros) override;

    /**
     * @brief Set the global plan for the local planner
     *
     * @param plan The plan to pass to the local planner
     * @return True if the plan was updated successfully
     */
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) override;

    /**
     * @brief Compute velocity commands to follow the global plan
     *
     * @param cmd_vel Will be filled with velocity command
     * @return True if a valid velocity command was computed
     */
    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override;

    /**
     * @brief Check if the goal pose has been achieved
     *
     * @return True if achieved
     */
    bool isGoalReached() override;

private:
    /**
     * @brief Convert ROS costmap to planner costmap format
     */
    bool convertCostmap_(plan_ga_planner::Costmap& costmap);

    /**
     * @brief Get current robot pose
     */
    bool getCurrentPose_(plan_ga_planner::Pose& pose);

    /**
     * @brief Get current robot velocity
     */
    bool getCurrentVelocity_(plan_ga_planner::Velocity& velocity);

    /**
     * @brief Estimate current acceleration from velocity history
     */
    bool getCurrentAcceleration_(plan_ga_planner::Acceleration& acceleration);

    /**
     * @brief Publish local plan for visualization
     */
    void publishPlan_(const plan_ga_planner::Trajectory& trajectory);

    /**
     * @brief Load parameters from parameter server
     */
    void loadParameters_();

    bool initialized_;
    std::string name_;
    tf2_ros::Buffer* tf_;
    costmap_2d::Costmap2DROS* costmap_ros_;

    std::unique_ptr<plan_ga_planner::PlannerCore> planner_core_;

    std::vector<geometry_msgs::PoseStamped> global_plan_;
    plan_ga_planner::ControlSequence current_control_sequence_;
    size_t control_index_;

    ros::Publisher plan_pub_;
    ros::NodeHandle private_nh_;

    // Velocity tracking for acceleration estimation
    plan_ga_planner::Velocity prev_velocity_;
    ros::Time prev_time_;
    bool have_prev_velocity_;
};

}  // namespace plan_ga_ros1

#endif  // PLAN_GA_ROS1_PLUGIN_H
