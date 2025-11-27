#ifndef PLAN_GA_PLANNER_PLANNER_CORE_H
#define PLAN_GA_PLANNER_PLANNER_CORE_H

#include <memory>
#include <string>
#include "plan_ga_planner/types.h"
#include "plan_ga_planner/onnx_inference.h"
#include "plan_ga_planner/costmap_processor.h"
#include "plan_ga_planner/trajectory_generator.h"
#include "plan_ga_planner/collision_checker.h"

namespace plan_ga_planner {

/**
 * @brief Core planning logic independent of ROS version
 *
 * This class encapsulates the main planning algorithm using ONNX inference.
 * It processes costmap data, runs the neural network, and generates control sequences.
 */
class PlannerCore {
public:
    /**
     * @brief Constructor
     *
     * @param model_path Path to ONNX model file
     */
    explicit PlannerCore(const std::string& model_path);

    /**
     * @brief Destructor
     */
    ~PlannerCore();

    /**
     * @brief Initialize planner with configuration
     *
     * @param config Planning configuration parameters
     * @return true if initialization successful
     */
    bool initialize(const PlannerConfig& config);

    /**
     * @brief Compute control sequence for current state
     *
     * @param costmap Local costmap (can be any size, will be windowed)
     * @param current_pose Robot pose [x, y, theta]
     * @param current_velocity Robot velocity [v_x, v_y, omega]
     * @param current_acceleration Robot acceleration [a_x, a_y, alpha]
     * @param goal_pose Goal pose [x, y, theta]
     * @param control_sequence Output control sequence
     * @return true if planning successful
     */
    bool computeControlSequence(
        const Costmap& costmap,
        const Pose& current_pose,
        const Velocity& current_velocity,
        const Acceleration& current_acceleration,
        const Pose& goal_pose,
        ControlSequence& control_sequence);

    /**
     * @brief Compute full trajectory from control sequence
     *
     * @param current_pose Starting pose
     * @param current_velocity Starting velocity
     * @param control_sequence Control commands
     * @param trajectory Output trajectory
     */
    void computeTrajectory(
        const Pose& current_pose,
        const Velocity& current_velocity,
        const ControlSequence& control_sequence,
        Trajectory& trajectory);

    /**
     * @brief Check if trajectory is collision-free
     *
     * @param costmap Costmap for collision checking
     * @param trajectory Trajectory to validate
     * @return true if no collisions detected
     */
    bool isTrajectoryValid(
        const Costmap& costmap,
        const Trajectory& trajectory) const;

    /**
     * @brief Check if goal is reached
     *
     * @param current_pose Current robot pose
     * @param goal_pose Goal pose
     * @return true if within goal tolerance
     */
    bool isGoalReached(
        const Pose& current_pose,
        const Pose& goal_pose) const;

    /**
     * @brief Get planner configuration
     */
    const PlannerConfig& getConfig() const { return config_; }

    /**
     * @brief Check if planner is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    /**
     * @brief Prepare input tensors for ONNX inference
     */
    bool prepareInputs_(
        const Costmap& costmap,
        const Pose& current_pose,
        const Velocity& current_velocity,
        const Acceleration& current_acceleration,
        const Pose& goal_pose,
        std::vector<float>& costmap_input,
        std::vector<float>& robot_state_input,
        std::vector<float>& goal_relative_input,
        std::vector<float>& costmap_metadata_input);

    /**
     * @brief Convert ONNX output to control sequence
     */
    void parseOutput_(
        const std::vector<float>& onnx_output,
        ControlSequence& control_sequence);

    /**
     * @brief Compute relative goal pose in robot frame
     */
    Pose computeRelativeGoal_(
        const Pose& current_pose,
        const Pose& goal_pose) const;

    /**
     * @brief Normalize angle to [-pi, pi]
     */
    double normalizeAngle_(double angle) const;

    /**
     * @brief Clip control commands to velocity limits
     */
    void clipControls_(ControlSequence& control_sequence);

    PlannerConfig config_;
    bool initialized_;

    std::unique_ptr<ONNXInference> onnx_inference_;
    std::unique_ptr<CostmapProcessor> costmap_processor_;
    std::unique_ptr<TrajectoryGenerator> trajectory_generator_;
    std::unique_ptr<CollisionChecker> collision_checker_;
};

}  // namespace plan_ga_planner

#endif  // PLAN_GA_PLANNER_PLANNER_CORE_H
