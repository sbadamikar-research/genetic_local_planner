#include "plan_ga_planner/planner_core.h"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace plan_ga_planner {

PlannerCore::PlannerCore(const std::string& model_path)
    : initialized_(false) {
    try {
        onnx_inference_ = std::make_unique<ONNXInference>(model_path);
    } catch (const std::exception& e) {
        std::cerr << "[PlannerCore] Failed to load ONNX model: " << e.what() << std::endl;
        onnx_inference_ = nullptr;
    }
}

PlannerCore::~PlannerCore() = default;

bool PlannerCore::initialize(const PlannerConfig& config) {
    config_ = config;

    // Verify ONNX model is loaded
    if (!onnx_inference_ || !onnx_inference_->isLoaded()) {
        std::cerr << "[PlannerCore] ONNX model not loaded" << std::endl;
        return false;
    }

    // Create components
    costmap_processor_ = std::make_unique<CostmapProcessor>(config_.costmap_window_size);
    trajectory_generator_ = std::make_unique<TrajectoryGenerator>();
    collision_checker_ = std::make_unique<CollisionChecker>(
        config_.footprint,
        config_.lethal_cost_threshold
    );

    initialized_ = true;
    std::cout << "[PlannerCore] Initialized successfully" << std::endl;

    return true;
}

bool PlannerCore::computeControlSequence(
    const Costmap& costmap,
    const Pose& current_pose,
    const Velocity& current_velocity,
    const Acceleration& current_acceleration,
    const Pose& goal_pose,
    ControlSequence& control_sequence) {

    if (!initialized_) {
        std::cerr << "[PlannerCore] Planner not initialized" << std::endl;
        return false;
    }

    // Prepare inputs
    std::vector<float> costmap_input;
    std::vector<float> robot_state_input;
    std::vector<float> goal_relative_input;
    std::vector<float> costmap_metadata_input;

    if (!prepareInputs_(costmap, current_pose, current_velocity, current_acceleration,
                       goal_pose, costmap_input, robot_state_input,
                       goal_relative_input, costmap_metadata_input)) {
        std::cerr << "[PlannerCore] Failed to prepare inputs" << std::endl;
        return false;
    }

    // Run ONNX inference
    std::vector<float> onnx_output;
    if (!onnx_inference_->infer(costmap_input, robot_state_input,
                                goal_relative_input, costmap_metadata_input,
                                onnx_output)) {
        std::cerr << "[PlannerCore] ONNX inference failed" << std::endl;
        return false;
    }

    // Parse output to control sequence
    parseOutput_(onnx_output, control_sequence);

    // Clip to velocity limits
    clipControls_(control_sequence);

    // Optional: Collision check
    if (config_.enable_collision_check) {
        Trajectory trajectory;
        computeTrajectory(current_pose, current_velocity, control_sequence, trajectory);

        if (!isTrajectoryValid(costmap, trajectory)) {
            if (config_.debug_mode) {
                std::cerr << "[PlannerCore] WARNING: Generated trajectory has collisions" << std::endl;
            }
            // Could implement recovery behavior here
            // For now, still return the trajectory (letting ROS handle recovery)
        }
    }

    return true;
}

void PlannerCore::computeTrajectory(
    const Pose& current_pose,
    const Velocity& current_velocity,
    const ControlSequence& control_sequence,
    Trajectory& trajectory) {

    trajectory_generator_->generateTrajectory(
        current_pose,
        current_velocity,
        control_sequence,
        trajectory
    );
}

bool PlannerCore::isTrajectoryValid(
    const Costmap& costmap,
    const Trajectory& trajectory) const {

    return collision_checker_->isTrajectoryValid(trajectory, costmap);
}

bool PlannerCore::isGoalReached(
    const Pose& current_pose,
    const Pose& goal_pose) const {

    // Check position tolerance
    double dx = current_pose[0] - goal_pose[0];
    double dy = current_pose[1] - goal_pose[1];
    double dist = std::sqrt(dx * dx + dy * dy);

    if (dist > config_.xy_goal_tolerance) {
        return false;
    }

    // Check orientation tolerance
    double dtheta = std::abs(normalizeAngle_(current_pose[2] - goal_pose[2]));
    if (dtheta > config_.yaw_goal_tolerance) {
        return false;
    }

    return true;
}

bool PlannerCore::prepareInputs_(
    const Costmap& costmap,
    const Pose& current_pose,
    const Velocity& current_velocity,
    const Acceleration& current_acceleration,
    const Pose& goal_pose,
    std::vector<float>& costmap_input,
    std::vector<float>& robot_state_input,
    std::vector<float>& goal_relative_input,
    std::vector<float>& costmap_metadata_input) {

    // 1. Process costmap window
    if (!costmap_processor_->processWindow(costmap, current_pose, costmap_input)) {
        return false;
    }

    // 2. Robot state: [x, y, theta, v_x, v_y, omega, a_x, a_y, alpha]
    robot_state_input = {
        static_cast<float>(current_pose[0]),
        static_cast<float>(current_pose[1]),
        static_cast<float>(current_pose[2]),
        static_cast<float>(current_velocity[0]),
        static_cast<float>(current_velocity[1]),
        static_cast<float>(current_velocity[2]),
        static_cast<float>(current_acceleration[0]),
        static_cast<float>(current_acceleration[1]),
        static_cast<float>(current_acceleration[2])
    };

    // 3. Relative goal
    Pose relative_goal = computeRelativeGoal_(current_pose, goal_pose);
    goal_relative_input = {
        static_cast<float>(relative_goal[0]),
        static_cast<float>(relative_goal[1]),
        static_cast<float>(relative_goal[2])
    };

    // 4. Costmap metadata: [inflation_decay, resolution]
    costmap_metadata_input = {
        static_cast<float>(costmap.inflation_decay),
        static_cast<float>(costmap.resolution)
    };

    return true;
}

void PlannerCore::parseOutput_(
    const std::vector<float>& onnx_output,
    ControlSequence& control_sequence) {

    control_sequence.clear();

    int num_steps = config_.num_control_steps;
    double dt = 1.0 / config_.control_frequency;

    for (int i = 0; i < num_steps; ++i) {
        int idx = i * 3;
        if (idx + 2 < static_cast<int>(onnx_output.size())) {
            ControlCommand cmd;
            cmd.v_x = onnx_output[idx];
            cmd.v_y = onnx_output[idx + 1];
            cmd.omega = onnx_output[idx + 2];
            cmd.dt = dt;

            control_sequence.push_back(cmd);
        }
    }
}

Pose PlannerCore::computeRelativeGoal_(
    const Pose& current_pose,
    const Pose& goal_pose) const {

    // Transform goal to robot frame
    double dx = goal_pose[0] - current_pose[0];
    double dy = goal_pose[1] - current_pose[1];

    double cos_theta = std::cos(-current_pose[2]);
    double sin_theta = std::sin(-current_pose[2]);

    Pose relative_goal;
    relative_goal[0] = dx * cos_theta - dy * sin_theta;
    relative_goal[1] = dx * sin_theta + dy * cos_theta;
    relative_goal[2] = normalizeAngle_(goal_pose[2] - current_pose[2]);

    return relative_goal;
}

double PlannerCore::normalizeAngle_(double angle) const {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

void PlannerCore::clipControls_(ControlSequence& control_sequence) {
    for (auto& cmd : control_sequence) {
        cmd.v_x = std::max(config_.min_v_x, std::min(config_.max_v_x, cmd.v_x));
        cmd.v_y = std::max(config_.min_v_y, std::min(config_.max_v_y, cmd.v_y));
        cmd.omega = std::max(config_.min_omega, std::min(config_.max_omega, cmd.omega));
    }
}

}  // namespace plan_ga_planner
