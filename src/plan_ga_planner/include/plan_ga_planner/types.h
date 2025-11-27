#ifndef PLAN_GA_PLANNER_TYPES_H
#define PLAN_GA_PLANNER_TYPES_H

#include <vector>
#include <array>
#include <cstdint>
#include <string>

namespace plan_ga_planner {

// Pose: [x, y, theta]
using Pose = std::array<double, 3>;

// Velocity: [v_x, v_y, omega]
using Velocity = std::array<double, 3>;

// Acceleration: [a_x, a_y, alpha]
using Acceleration = std::array<double, 3>;

/**
 * @brief Single control command
 */
struct ControlCommand {
    double v_x;     // Linear velocity x (m/s)
    double v_y;     // Linear velocity y (m/s)
    double omega;   // Angular velocity (rad/s)
    double dt;      // Time step (s)

    ControlCommand() : v_x(0.0), v_y(0.0), omega(0.0), dt(0.1) {}

    ControlCommand(double vx, double vy, double w, double time_step = 0.1)
        : v_x(vx), v_y(vy), omega(w), dt(time_step) {}
};

// Sequence of control commands
using ControlSequence = std::vector<ControlCommand>;

/**
 * @brief Costmap representation
 */
struct Costmap {
    std::vector<uint8_t> data;  // Flattened costmap values (0-255)
    int width;                  // Grid width (pixels)
    int height;                 // Grid height (pixels)
    double resolution;          // Meters per cell
    double origin_x;            // World coordinates of (0,0) cell
    double origin_y;
    double inflation_decay;     // Inflation decay factor

    Costmap() : width(0), height(0), resolution(0.05),
                origin_x(0.0), origin_y(0.0), inflation_decay(0.8) {}

    /**
     * @brief Get cost at specific grid cell
     * @param x Grid x coordinate
     * @param y Grid y coordinate
     * @return Cost value (0-255), or 0 if out of bounds
     */
    uint8_t getCost(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 0;
        }
        return data[y * width + x];
    }

    /**
     * @brief Set cost at specific grid cell
     */
    void setCost(int x, int y, uint8_t cost) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[y * width + x] = cost;
        }
    }

    /**
     * @brief Convert world coordinates to grid coordinates
     */
    void worldToGrid(double wx, double wy, int& mx, int& my) const {
        mx = static_cast<int>((wx - origin_x) / resolution);
        my = static_cast<int>((wy - origin_y) / resolution);
    }

    /**
     * @brief Convert grid coordinates to world coordinates
     */
    void gridToWorld(int mx, int my, double& wx, double& wy) const {
        wx = origin_x + (mx + 0.5) * resolution;
        wy = origin_y + (my + 0.5) * resolution;
    }
};

/**
 * @brief Trajectory point
 */
struct TrajectoryPoint {
    Pose pose;
    Velocity velocity;
    double time_from_start;

    TrajectoryPoint() : time_from_start(0.0) {
        pose.fill(0.0);
        velocity.fill(0.0);
    }
};

// Full trajectory
using Trajectory = std::vector<TrajectoryPoint>;

// Robot footprint polygon (vertices in robot frame)
using FootprintPolygon = std::vector<std::array<double, 2>>;

/**
 * @brief Planner configuration
 */
struct PlannerConfig {
    // Model parameters
    std::string model_path;
    int num_control_steps;
    double control_frequency;
    double time_horizon;

    // Velocity limits
    double max_v_x;
    double min_v_x;
    double max_v_y;
    double min_v_y;
    double max_omega;
    double min_omega;

    // Costmap parameters
    int costmap_window_size;

    // Safety parameters
    uint8_t lethal_cost_threshold;
    bool enable_collision_check;

    // Robot footprint
    FootprintPolygon footprint;

    // Goal tolerance
    double xy_goal_tolerance;
    double yaw_goal_tolerance;

    // Debug
    bool debug_mode;
    bool publish_local_plan;

    /**
     * @brief Constructor with default values
     */
    PlannerConfig()
        : model_path("/models/planner_policy.onnx"),
          num_control_steps(20),
          control_frequency(10.0),
          time_horizon(2.0),
          max_v_x(1.0),
          min_v_x(-0.5),
          max_v_y(0.5),
          min_v_y(-0.5),
          max_omega(1.0),
          min_omega(-1.0),
          costmap_window_size(50),
          lethal_cost_threshold(253),
          enable_collision_check(true),
          xy_goal_tolerance(0.1),
          yaw_goal_tolerance(0.1),
          debug_mode(false),
          publish_local_plan(true) {
        // Default square footprint (0.4m x 0.4m)
        footprint = {
            {{-0.2, -0.2}},
            {{0.2, -0.2}},
            {{0.2, 0.2}},
            {{-0.2, 0.2}}
        };
    }
};

}  // namespace plan_ga_planner

#endif  // PLAN_GA_PLANNER_TYPES_H
