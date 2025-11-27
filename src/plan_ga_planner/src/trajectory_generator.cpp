#include "plan_ga_planner/trajectory_generator.h"
#include <cmath>

namespace plan_ga_planner {

TrajectoryGenerator::TrajectoryGenerator() {}

void TrajectoryGenerator::generateTrajectory(
    const Pose& current_pose,
    const Velocity& current_velocity,
    const ControlSequence& control_sequence,
    Trajectory& trajectory) {

    trajectory.clear();
    trajectory.reserve(control_sequence.size() + 1);

    // Add starting point
    TrajectoryPoint start_point;
    start_point.pose = current_pose;
    start_point.velocity = current_velocity;
    start_point.time_from_start = 0.0;
    trajectory.push_back(start_point);

    // Forward simulate
    Pose pose = current_pose;
    Velocity velocity = current_velocity;
    double time = 0.0;

    for (const auto& command : control_sequence) {
        Pose next_pose;
        Velocity next_velocity;

        // Integrate one step
        integrateStep_(pose, velocity, command, next_pose, next_velocity);

        // Add to trajectory
        time += command.dt;
        TrajectoryPoint point;
        point.pose = next_pose;
        point.velocity = next_velocity;
        point.time_from_start = time;
        trajectory.push_back(point);

        // Update for next iteration
        pose = next_pose;
        velocity = next_velocity;
    }
}

void TrajectoryGenerator::integrateStep_(
    const Pose& pose,
    const Velocity& velocity,
    const ControlCommand& command,
    Pose& next_pose,
    Velocity& next_velocity) {

    // Simple Euler integration
    // For more accuracy, could use RK4 or similar

    double x = pose[0];
    double y = pose[1];
    double theta = pose[2];

    double v_x = command.v_x;
    double v_y = command.v_y;
    double omega = command.omega;
    double dt = command.dt;

    // Update orientation first
    double next_theta = theta + omega * dt;
    next_theta = normalizeAngle_(next_theta);

    // Update position (in global frame)
    // Robot velocities are in robot frame, transform to global
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    double vx_global = v_x * cos_theta - v_y * sin_theta;
    double vy_global = v_x * sin_theta + v_y * cos_theta;

    double next_x = x + vx_global * dt;
    double next_y = y + vy_global * dt;

    // Set next pose
    next_pose[0] = next_x;
    next_pose[1] = next_y;
    next_pose[2] = next_theta;

    // Set next velocity (assumes instantaneous velocity changes)
    // For more realistic dynamics, would model acceleration
    next_velocity[0] = v_x;
    next_velocity[1] = v_y;
    next_velocity[2] = omega;
}

double TrajectoryGenerator::normalizeAngle_(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

}  // namespace plan_ga_planner
