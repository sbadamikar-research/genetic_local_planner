#ifndef PLAN_GA_PLANNER_TRAJECTORY_GENERATOR_H
#define PLAN_GA_PLANNER_TRAJECTORY_GENERATOR_H

#include "plan_ga_planner/types.h"

namespace plan_ga_planner {

/**
 * @brief Generates trajectory from control sequence
 *
 * Forward simulates robot motion given control commands.
 */
class TrajectoryGenerator {
public:
    /**
     * @brief Constructor
     */
    TrajectoryGenerator();

    /**
     * @brief Generate trajectory from control sequence
     *
     * @param current_pose Starting pose
     * @param current_velocity Starting velocity
     * @param control_sequence Control commands
     * @param trajectory Output trajectory
     */
    void generateTrajectory(
        const Pose& current_pose,
        const Velocity& current_velocity,
        const ControlSequence& control_sequence,
        Trajectory& trajectory);

private:
    /**
     * @brief Integrate one timestep with differential drive dynamics
     *
     * @param pose Current pose [x, y, theta]
     * @param velocity Current velocity [v_x, v_y, omega]
     * @param command Control command
     * @param next_pose Output next pose
     * @param next_velocity Output next velocity
     */
    void integrateStep_(
        const Pose& pose,
        const Velocity& velocity,
        const ControlCommand& command,
        Pose& next_pose,
        Velocity& next_velocity);

    /**
     * @brief Normalize angle to [-pi, pi]
     */
    double normalizeAngle_(double angle);
};

}  // namespace plan_ga_planner

#endif  // PLAN_GA_PLANNER_TRAJECTORY_GENERATOR_H
