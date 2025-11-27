#ifndef PLAN_GA_PLANNER_COLLISION_CHECKER_H
#define PLAN_GA_PLANNER_COLLISION_CHECKER_H

#include "plan_ga_planner/types.h"

namespace plan_ga_planner {

/**
 * @brief Checks trajectories for collisions
 *
 * Uses robot footprint and costmap to validate trajectories.
 */
class CollisionChecker {
public:
    /**
     * @brief Constructor
     *
     * @param footprint Robot footprint polygon
     * @param lethal_threshold Costmap value considered lethal
     */
    CollisionChecker(const FootprintPolygon& footprint, uint8_t lethal_threshold = 253);

    /**
     * @brief Check if trajectory is collision-free
     *
     * @param trajectory Trajectory to check
     * @param costmap Costmap for collision checking
     * @return true if no collisions detected
     */
    bool isTrajectoryValid(
        const Trajectory& trajectory,
        const Costmap& costmap);

    /**
     * @brief Check if single pose is collision-free
     *
     * @param pose Robot pose
     * @param costmap Costmap
     * @return true if no collision
     */
    bool isPoseValid(
        const Pose& pose,
        const Costmap& costmap);

    /**
     * @brief Get maximum cost along trajectory
     *
     * @param trajectory Trajectory
     * @param costmap Costmap
     * @return Maximum cost value encountered
     */
    uint8_t getMaxCost(
        const Trajectory& trajectory,
        const Costmap& costmap);

private:
    /**
     * @brief Check footprint at given pose
     *
     * @param pose Robot pose
     * @param costmap Costmap
     * @param max_cost Output maximum cost in footprint
     * @return true if valid (below lethal threshold)
     */
    bool checkFootprint_(
        const Pose& pose,
        const Costmap& costmap,
        uint8_t& max_cost);

    /**
     * @brief Transform footprint vertex to world frame
     */
    void transformFootprintVertex_(
        const Pose& pose,
        double local_x,
        double local_y,
        double& world_x,
        double& world_y);

    FootprintPolygon footprint_;
    uint8_t lethal_threshold_;
};

}  // namespace plan_ga_planner

#endif  // PLAN_GA_PLANNER_COLLISION_CHECKER_H
