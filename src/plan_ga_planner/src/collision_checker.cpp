#include "plan_ga_planner/collision_checker.h"
#include <cmath>
#include <algorithm>

namespace plan_ga_planner {

CollisionChecker::CollisionChecker(const FootprintPolygon& footprint, uint8_t lethal_threshold)
    : footprint_(footprint), lethal_threshold_(lethal_threshold) {}

bool CollisionChecker::isTrajectoryValid(
    const Trajectory& trajectory,
    const Costmap& costmap) {

    for (const auto& point : trajectory) {
        if (!isPoseValid(point.pose, costmap)) {
            return false;
        }
    }

    return true;
}

bool CollisionChecker::isPoseValid(
    const Pose& pose,
    const Costmap& costmap) {

    uint8_t max_cost;
    return checkFootprint_(pose, costmap, max_cost);
}

uint8_t CollisionChecker::getMaxCost(
    const Trajectory& trajectory,
    const Costmap& costmap) {

    uint8_t max_cost = 0;

    for (const auto& point : trajectory) {
        uint8_t cost;
        checkFootprint_(point.pose, costmap, cost);
        max_cost = std::max(max_cost, cost);
    }

    return max_cost;
}

bool CollisionChecker::checkFootprint_(
    const Pose& pose,
    const Costmap& costmap,
    uint8_t& max_cost) {

    max_cost = 0;

    // Check each vertex of the footprint
    for (const auto& vertex : footprint_) {
        double world_x, world_y;
        transformFootprintVertex_(pose, vertex[0], vertex[1], world_x, world_y);

        // Convert to grid coordinates
        int mx, my;
        costmap.worldToGrid(world_x, world_y, mx, my);

        // Get cost
        uint8_t cost = costmap.getCost(mx, my);
        max_cost = std::max(max_cost, cost);

        // Check if lethal
        if (cost >= lethal_threshold_) {
            return false;
        }
    }

    // Also check center point
    int center_mx, center_my;
    costmap.worldToGrid(pose[0], pose[1], center_mx, center_my);
    uint8_t center_cost = costmap.getCost(center_mx, center_my);
    max_cost = std::max(max_cost, center_cost);

    if (center_cost >= lethal_threshold_) {
        return false;
    }

    // Check edges by sampling points along footprint edges
    for (size_t i = 0; i < footprint_.size(); ++i) {
        size_t next = (i + 1) % footprint_.size();

        // Sample 5 points along each edge
        for (int sample = 1; sample < 5; ++sample) {
            double t = sample / 5.0;
            double local_x = footprint_[i][0] * (1.0 - t) + footprint_[next][0] * t;
            double local_y = footprint_[i][1] * (1.0 - t) + footprint_[next][1] * t;

            double world_x, world_y;
            transformFootprintVertex_(pose, local_x, local_y, world_x, world_y);

            int mx, my;
            costmap.worldToGrid(world_x, world_y, mx, my);

            uint8_t cost = costmap.getCost(mx, my);
            max_cost = std::max(max_cost, cost);

            if (cost >= lethal_threshold_) {
                return false;
            }
        }
    }

    return true;
}

void CollisionChecker::transformFootprintVertex_(
    const Pose& pose,
    double local_x,
    double local_y,
    double& world_x,
    double& world_y) {

    double x = pose[0];
    double y = pose[1];
    double theta = pose[2];

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    // Transform from robot frame to world frame
    world_x = x + local_x * cos_theta - local_y * sin_theta;
    world_y = y + local_x * sin_theta + local_y * cos_theta;
}

}  // namespace plan_ga_planner
