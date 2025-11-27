#include "plan_ga_planner/costmap_processor.h"
#include <cmath>
#include <iostream>

namespace plan_ga_planner {

CostmapProcessor::CostmapProcessor(int window_size)
    : window_size_(window_size) {}

bool CostmapProcessor::processWindow(
    const Costmap& costmap,
    const Pose& center_pose,
    std::vector<float>& output_data) {

    // Extract window
    std::vector<uint8_t> window_data;
    if (!extractWindow_(costmap, center_pose, window_data)) {
        return false;
    }

    // Normalize to [0, 1]
    normalize_(window_data, output_data);

    return true;
}

bool CostmapProcessor::extractWindow_(
    const Costmap& costmap,
    const Pose& center_pose,
    std::vector<uint8_t>& window_data) {

    window_data.resize(window_size_ * window_size_, 0);

    // Convert robot pose to grid coordinates
    int center_mx, center_my;
    costmap.worldToGrid(center_pose[0], center_pose[1], center_mx, center_my);

    // Calculate window bounds
    int half_window = window_size_ / 2;
    int start_mx = center_mx - half_window;
    int start_my = center_my - half_window;

    // Extract window with boundary handling
    for (int wy = 0; wy < window_size_; ++wy) {
        for (int wx = 0; wx < window_size_; ++wx) {
            int mx = start_mx + wx;
            int my = start_my + wy;

            // Get cost (returns 0 if out of bounds)
            uint8_t cost = costmap.getCost(mx, my);

            // Store in window (row-major order)
            window_data[wy * window_size_ + wx] = cost;
        }
    }

    return true;
}

void CostmapProcessor::normalize_(
    const std::vector<uint8_t>& input,
    std::vector<float>& output) {

    output.resize(input.size());

    // Normalize [0, 255] -> [0, 1]
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]) / 255.0f;
    }
}

}  // namespace plan_ga_planner
