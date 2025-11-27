#ifndef PLAN_GA_PLANNER_COSTMAP_PROCESSOR_H
#define PLAN_GA_PLANNER_COSTMAP_PROCESSOR_H

#include "plan_ga_planner/types.h"
#include <vector>

namespace plan_ga_planner {

/**
 * @brief Processes costmap for neural network input
 *
 * Extracts local window, normalizes values, and prepares for inference.
 */
class CostmapProcessor {
public:
    /**
     * @brief Constructor
     *
     * @param window_size Size of the square window (e.g., 50 for 50x50)
     */
    explicit CostmapProcessor(int window_size = 50);

    /**
     * @brief Extract and process local costmap window
     *
     * @param costmap Full costmap
     * @param center_pose Robot pose (center of window)
     * @param output_data Flattened and normalized output (window_size * window_size floats)
     * @return true if successful
     */
    bool processWindow(
        const Costmap& costmap,
        const Pose& center_pose,
        std::vector<float>& output_data);

    /**
     * @brief Get window size
     */
    int getWindowSize() const { return window_size_; }

private:
    /**
     * @brief Extract window centered at robot pose
     */
    bool extractWindow_(
        const Costmap& costmap,
        const Pose& center_pose,
        std::vector<uint8_t>& window_data);

    /**
     * @brief Normalize costmap values [0, 255] -> [0, 1]
     */
    void normalize_(
        const std::vector<uint8_t>& input,
        std::vector<float>& output);

    int window_size_;
};

}  // namespace plan_ga_planner

#endif  // PLAN_GA_PLANNER_COSTMAP_PROCESSOR_H
