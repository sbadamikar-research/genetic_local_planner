#ifndef PLAN_GA_PLANNER_ONNX_INFERENCE_H
#define PLAN_GA_PLANNER_ONNX_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace plan_ga_planner {

/**
 * @brief ONNX Runtime inference wrapper
 *
 * Handles loading ONNX model and running inference with proper memory management.
 * Thread-safe for single-threaded use.
 */
class ONNXInference {
public:
    /**
     * @brief Constructor
     *
     * @param model_path Path to .onnx model file
     * @param num_threads Number of threads for inference (default: 1)
     */
    explicit ONNXInference(const std::string& model_path, int num_threads = 1);

    /**
     * @brief Destructor
     */
    ~ONNXInference();

    /**
     * @brief Run inference on input tensors
     *
     * @param costmap_input Flattened costmap (50*50 = 2500 floats), normalized [0,1]
     * @param robot_state_input Robot state (9 floats): [x, y, theta, v_x, v_y, omega, a_x, a_y, alpha]
     * @param goal_relative_input Relative goal (3 floats): [dx, dy, dtheta]
     * @param costmap_metadata_input Costmap metadata (2 floats): [inflation_decay, resolution]
     * @param output Output control sequence (num_steps * 3 floats)
     * @return true if inference successful
     */
    bool infer(
        const std::vector<float>& costmap_input,
        const std::vector<float>& robot_state_input,
        const std::vector<float>& goal_relative_input,
        const std::vector<float>& costmap_metadata_input,
        std::vector<float>& output);

    /**
     * @brief Get number of control steps in output
     */
    int getNumControlSteps() const { return num_control_steps_; }

    /**
     * @brief Get costmap size
     */
    int getCostmapSize() const { return costmap_size_; }

    /**
     * @brief Check if model is loaded
     */
    bool isLoaded() const { return session_ != nullptr; }

private:
    /**
     * @brief Load ONNX model
     */
    void loadModel_(const std::string& model_path);

    /**
     * @brief Setup input/output names
     */
    void setupInputOutputNames_();

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_name_strings_;
    std::vector<std::string> output_name_strings_;

    int costmap_size_;
    int num_control_steps_;
};

}  // namespace plan_ga_planner

#endif  // PLAN_GA_PLANNER_ONNX_INFERENCE_H
