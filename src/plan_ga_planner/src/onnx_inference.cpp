#include "plan_ga_planner/onnx_inference.h"
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace plan_ga_planner {

ONNXInference::ONNXInference(const std::string& model_path, int num_threads)
    : costmap_size_(50), num_control_steps_(20) {

    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PlanGAPlanner");

    // Session options
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(num_threads);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load model
    loadModel_(model_path);
    setupInputOutputNames_();
}

ONNXInference::~ONNXInference() = default;

void ONNXInference::loadModel_(const std::string& model_path) {
    try {
        // Load model (ONNX Runtime expects std::string or wchar_t*)
        #ifdef _WIN32
            // Convert to wide string for Windows
            std::wstring wide_path(model_path.begin(), model_path.end());
            session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), *session_options_);
        #else
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        #endif

        std::cout << "[ONNX] Model loaded successfully: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
    }
}

void ONNXInference::setupInputOutputNames_() {
    // Input names: costmap, robot_state, goal_relative, costmap_metadata
    input_name_strings_ = {"costmap", "robot_state", "goal_relative", "costmap_metadata"};
    input_names_.clear();
    for (const auto& name : input_name_strings_) {
        input_names_.push_back(name.c_str());
    }

    // Output name: control_sequence
    output_name_strings_ = {"control_sequence"};
    output_names_.clear();
    for (const auto& name : output_name_strings_) {
        output_names_.push_back(name.c_str());
    }
}

bool ONNXInference::infer(
    const std::vector<float>& costmap_input,
    const std::vector<float>& robot_state_input,
    const std::vector<float>& goal_relative_input,
    const std::vector<float>& costmap_metadata_input,
    std::vector<float>& output) {

    try {
        // Validate input sizes
        if (costmap_input.size() != static_cast<size_t>(costmap_size_ * costmap_size_)) {
            std::cerr << "[ONNX] Invalid costmap input size: " << costmap_input.size()
                      << ", expected: " << (costmap_size_ * costmap_size_) << std::endl;
            return false;
        }
        if (robot_state_input.size() != 9) {
            std::cerr << "[ONNX] Invalid robot state size: " << robot_state_input.size() << std::endl;
            return false;
        }
        if (goal_relative_input.size() != 3) {
            std::cerr << "[ONNX] Invalid goal relative size: " << goal_relative_input.size() << std::endl;
            return false;
        }
        if (costmap_metadata_input.size() != 2) {
            std::cerr << "[ONNX] Invalid costmap metadata size: " << costmap_metadata_input.size() << std::endl;
            return false;
        }

        // Create memory info
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensors
        std::vector<Ort::Value> input_tensors;

        // 1. Costmap: (1, 1, 50, 50)
        std::vector<int64_t> costmap_shape = {1, 1, costmap_size_, costmap_size_};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(costmap_input.data()),
            costmap_input.size(),
            costmap_shape.data(),
            costmap_shape.size()
        ));

        // 2. Robot state: (1, 9)
        std::vector<int64_t> state_shape = {1, 9};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(robot_state_input.data()),
            robot_state_input.size(),
            state_shape.data(),
            state_shape.size()
        ));

        // 3. Goal relative: (1, 3)
        std::vector<int64_t> goal_shape = {1, 3};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(goal_relative_input.data()),
            goal_relative_input.size(),
            goal_shape.data(),
            goal_shape.size()
        ));

        // 4. Costmap metadata: (1, 2)
        std::vector<int64_t> metadata_shape = {1, 2};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(costmap_metadata_input.data()),
            costmap_metadata_input.size(),
            metadata_shape.data(),
            metadata_shape.size()
        ));

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_.data(),
            output_names_.size()
        );

        // Extract output: (1, num_steps, 3)
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = num_control_steps_ * 3;
        output.assign(output_data, output_data + output_size);

        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[ONNX] Inference error: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace plan_ga_planner
