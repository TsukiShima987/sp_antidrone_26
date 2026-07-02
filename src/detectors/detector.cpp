#include "include/detector.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <iostream>

Detector::Detector(const std::string& config_path) {
    const auto config = YAML::LoadFile(config_path);

    std::string option = config["detector_option"].as<std::string>();

    if (option == "uav") {
        detector_ = std::make_unique<LightBarDetector>();
    } else if (option == "yolo") {
        detector_ = std::make_unique<YOLODetector>();
    } else {
        throw std::invalid_argument("detector_option must be \"uav\" or \"yolo\", got: " + option);
    }

    std::cout << "Detector initialized with option: " << option << std::endl;
}

std::vector<UAVTarget> Detector::detect(const cv::Mat& frame,
                                         std::chrono::steady_clock::time_point timestamp) {
    return detector_->detect(frame, timestamp);
}

cv::Mat Detector::visualize(const cv::Mat& frame,
                              const std::vector<UAVTarget>& targets) {
    return detector_->visualize(frame, targets);
}
