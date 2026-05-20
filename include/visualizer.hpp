#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <string>
#include "detector.hpp"

class UAVDetectorVisualizer {
private:
    UAVDetector detector;
    cv::Scalar colors[3];
public:
    UAVDetectorVisualizer();
    void visualizeFrame(cv::Mat& frame, std::vector<UAVTarget>& targets);

// private:
    void drawTarget(cv::Mat& frame, const UAVTarget& target);
    void showDetectionInfo(cv::Mat& frame, const std::vector<UAVTarget>& targets);
};
