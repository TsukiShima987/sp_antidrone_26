#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <cmath>
#include <string>
#include <memory>
#include <chrono>
#include "../tools/solver.hpp"
#include "../TensorRT-YOLO/include/trtyolo.hpp"
#include <Eigen/Dense>
#include <optional>

struct UAVTarget{
    std::vector<cv::Point2f> roi;
    cv::Rect2f bounding_box;
    cv::Point2f center;

    cv::RotatedRect top_lb;
    cv::RotatedRect bottom_lb;
    float lb_length;
    float lb_spacing;

    cv::Point3f position;
    cv::Point3f velocity;
    float distance;
    float yaw;
    float pitch;

    float confidence;
    int id;
};

// Plain structs replacing ROS messages
struct Bbox {
    int x_min, y_min, x_max, y_max;
    int class_id;
    float class_confidence;

    float distance;
    float yaw;
    float pitch;
    cv::Point3f position;
    cv::Point3f velocity;

    float confidence;
    int id;
};

struct CarBbox {
    int img_height;
    int img_width;
    std::vector<Bbox> bboxs;
};

// ============================================================================
// Abstract base class for UAV detectors
// ============================================================================
class BaseDetector {
public:
    BaseDetector();
    virtual ~BaseDetector() = default;

    virtual std::vector<UAVTarget> detect(const cv::Mat& frame,
        std::chrono::steady_clock::time_point timestamp) = 0;
    virtual cv::Mat visualize(const cv::Mat& frame,
        const std::vector<UAVTarget>& targets) = 0;

protected:
    cv::Matx33d camera_matrix;
    cv::Mat dist_coeffs;
    const float real_spacing = 0.042f;
    const float real_object_height = 0.067f;
    std::string config_path = "io/configs/camera.yaml";
    cv::Matx44d T_camera2gimbal;
    int next_id = 0;

    void estimatePose(UAVTarget& target, float pixel_spacing, float real_size,
        const cv::Point2f& center);
    cv::Point3d computeLaserAimPoint(const cv::Point3d& target_cam);
    int assignID(const UAVTarget& target);
};

// ============================================================================
// Traditional light-bar based detector
// ============================================================================
class LightBarDetector : public BaseDetector {
public:
    LightBarDetector();

    std::vector<UAVTarget> detect(const cv::Mat& frame,
        std::chrono::steady_clock::time_point timestamp) override;
    cv::Mat visualize(const cv::Mat& frame,
        const std::vector<UAVTarget>& targets) override;

private:
    struct DetectionParams {
        float min_length = 5;
        float max_length = 500;
        float min_ratio = 1.0;
        float max_ratio = 2.0;
        float max_angle_diff = 30.0;
        float min_spacing_ratio = 1.0;
        float max_spacing_ratio = 4.0;
        float min_confidence = 0.5;
    } params;

    void multiThresholdBinary(const cv::Mat& src, std::vector<cv::Mat>& binarys);
    void detectLightPairs(const cv::Mat& binary,
        std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs);
    bool isValidPair(const cv::RotatedRect& r1, const cv::RotatedRect& r2);
    void removeDuplicates(std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs);
    UAVTarget createUAVTarget(const cv::RotatedRect& top, const cv::RotatedRect& bottom,
        const cv::Mat& frame);
    cv::Rect2f calculateBoundingBox(const cv::RotatedRect& top, const cv::RotatedRect& bottom);
    std::vector<cv::Point2f> calculateROIVertices(const cv::RotatedRect& top,
        const cv::RotatedRect& bottom);
    float calculateConfidence(const UAVTarget& target);
    bool validateTarget(const UAVTarget& target);
};

// ============================================================================
// YOLO-based detector
// ============================================================================
class YOLODetector : public BaseDetector {
public:
    YOLODetector();

    std::vector<UAVTarget> detect(const cv::Mat& frame,
        std::chrono::steady_clock::time_point timestamp) override;
    cv::Mat visualize(const cv::Mat& frame,
        const std::vector<UAVTarget>& targets) override;

private:
    std::shared_ptr<trtyolo::DetectModel> model_;
    CarBbox last_car_bboxs_;  // cached from last detect() for visualization

    std::pair<Bbox, CarBbox> detectOnce(const cv::Mat& image);
    cv::Rect getRect(cv::Mat& img, const trtyolo::Box& bbox);
    void drawCarBbox(const CarBbox& car_bboxs, cv::Mat& frame);
};

// ============================================================================
// Detector factory — reads config and selects the appropriate detector
// ============================================================================
class Detector {
public:
    explicit Detector(const std::string& config_path);

    std::vector<UAVTarget> detect(const cv::Mat& frame,
        std::chrono::steady_clock::time_point timestamp);
    cv::Mat visualize(const cv::Mat& frame,
        const std::vector<UAVTarget>& targets);

private:
    std::unique_ptr<BaseDetector> detector_;
};
