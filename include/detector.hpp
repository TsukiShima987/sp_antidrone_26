#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <cmath>
#include <string>
#include <memory>
#include <chrono>
// #include "io/gimbal/gimbal.hpp"
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

class UAVDetector{
private:
    struct DetectionParams{
        float min_length = 5;
        float max_length = 500;
        float min_ratio = 1.0;
        float max_ratio = 2.0;
        float max_angle_diff = 30.0;
        float min_spacing_ratio = 1.0;
        float max_spacing_ratio = 4.0;
        float min_confidence = 0.5;
    } params;

    int next_id = 0;

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    const float real_spacing = 0.042f;
    const float real_object_height = 0.067f;
    std::string config_path = "io/configs/camera.yaml";

    std::shared_ptr<trtyolo::DetectModel> model_;

    cv::Mat T_camera2gimbal;
    bool yolo_detection = false;
    bool uav_detection = false;
public:
    UAVDetector();

    std::vector<UAVTarget> detectUAVs(const cv::Mat& frame, std::chrono::steady_clock::time_point timestamp);
    std::tuple<Bbox, std::vector<UAVTarget>, cv::Mat> detectYolos(const cv::Mat& frame, std::chrono::steady_clock::time_point timestamp);
    void estimatePoseYolo(Bbox& bbox, std::chrono::steady_clock::time_point timestamp);
    void estimatePose(UAVTarget& target, std::chrono::steady_clock::time_point timestamp);

    std::pair<Bbox, cv::Mat> detect_once(cv::Mat frame);

private:
    cv::Point3d computeLaserAimPoint(const cv::Point3d& target_cam);
    cv::Rect get_rect(cv::Mat &img, const trtyolo::Box& bbox);
    void draw_car_bbox(CarBbox car_bboxs, cv::Mat& frame);

    void multiThresholdBinary(const cv::Mat& src, std::vector<cv::Mat>& binarys);
    void detectLightPairs(const cv::Mat& binary, std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs);
    bool isValidPair(const cv::RotatedRect& r1, const cv::RotatedRect& r2);
    void removeDuplicates(std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs);

    UAVTarget createUAVTarget(const cv::RotatedRect& top, const cv::RotatedRect& bottom, const cv::Mat& frame, std::chrono::steady_clock::time_point timestamp);
    cv::Rect2f calculateBoundingBox(const cv::RotatedRect& top, const cv::RotatedRect& bottom);
    std::vector<cv::Point2f> calculateROIVertices(const cv::RotatedRect& top, const cv::RotatedRect& bottom);
    float calculateConfidence(const UAVTarget& target);
    bool validateTarget(const UAVTarget& target);
    int assignID(const UAVTarget& target);
};
