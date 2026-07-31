#include "include/detector.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>

YOLODetector::YOLODetector() {
    std::string config_path = "config/antidrone.yaml";
    const auto config = YAML::LoadFile(config_path);

    std::string drone_engine_file;
    drone_engine_file = config["drone_engine_file"].as<std::string>();
    trtyolo::InferOption option;
    option.enableSwapRB();
    model_ = std::make_shared<trtyolo::DetectModel>(drone_engine_file, option);
    std::cout << "YOLODetector initialized" << std::endl;
}

std::vector<UAVTarget> YOLODetector::detect(const cv::Mat& frame,
                                             std::chrono::steady_clock::time_point /*timestamp*/) {
    std::vector<UAVTarget> targets;

    Bbox maxbbox;
    std::tie(maxbbox, last_car_bboxs_) = detectOnce(frame);

    if (maxbbox.class_confidence < 0) {
        return targets;  // no detection
    }

    UAVTarget target;
    cv::Point2f center((maxbbox.x_min + maxbbox.x_max) / 2.0f,
                       (maxbbox.y_min + maxbbox.y_max) / 2.0f);
    target.center = center;
    target.confidence = maxbbox.class_confidence;
    target.bounding_box = cv::Rect2f(cv::Point2f(maxbbox.x_min, maxbbox.y_min),
                                      cv::Point2f(maxbbox.x_max, maxbbox.y_max));

    float pixel_spacing = static_cast<float>(maxbbox.y_max - maxbbox.y_min);
    estimatePose(target, pixel_spacing, real_object_height, center);

    target.id = assignID(target);

    targets.push_back(target);
    return targets;
}

cv::Mat YOLODetector::visualize(const cv::Mat& frame,
                                 const std::vector<UAVTarget>& targets) {
    cv::Mat display = frame.clone();
    cv::cvtColor(display, display, cv::COLOR_BGR2RGB);

    // Draw all YOLO detections from last inference
    // drawCarBbox(last_car_bboxs_, display);

    cv::cvtColor(display, display, cv::COLOR_RGB2BGR);

    // Draw UAVTarget overlays
    for (const auto& target : targets) {
        cv::Scalar color(0x27, 0xC1, 0x36);  // green
        cv::rectangle(display, target.bounding_box, color, 2);

        std::string label = cv::format("ID:%d Conf:%.2f Dist:%.1fm",
                                       target.id, target.confidence, target.distance);
        cv::putText(display, label,
                    cv::Point(target.bounding_box.x, target.bounding_box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }

    cv::resize(display, display, cv::Size(1920, 1280));
    return display;
}

std::pair<Bbox, CarBbox> YOLODetector::detectOnce(const cv::Mat& image) {
    CarBbox car_bboxs;
    car_bboxs.img_height = image.rows;
    car_bboxs.img_width = image.cols;

    cv::Mat rgb = image.clone();
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

    trtyolo::Image input_image(rgb.data, rgb.cols, rgb.rows);
    trtyolo::DetectRes result = model_->predict(input_image);

    Bbox max_confidence_bbox;
    max_confidence_bbox.class_confidence = -1.0f;

    for (size_t j = 0; j < result.num; j++) {
        cv::Rect r = getRect(rgb, result.boxes[j]);

        Bbox bbox;
        bbox.x_min = r.x;
        bbox.y_min = r.y;
        bbox.x_max = r.x + r.width;
        bbox.y_max = r.y + r.height;

        bbox.class_confidence = result.scores[j];
        bbox.class_id = result.classes[j];

        car_bboxs.bboxs.push_back(bbox);

        if (bbox.class_confidence > max_confidence_bbox.class_confidence) {
            max_confidence_bbox = bbox;
        }
    }

    return std::make_pair(max_confidence_bbox, car_bboxs);
}

cv::Rect YOLODetector::getRect(cv::Mat& /*img*/, const trtyolo::Box& bbox) {
    float left = bbox.left;
    float top = bbox.top;
    float right = bbox.right;
    float bottom = bbox.bottom;
    return cv::Rect(cv::Point(left, top), cv::Point(right, bottom));
}

void YOLODetector::drawCarBbox(const CarBbox& car_bboxs, cv::Mat& frame) {
    for (auto bbox : car_bboxs.bboxs) {
        cv::Scalar color = (bbox.class_id < 6) ? cv::Scalar(255, 128, 0)
                                                : cv::Scalar(50, 50, 255);
        if (bbox.x_min > 0 || bbox.y_min > 20 || bbox.x_max < frame.cols - 50
            || bbox.y_max < frame.rows) {
            cv::rectangle(frame, cv::Point(bbox.x_min, bbox.y_min),
                          cv::Point(bbox.x_max, bbox.y_max), color, 10);
            cv::putText(frame, std::to_string((bbox.class_id) % 6 + 1),
                        cv::Point(bbox.x_min + 40, bbox.y_min - 10),
                        cv::FONT_HERSHEY_PLAIN, 6, color, 6);
            cv::circle(frame, cv::Point((bbox.x_min + bbox.x_max) / 2,
                                         (bbox.y_max + bbox.y_min) / 2), 2, color, 10);
        }
    }
}
