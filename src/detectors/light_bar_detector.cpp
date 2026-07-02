#include "include/detector.hpp"
#include <iostream>

LightBarDetector::LightBarDetector() {
    cv::namedWindow("binary", 0);
    std::cout << "LightBarDetector initialized" << std::endl;
}

std::vector<UAVTarget> LightBarDetector::detect(const cv::Mat& frame,
                                                 std::chrono::steady_clock::time_point /*timestamp*/) {
    std::vector<UAVTarget> targets;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Mat> binarys;
    multiThresholdBinary(gray, binarys);

    std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> light_pairs;
    for (const auto& binary : binarys) {
        detectLightPairs(binary, light_pairs);
    }

    removeDuplicates(light_pairs);

    std::vector<UAVTarget> valid_targets;
    for (const auto& pair : light_pairs) {
        UAVTarget target = createUAVTarget(pair.first, pair.second, frame);
        if (validateTarget(target)) {
            valid_targets.push_back(target);
        }
    }

    if (!valid_targets.empty()) {
        auto best_it = std::max_element(valid_targets.begin(), valid_targets.end(),
            [](const UAVTarget& a, const UAVTarget& b) {
                return a.confidence < b.confidence;
            });
        UAVTarget merged = *best_it;

        merged.id = assignID(merged);

        float pixel_spacing = cv::norm(merged.top_lb.center - merged.bottom_lb.center);
        estimatePose(merged, pixel_spacing, real_spacing, merged.center);

        std::cout << "ID:" << merged.id << ", Confidence:" << merged.confidence << std::endl;
        std::cout << "yaw:" << merged.yaw * 180.0f / CV_PI
                  << ", pitch" << merged.pitch * 180.0f / CV_PI << std::endl;
        targets.push_back(merged);
    }

    return targets;
}

cv::Mat LightBarDetector::visualize(const cv::Mat& frame,
                                     const std::vector<UAVTarget>& targets) {
    cv::Mat display = frame.clone();

    for (const auto& target : targets) {
        // Color based on confidence
        cv::Scalar color;
        if (target.confidence > 0.8) {
            color = cv::Scalar(0, 255, 0);       // green - high
        } else if (target.confidence > 0.6) {
            color = cv::Scalar(0, 255, 255);     // yellow - medium
        } else {
            color = cv::Scalar(0, 0, 255);       // red - low
        }

        // Draw light bars (rotated rects)
        cv::Point2f topPts[4], bottomPts[4];
        target.top_lb.points(topPts);
        target.bottom_lb.points(bottomPts);

        for (int i = 0; i < 4; i++) {
            cv::line(display, topPts[i], topPts[(i + 1) % 4], color, 2);
            cv::line(display, bottomPts[i], bottomPts[(i + 1) % 4], color, 2);
        }

        // Light bar centers
        cv::circle(display, target.top_lb.center, 3, color, -1);
        cv::circle(display, target.bottom_lb.center, 3, color, -1);

        // Bounding box
        cv::rectangle(display, target.bounding_box, color, 1);

        // ROI vertices
        if (target.roi.size() == 4) {
            for (int i = 0; i < 4; i++) {
                cv::line(display, target.roi[i], target.roi[(i + 1) % 4],
                         cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
        }

        // Center point
        cv::circle(display, target.center, 4, cv::Scalar(255, 255, 255), -1);

        // Label
        std::string label = cv::format("ID:%d Conf:%.2f", target.id, target.confidence);
        cv::putText(display, label,
                    cv::Point(target.bounding_box.x, target.bounding_box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    // Stats overlay
    cv::rectangle(display, cv::Rect(5, 5, 200, 70), cv::Scalar(0, 0, 0), -1);
    std::string info = cv::format("Detections: %d", (int)targets.size());
    cv::putText(display, info, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    int y_offset = 45;
    for (size_t i = 0; i < std::min(targets.size(), (size_t)3); i++) {
        std::string target_info = cv::format("ID:%d C:%.2f L:%.1f",
            targets[i].id, targets[i].confidence, targets[i].lb_length);
        cv::putText(display, target_info, cv::Point(10, y_offset + (int)i * 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    return display;
}

void LightBarDetector::multiThresholdBinary(const cv::Mat& src, std::vector<cv::Mat>& binarys) {
    std::vector<int> thresholds = {50};

    for (int thresh : thresholds) {
        cv::Mat binary;
        cv::threshold(src, binary, thresh, 255, cv::THRESH_BINARY);

        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernal);

        // cv::resizeWindow("binary", cv::Size(1920, 1280));
        // cv::imshow("binary", binary);

        binarys.push_back(binary);
    }
}

void LightBarDetector::detectLightPairs(const cv::Mat& binary,
                                         std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> lights;

    for (const auto& contour : contours) {
        cv::RotatedRect rect = cv::minAreaRect(contour);

        float length = std::max(rect.size.width, rect.size.height);
        float width = std::min(rect.size.width, rect.size.height);
        float ratio = length / width;

        if (length >= params.min_length && length <= params.max_length
            && ratio >= params.min_ratio && ratio <= params.max_ratio) {
            lights.push_back(rect);
        }
    }

    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            if (isValidPair(lights[i], lights[j])) {
                pairs.push_back({lights[i], lights[j]});
            }
        }
    }
}

bool LightBarDetector::isValidPair(const cv::RotatedRect& r1, const cv::RotatedRect& r2) {
    float angle1 = r1.angle;
    float angle2 = r2.angle;
    float angle_diff = std::abs(angle1 - angle2);
    angle_diff = std::min(angle_diff, 180.0f - angle_diff);

    if (angle_diff > params.max_angle_diff) return false;

    float len1 = std::max(r1.size.width, r1.size.height);
    float len2 = std::max(r2.size.width, r2.size.height);

    float spacing = cv::norm(r1.center - r2.center);
    float avg_length = (len1 + len2) / 2;

    if (std::abs(r1.center.x - r2.center.x) / avg_length > 0.5) return false;

    float spacing_ratio = spacing / avg_length;

    if (spacing_ratio < params.min_spacing_ratio || spacing_ratio > params.max_spacing_ratio)
        return false;

    return true;
}

void LightBarDetector::removeDuplicates(
        std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs) {
    const float DIST_THRESH = 20;

    for (auto it = pairs.begin(); it != pairs.end(); ) {
        bool duplicate = false;
        cv::Point2f center1 = (it->first.center + it->second.center) / 2;

        for (auto jt = pairs.begin(); jt != it; ++jt) {
            cv::Point2f center2 = (jt->first.center + jt->second.center) / 2;
            if (cv::norm(center1 - center2) < DIST_THRESH) {
                duplicate = true;
                break;
            }
        }

        if (duplicate) it = pairs.erase(it);
        else ++it;
    }
}

UAVTarget LightBarDetector::createUAVTarget(const cv::RotatedRect& top,
                                              const cv::RotatedRect& bottom,
                                              const cv::Mat& /*frame*/) {
    UAVTarget target;

    if (top.center.y < bottom.center.y) {
        target.top_lb = top;
        target.bottom_lb = bottom;
    } else {
        target.bottom_lb = top;
        target.top_lb = bottom;
    }

    target.center = (target.top_lb.center + target.bottom_lb.center) / 2;

    float top_len = std::max(target.top_lb.size.width, target.top_lb.size.height);
    float bottom_len = std::max(target.bottom_lb.size.width, target.bottom_lb.size.height);
    target.lb_length = (top_len + bottom_len) / 2;
    target.lb_spacing = target.bottom_lb.center.y - target.top_lb.center.y;

    target.bounding_box = calculateBoundingBox(target.top_lb, target.bottom_lb);
    target.roi = calculateROIVertices(target.top_lb, target.bottom_lb);

    target.confidence = calculateConfidence(target);

    return target;
}

cv::Rect2f LightBarDetector::calculateBoundingBox(const cv::RotatedRect& top,
                                                    const cv::RotatedRect& bottom) {
    std::vector<cv::Point2f> points;

    cv::Point2f topPts[4], bottomPts[4];
    top.points(topPts);
    bottom.points(bottomPts);
    for (int i = 0; i < 4; ++i) {
        points.push_back(topPts[i]);
        points.push_back(bottomPts[i]);
    }

    float minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
    for (const auto& p : points) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }

    return cv::Rect2f(minX, minY, maxX - minX, maxY - minY);
}

std::vector<cv::Point2f> LightBarDetector::calculateROIVertices(const cv::RotatedRect& top,
                                                                  const cv::RotatedRect& bottom) {
    std::vector<cv::Point2f> vertices(4);

    cv::Point2f topPts[4], bottomPts[4];
    top.points(topPts);
    bottom.points(bottomPts);

    float leftX = std::min({topPts[0].x, topPts[1].x, topPts[2].x, topPts[3].x,
                            bottomPts[0].x, bottomPts[1].x, bottomPts[2].x, bottomPts[3].x});
    float rightX = std::max({topPts[0].x, topPts[1].x, topPts[2].x, topPts[3].x,
                             bottomPts[0].x, bottomPts[1].x, bottomPts[2].x, bottomPts[3].x});

    vertices[0] = cv::Point2f(leftX, topPts[0].y);
    vertices[1] = cv::Point2f(rightX, topPts[0].y);
    vertices[2] = cv::Point2f(rightX, bottomPts[3].y);
    vertices[3] = cv::Point2f(leftX, bottomPts[3].y);

    return vertices;
}

float LightBarDetector::calculateConfidence(const UAVTarget& target) {
    float confidence = 0;

    float angle_diff = std::abs(target.top_lb.angle - target.bottom_lb.angle);
    angle_diff = std::min(angle_diff, 180.0f - angle_diff);
    confidence += (1.0f - angle_diff / params.max_angle_diff) * 0.2f;

    float top_len = std::max(target.top_lb.size.width, target.top_lb.size.height);
    float bottom_len = std::max(target.bottom_lb.size.width, target.bottom_lb.size.height);
    float len_ratio = std::min(top_len, bottom_len) / std::max(top_len, bottom_len);
    confidence += len_ratio * 0.2f;

    float spacing_ratio = target.lb_spacing / target.lb_length;
    if (spacing_ratio >= params.min_spacing_ratio && spacing_ratio <= params.max_spacing_ratio) {
        float ideal_ratio = (params.min_spacing_ratio + params.max_spacing_ratio) / 2;
        float ratio_diff = std::abs(spacing_ratio - ideal_ratio) / ideal_ratio;
        confidence += (1.0f - ratio_diff) * 0.6f;
    }

    return confidence;
}

bool LightBarDetector::validateTarget(const UAVTarget& target) {
    return target.confidence >= params.min_confidence;
}
