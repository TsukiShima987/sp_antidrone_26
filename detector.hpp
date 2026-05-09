#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <deque>
#include <cmath>
#include <string>
#include "io/gimbal/gimbal.hpp"
#include "tools/targetyawpitch.hpp"
#include "tools/solver.hpp"
#include <yaml-cpp/yaml.h>

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

    UAVTarget() : confidence(0), id(-1), distance(0), yaw(0), pitch(0) {}
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
        float min_confidence = 0.8;
    } params;

    int next_id = 0;

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 30164.346973727235, 0,                  1585.3978670158735, 
                                                       0,                  29990.338352363626, 1027.6312267976482, 
                                                       0,                  0,                  1                  );
    cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << 0.47187615502896024, 
                                                     23.566194187549524,
                                                     0.037885849822472041,
                                                     0.0028823047400091048,
                                                     0                     );
    const float real_spacing = 0.042f;

    std::string config_path = "io/configs/camera.yaml";
    std::string transform_path = "config/camera2gimbal.yaml";
    io::Gimbal gimbal;

    cv::Mat T_camera2gimbal;

public:
    UAVDetector() : gimbal(config_path){
        cv::namedWindow("binary", 0);

        T_camera2gimbal = cv::Mat::eye(4, 4, CV_64F);   // 先设为单位阵保证安全
        try {
            YAML::Node config = YAML::LoadFile(transform_path);
            if (config["T_camera2gimbal"] && config["T_camera2gimbal"].IsSequence()) {
                auto rows = config["T_camera2gimbal"];
                for (size_t i = 0; i < 4; ++i) {
                    auto row = rows[i];
                    for (size_t j = 0; j < 4; ++j) {
                        T_camera2gimbal.at<double>(i, j) = row[j].as<double>();
                    }
                }
            } else {
                std::cerr << "Missing or invalid T_camera2gimbal in yaml" << std::endl;
            }
        } catch (const YAML::Exception& e) {
            std::cerr << "YAML parse error: " << e.what() << std::endl;
        }

    }

    std::vector<UAVTarget> detectUAVs(const cv::Mat& frame, std::chrono::steady_clock::time_point timestamp)
    {
        std::vector<UAVTarget> targets;   // 最终返回结果（合并后最多一个）

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Mat> binarys;
        multiThresholdBinary(gray, binarys);

        std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> light_pairs;
        for (const auto& binary : binarys)
        {
            detectLightPairs(binary, light_pairs);
        }

        removeDuplicates(light_pairs);

        // ---------- 收集所有有效目标----------
        std::vector<UAVTarget> valid_targets;
        for (const auto& pair : light_pairs)
        {
            UAVTarget target = createUAVTarget(pair.first, pair.second, frame, timestamp);
            if (validateTarget(target))
            {
                valid_targets.push_back(target);
            }
        }

        if (!valid_targets.empty())
        {
            auto best_it = std::max_element(valid_targets.begin(), valid_targets.end(),
                [](const UAVTarget& a, const UAVTarget& b) {
                    return a.confidence < b.confidence;
                });
            UAVTarget merged = *best_it;

            cv::Point2f avg_center(0.f, 0.f);
            for (const auto& t : valid_targets)
            {
                avg_center += t.center;
            }
            avg_center *= (1.0f / valid_targets.size());
            merged.center = avg_center;

            merged.id = assignID(merged);

            estimatePose(merged, timestamp);

            auto gs = gimbal.state();
            std::cout << "ID:" << merged.id << ", Confidence:" << merged.confidence << std::endl;
            std::cout << "yaw:" << merged.yaw * 180.0f / CV_PI << ", pitch" << merged.pitch * 180.0f / CV_PI << std::endl;
            gimbal.send(1, 0, -merged.yaw, 0, 0, merged.pitch, 0, 0);

            targets.push_back(merged);
        }
        else 
        {
            //gimbal.send(0, 0, 0, 0, 0, 0, 0, 0);
        }

        return targets;
    }

    void estimatePose(UAVTarget& target, std::chrono::steady_clock::time_point timestamp)
    {
        float fx = camera_matrix.at<double>(0, 0);
        float fy = camera_matrix.at<double>(1, 1);
        float cx = camera_matrix.at<double>(0, 2);
        float cy = camera_matrix.at<double>(1, 2);
        float pixel_spacing = cv::norm(target.top_lb.center - target.bottom_lb.center);

        double z = (fy * real_spacing) / pixel_spacing;
        double x = (target.center.x - cx) * z / fx;
        double y = (target.center.y - cy) * z / fy;
        double distance = cv::norm(cv::Point3f(x, y, z));

        // tools::TargetYawPitch c2l;
        // auto p_laser = c2l.TargetXYZ(distance);
        // x = p_laser.x;
        // y = p_laser.y;
        // z = p_laser.z;

        cv::Mat p_camera = (cv::Mat_<double>(4, 1) << x*1000, y*1000, z*1000, 1.0); 
        std::cout << "p_camera" << p_camera << std::endl;
        cv::Mat p_gimbal_h = T_camera2gimbal * p_camera;
        std::cout << "p_gimbal_h" << p_gimbal_h << std::endl;
        cv::Point3d rel_gim(p_gimbal_h.at<double>(0), p_gimbal_h.at<double>(1), p_gimbal_h.at<double>(2));

        tools::Solver solver;
        auto q = gimbal.q(timestamp);
        solver.set_R_gimbal2world(q);
        Eigen::Vector3d p_gimbal(rel_gim.x, rel_gim.y, rel_gim.z);
        Eigen::Vector3d p_world = solver.R_gimbal2world() * p_gimbal;

        x = p_world.x();
        y = p_world.y();
        z = p_world.z();
        std::cout << "p_world: " << p_world << std::endl; 

        auto gs = gimbal.state();
        std::cout << "gimbal yaw: " << gs.yaw / CV_PI * 180.0 << std::endl;
        std::cout << "gimbal pitch: " << gs.pitch / CV_PI * 180.0 << std::endl;

        target.position = cv::Point3f(x, y, z);
        target.distance = cv::norm(target.position);
        target.yaw = -std::atan2(y, x);
        target.pitch = -std::atan2(z, sqrt(x * x + y * y));
    }

private:
    void multiThresholdBinary(const cv::Mat& src, std::vector<cv::Mat>& binarys)
    {
        std::vector<int> thresholds = {125};

        for (int thresh : thresholds)
        {
            cv::Mat binary;
            cv::threshold(src, binary, thresh, 255, cv::THRESH_BINARY);

            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernal);

            cv::resizeWindow("binary", cv::Size(1920, 1280));
            cv::imshow("binary", binary);

            binarys.push_back(binary);
        }
    }

    void detectLightPairs(const cv::Mat& binary, std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs)
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::RotatedRect> lights;

        for (const auto& contour : contours)
        {
            cv::RotatedRect rect = cv::minAreaRect(contour);

            float length = std::max(rect.size.width, rect.size.height);
            float width = std::min(rect.size.width, rect.size.height);
            float ratio = length / width;

            if (length >= params.min_length && length <= params.max_length && ratio >= params.min_ratio && ratio <= params.max_ratio)
            {
                lights.push_back(rect);
            }
        }

        for (size_t i = 0; i < lights.size(); ++i)
        {
            for (size_t j = i + 1; j < lights.size(); ++j)
            {
                if (isValidPair(lights[i], lights[j]))
                {
                    pairs.push_back({lights[i], lights[j]});
                }
            }
        }
    }

    bool isValidPair(const cv::RotatedRect& r1, const cv::RotatedRect& r2)
    {
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

        if (spacing_ratio < params.min_spacing_ratio || spacing_ratio > params.max_spacing_ratio) return false;

        return true;
    }

    void removeDuplicates(std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>>& pairs)
    {
        const float DIST_THRESH = 20;

        for (auto it = pairs.begin(); it != pairs.end(); )
        {
            bool duplicate = false;
            cv::Point2f center1 = (it->first.center + it->second.center) / 2;

            for (auto jt = pairs.begin(); jt != it; ++jt)
            {
                cv::Point2f center2 = (jt->first.center + jt->second.center) / 2;
                if (cv::norm(center1 - center2) < DIST_THRESH)
                {
                    duplicate = true;
                    break;
                }
            }

            if (duplicate) it = pairs.erase(it);
            else ++it;
        }
    }

    UAVTarget createUAVTarget(const cv::RotatedRect& top, const cv::RotatedRect& bottom, const cv::Mat& frame, std::chrono::steady_clock::time_point timestamp)
    {
        UAVTarget target;

        if (top.center.y < bottom.center.y)
        {
            target.top_lb = top;
            target.bottom_lb = bottom;
        }
        else
        {
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

    cv::Rect2f calculateBoundingBox(const cv::RotatedRect& top, const cv::RotatedRect& bottom)
    {
        std::vector<cv::Point2f> points;

        cv::Point2f topPts[4], bottomPts[4];
        top.points(topPts);
        bottom.points(bottomPts);
        for (int i = 0; i < 4; ++i)
        {
            points.push_back(topPts[i]);
            points.push_back(bottomPts[i]);
        }

        float minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
        for (const auto& p : points)
        {
            minX = std::min(minX, p.x);
            minY = std::min(minY, p.y);
            maxX = std::max(maxX, p.x);
            maxY = std::max(maxY, p.y);
        }

        return cv::Rect2f(minX, minY, maxX - minX, maxY - minY); 
    }

    std::vector<cv::Point2f> calculateROIVertices(const cv::RotatedRect& top, const cv::RotatedRect& bottom)
    {
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

    float calculateConfidence(const UAVTarget& target)
    {
        float confidence = 0;

        float angle_diff = std::abs(target.top_lb.angle - target.bottom_lb.angle);
        angle_diff = std::min(angle_diff, 180.0f - angle_diff);
        confidence += (1.0f - angle_diff / params.max_angle_diff) * 0.2f;

        float top_len = std::max(target.top_lb.size.width, target.top_lb.size.height);
        float bottom_len = std::max(target.bottom_lb.size.width, target.bottom_lb.size.height);
        float len_ratio = std::min(top_len, bottom_len) / std::max(top_len, bottom_len);
        confidence += len_ratio * 0.2f;

        float spacing_ratio = target.lb_spacing / target.lb_length;
        if (spacing_ratio >= params.min_spacing_ratio && spacing_ratio <= params.max_spacing_ratio)
        {
            float ideal_ratio = (params.min_spacing_ratio + params.max_spacing_ratio) / 2;
            float ratio_diff = std::abs(spacing_ratio - ideal_ratio) / ideal_ratio;
            confidence += (1.0f - ratio_diff) * 0.6f;
        }

        return confidence;
    }

    bool validateTarget(const UAVTarget& target)
    {
        return target.confidence >= params.min_confidence;
    }

    int assignID(const UAVTarget& target)
    {
        return next_id++;
    }
};