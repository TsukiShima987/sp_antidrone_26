#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "../io/camera.hpp"
#include "../include/visualizer.hpp"
#include "../include/detector.hpp"

UAVDetectorVisualizer::UAVDetectorVisualizer()
{
    colors[0] = cv::Scalar(0, 255, 0);
    colors[1] = cv::Scalar(0, 255, 255);
    colors[2] = cv::Scalar(0, 0, 255);
}

void UAVDetectorVisualizer::visualizeFrame(cv::Mat& frame, std::vector<UAVTarget>& targets) {
    // 检测无人机
    // std::vector<UAVTarget> targets = detector.detectUAVs(frame, timestamp);

    // 在图像上绘制检测结果
    for (const auto& target : targets) {
        drawTarget(frame, target);
    }

    // 显示检测信息
    showDetectionInfo(frame, targets);
}

void UAVDetectorVisualizer::drawTarget(cv::Mat& frame, const UAVTarget& target) {
    // 根据置信度选择颜色
    cv::Scalar color;
    if (target.confidence > 0.8) {
        color = colors[0]; // 绿色 - 高置信度
    } else if (target.confidence > 0.6) {
        color = colors[1]; // 黄色 - 中等置信度
    } else {
        color = colors[2]; // 红色 - 低置信度
    }

    // 绘制两个光条（使用旋转矩形）
    cv::Point2f topPts[4], bottomPts[4];
    target.top_lb.points(topPts);
    target.bottom_lb.points(bottomPts);

    // 绘制光条轮廓
    for (int i = 0; i < 4; i++) {
        cv::line(frame, topPts[i], topPts[(i+1)%4], color, 2);
        cv::line(frame, bottomPts[i], bottomPts[(i+1)%4], color, 2);
    }

    // 绘制光条中心点
    cv::circle(frame, target.top_lb.center, 3, color, -1);
    cv::circle(frame, target.bottom_lb.center, 3, color, -1);

    // 绘制边界框
    cv::rectangle(frame, target.bounding_box, color, 1);

    // 绘制ROI顶点（四边形）
    if (target.roi.size() == 4) {
        for (int i = 0; i < 4; i++) {
            cv::line(frame, target.roi[i], target.roi[(i+1)%4], 
                    cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
    }

    // 绘制中心点
    cv::circle(frame, target.center, 4, cv::Scalar(255, 255, 255), -1);

    // 添加标签
    std::string label = cv::format("ID:%d Conf:%.2f", target.id, target.confidence);
    cv::putText(frame, label, 
               cv::Point(target.bounding_box.x, target.bounding_box.y - 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

void UAVDetectorVisualizer::showDetectionInfo(cv::Mat& frame, const std::vector<UAVTarget>& targets) {
    // 在左上角显示统计信息
    cv::rectangle(frame, cv::Rect(5, 5, 200, 70), cv::Scalar(0, 0, 0), -1);

    std::string info = cv::format("Detections: %d", (int)targets.size());
    cv::putText(frame, info, cv::Point(10, 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // 显示每个目标的详细信息
    int y_offset = 45;
    for (size_t i = 0; i < std::min(targets.size(), (size_t)3); i++) {
        std::string target_info = cv::format("ID:%d C:%.2f L:%.1f", 
            targets[i].id, 
            targets[i].confidence,
            targets[i].lb_length);
        cv::putText(frame, target_info, cv::Point(10, y_offset + i*20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
