#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "io/camera.hpp"
#include "detector.hpp"

class UAVDetectorVisualizer {
private:
    UAVDetector detector;
    cv::Scalar colors[3] = {
        cv::Scalar(0, 255, 0),   // 绿色 - 正常检测
        cv::Scalar(0, 255, 255), // 黄色 - 中等置信度
        cv::Scalar(0, 0, 255)    // 红色 - 低置信度
    };
    
public:
    void visualizeFrame(cv::Mat& frame, std::chrono::steady_clock::time_point timestamp) {
        // 检测无人机
        std::vector<UAVTarget> targets = detector.detectUAVs(frame, timestamp);
        
        // 在图像上绘制检测结果
        for (const auto& target : targets) {
            drawTarget(frame, target);
        }
        
        // 显示检测信息
        showDetectionInfo(frame, targets);
    }
    
private:
    void drawTarget(cv::Mat& frame, const UAVTarget& target) {
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
    
    void showDetectionInfo(cv::Mat& frame, const std::vector<UAVTarget>& targets) {
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
};

int main(int argc, char** argv) {
    UAVDetectorVisualizer visualizer;
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    cv::namedWindow("UAV Detector - Camera", 0);

    // 解析命令行参数
    if (argc < 2) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  For camera: " << argv[0] << " camera [camera_id]" << std::endl;
        std::cout << "  For video:  " << argv[0] << " video <video_path>" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " camera 0" << std::endl;
        std::cout << "  " << argv[0] << " video test.mp4" << std::endl;
        return -1;
    }
    
    std::string mode = argv[1];
    
    if (mode == "camera") {
        // // 摄像头模式
        // int camera_id = 1;
        // if (argc >= 3) {
        //     camera_id = std::stoi(argv[2]);
        // }
        
        // cv::VideoCapture cap(camera_id, cv::CAP_V4L2);
        // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        // if (!cap.isOpened()) {
        //     std::cerr << "Error: Could not open camera " << camera_id << std::endl;
        //     return -1;
        // }
        
        // std::cout << "Camera mode - Press 'q' to quit, 's' to save current frame" << std::endl;
        
        // while (true) {
        //     cap >> frame;
        //     if (frame.empty()) {
        //         std::cerr << "Error: Empty frame from camera" << std::endl;
        //         break;
        //     }
            
        //     timestamp = cv::getTickCount() / cv::getTickFrequency();
        //     visualizer.visualizeFrame(frame, timestamp);
            
        //     cv::imshow("UAV Detector - Camera", frame);
            
        //     char key = cv::waitKey(1);
        //     if (key == 'q') break;
        //     if (key == 's') {
        //         cv::imwrite("detection_capture.jpg", frame);
        //         std::cout << "Frame saved as detection_capture.jpg" << std::endl;
        //     }
        // }
        
        // cap.release();
        auto config = YAML::LoadFile("config/camera.yaml");
        std::unique_ptr<io::Camera> camera;
        auto needed_file_ = std::make_unique<std::string>();
        *needed_file_ = config["camera_config_file"].as<std::string>();
        camera = std::make_unique<io::Camera>(*needed_file_);
        while (true)
        {
            cv::Mat frame;
            std::chrono::steady_clock::time_point timestamp;
            camera->read(frame, timestamp);
            if (frame.empty()) {
                std::cerr << "Error: Empty frame from camera" << std::endl;
                break;
            }
            visualizer.visualizeFrame(frame, timestamp);

            cv::resizeWindow("UAV Detector - Camera", cv::Size(1920, 1280));
            cv::imshow("UAV Detector - Camera", frame);

            char key = cv::waitKey(1);
            if (key == 'q') break;
            if (key == 's') {
                cv::imwrite("detection_capture.jpg", frame);
                std::cout << "Frame saved as detection_capture.jpg" << std::endl;
            }        
        }
    }
    else if (mode == "video") {
        // 视频文件模式
        if (argc < 3) {
            std::cerr << "Error: Please provide video path" << std::endl;
            std::cout << "Usage: " << argv[0] << " video <video_path>" << std::endl;
            return -1;
        }
        
        std::string video_path = argv[2];
        cv::VideoCapture cap(video_path);
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << video_path << std::endl;
            return -1;
        }
        
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        int frame_count = 0;
        
        std::cout << "Video mode - File: " << video_path << std::endl;
        std::cout << "FPS: " << fps << ", Total frames: " << total_frames << std::endl;
        std::cout << "Controls: SPACE - pause/resume, 'q' - quit, 's' - save frame" << std::endl;
        
        bool paused = false;
        
        while (true) {
            if (!paused) {
                cap >> frame;
                frame_count++;
                
                if (frame.empty()) {
                    std::cout << "End of video" << std::endl;
                    break;
                }
                
                timestamp = std::chrono::steady_clock::now();
                visualizer.visualizeFrame(frame, timestamp);
            }
            
            // 显示进度
            std::string progress = cv::format("Frame: %d/%d (%.1f%%)", 
                frame_count, total_frames, 
                (float)frame_count / total_frames * 100);
            cv::putText(frame, progress, cv::Point(frame.cols - 200, 25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            
            cv::imshow("UAV Detector - Video", frame);
            
            char key = cv::waitKey(paused ? 0 : 1);
            if (key == 'q') break;
            if (key == ' ') paused = !paused;
            if (key == 's') {
                cv::imwrite(cv::format("detection_frame_%d.jpg", frame_count), frame);
                std::cout << "Frame " << frame_count << " saved" << std::endl;
            }
        }
        
        cap.release();
    }
    else {
        std::cerr << "Error: Unknown mode. Use 'camera' or 'video'" << std::endl;
        return -1;
    }
    
    cv::destroyAllWindows();
    return 0;
}