#include "aimer.hpp"
#include "visualizer.hpp"
#include "scanner.hpp"
#include <yaml-cpp/yaml.h>


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
        std::string config_path = "config/antidrone.yaml";
        // auto config = YAML::LoadFile(config_path);
        std::unique_ptr<io::Camera> camera;
        // auto needed_file_ = std::make_unique<std::string>();
        // *needed_file_ = config["camera_config_file"].as<std::string>();
        camera = std::make_unique<io::Camera>(config_path);
        Aimer aimer(config_path);
        io::Gimbal gimbal(config_path);
        UAVDetector detector;
        aimer.set_gimbal(&gimbal);
        Scanner scanner(config_path);
        while (true)
        {
            cv::Mat frame;
            std::chrono::steady_clock::time_point timestamp;
            camera->read(frame, timestamp);
            if (frame.empty()) {
                std::cerr << "Error: Empty frame from camera" << std::endl;
                break;
            }
            std::vector<UAVTarget> targets = detector.detectUAVs(frame, timestamp);
            visualizer.visualizeFrame(frame, targets);

            auto q_s = gimbal.state();
            std::cout << "Gimbal State - Yaw: " << q_s.yaw * 180 / M_PI
                      << ", Pitch: " << q_s.pitch * 180 / M_PI << std::endl;
            if (!targets.empty()) {
                scanner.reset(timestamp);
                auto [yaw, pitch] = aimer.aim(targets[0], timestamp);
                std::cout << "Aiming at target - Yaw: " << yaw * 180 / M_PI
                          << " degrees, Pitch: " << pitch * 180 / M_PI << " degrees" << std::endl;
                gimbal.send(true, false, yaw, 0, 0, pitch, 0, 0);
            } else {
                auto scan_pos = scanner.update(timestamp);
                if (scan_pos) {
                    std::cout << "Scanning - Yaw: " << scan_pos->first * 180 / M_PI
                              << "°, Pitch: " << scan_pos->second * 180 / M_PI << "°" << std::endl;
                    gimbal.send(true, false, scan_pos->first, 0, 0, scan_pos->second, 0, 0);
                }
            }

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
                // visualizer.visualizeFrame(frame, timestamp);
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