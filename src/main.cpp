#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "antidrone_node.hpp"
#include "detector.hpp"
#include "tools/plotter.hpp"

int main(int argc, char** argv)
{
    std::string mode = "camera";
    if (argc >= 2) {
        mode = argv[1];
    }

    // ---- 视频模式 (离线检测，无需ROS) ----
    if (mode == "video") {
        std::string config_path = "config/antidrone.yaml";
        auto config = YAML::LoadFile(config_path);

        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " video <video_path>" << std::endl;
            return -1;
        }

        Detector detector(config_path);
        tools::Plotter plotter("127.0.0.1", 9870);

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
        cv::Mat frame;

        while (true) {
            if (!paused) {
                cap >> frame;
                frame_count++;
                if (frame.empty()) {
                    std::cout << "End of video" << std::endl;
                    break;
                }
            }

            auto timestamp = std::chrono::steady_clock::now();
            std::vector<UAVTarget> targets;
            targets = detector.detect(frame, timestamp);

            {
                nlohmann::json j;
                j["type"] = "detection";
                j["has_target"] = !targets.empty();
                j["frame_count"] = frame_count;
                if (!targets.empty()) {
                    const auto& t = targets[0];
                    j["id"] = t.id;
                    j["confidence"] = t.confidence;
                    j["center_x"] = t.center.x;
                    j["center_y"] = t.center.y;
                    j["bbox_x"] = t.bounding_box.x;
                    j["bbox_y"] = t.bounding_box.y;
                    j["bbox_w"] = t.bounding_box.width;
                    j["bbox_h"] = t.bounding_box.height;
                    j["pos_x"] = t.position.x;
                    j["pos_y"] = t.position.y;
                    j["pos_z"] = t.position.z;
                    j["distance"] = t.distance;
                }
                plotter.plot(j);
            }

            cv::Mat display = detector.visualize(frame, targets);
            cv::resizeWindow("UAV Detector - Camera", cv::Size(1920, 1280));
            cv::imshow("UAV Detector - Camera", display);

            char key = cv::waitKey(1);
            if (key == 'q') break;
        }

        cap.release();
        cv::destroyAllWindows();
        return 0;
    }

    // ---- 相机模式 (ROS节点，相机循环在独立线程) ----
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AntidroneNode>();
    node->start();  // 启动相机处理线程
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
