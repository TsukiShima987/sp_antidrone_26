#include "solver.hpp"
#include "scanner.hpp"
#include <yaml-cpp/yaml.h>
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/math_tools.hpp"
#include "io/camera.hpp"
#include "tracker.hpp"
#include "detector.hpp"
#include "scanner.hpp"
#include "target.hpp"

int main(int argc, char** argv) {
    cv::Mat frame;

    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_timestamp = timestamp;
    bool first_measurement = true;

    cv::namedWindow("UAV Detector - Camera", 0);
    double fps_ = 30.0;            

    std::string config_path = "config/antidrone.yaml";
    auto config = YAML::LoadFile(config_path);

    bool is_recording_ = config["record_video"].as<bool>();

    bool scan_on = config["scan_option"].as<bool>();
    bool start_check = config["start_check"].as<bool>();

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
  
        std::unique_ptr<io::Camera> camera;
        // auto needed_file_ = std::make_unique<std::string>();
        // *needed_file_ = config["camera_config_file"].as<std::string>();
        camera = std::make_unique<io::Camera>(config_path);
        Solver solver(config_path);
        io::Gimbal gimbal(config_path);
        Detector detector(config_path);
        tools::Tracker tracker;
        solver.set_gimbal(&gimbal);
        Scanner scanner(config_path);

        tools::Recorder recorder(fps_);
        tools::Plotter plotter("127.0.0.1", 9870);
        solver.set_plotter(&plotter);

        if (start_check)
        {
            std::cout << "Sending gimbal reset command for 10 seconds..." << std::endl;
            auto reset_start = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - reset_start).count() < 10.0) {
                gimbal.send(1 , 1, 0, 0, 0, 0, 0, 0);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            std::cout << "Gimbal reset finished. Entering main loop." << std::endl;
        }

        while (true)
        {
            cv::Mat frame;
            std::chrono::steady_clock::time_point timestamp;
            camera->read(frame, timestamp);
            if (frame.empty()) {
                std::cerr << "Error: Empty frame from camera" << std::endl;
                break;
            }
            double dt = 0.0;
            if (!first_measurement) {
                dt = std::chrono::duration<double>(timestamp - last_timestamp).count();
                if (dt > 0.1) dt = 0.1;
            }
            last_timestamp = timestamp;
            first_measurement = false;

            auto q = gimbal.q(timestamp);
            if (is_recording_) {
                        // video_writer_.write(frame);
                    recorder.record(frame, q, timestamp);
            }

            // visualizer.visualizeFrame(frame, timestamp);
                // 检测无人机
            std::vector<UAVTarget> targets;

            targets = detector.detect(frame, timestamp);
            cv::Mat display = detector.visualize(frame, targets);
            cv::resizeWindow("UAV Detector - Camera", cv::Size(1920, 1280));
            cv::imshow("UAV Detector - Camera", display);

            char key = cv::waitKey(1);
            if (key == 'q') break;


            // // --- plot frame timing ---
            // {
            //     nlohmann::json j;
            //     j["type"] = "timing";
            //     j["dt"] = dt;
            //     j["fps"] = (dt > 0) ? (1.0 / dt) : 0.0;
            //     plotter.plot(j);
            // }

            auto q_s = gimbal.state();

            // --- plot gimbal state ---
            {
                nlohmann::json j;
                j["type"] = "gimbal_state";
                j["yaw_deg"] = tools::rad2deg(q_s.yaw);
                j["pitch_deg"] = tools::rad2deg(q_s.pitch);
                j["yaw_vel"] = q_s.yaw_vel;
                j["pitch_vel"] = q_s.pitch_vel;
                j["bullet_speed"] = q_s.bullet_speed;
                j["supercap_voltage"] = q_s.supercap_voltage;
                plotter.plot(j);
            }

            std::cout << "Gimbal State - Yaw: " << tools::rad2deg(q_s.yaw)
                      << ", Pitch: " << tools::rad2deg(q_s.pitch) << std::endl;

            if (!targets.empty()) {
                scanner.reset(timestamp);

                // --- plot raw detection ---
                {
                    const auto& t = targets[0];
                    nlohmann::json j;
                    j["type"] = "detection";
                    j["has_target"] = true;
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
                    plotter.plot(j);
                }

                solver.solve(targets[0], timestamp);
                std::cout << "Aiming at target - Yaw: " << tools::rad2deg(targets[0].yaw)
                          << " degrees, Pitch: " << tools::rad2deg(targets[0].pitch) << " degrees" << std::endl;

                // --- plot raw aim angles ---
                {
                    nlohmann::json j;
                    j["type"] = "aim";
                    j["yaw_raw_deg"] = tools::rad2deg(targets[0].yaw);
                    j["pitch_raw_deg"] = tools::rad2deg(targets[0].pitch);
                    plotter.plot(j);
                }
                // 计算从相机读取到这一行的时间间隔
                auto time_now = std::chrono::steady_clock::now();
                double time_interval = std::chrono::duration<double>(time_now - timestamp).count();
                {
                    nlohmann::json j;
                    j["type"] = "time_interval";
                    j["time_interval"] = time_interval;
                    plotter.plot(j);
                }

                tracker.update(targets[0], dt);
                const auto& ekf_data = tracker.data();

                // // --- plot tracker / EKF diagnostics ---
                // {
                //     nlohmann::json j;
                //     j["type"] = "tracker";
                //     j["yaw_filt_deg"] = tools::rad2deg(yaw_filt);
                //     j["pitch_filt_deg"] = tools::rad2deg(pitch_filt);
                //     j["yaw_raw_deg"] = tools::rad2deg(yaw_raw);
                //     j["pitch_raw_deg"] = tools::rad2deg(pitch_raw);
                //     j["nis"] = (ekf_data.count("nis") ? ekf_data.at("nis") : 0.0);
                //     j["nees"] = (ekf_data.count("nees") ? ekf_data.at("nees") : 0.0);
                //     j["nis_fail"] = (ekf_data.count("nis_fail") ? ekf_data.at("nis_fail") : 0.0);
                //     j["nees_fail"] = (ekf_data.count("nees_fail") ? ekf_data.at("nees_fail") : 0.0);
                //     plotter.plot(j);
                // }

                std::cout << "Filtered - Yaw: " << tools::rad2deg(targets[0].predict_yaw)
                          << "°, Pitch: " << tools::rad2deg(targets[0].predict_pitch) << "°" << std::endl;

                std::cout << "EKF Data - NIS: " << (ekf_data.count("nis") ? ekf_data.at("nis") : 0)
                          << ", NEES: " << (ekf_data.count("nees") ? ekf_data.at("nees") : 0)
                          << ", NIS Fail: " << (ekf_data.count("nis_fail") ? ekf_data.at("nis_fail") : 0)
                          << ", NEES Fail: " << (ekf_data.count("nees_fail") ? ekf_data.at("nees_fail") : 0)
                          << std::endl;

                // Send to gimbal
                gimbal.send(true, false, targets[0].yaw, 0, 0, targets[0].pitch, 0, 0);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

            } else {
                // --- plot no-target status ---
                {
                    nlohmann::json j;
                    j["type"] = "detection";
                    j["has_target"] = false;
                    plotter.plot(j);
                }

                if (scan_on){
                    auto scan_pos = scanner.update(timestamp);
                    if (scan_pos) {
                        std::cout << "Scanning - Yaw: " << tools::rad2deg(scan_pos->first)
                                << "°, Pitch: " << tools::rad2deg(scan_pos->second) << "°" << std::endl;

                        // --- plot scan position ---
                        {
                            nlohmann::json j;
                            j["type"] = "scan";
                            j["yaw_deg"] = tools::rad2deg(scan_pos->first);
                            j["pitch_deg"] = tools::rad2deg(scan_pos->second);
                            plotter.plot(j);
                        }

                        gimbal.send(true, false, scan_pos->first, 0, 0, scan_pos->second, 0, 0);
                    }
                }
                else {
                    continue;
                }
            }
        }
        is_recording_ = false;
    }
    else if (mode == "video") {
        // 视频文件模式
        if (argc < 3) {
            std::cerr << "Error: Please provide video path" << std::endl;
            std::cout << "Usage: " << argv[0] << " video <video_path>" << std::endl;
            return -1;
        }

        Detector detector(config_path);
        Scanner scanner(config_path);
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
        
        while (true) {
            if (!paused) {
                cap >> frame;
                frame_count++;
                
                if (frame.empty()) {
                    std::cout << "End of video" << std::endl;
                    break;
                }
                
                timestamp = std::chrono::steady_clock::now();
            }
            
            std::vector<UAVTarget> targets;

            targets = detector.detect(frame, timestamp);

            // --- plot detection data (video mode) ---
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
    }
    else {
        std::cerr << "Error: Unknown mode. Use 'camera' or 'video'" << std::endl;
        return -1;
    }
    
    cv::destroyAllWindows();
    return 0;
}