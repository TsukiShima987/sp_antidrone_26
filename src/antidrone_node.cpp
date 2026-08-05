#include "antidrone_node.hpp"

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "solver.hpp"
#include "detector.hpp"
#include "tracker.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/math_tools.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"

AntidroneNode::AntidroneNode()
: Node("antidrone_node")
{
    config_path_ = "config/antidrone.yaml";
    auto config = YAML::LoadFile(config_path_);

    listen_game_status_ = config["listen_game_status"] ?
        config["listen_game_status"].as<bool>() : true;

    level3_exposure_ms_ = config["level3_exposure_ms"] ?
        config["level3_exposure_ms"].as<double>() : 3.0;
    level3_countdown_s_ = config["level3_countdown_s"] ?
        config["level3_countdown_s"].as<int>() : 180;

    game_status_sub_ = this->create_subscription<radar_msgs::msg::GameStatus>(
        "/game_status", 10,
        std::bind(&AntidroneNode::gameStatusCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "AntidroneNode initialized");
}

AntidroneNode::~AntidroneNode()
{
    quit_ = true;
    if (camera_thread_.joinable()) {
        camera_thread_.join();
    }
}

void AntidroneNode::gameStatusCallback(const radar_msgs::msg::GameStatus::ConstPtr &msg)
{
    game_status_ = *msg;

    // 比赛阶段为4(比赛中) 且 剩余时间 < level3_countdown_s_ 时，触发level3切换
    if (!level3_triggered_ &&
        msg->game_progress == 4 &&
        msg->stage_remain_time < static_cast<uint16_t>(level3_countdown_s_))
    {
        level3_triggered_ = true;
        RCLCPP_INFO(this->get_logger(),
            "Level-3 trigger: game_progress=%d, remain=%d s, exposure=%.1f ms",
            msg->game_progress, msg->stage_remain_time, level3_exposure_ms_);
    }
}

void AntidroneNode::start()
{
    camera_thread_ = std::thread(&AntidroneNode::cameraLoop, this);
}

void AntidroneNode::cameraLoop()
{
    auto config = YAML::LoadFile(config_path_);

    bool is_recording_ = config["record_video"].as<bool>();
    double target_lost_timeout_s = config["target_lost_timeout_s"] ?
        config["target_lost_timeout_s"].as<double>() : 0.0;
    bool start_check = config["start_check"].as<bool>();

    std::unique_ptr<io::Camera> camera;
    camera = std::make_unique<io::Camera>(config_path_);
    Solver solver(config_path_);
    io::Gimbal gimbal(config_path_);
    Detector detector(config_path_);
    tools::Tracker tracker(config_path_);

    tools::Recorder recorder(30.0);
    tools::Plotter plotter("127.0.0.1", 9870);

    cv::namedWindow("UAV Detector - Camera", 0);

    std::chrono::steady_clock::time_point last_timestamp = std::chrono::steady_clock::now();
    bool first_measurement = true;

    // 目标丢失超时保护
    auto last_detection_time = std::chrono::steady_clock::now();
    float last_yaw = 0.0f;
    float last_pitch = 0.0f;

    // 激光标定模式状态
    bool calib_mode = false;
    const double calib_step = 0.0002;  // 每次按键调整约 0.011°

    if (start_check) {
        RCLCPP_INFO(this->get_logger(), "Sending gimbal reset command for 10 seconds...");
        auto reset_start = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(
            std::chrono::steady_clock::now() - reset_start).count() < 10.0)
        {
            gimbal.send(1, 1, 0, 0, 0, 0, 0, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        RCLCPP_INFO(this->get_logger(), "Gimbal reset finished. Entering main loop.");
    }

    while (!quit_)
    {
        // level-3 切换：比赛剩余时间不足时，切换曝光和模型（仅触发一次）
        if (level3_triggered_.load() && !level3_applied_) {
            camera->setExposure(level3_exposure_ms_);
            detector.switchToLevel3();
            level3_applied_ = true;
            RCLCPP_INFO(this->get_logger(),
                "Level-3 applied: exposure=%.1f ms, model switched", level3_exposure_ms_);
        }

        cv::Mat frame;
        std::chrono::steady_clock::time_point timestamp;
        camera->read(frame, timestamp);
        if (frame.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Empty frame from camera");
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
            recorder.record(frame, q, timestamp);
        }

        // 检测无人机
        std::vector<UAVTarget> targets;
        targets = detector.detect(frame, timestamp);
        cv::Mat display = detector.visualize(frame, targets);
        cv::resizeWindow("UAV Detector - Camera", cv::Size(1920, 1280));

        // ---- 标定模式视觉提示 ----
        if (calib_mode) {
            auto d = detector.getLaserDirection();
            std::string info = cv::format("CALIB MODE | d0: [%.6f, %.6f, %.6f] | Step: %.4f rad",
                                          d.x(), d.y(), d.z(), calib_step);
            cv::putText(display, info, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            cv::putText(display, "Arrows: adjust | Enter: save | ESC: discard",
                        cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow("UAV Detector - Camera", display);

        // ---- 键盘处理 ----
        char key = cv::waitKey(1);

        if (calib_mode) {
            switch (key) {
                case 81:  // 左箭头
                    detector.adjustLaserDirection(0, +calib_step);
                    break;
                case 83:  // 右箭头
                    detector.adjustLaserDirection(0, -calib_step);
                    break;
                case 82:  // 上箭头
                    detector.adjustLaserDirection(-calib_step, 0);
                    break;
                case 84:  // 下箭头
                    detector.adjustLaserDirection(+calib_step, 0);
                    break;
                case 13:  // Enter: 保存并退出标定
                case 10:
                    detector.saveCalibration("config/antidrone_calibrated.yaml");
                    calib_mode = false;
                    RCLCPP_INFO(this->get_logger(),
                        "Laser calibration saved to antidrone_calibrated.yaml");
                    break;
                case 27:  // ESC: 放弃标定，恢复原始值
                    detector.reloadLaserParams("config/antidrone.yaml");
                    calib_mode = false;
                    RCLCPP_INFO(this->get_logger(),
                        "Laser calibration discarded, restored original values");
                    break;
                default:
                    break;
            }
        } else {
            if (key == 'a') {
                calib_mode = true;
                RCLCPP_INFO(this->get_logger(),
                    "Entered laser calibration mode. Arrows: adjust | Enter: save | ESC: discard");
            } else if (key == 'q') {
                break;
            }
        }

        auto q_s = gimbal.state();

        // --- plot: quat_euler + gimbal_state ---
        {
            Eigen::Vector3d rpy = q.toRotationMatrix().eulerAngles(2, 1, 0);
            nlohmann::json j;
            j["type"] = "quat_euler";
            j["roll_deg_q"]  = tools::rad2deg(rpy[2]);
            j["pitch_deg_q"] = tools::rad2deg(rpy[1]);
            j["yaw_deg_q"]   = tools::rad2deg(rpy[0]);
            plotter.plot(j);
        }
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

        if (!targets.empty()) {
            solver.set_R_gimbal2world(q);
            solver.solve(targets[0]);

            auto time_now = std::chrono::steady_clock::now();
            double time_interval = std::chrono::duration<double>(time_now - timestamp).count();

            tracker.update(targets[0], dt);
            const auto& ekf_data = tracker.data();

            last_detection_time = std::chrono::steady_clock::now();
            last_yaw = targets[0].predict_yaw;
            last_pitch = targets[0].predict_pitch;

            gimbal.send(true, false, targets[0].predict_yaw, 0, 0,
                            targets[0].predict_pitch, 0, 0);
            

            // --- plot ---
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
            {
                nlohmann::json j;
                j["type"] = "aim";
                j["yaw_raw_deg"] = tools::rad2deg(targets[0].yaw);
                j["pitch_raw_deg"] = tools::rad2deg(targets[0].pitch);
                plotter.plot(j);
            }
            {
                nlohmann::json j;
                j["type"] = "time_interval";
                j["time_interval"] = time_interval;
                plotter.plot(j);
            }
            {
                nlohmann::json j;
                j["type"] = "tracker";
                j["yaw_filt_deg"] = tools::rad2deg(targets[0].predict_yaw);
                j["pitch_filt_deg"] = tools::rad2deg(targets[0].predict_pitch);
                plotter.plot(j);
            }
        } else {
            double time_since_last = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - last_detection_time).count();
            if (time_since_last < target_lost_timeout_s) {
                gimbal.send(true, false, last_yaw, 0, 0, last_pitch, 0, 0);
            } 
            else if (level3_triggered_.load()){
                gimbal.send(true, true, 0, 0, 0, 0, 0, 0);
            }
            else {
                gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
            }
        }
    }

    cv::destroyAllWindows();

    // 通知主线程退出 spin
    rclcpp::shutdown();
}
