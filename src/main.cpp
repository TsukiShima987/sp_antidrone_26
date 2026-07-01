#include "aimer.hpp"
#include "visualizer.hpp"
#include "scanner.hpp"
#include <yaml-cpp/yaml.h>
#include "tools/recorder.hpp"
#include "tools/extended_kalman_filter.hpp"

int main(int argc, char** argv) {
    UAVDetectorVisualizer visualizer;
    cv::Mat frame;

    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    //from here to checkpoint it is for ekf
    tools::ExtendedKalmanFilter* ekf = nullptr;  
    std::chrono::steady_clock::time_point last_timestamp = timestamp;
    bool first_measurement = true;

    auto x_add = [](const Eigen::VectorXd& x, const Eigen::VectorXd& delta) {
    Eigen::VectorXd x_new = x + delta;
    // Normalise yaw to [-π, π]
    x_new(0) = std::atan2(std::sin(x_new(0)), std::cos(x_new(0)));
    // Normalise pitch to [-π, π] (or clamp to [-π/2, π/2] if needed)
    x_new(1) = std::atan2(std::sin(x_new(1)), std::cos(x_new(1)));
    return x_new;
    };

    auto z_subtract = [](const Eigen::VectorXd& z, const Eigen::VectorXd& hx) {
    Eigen::VectorXd diff = z - hx;
    diff(0) = std::atan2(std::sin(diff(0)), std::cos(diff(0))); // yaw
    diff(1) = std::atan2(std::sin(diff(1)), std::cos(diff(1))); // pitch
    return diff;
    };

    auto computeQ = [](double dt, double q_yaw = 0.35, double q_pitch = 0.23) {
    Eigen::MatrixXd Q(4,4);
    Q.setZero();
    double dt2 = dt*dt;
    double dt3 = dt2*dt;
    Q(0,0) = q_yaw * dt3/3.0;   Q(0,2) = q_yaw * dt2/2.0;
    Q(1,1) = q_pitch * dt3/3.0; Q(1,3) = q_pitch * dt2/2.0;
    Q(2,0) = q_yaw * dt2/2.0;   Q(2,2) = q_yaw * dt;
    Q(3,1) = q_pitch * dt2/2.0; Q(3,3) = q_pitch * dt;
    return Q;
    };
    // CheckPoint

    cv::namedWindow("UAV Detector - Camera", 0);
    double fps_ = 30.0;            

    std::string config_path = "config/antidrone.yaml";
    auto config = YAML::LoadFile(config_path);

    bool yolo_detection = config["yolo_option"].as<bool>();
    bool uav_detection = config["uav_option"].as<bool>();
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
        Aimer aimer(config_path);
        io::Gimbal gimbal(config_path);
        UAVDetector detector;
        aimer.set_gimbal(&gimbal);
        Scanner scanner(config_path);

        tools::Recorder recorder(fps_);

        // video_writer_.open(output_video_path_, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps_, ou4tput_size_); // h264
        // if (!video_writer_.isOpened()) {
        //     std::cerr << "Error: Could not open video writer" << std::endl;
        //     return -1;
        // }
        
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
            Bbox maxbbox;
            cv::Mat img;

            if (uav_detection){
                targets = detector.detectUAVs(frame, timestamp);
                visualizer.visualizeFrame(frame, targets);
                cv::resizeWindow("UAV Detector - Camera", cv::Size(1920, 1280));
                cv::imshow("UAV Detector - Camera", frame);

                char key = cv::waitKey(1);
                if (key == 'q') break;

            }

            else if (yolo_detection) {
                std::tie(maxbbox, targets, img) = detector.detectYolos(frame, timestamp);
                if (targets[0].confidence == -1.0f) {
                    targets.clear(); // 如果没有检测到有效目标，清空列表
                }
                cv::resizeWindow("YOLO Detector - Camera", cv::Size(1920, 1280));
                cv::imshow("YOLO Detector - Camera", img);

                char key = cv::waitKey(1);
                if (key == 'q') break;

            }


            auto q_s = gimbal.state();
            std::cout << "Gimbal State - Yaw: " << q_s.yaw * 180 / M_PI
                      << ", Pitch: " << q_s.pitch * 180 / M_PI << std::endl;
            if (!targets.empty()) {
                scanner.reset(timestamp);
                auto [yaw_raw, pitch_raw] = aimer.aim(targets[0], timestamp);
                std::cout << "Aiming at target - Yaw: " << yaw_raw * 180 / M_PI
                          << " degrees, Pitch: " << pitch_raw * 180 / M_PI << " degrees" << std::endl;
                

                if (ekf == nullptr) {
                    Eigen::VectorXd x0(4);
                    x0 << yaw_raw, pitch_raw, 0.0, 0.0;
                    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4,4) * 0.1;
                    ekf = new tools::ExtendedKalmanFilter(x0, P0, x_add);
                    std::cout << "EKF initialized" << std::endl;
                }

                // Predict step
                if (dt > 0) {
                    Eigen::MatrixXd F(4,4);
                    F << 1, 0, dt, 0,
                        0, 1, 0, dt,
                        0, 0, 1, 0,
                        0, 0, 0, 1;
                    Eigen::MatrixXd Q = computeQ(dt, 0.35, 0.23);
                    ekf->predict(F, Q);
                }                
                 // Update step
                
                Eigen::VectorXd z(2);
                z << yaw_raw, pitch_raw;
                Eigen::MatrixXd H(2,4);
                H << 1, 0, 0, 0,
                    0, 1, 0, 0;
                Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2,2);
                R.diagonal() << 0.003, 0.003;   // tune

                auto h = [](const Eigen::VectorXd& x) {
                    Eigen::VectorXd z_pred(2);
                    z_pred << x(0), x(1);
                    return z_pred;
                };

                ekf->update(z, H, R, h, z_subtract);

                Eigen::VectorXd x_filt = ekf->getState();
                double yaw_filt = x_filt(0);
                double pitch_filt = x_filt(1);

                std::cout << "Filtered - Yaw: " << yaw_filt * 180 / M_PI
                        << "°, Pitch: " << pitch_filt * 180 / M_PI << "°" << std::endl;

                std::cout << "EKF Data - NIS: " << ekf->data["nis"] 
                          << ", NEES: " << ekf->data["nees"] 
                          << ", NIS Fail: " << ekf->data["nis_fail"] 
                          << ", NEES Fail: " << ekf->data["nees_fail"] 
                          << std::endl;

                // Send to gimbal
                gimbal.send(true, false, yaw_filt, 0, 0, pitch_filt + (0.26 * M_PI / 180), 0, 0);

            } else {
                if (scan_on){
                    auto scan_pos = scanner.update(timestamp);
                    if (scan_pos) {
                        std::cout << "Scanning - Yaw: " << scan_pos->first * 180 / M_PI
                                << "°, Pitch: " << scan_pos->second * 180 / M_PI << "°" << std::endl;
                        gimbal.send(true, false, scan_pos->first, 0, 0, scan_pos->second, 0, 0);
                    }
                }
                else {
                    continue;
                }
            }



        }
        delete ekf;
        // video_writer_.release();
        is_recording_ = false;
    }
    else if (mode == "video") {
        // 视频文件模式
        if (argc < 3) {
            std::cerr << "Error: Please provide video path" << std::endl;
            std::cout << "Usage: " << argv[0] << " video <video_path>" << std::endl;
            return -1;
        }

        UAVDetector detector;
        Scanner scanner(config_path);

        
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
            
            std::vector<UAVTarget> targets;
            Bbox maxbbox;
            cv::Mat img;

            if (uav_detection){
                targets = detector.detectUAVs(frame, timestamp);
                visualizer.visualizeFrame(frame, targets);
                cv::resizeWindow("UAV Detector - Camera", cv::Size(1920, 1280));
                cv::imshow("UAV Detector - Camera", frame);

                char key = cv::waitKey(1);
                if (key == 'q') break;

            }

            else if (yolo_detection) {
                std::tie(maxbbox, targets, img) = detector.detectYolos(frame, timestamp);
                if (targets[0].confidence == -1.0f) {
                    targets.clear(); // 如果没有检测到有效目标，清空列表
                }
                cv::resizeWindow("YOLO Detector - Camera", cv::Size(1920, 1280));
                cv::imshow("YOLO Detector - Camera", img);

                char key = cv::waitKey(1);
                if (key == 'q') break;

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