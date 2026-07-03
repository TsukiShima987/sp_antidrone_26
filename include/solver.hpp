// #include "detector.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "target.hpp"
#include "../io/gimbal/gimbal.hpp"
// #include "../io/camera.hpp"
#include "tools/plotter.hpp"


class Solver {
public:
    Solver(std::string config_path) : 
        R_gimbal2world_(Eigen::Matrix3d::Identity()), 
        R_gimbal2imubody_(Eigen::Matrix<double,3,3,Eigen::RowMajor>::Identity()) 
    {
        T_camera2gimbal = cv::Mat::eye(4, 4, CV_64F);
        try {
            YAML::Node config = YAML::LoadFile(config_path);
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

    void set_gimbal(io::Gimbal * gimbal_ptr) {
        gimbal = gimbal_ptr;
    }

    void set_plotter(tools::Plotter * plotter_ptr) {
        plotter = plotter_ptr;
    }

    void solve(UAVTarget& target, std::chrono::steady_clock::time_point timestamp) {
        double x = target.position.x;
        double y = target.position.y;
        double z = target.position.z;
        std::cout << "Target position (camera frame) - x: " << x << " m, y: " << y << " m, z: " << z << " m" << std::endl;

        cv::Mat p_camera = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);

        cv::Mat p_gimbal_h = T_camera2gimbal * p_camera;
        std::cout << "Target position (gimbal frame) - x: " << p_gimbal_h.at<double>(0) << " m, y: " << p_gimbal_h.at<double>(1) << " m, z: " << p_gimbal_h.at<double>(2) << " m" << std::endl;
        cv::Point3d rel_gim(p_gimbal_h.at<double>(0), p_gimbal_h.at<double>(1), p_gimbal_h.at<double>(2));

        // tools::Solver solver;
        Eigen::Vector3d p_gimbal(rel_gim.x, rel_gim.y, rel_gim.z);
        auto q = gimbal->q(timestamp);
        set_R_gimbal2world(q);
        Eigen::Vector3d p_world = gimbal2world() * p_gimbal;
        std::cout << "Target position (world frame) - x: " << p_world.x() << " m, y: " << p_world.y() << " m, z: " << p_world.z() << " m" << std::endl;

        // --- plot target position in all three coordinate frames ---
        if (plotter) {
            nlohmann::json j;
            j["type"] = "target_position";
            j["camera"]["x"] = x;
            j["camera"]["y"] = y;
            j["camera"]["z"] = z;
            j["gimbal"]["x"] = rel_gim.x;
            j["gimbal"]["y"] = rel_gim.y;
            j["gimbal"]["z"] = rel_gim.z;
            j["world"]["x"] = p_world.x();
            j["world"]["y"] = p_world.y();
            j["world"]["z"] = p_world.z();
            plotter->plot(j);
        }

        double world_x = p_world.x();
        double world_y = p_world.y();
        double world_z = p_world.z();

        target.yaw = std::atan2(world_y, world_x);
        target.pitch = -std::atan2(world_z, sqrt(world_x * world_x + world_y * world_y));
    }

private:
    io::Gimbal * gimbal = nullptr;
    tools::Plotter * plotter = nullptr;
    cv::Mat T_camera2gimbal;
    
    Eigen::Matrix3d R_gimbal2imubody_;
    Eigen::Matrix3d R_gimbal2world_;

    Eigen::Matrix3d gimbal2world() const { return R_gimbal2world_; }

    void set_R_gimbal2world(const Eigen::Quaterniond & q)
    {
        Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
        R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
    }
};