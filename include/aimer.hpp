#include "detector.hpp"
#include <yaml-cpp/yaml.h>
#include "../io/gimbal/gimbal.hpp"
#include "../io/camera.hpp"


class Aimer {
public:
    Aimer(std::string config_path) {
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

    std::pair<double, double> aim(const UAVTarget& target, std::chrono::steady_clock::time_point timestamp) {
        double x = target.position.x;
        double y = target.position.y;
        double z = target.position.z;
        std::cout << "Target position (camera frame) - x: " << x << " m, y: " << y << " m, z: " << z << " m" << std::endl;

        cv::Mat p_camera = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);

        cv::Mat p_gimbal_h = T_camera2gimbal * p_camera;
        std::cout << "Target position (gimbal frame) - x: " << p_gimbal_h.at<double>(0) << " m, y: " << p_gimbal_h.at<double>(1) << " m, z: " << p_gimbal_h.at<double>(2) << " m" << std::endl;
        cv::Point3d rel_gim(p_gimbal_h.at<double>(0), p_gimbal_h.at<double>(1), p_gimbal_h.at<double>(2));

        tools::Solver solver;
        auto q = gimbal->q(timestamp);
        solver.set_R_gimbal2world(q);
        Eigen::Vector3d p_gimbal(rel_gim.x, rel_gim.y, rel_gim.z);
        Eigen::Vector3d p_world = solver.R_gimbal2world() * p_gimbal;
        std::cout << "Target position (world frame) - x: " << p_world.x() << " m, y: " << p_world.y() << " m, z: " << p_world.z() << " m" << std::endl;

        double world_x = p_world.x();
        double world_y = p_world.y();
        double world_z = p_world.z();

        double yaw = std::atan2(world_y, world_x);
        double pitch = -std::atan2(world_z, sqrt(world_x * world_x + world_y * world_y));

        return std::make_pair(yaw, pitch);
    }

private:
    io::Gimbal * gimbal;
    cv::Mat T_camera2gimbal;
};