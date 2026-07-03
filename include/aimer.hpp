#include "detector.hpp"
#include <yaml-cpp/yaml.h>
#include "../io/gimbal/gimbal.hpp"
#include "../io/camera.hpp"
#include "tools/plotter.hpp"


class Aimer {
public:
    Aimer(std::string config_path) {
        const auto config = YAML::LoadFile(config_path);

        auto T_camera2gimbal_data = config["T_camera2gimbal"].as<std::vector<double>>();
        T_camera2gimbal = cv::Matx44d(T_camera2gimbal_data.data());
    }

    void set_gimbal(io::Gimbal * gimbal_ptr) {
        gimbal = gimbal_ptr;
    }

    void set_plotter(tools::Plotter * plotter_ptr) {
        plotter = plotter_ptr;
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

        double yaw = std::atan2(world_y, world_x);
        double pitch = -std::atan2(world_z, sqrt(world_x * world_x + world_y * world_y));

        return std::make_pair(yaw, pitch);
    }

private:
    io::Gimbal * gimbal = nullptr;
    tools::Plotter * plotter = nullptr;
    cv::Matx44d T_camera2gimbal;
};