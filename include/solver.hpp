// #include "detector.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "target.hpp"
#include "../io/gimbal/gimbal.hpp"
// #include "../io/camera.hpp"
#include "tools/plotter.hpp"
#include "tools/math_tools.hpp"


class Solver {
public:
    Solver(std::string config_path) : 
        R_gimbal2world_(Eigen::Matrix3d::Identity()), 
        R_gimbal2imubody_(Eigen::Matrix<double,3,3,Eigen::RowMajor>::Identity()) 
    {
        const auto config = YAML::LoadFile(config_path);

        auto T_camera2gimbal_data = config["T_camera2gimbal"].as<std::vector<double>>();
        T_camera2gimbal = cv::Matx44d(T_camera2gimbal_data.data());

    }

    void solve(UAVTarget& target) {
        double x = target.position.x;
        double y = target.position.y;
        double z = target.position.z;

        cv::Mat p_camera = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);

        cv::Mat p_gimbal_h = T_camera2gimbal * p_camera;
        cv::Point3d rel_gim(p_gimbal_h.at<double>(0), p_gimbal_h.at<double>(1), p_gimbal_h.at<double>(2));

        Eigen::Vector3d p_gimbal(rel_gim.x, rel_gim.y, rel_gim.z);

        Eigen::Vector3d p_world = gimbal2world() * p_gimbal;

        double world_x = p_world.x();
        double world_y = p_world.y();
        double world_z = p_world.z();

        target.yaw = std::atan2(world_y, world_x);
        target.pitch = -std::atan2(world_z, sqrt(world_x * world_x + world_y * world_y));
    }

    void set_R_gimbal2world(const Eigen::Quaterniond & q)
    {
        Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
        R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
    }

private:
    cv::Matx44d T_camera2gimbal;
    
    Eigen::Matrix3d R_gimbal2imubody_;
    Eigen::Matrix3d R_gimbal2world_;

    Eigen::Matrix3d gimbal2world() const { return R_gimbal2world_; }

};