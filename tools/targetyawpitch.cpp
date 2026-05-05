#include "targetyawpitch.hpp"
#include <fmt/chrono.h>
#include <filesystem>
#include <string>
#include "math_tools.hpp"
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

namespace tools
{
    TargetYawPitch::TargetYawPitch()
    {
        line_point << 34.41870511, 13.87355278,  0.0;
        line_direction << -0.01766465,  0.00665305, -0.99982183;      

        K << 30164.346973727235, 0,                  1585.3978670158735, 
             0,                  29990.338352363626, 1027.6312267976482, 
             0,                  0,                  1     ;             


    };
    std::tuple<double, double, double> TargetYawPitch::TargetYawPitch_Calculator(double dist,double current_yaw, double current_pitch)
    {
        Eigen::Vector3d target_cam(0, 0, dist*1000);
        Eigen::Vector3d v = target_cam - line_point;

        double t = v.dot(line_direction);  

        Eigen::Vector3d closest_point = line_point + t * line_direction;

        Eigen::Vector3d laser_dir = closest_point.normalized();
        double laser_yaw = atan2(laser_dir.x(), laser_dir.z());
        double laser_pitch = atan2(laser_dir.y(), laser_dir.z());
        // the angles are in radians 
        double target_yaw = current_yaw - laser_yaw;
        double target_pitch = current_pitch - laser_pitch;

        return std::make_tuple(target_yaw, target_pitch, closest_point.z());
    }

    cv::Point3d TargetYawPitch::TargetXYZ(double dist,double current_yaw, double current_pitch)
    {
        Eigen::Vector3d target_cam(0, 0, dist*1000);
        Eigen::Vector3d v = target_cam - line_point;

        double t = v.dot(line_direction);  

        Eigen::Vector3d closest_point = line_point + t * line_direction;
        return cv::Point3d(closest_point.x(), closest_point.y(), closest_point.z());
    }

}