#include "camera2gimbal.hpp"
#include "math_tools.hpp"

namespace tools
{
    Camera2Gimball::Camera2Gimball()
    {
    };
    cv::Point3d Camera2Gimball::Camera2Gimballt(cv::Point3f cam_point)
    {
        double point_x = cam_point.z + 87.15;
        double point_y = 15 - cam_point.x; //not accuarate prob
        double point_z = 21.5 - cam_point.y;
        cv::Point3d gim_point(point_x, point_y, point_z);
        return gim_point;
    };
    cv::Point3d Camera2Gimball::Camera2GimballYawPitch2Point(double dist,double current_yaw, double current_pitch)
    {
        Eigen::Vector3d point_ypd(current_yaw, current_pitch, dist);
        Eigen::Vector3d point_xyz = tools::ypd2xyz(point_ypd);
        std::cout << "point in camera: " << point_xyz << std::endl;
        double point_x = point_xyz.z() + 87.15;
        double point_y = 15 - point_xyz.x(); 
        double point_z = 21.5 - point_xyz.y();
        cv::Point3d gim_point(point_x, point_y, point_z);
        return gim_point;
    };

}