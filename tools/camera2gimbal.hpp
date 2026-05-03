#ifndef CAMERA2GIMBALL_HPP
#define CAMERA2GIMBALL_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <fstream>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <string>
#include <utility>
#include <fmt/chrono.h>
#include <filesystem>
#include <string>
#include "math_tools.hpp"
#include <condition_variable>
#include <opencv2/opencv.hpp>

#define BUFFER_SIZE 10

namespace tools
{
  class Camera2Gimball
  {
  public:
    Camera2Gimball();

    cv::Point3d Camera2Gimballt(cv::Point3f cam_point);
    cv::Point3d Camera2GimballYawPitch2Point(double dist,double current_yaw, double current_pitch);

  };
}

#endif // CAMERA2GIMBALL_HPP