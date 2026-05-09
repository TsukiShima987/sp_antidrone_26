#ifndef TARGETYAWPITCH_HPP
#define TARGETYAWPITCH_HPP

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

#define BUFFER_SIZE 10

namespace tools
{
  class TargetYawPitch
  {
  public:
    TargetYawPitch();

    std::tuple<double, double, double> TargetYawPitch_Calculator(double dist,double current_yaw, double current_pitch);
    cv::Point3d TargetXYZ(double dist);

  private:
    Eigen::Vector3d line_point;      
    Eigen::Vector3d line_direction;
    Eigen::Matrix3d K;               
  };
}

#endif // TARGETYAWPITCH_HPP