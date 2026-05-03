#include "solver.hpp"
#include <chrono>
#include <vector>

namespace tools
{
  Solver::Solver() : R_gimbal2world_(Eigen::Matrix3d::Identity()), R_gimbal2imubody_(Eigen::Matrix<double,3,3,Eigen::RowMajor>::Identity())
  {
  }

  Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

  void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
  {
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
  }
}