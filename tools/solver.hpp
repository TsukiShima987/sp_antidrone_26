#ifndef TOOLS__SOLVER_HPP
#define TOOLS__SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

namespace tools
{
    class Solver{
    public:
        Solver();
        Eigen::Matrix3d R_gimbal2world() const;
        void set_R_gimbal2world(const Eigen::Quaterniond & q);

    private:
        Eigen::Matrix3d R_gimbal2imubody_;
        Eigen::Matrix3d R_gimbal2world_;

    };
}

#endif  // TOOLS__SOLVER_HPP