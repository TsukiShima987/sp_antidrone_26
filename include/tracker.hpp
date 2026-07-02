#pragma once

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <utility>

namespace tools {

class ExtendedKalmanFilter;

class Tracker {
public:
    Tracker();
    ~Tracker();

    /// Update filter with a new (yaw, pitch) measurement.
    /// @returns filtered (yaw, pitch) in radians.
    std::pair<double, double> update(double yaw_raw, double pitch_raw, double dt);

    /// Reset filter — next update() will re-initialize.
    void reset();

    bool isInitialized() const { return ekf_ != nullptr; }

    /// Access EKF diagnostics (NIS, NEES, etc.)
    const std::map<std::string, double>& data() const;

private:
    std::unique_ptr<ExtendedKalmanFilter> ekf_;

    // Noise parameters
    static constexpr double q_yaw_   = 0.35;
    static constexpr double q_pitch_ = 0.23;
    static constexpr double r_yaw_   = 0.003;
    static constexpr double r_pitch_ = 0.003;

    // Angle-aware vector addition / subtraction
    static Eigen::VectorXd xAdd(const Eigen::VectorXd& x, const Eigen::VectorXd& delta);
    static Eigen::VectorXd zSubtract(const Eigen::VectorXd& z, const Eigen::VectorXd& hx);

    // Process noise covariance
    static Eigen::MatrixXd computeQ(double dt);

    // Observation model h(x) = [yaw, pitch]^T
    static Eigen::VectorXd observationModel(const Eigen::VectorXd& x);
};

}  // namespace tools
