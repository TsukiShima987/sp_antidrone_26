#pragma once

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "target.hpp"

namespace tools {

class ExtendedKalmanFilter;

class Tracker {
public:
    explicit Tracker(const std::string& config_path);
    ~Tracker();

    /// Update filter with a target's (yaw, pitch) measurement.
    /// Reads target.yaw / target.pitch, writes target.predict_yaw / target.predict_pitch.
    void update(UAVTarget& target, double dt);

    /// Reset filter — next update() will re-initialize.
    void reset();

    bool isInitialized() const { return ekf_ != nullptr; }

    /// Access EKF diagnostics (NIS, NEES, etc.)
    const std::map<std::string, double>& data() const;

private:
    std::unique_ptr<ExtendedKalmanFilter> ekf_;

    // Noise parameters (loaded from config)
    double q_yaw_;
    double q_pitch_;
    double r_yaw_;
    double r_pitch_;

    // Delay parameters (loaded from config)
    double calculating_delay_;
    double excuting_delay_;
    double adjust_delay_;

    // Angle-aware vector addition / subtraction
    static Eigen::VectorXd xAdd(const Eigen::VectorXd& x, const Eigen::VectorXd& delta);
    static Eigen::VectorXd zSubtract(const Eigen::VectorXd& z, const Eigen::VectorXd& hx);

    // Process noise covariance
    Eigen::MatrixXd computeQ(double dt);

    // Observation model h(x) = [yaw, pitch]^T
    static Eigen::VectorXd observationModel(const Eigen::VectorXd& x);
};

}  // namespace tools
