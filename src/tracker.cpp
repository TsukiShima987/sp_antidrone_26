#include "include/tracker.hpp"
#include "tools/extended_kalman_filter.hpp"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>

namespace tools {

Tracker::Tracker(const std::string& config_path) {
    auto config = YAML::LoadFile(config_path);
    q_yaw_              = config["tracker_q_yaw"].as<double>();
    q_pitch_            = config["tracker_q_pitch"].as<double>();
    r_yaw_              = config["tracker_r_yaw"].as<double>();
    r_pitch_            = config["tracker_r_pitch"].as<double>();
    calculating_delay_  = config["tracker_calculating_delay"].as<double>();
    excuting_delay_     = config["tracker_excuting_delay"].as<double>();
    adjust_delay_       = config["tracker_adjust_delay"].as<double>();
}

Tracker::~Tracker() = default;

void Tracker::update(UAVTarget& target, double dt) {
    double yaw_raw = target.yaw;
    double pitch_raw = target.pitch;
    // --- first measurement: initialize EKF ---
    if (!ekf_) {
        Eigen::VectorXd x0(4);
        x0 << yaw_raw, pitch_raw, 0.0, 0.0;
        Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4) * 0.1;
        ekf_ = std::make_unique<ExtendedKalmanFilter>(x0, P0, xAdd);
        std::cout << "Tracker: EKF initialized" << std::endl;
    }

    // --- predict ---
    if (dt > 0) {
        Eigen::MatrixXd F(4, 4);
        F << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1,  0,
             0, 0, 0,  1;
        Eigen::MatrixXd Q = computeQ(dt);
        ekf_->predict(F, Q);
    }

    // --- update ---
    Eigen::VectorXd z(2);
    z << yaw_raw, pitch_raw;

    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2, 2);
    R.diagonal() << r_yaw_, r_pitch_;

    ekf_->update(z, H, R, observationModel, zSubtract);

    double delay = calculating_delay_ + excuting_delay_ + adjust_delay_;

    Eigen::VectorXd x_filt = ekf_->getState();
    target.predict_yaw = x_filt(0) + x_filt(2) * delay;
    target.predict_pitch = x_filt(1) + x_filt(3) * delay;
}

void Tracker::reset() {
    ekf_.reset();
}

const std::map<std::string, double>& Tracker::data() const {
    // Return a reference to the EKF's data map (empty static fallback if ekf is null)
    static const std::map<std::string, double> empty;
    return ekf_ ? ekf_->data : empty;
}

// ---------------------------------------------------------------------------
// Private helper functions
// ---------------------------------------------------------------------------

Eigen::VectorXd Tracker::xAdd(const Eigen::VectorXd& x, const Eigen::VectorXd& delta) {
    Eigen::VectorXd x_new = x + delta;
    // Normalize yaw to [-π, π]
    x_new(0) = std::atan2(std::sin(x_new(0)), std::cos(x_new(0)));
    // Normalize pitch to [-π, π]
    x_new(1) = std::atan2(std::sin(x_new(1)), std::cos(x_new(1)));
    return x_new;
}

Eigen::VectorXd Tracker::zSubtract(const Eigen::VectorXd& z, const Eigen::VectorXd& hx) {
    Eigen::VectorXd diff = z - hx;
    diff(0) = std::atan2(std::sin(diff(0)), std::cos(diff(0)));  // yaw
    diff(1) = std::atan2(std::sin(diff(1)), std::cos(diff(1)));  // pitch
    return diff;
}

Eigen::MatrixXd Tracker::computeQ(double dt) {
    Eigen::MatrixXd Q(4, 4);
    Q.setZero();
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    Q(0, 0) = q_yaw_   * dt3 / 3.0;   Q(0, 2) = q_yaw_   * dt2 / 2.0;
    Q(1, 1) = q_pitch_ * dt3 / 3.0;   Q(1, 3) = q_pitch_ * dt2 / 2.0;
    Q(2, 0) = q_yaw_   * dt2 / 2.0;   Q(2, 2) = q_yaw_   * dt;
    Q(3, 1) = q_pitch_ * dt2 / 2.0;   Q(3, 3) = q_pitch_ * dt;
    return Q;
}

Eigen::VectorXd Tracker::observationModel(const Eigen::VectorXd& x) {
    Eigen::VectorXd z_pred(2);
    z_pred << x(0), x(1);
    return z_pred;
}

}  // namespace tools
