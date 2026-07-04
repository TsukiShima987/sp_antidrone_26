#include "include/detector.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <Eigen/Geometry>

BaseDetector::BaseDetector() {
    std::string config_path = "config/antidrone.yaml";
    const auto config = YAML::LoadFile(config_path);

    auto camera_matrix_data = config["camera_matrix"].as<std::vector<double>>();
    auto distort_coeffs_data = config["distort_coeffs"].as<std::vector<double>>();
    auto laser_offset_data = config["laser_offset"].as<std::vector<double>>();
    auto laser_direction_data = config["laser_direction"].as<std::vector<double>>();

    camera_matrix = cv::Matx33d(camera_matrix_data.data());
    dist_coeffs = cv::Mat(distort_coeffs_data);
    S0 = Eigen::Vector3d(laser_offset_data.data());
    d0 = Eigen::Vector3d(laser_direction_data.data());
    d0.normalize();
}

void BaseDetector::estimatePose(UAVTarget& target, float pixel_spacing, float real_size,
                                 const cv::Point2f& center) {
    float fx = camera_matrix(0, 0);
    float fy = camera_matrix(1, 1);
    float cx = camera_matrix(0, 2);
    float cy = camera_matrix(1, 2);

    double z = (fy * real_size) / pixel_spacing;
    double x = (center.x - cx) * z / fx;
    double y = (center.y - cy) * z / fy;

    target.position = computeLaserAimPoint(cv::Point3d(x, y, z));
    target.distance = cv::norm(target.position);
}

cv::Point3d BaseDetector::computeLaserAimPoint(const cv::Point3d& target_cam) {
    
    using namespace Eigen;

    Vector3d p_cam(target_cam.x, target_cam.y, target_cam.z);
    double dist = p_cam.norm();
    if (dist < 1e-6) return target_cam;

    const double tol = 1e-8, step = 1e-6;
    const int maxIter = 50;

    auto buildR = [](double a, double t) -> Matrix3d {
        return AngleAxisd(t, Vector3d(cos(a), sin(a), 0)).toRotationMatrix();
    };

    // Constraint: R^{-1} * p_cam = S0 + λ * d0
    // After gimbal rotates by R, target (in new camera frame) lies on laser ray
    auto residual = [&](double a, double t) -> Vector2d {
        Matrix3d R = buildR(a, t);
        Vector3d diff = R.transpose() * p_cam - S0;
        return diff.cross(d0).head<2>();
    };

    auto canonicalize = [](double &a, double &t) {
        if (t < 0) { t = -t; a += M_PI; }
        a = fmod(a, 2 * M_PI);
        if (a < 0) a += 2 * M_PI;
    };

    // Distance-adaptive: ~1 mrad angular tolerance
    auto valid = [&](double a, double t) -> bool {
        Matrix3d R = buildR(a, t);
        Vector3d diff = R.transpose() * p_cam - S0;
        double cross_threshold = std::max(1e-3 * dist, 1e-3);
        return diff.dot(d0) > 0 && diff.cross(d0).norm() < cross_threshold;
    };

    std::vector<double> seedsA = {0., M_PI/2, M_PI, 3*M_PI/2};
    std::vector<double> seedsT = {0.001, 0.01, 0.1, M_PI/4, M_PI/2, 3*M_PI/4};
    double bestA = 0, bestT = 0, bestRes = 1e20;

    for (double a0 : seedsA) {
        for (double t0 : seedsT) {
            double a = a0, t = t0;
            canonicalize(a, t);
            for (int i = 0; i < maxIter; ++i) {
                Vector2d f = residual(a, t);
                double r = f.norm();
                if (r < tol) {
                    if (valid(a, t)) { bestA = a; bestT = t; bestRes = r; goto done; }
                    else break;
                }
                Matrix2d J;
                for (int j = 0; j < 2; ++j) {
                    double da = (j == 0 ? step : 0), dt = (j == 1 ? step : 0);
                    J.col(j) = (residual(a + da, t + dt) - f) / step;
                }
                Vector2d delta = J.fullPivLu().solve(-f);

                double max_step = 0.5;
                double dn = delta.norm();
                if (dn > max_step) delta *= max_step / dn;

                // Armijo backtracking line search
                double step_size = 1.0;
                double f_norm = r;
                bool accepted = false;
                double anew = a, tnew = t;
                for (int ls = 0; ls < 8; ++ls) {
                    anew = a + step_size * delta(0);
                    tnew = t + step_size * delta(1);
                    canonicalize(anew, tnew);
                    if (residual(anew, tnew).norm() < f_norm) {
                        accepted = true;
                        break;
                    }
                    step_size *= 0.5;
                }
                if (!accepted) break;

                if (std::abs(anew - a) < 1e-12 && std::abs(tnew - t) < 1e-12) {
                    if (valid(anew, tnew)) {
                        bestA = anew; bestT = tnew;
                        bestRes = residual(anew, tnew).norm();
                        goto done;
                    }
                    break;
                }
                a = anew; t = tnew;
            }
        }
    }
done:
    if (bestRes < tol && valid(bestA, bestT)) {
        Matrix3d R = buildR(bestA, bestT);
        Vector3d aim_dir = R * Vector3d(0, 0, 1);
        Vector3d aim_pt = aim_dir * dist;
        return cv::Point3d(aim_pt.x(), aim_pt.y(), aim_pt.z());
    }
    // Fallback: small-angle approximation
    return cv::Point3d(target_cam.x - S0.x(),
                       target_cam.y - S0.y(),
                       target_cam.z);
}

int BaseDetector::assignID(const UAVTarget& /*target*/) {
    return next_id++;
}
