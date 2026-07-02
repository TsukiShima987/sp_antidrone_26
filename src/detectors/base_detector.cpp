#include "include/detector.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <Eigen/Geometry>

BaseDetector::BaseDetector() {
    camera_matrix = (cv::Mat_<double>(3, 3) << 30164.346973727235, 0,                  1585.3978670158735,
                                               0,                  29990.338352363626, 1027.6312267976482,
                                               0,                  0,                  1                  );
    dist_coeffs = (cv::Mat_<double>(5, 1) << 0.47187615502896024,
                                               23.566194187549524,
                                               0.037885849822472041,
                                               0.0028823047400091048,
                                               0                     );
    std::string config_path = "config/antidrone.yaml";
    const auto config = YAML::LoadFile(config_path);

    T_camera2gimbal = cv::Mat::eye(4, 4, CV_64F);
    try {
        if (config["T_camera2gimbal"] && config["T_camera2gimbal"].IsSequence()) {
            auto rows = config["T_camera2gimbal"];
            for (size_t i = 0; i < 4; ++i) {
                auto row = rows[i];
                for (size_t j = 0; j < 4; ++j) {
                    T_camera2gimbal.at<double>(i, j) = row[j].as<double>();
                }
            }
        } else {
            std::cerr << "Missing or invalid T_camera2gimbal in yaml" << std::endl;
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parse error: " << e.what() << std::endl;
    }
}

void BaseDetector::estimatePose(UAVTarget& target, float pixel_spacing, float real_size,
                                 const cv::Point2f& center) {
    float fx = camera_matrix.at<double>(0, 0);
    float fy = camera_matrix.at<double>(1, 1);
    float cx = camera_matrix.at<double>(0, 2);
    float cy = camera_matrix.at<double>(1, 2);

    double z = (fy * real_size) / pixel_spacing;
    double x = (center.x - cx) * z / fx;
    double y = (center.y - cy) * z / fy;

    cv::Point3d aim_point = computeLaserAimPoint(cv::Point3d(x, y, z));
    x = -aim_point.x;
    y = -aim_point.y;
    z = aim_point.z;

    target.position = cv::Point3f(x, y, z);
    target.distance = cv::norm(target.position);
}

cv::Point3d BaseDetector::computeLaserAimPoint(const cv::Point3d& target_cam) {
    using namespace Eigen;

    const Vector3d S0(36.71872987, -7.4622397, 1.0);
    Vector3d d0(0.00409691, 0.00001795, 0.99998631);
    d0.normalize();

    Vector3d p_cam(target_cam.x * 1000.0,
                   target_cam.y * 1000.0,
                   target_cam.z * 1000.0);
    double dist_mm = p_cam.norm();
    if (dist_mm < 1e-6) return target_cam;

    const double tol = 1e-9, step = 1e-7;
    const int maxIter = 50;

    auto buildR = [](double a, double t) -> Matrix3d {
        return AngleAxisd(t, Vector3d(cos(a), sin(a), 0)).toRotationMatrix();
    };

    auto residual = [&](double a, double t) -> Vector2d {
        Matrix3d R = buildR(a, t);
        Vector3d diff = R * p_cam - S0;
        return diff.cross(d0).head<2>();
    };

    auto normalize = [](double &a, double &t) {
        if (t < 0) { t = -t; a += M_PI; }
        a = fmod(a, 2 * M_PI);
        if (a < 0) a += 2 * M_PI;
    };

    auto valid = [&](double a, double t) -> bool {
        Matrix3d R = buildR(a, t);
        Vector3d diff = R * p_cam - S0;
        return diff.dot(d0) > 0 && diff.cross(d0).norm() < 1e-8;
    };

    std::vector<double> seedsA = {0., M_PI/2, M_PI, 3*M_PI/2};
    std::vector<double> seedsT = {M_PI/4, M_PI/2, 3*M_PI/4};
    double bestA = 0, bestT = 0, bestRes = 1e20;

    for (double a0 : seedsA) {
        for (double t0 : seedsT) {
            double a = a0, t = t0;
            normalize(a, t);
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
                double anew = a + delta(0), tnew = t + delta(1);
                normalize(anew, tnew);
                if (std::abs(delta(0)) < 1e-12 && std::abs(delta(1)) < 1e-12) {
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
    if (bestRes < 1e-7 && valid(bestA, bestT)) {
        Matrix3d R = buildR(bestA, bestT);
        Vector3d aim_dir = R * Vector3d(0, 0, 1);
        Vector3d aim_pt_mm = aim_dir * dist_mm;
        return cv::Point3d(aim_pt_mm.x() / 1000.0,
                           aim_pt_mm.y() / 1000.0,
                           aim_pt_mm.z() / 1000.0);
    }
    return target_cam;
}

int BaseDetector::assignID(const UAVTarget& /*target*/) {
    return next_id++;
}
