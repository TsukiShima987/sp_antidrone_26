#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <yaml-cpp/yaml.h>

class Scanner {
public:
    struct Config {
        double center_yaw = 0.0;
        double center_pitch = 0.0;
        double yaw_step = 5.0 * M_PI / 180.0;
        double pitch_step = 3.0 * M_PI / 180.0;
        int cols = 5;
        int rows = 4;
        int64_t timeout_ms = 1000;
    };

    Scanner() = default;

    explicit Scanner(const Config& config)
        : config_(config)
        , last_target_time_(std::chrono::steady_clock::now()) {}

    explicit Scanner(const std::string& config_path) {
        try {
            YAML::Node node = YAML::LoadFile(config_path);
            if (node["scan_center_yaw_deg"])
                config_.center_yaw = node["scan_center_yaw_deg"].as<double>() * M_PI / 180.0;
            if (node["scan_center_pitch_deg"])
                config_.center_pitch = node["scan_center_pitch_deg"].as<double>() * M_PI / 180.0;
            if (node["scan_yaw_step_deg"])
                config_.yaw_step = node["scan_yaw_step_deg"].as<double>() * M_PI / 180.0;
            if (node["scan_pitch_step_deg"])
                config_.pitch_step = node["scan_pitch_step_deg"].as<double>() * M_PI / 180.0;
            if (node["scan_cols"])
                config_.cols = node["scan_cols"].as<int>();
            if (node["scan_rows"])
                config_.rows = node["scan_rows"].as<int>();
            if (node["scan_timeout_ms"])
                config_.timeout_ms = node["scan_timeout_ms"].as<int64_t>();
        } catch (const YAML::Exception& e) {
            std::cerr << "Scanner: YAML parse error: " << e.what() << std::endl;
        }
    }

    void reset(std::chrono::steady_clock::time_point t) {
        last_target_time_ = t;
        scanning_ = false;
    }

    std::optional<std::pair<double, double>> update(
        std::chrono::steady_clock::time_point now)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_target_time_).count();

        if (elapsed <= config_.timeout_ms) return std::nullopt;

        if (!scanning_) {
            scanning_ = true;
            step_ = 0;
        }

        int row = step_ / config_.cols;
        int col = step_ % config_.cols;
        if (row % 2 == 1) col = config_.cols - 1 - col;
        if (row >= config_.rows) {
            step_ = 0;
            row = 0;
            col = 0;
        }

        double yaw_off = (col - config_.cols / 2) * config_.yaw_step;
        double pitch_off = (row - config_.rows / 2) * config_.pitch_step;

        std::pair<double, double> target = {
            config_.center_yaw + yaw_off,
            config_.center_pitch + pitch_off
        };
        step_++;
        return target;
    }

private:
    Config config_;
    bool scanning_ = false;
    std::chrono::steady_clock::time_point last_target_time_;
    int step_ = 0;
};
