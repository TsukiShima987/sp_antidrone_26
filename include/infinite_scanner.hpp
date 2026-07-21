#pragma once

#include <chrono>
#include <cmath>
#include <optional>
#include <utility>
#include <yaml-cpp/yaml.h>

/// ∞ 形扫描器 (Lissajous 曲线, 频率比 1:2)
///
/// 波形叠加:
///   X 轴 (yaw)   = A_yaw  * sin(ωt)
///   Y 轴 (pitch) = A_pitch * sin(2ωt + phase)
///
/// sin(ωt) 和 sin(2ωt) 叠加产生 ∞ 形轨迹
class InfiniteScanner {
public:
    struct Config {
        double center_yaw   = 0.0;              // yaw 扫描中心   [rad]
        double center_pitch = 0.2;              // pitch 扫描中心 [rad]
        double amp_yaw      = 10.0 * M_PI / 180.0;   // yaw 振幅   [rad]
        double amp_pitch     = 5.0  * M_PI / 180.0;   // pitch 振幅 [rad]
        double period_s     = 3.0;                    // 扫描周期 [s]
        double phase         = 0.0;                    // pitch 通道相位偏移 [rad]
        int64_t timeout_ms   = 1000;                   // 目标丢失后多久开始扫描 [ms]
    };

    InfiniteScanner() = default;

    explicit InfiniteScanner(const Config& config)
        : config_(config) {}

    explicit InfiniteScanner(const std::string& config_path) {
        try {
            YAML::Node node = YAML::LoadFile(config_path);
            if (node["inf_scan_center_yaw_deg"])
                config_.center_yaw = node["inf_scan_center_yaw_deg"].as<double>() * M_PI / 180.0;
            if (node["inf_scan_center_pitch_deg"])
                config_.center_pitch = node["inf_scan_center_pitch_deg"].as<double>() * M_PI / 180.0;
            if (node["inf_scan_amp_yaw_deg"])
                config_.amp_yaw = node["inf_scan_amp_yaw_deg"].as<double>() * M_PI / 180.0;
            if (node["inf_scan_amp_pitch_deg"])
                config_.amp_pitch = node["inf_scan_amp_pitch_deg"].as<double>() * M_PI / 180.0;
            if (node["inf_scan_period_s"])
                config_.period_s = node["inf_scan_period_s"].as<double>();
            if (node["inf_scan_phase_deg"])
                config_.phase = node["inf_scan_phase_deg"].as<double>() * M_PI / 180.0;
            if (node["inf_scan_timeout_ms"])
                config_.timeout_ms = node["inf_scan_timeout_ms"].as<int64_t>();
        } catch (const YAML::Exception& e) {
            std::cerr << "InfiniteScanner: YAML parse error: " << e.what() << std::endl;
        }
    }

    void reset(std::chrono::steady_clock::time_point t) {
        last_target_time_ = t;
        scanning_ = false;
    }

    std::optional<std::pair<double, double>> update(
        std::chrono::steady_clock::time_point now)
    {
        // 目标未丢失，不扫描
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_target_time_).count();
        if (elapsed <= config_.timeout_ms) return std::nullopt;

        // 开始扫描时记录起始时间
        if (!scanning_) {
            scanning_ = true;
            t0_ = now;
        }

        double t = std::chrono::duration<double>(now - t0_).count();
        double omega = 2.0 * M_PI / config_.period_s;

        // ∞ 形 Lissajous:  yaw = sin(ωt), pitch = sin(2ωt)
        double yaw   = config_.center_yaw  + config_.amp_yaw  * std::sin(2.0 * omega * t);
        double pitch = config_.center_pitch + config_.amp_pitch * std::sin(3.0 * omega * t + config_.phase);

        return std::make_pair(yaw, pitch);
    }

private:
    Config config_;
    bool scanning_ = false;
    std::chrono::steady_clock::time_point last_target_time_;
    std::chrono::steady_clock::time_point t0_;
};
