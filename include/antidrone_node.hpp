#ifndef ANTIDRONE_NODE_HPP
#define ANTIDRONE_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <radar_msgs/msg/game_status.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

class AntidroneNode : public rclcpp::Node
{
public:
    AntidroneNode();
    ~AntidroneNode();

    /// Launch the camera processing loop in a separate thread.
    void start();

private:
    // ---- ROS ----
    rclcpp::Subscription<radar_msgs::msg::GameStatus>::SharedPtr game_status_sub_;

    // ---- Config ----
    std::string config_path_;
    std::string mode_;

    // ---- Game state ----
    radar_msgs::msg::GameStatus game_status_;
    bool listen_game_status_ = true;

    // ---- Level-3 switch ----
    double level3_exposure_ms_ = 3.0;
    int level3_countdown_s_ = 180;
    std::atomic<bool> level3_triggered_{false};
    bool level3_applied_ = false;  // 仅相机线程访问，无需原子

    // ---- Thread ----
    std::atomic<bool> quit_{false};
    std::thread camera_thread_;

    // ---- Callbacks ----
    void gameStatusCallback(const radar_msgs::msg::GameStatus::ConstPtr &msg);

    // ---- Camera loop (runs in separate thread) ----
    void cameraLoop();
};

#endif // ANTIDRONE_NODE_HPP
