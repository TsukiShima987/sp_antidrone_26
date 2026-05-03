#include "subscribe2location.hpp"
namespace io
{
    Subscribe2Location::Subscribe2Location(): Node("location_subscriber")
    {
        subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "location_sender", 10, std::bind(&Subscribe2Location::callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "location_subscriber node initialized.");
    }
    Subscribe2Location::~Subscribe2Location()
    {
        RCLCPP_INFO(this->get_logger(), "location_subscriber node shutting down.");
    }
    void Subscribe2Location::start()
    {
        RCLCPP_INFO(this->get_logger(), "location_subscriber node Starting to spin...");
        rclcpp::spin(this->shared_from_this());
    }
    void Subscribe2Location::callback(const geometry_msgs::msg::Point::SharedPtr msg)
    {
        // queue_.push(*msg);
        tools::logger()->info("Received location data: x={}, y={}, z={}", msg->x, msg->y, msg->z);
        info_ = {msg->x,msg->y,msg->z,true};
    }
    LocationInfo Subscribe2Location::subscribe_data()
    {
        // if (queue_.empty()) {
        //     RCLCPP_INFO(this->get_logger(), "No location_data detected !");
        //     LocationInfo info{0.0,0.0,0.0,false};
        // }
        // else
        // {
        //     tools::logger()->info("size: {}",queue_.size());
        // }
        // geometry_msgs::msg::Point msg;
        // queue_.back(msg);
        // LocationInfo info{msg.x,msg.y,msg.z,true};
        return info_;
    }

}