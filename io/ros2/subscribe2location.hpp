#ifndef IO__SUBSCRIBE2LOCATION_HPP
#define IO__SUBSCRIBE2LOCATION_HPP

#include <rclcpp/rclcpp.hpp>
#include "geometry_msgs/msg/point.hpp"
#include <vector>
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct LocationInfo
{
    double x;
    double y;
    double z;
    bool value;
};
class Subscribe2Location : public rclcpp::Node
{
public:
  Subscribe2Location();

  ~Subscribe2Location();

  void start();

  LocationInfo subscribe_data();

private:
  void callback(const geometry_msgs::msg::Point::SharedPtr msg);
  LocationInfo info_={0,0,0,false};
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr subscription_;

};

}  // namespace io

#endif  // IO__SUBSCRIBE2NAV_HPP
