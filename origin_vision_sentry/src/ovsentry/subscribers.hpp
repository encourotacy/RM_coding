#ifndef OVSENTRY__SUBSCRIBERS_HPP
#define OVSENTRY__SUBSCRIBERS_HPP

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include "types.hpp"

namespace ovsentry
{

class ArmorIgnoreSubscriber
{
public:
  ArmorIgnoreSubscriber(
    const std::string & topic = "/request_auto_aim_ignore",
    const std::string & msg_type = "rm_interfaces/msg/RequestAutoAimIgnore");
  ~ArmorIgnoreSubscriber();

  ArmorTargetMask mask() const;

private:
  void callback(const std::shared_ptr<rclcpp::SerializedMessage> & message);

  mutable std::mutex mutex_;
  ArmorTargetMask mask_;
  bool self_initialized_ = false;
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::GenericSubscription> subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
};

class BuffRequestSubscriber
{
public:
  explicit BuffRequestSubscriber(const std::string & topic = "/request_buff");
  ~BuffRequestSubscriber();

  bool requested() const;

private:
  std::atomic_bool request_buff_{false};
  bool self_initialized_ = false;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
};

inline ArmorTargetMask read_nav_armor_target_mask(const ArmorIgnoreSubscriber & subscriber)
{
  return subscriber.mask();
}

}  // namespace ovsentry

#endif  // OVSENTRY__SUBSCRIBERS_HPP
