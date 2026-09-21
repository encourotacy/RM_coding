#include "subscribers.hpp"

#include <algorithm>
#include <exception>

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>

#include "tools/logger.hpp"

namespace ovsentry
{
namespace
{

std::vector<uint8_t> deserialize_ignore_ids(const rclcpp::SerializedMessage & serialized_message)
{
  const auto & raw = serialized_message.get_rcl_serialized_message();
  eprosima::fastcdr::FastBuffer buffer(reinterpret_cast<char *>(raw.buffer), raw.buffer_length);
  eprosima::fastcdr::Cdr cdr(buffer);
  cdr.read_encapsulation();

  uint32_t count = 0;
  cdr >> count;

  std::vector<uint8_t> ids;
  ids.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t id = 0;
    cdr >> id;
    ids.push_back(id);
  }

  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

}  // namespace

ArmorIgnoreSubscriber::ArmorIgnoreSubscriber(const std::string & topic, const std::string & msg_type)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
    self_initialized_ = true;
  }

  node_ = std::make_shared<rclcpp::Node>("auto_aim_ignore_subscriber");
  try {
    subscription_ = node_->create_generic_subscription(
      topic, msg_type, rclcpp::SensorDataQoS(),
      [this](const std::shared_ptr<rclcpp::SerializedMessage> message) { this->callback(message); });

    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    spin_thread_ = std::thread([this]() { executor_->spin(); });
    tools::logger()->info("[AutoAimIgnore] Subscribed '{}' as '{}'.", topic, msg_type);
  } catch (const std::exception & e) {
    tools::logger()->warn("[AutoAimIgnore] Failed to subscribe '{}': {}", topic, e.what());
  }
}

ArmorIgnoreSubscriber::~ArmorIgnoreSubscriber()
{
  if (executor_) executor_->cancel();
  if (spin_thread_.joinable()) spin_thread_.join();
  if (executor_ && node_) executor_->remove_node(node_);
  if (self_initialized_ && rclcpp::ok()) rclcpp::shutdown();
}

ArmorTargetMask ArmorIgnoreSubscriber::mask() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mask_;
}

void ArmorIgnoreSubscriber::callback(const std::shared_ptr<rclcpp::SerializedMessage> & message)
{
  try {
    auto ids = deserialize_ignore_ids(*message);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      mask_.enabled = !ids.empty();
      mask_.ignored_ids = ids;
    }
    for (const auto id : ids) {
      tools::logger()->info("[AutoAimIgnore] ignore armor id: {}", static_cast<int>(id));
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[AutoAimIgnore] Failed to parse ignore ids: {}", e.what());
  }
}

BuffRequestSubscriber::BuffRequestSubscriber(const std::string & topic)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
    self_initialized_ = true;
  }

  node_ = std::make_shared<rclcpp::Node>("buff_request_subscriber");
  subscription_ = node_->create_subscription<std_msgs::msg::Bool>(
    topic, 10, [this](const std_msgs::msg::Bool::SharedPtr message) {
      request_buff_.store(message->data);
      tools::logger()->info("[BuffRequest] request_buff={}", message->data ? "true" : "false");
    });

  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spin_thread_ = std::thread([this]() { executor_->spin(); });
  tools::logger()->info("[BuffRequest] Subscribed '{}'.", topic);
}

BuffRequestSubscriber::~BuffRequestSubscriber()
{
  if (executor_) executor_->cancel();
  if (spin_thread_.joinable()) spin_thread_.join();
  if (executor_ && node_) executor_->remove_node(node_);
  if (self_initialized_ && rclcpp::ok()) rclcpp::shutdown();
}

bool BuffRequestSubscriber::requested() const { return request_buff_.load(); }

}  // namespace ovsentry
