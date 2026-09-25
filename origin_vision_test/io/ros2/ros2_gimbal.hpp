#ifndef IO__ROS2_GIMBAL_HPP
#define IO__ROS2_GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tools/thread_safe_queue.hpp"

// [[cyclonedds]] typed message headers (replaces manual Fast CDR serialization)
#include "rm_interfaces/msg/gimbal.hpp"
#include "rm_interfaces/msg/gimbal_cmd.hpp"

namespace io
{
struct ROS2GimbalState
{
  double yaw;
  double yaw_vel;
  double pitch;
  double pitch_vel;
  double bullet_speed;
  double big_yaw;
};

class ROS2Gimbal
{
public:
  explicit ROS2Gimbal(const std::string & config_path);
  ~ROS2Gimbal();

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
  Eigen::Quaterniond imu_at_image(std::chrono::steady_clock::time_point image_timestamp);
  double big_yaw_at_image(std::chrono::steady_clock::time_point image_timestamp);
  double offset_ms() const;

  void send(const io::Command & command);
  void send(const io::Command & command, double big_yaw, double small_yaw);

  // Inputs and ROS2 command fields use rad / rad/s / rad/s^2.
  void send_mpc(
    bool control, bool fire, double big_yaw, double small_yaw, double pitch, double yaw_vel,
    double pitch_vel, double yaw_acc, double pitch_acc, uint8_t armor_id = 0, double vx = 0.0,
    double vy = 0.0, double distance = 0.0);

  double bullet_speed() const;
  Mode mode() const;
  ROS2GimbalState state() const;

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
    double big_yaw_rad = 0.0;
  };

  struct BridgeConfig
  {
    std::string node_name = "nav_gimbal_bridge";
    std::string status_topic = "/serial/gimbal_status";
    std::string cmd_topic = "/serial/process_gimbal";
    // [[deprecated(cyclonedds)]] no longer needed with typed messages
    // std::string status_msg_type;
    // std::string cmd_msg_type;
  };

  BridgeConfig load_bridge_config(const std::string & config_path);
  void configure_topics();
  // [[deprecated(cyclonedds)]] void status_callback(const std::shared_ptr<rclcpp::SerializedMessage> & message);
  void status_callback(const rm_interfaces::msg::Gimbal::SharedPtr msg);
  bool prime_queue_if_ready();
  Eigen::Quaterniond latest_q() const;
  double latest_big_yaw() const;
  double big_yaw_at(std::chrono::steady_clock::time_point timestamp);

  tools::ThreadSafeQueue<IMUData> queue_{1000};
  std::shared_ptr<rclcpp::Node> node_;
  // [[deprecated(cyclonedds)]] old generic publisher (required Fast CDR)
  // std::shared_ptr<rclcpp::GenericPublisher> cmd_publisher_;
  std::shared_ptr<rclcpp::Publisher<rm_interfaces::msg::GimbalCmd>> cmd_publisher_;
  // [[deprecated(cyclonedds)]] old generic subscription (required Fast CDR)
  // std::shared_ptr<rclcpp::GenericSubscription> status_subscription_;
  std::shared_ptr<rclcpp::Subscription<rm_interfaces::msg::Gimbal>> status_subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;

  std::atomic<bool> self_initialized_{false};
  std::atomic<size_t> sample_count_{0};
  std::atomic<bool> has_latest_q_{false};
  bool queue_primed_{false};

  mutable std::mutex mtx_;
  IMUData data_prev_;
  IMUData data_ahead_;
  IMUData data_behind_;
  Eigen::Quaterniond latest_q_{Eigen::Quaterniond::Identity()};
  bool has_prev_{false};
  Mode mode_{Mode::idle};
  double yaw_ = 0.0;
  double pitch_ = 0.0;
  double yaw_vel_ = 0.0;
  double pitch_vel_ = 0.0;
  double bullet_speed_ = 23.0;
  double big_yaw_ = 0.0;

  BridgeConfig bridge_config_;
  std::chrono::microseconds imu_query_offset_{-1000};
};

}  // namespace io

#endif  // IO__ROS2_GIMBAL_HPP
