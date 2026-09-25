#include "ros2_gimbal.hpp"

// [[deprecated(cyclonedds)]] Fast CDR manual serialization no longer needed
// #include <fastcdr/Cdr.h>
// #include <fastcdr/FastBuffer.h>

#include <algorithm>
#include <cmath>
// [[deprecated(cyclonedds)]]
// #include <cstring>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include "ros2_gimbal_internal.hpp"

namespace io
{
namespace
{
inline double deg2rad(double deg) { return deg * M_PI / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

std::chrono::microseconds read_imu_query_offset(const YAML::Node & yaml)
{
  double offset_s = -0.001;
  if (yaml["timing"].IsDefined() && yaml["timing"]["offset"].IsDefined()) {
    offset_s = yaml["timing"]["offset"].as<double>();
  }
  return std::chrono::microseconds(static_cast<int64_t>(std::llround(offset_s * 1e6)));
}

Eigen::Quaterniond quaternion_from_deg_euler(double yaw_deg, double pitch_deg, double roll_deg)
{
  return Eigen::AngleAxisd(deg2rad(yaw_deg), Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(deg2rad(pitch_deg), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(deg2rad(roll_deg), Eigen::Vector3d::UnitX());
}

bool is_valid_quaternion(const Eigen::Quaterniond & q)
{
  const double norm = q.norm();
  return std::isfinite(norm) && std::abs(norm - 1.0) < 1e-2;
}

// [[deprecated(cyclonedds)]] typed messages replace generic type candidates
/*
std::vector<std::pair<std::string, std::string>> build_type_candidates(
  const std::string & status_msg_type, const std::string & cmd_msg_type)
{
  std::vector<std::pair<std::string, std::string>> candidates;
  std::set<std::pair<std::string, std::string>> seen;

  auto append = [&](const std::string & status_type, const std::string & cmd_type) {
    const auto key = std::make_pair(status_type, cmd_type);
    if (status_type.empty() || cmd_type.empty() || seen.count(key) > 0) return;
    seen.insert(key);
    candidates.push_back(key);
  };

  append(status_msg_type, cmd_msg_type);
  append("rm_interfaces/msg/Gimbal", "rm_interfaces/msg/GimbalCmd");

  return candidates;
}
*/

// [[deprecated(cyclonedds)]] manual Fast CDR deserialization replaced by typed message fields
/*
void deserialize_gimbal_status(
  const rclcpp::SerializedMessage & serialized_message, double & pitch_deg, double & roll_deg,
  double & yaw_deg, uint8_t & mode, Eigen::Quaterniond & q, double & yaw_vel_deg,
  double & pitch_vel_deg, double & bullet_speed, double & big_yaw_deg)
{
  const auto & raw = serialized_message.get_rcl_serialized_message();
  eprosima::fastcdr::FastBuffer buffer(reinterpret_cast<char *>(raw.buffer), raw.buffer_length);
  eprosima::fastcdr::Cdr cdr(buffer);
  cdr.read_encapsulation();

  float pitch = 0.0f;
  float roll = 0.0f;
  float yaw = 0.0f;
  float big_yaw = 0.0f;
  float w = 0.0f;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float small_yaw_speed = 0.0f;
  float pitch_speed = 0.0f;
  float bullet_speed_raw = 0.0f;
  uint8_t bullet_shoot_num = 0;

  cdr >> pitch;
  cdr >> roll;
  cdr >> yaw;
  cdr >> big_yaw;
  cdr >> mode;
  cdr >> w;
  cdr >> x;
  cdr >> y;
  cdr >> z;
  cdr >> small_yaw_speed;
  cdr >> pitch_speed;
  cdr >> bullet_speed_raw;
  cdr >> bullet_shoot_num;
  (void)bullet_shoot_num;

  pitch_deg = pitch;
  roll_deg = roll;
  yaw_deg = yaw;
  q = Eigen::Quaterniond(w, x, y, z);
  yaw_vel_deg = small_yaw_speed;
  pitch_vel_deg = pitch_speed;
  bullet_speed = bullet_speed_raw;
  big_yaw_deg = big_yaw;
}
*/

// [[deprecated(cyclonedds)]] manual Fast CDR serialization replaced by typed message construction
/*
rclcpp::SerializedMessage serialize_gimbal_mpc_cmd(
  bool control, bool fire, double big_yaw_rad, double small_yaw_rad, double pitch_rad,
  double yaw_vel_rad, double pitch_vel_rad, double yaw_acc_rad, double pitch_acc_rad,
  uint8_t armor_id, double vx, double vy, double distance)
{
  const bool fire_advice = control && fire;
  const double big_yaw_deg = control ? rad2deg(big_yaw_rad) : 0.0;
  const double small_yaw_deg = control ? rad2deg(small_yaw_rad) : 0.0;
  const double pitch_deg = control ? rad2deg(-pitch_rad) : 0.0;
  const double yaw_vel_deg = control ? rad2deg(yaw_vel_rad) : 0.0;
  const double pitch_vel_deg = control ? rad2deg(-pitch_vel_rad) : 0.0;
  const double yaw_acc_deg = control ? rad2deg(yaw_acc_rad) : 0.0;
  const double pitch_acc_deg = control ? rad2deg(-pitch_acc_rad) : 0.0;

  eprosima::fastcdr::FastBuffer buffer;
  eprosima::fastcdr::Cdr cdr(buffer);
  cdr.serialize_encapsulation();

  const auto now_ns = rclcpp::Clock(RCL_SYSTEM_TIME).now().nanoseconds();
  const int32_t sec = static_cast<int32_t>(now_ns / 1000000000LL);
  const uint32_t nanosec = static_cast<uint32_t>(now_ns % 1000000000LL);
  const std::string frame_id;

  cdr << sec;
  cdr << nanosec;
  cdr << frame_id;

  cdr << armor_id;
  cdr << vx;
  cdr << vy;

  cdr << fire_advice;
  cdr << big_yaw_deg;
  cdr << small_yaw_deg;
  cdr << pitch_deg;
  cdr << yaw_vel_deg;
  cdr << pitch_vel_deg;
  cdr << yaw_acc_deg;
  cdr << pitch_acc_deg;
  cdr << distance;
  const auto serialized_size = cdr.get_serialized_data_length();
  rclcpp::SerializedMessage message(serialized_size);
  auto & raw = message.get_rcl_serialized_message();
  std::memcpy(raw.buffer, buffer.getBuffer(), serialized_size);
  raw.buffer_length = serialized_size;
  return message;
}

rclcpp::SerializedMessage serialize_gimbal_cmd(
  const io::Command & command, double big_yaw_rad, double small_yaw_rad)
{
  return serialize_gimbal_mpc_cmd(
    command.control, command.shoot, big_yaw_rad, small_yaw_rad, command.pitch, 0.0, 0.0, 0.0,
    0.0, 0, 0.0, 0.0, 0.0); // 默认 ID=0, 速度=0
}
*/

// [[cyclonedds]] typed message constructor for GimbalCmd
rm_interfaces::msg::GimbalCmd build_gimbal_cmd(
  bool control, bool fire, double big_yaw_rad, double small_yaw_rad, double pitch_rad,
  double yaw_vel_rad, double pitch_vel_rad, double yaw_acc_rad, double pitch_acc_rad,
  uint8_t armor_id, double vx, double vy, double distance)
{
  rm_interfaces::msg::GimbalCmd msg;
  const auto now_ns = rclcpp::Clock(RCL_SYSTEM_TIME).now().nanoseconds();
  msg.header.stamp.sec = static_cast<int32_t>(now_ns / 1000000000LL);
  msg.header.stamp.nanosec = static_cast<uint32_t>(now_ns % 1000000000LL);
  msg.header.frame_id = "";
  msg.enemy_id = armor_id;
  msg.vx = vx;
  msg.vy = vy;
  msg.fire_advice = control && fire;
  msg.big_yaw = control ? rad2deg(big_yaw_rad) : 0.0;
  msg.small_yaw = control ? rad2deg(small_yaw_rad) : 0.0;
  msg.pitch = control ? rad2deg(-pitch_rad) : 0.0;
  msg.yaw_vel_deg = control ? rad2deg(yaw_vel_rad) : 0.0;
  msg.pitch_vel_deg = control ? rad2deg(-pitch_vel_rad) : 0.0;
  msg.yaw_acc_deg = control ? rad2deg(yaw_acc_rad) : 0.0;
  msg.pitch_acc_deg = control ? rad2deg(-pitch_acc_rad) : 0.0;
  msg.target_distance = distance;
  return msg;
}

rm_interfaces::msg::GimbalCmd build_gimbal_cmd_from_command(
  const io::Command & command, double big_yaw_rad, double small_yaw_rad)
{
  return build_gimbal_cmd(
    command.control, command.shoot, big_yaw_rad, small_yaw_rad, command.pitch, 0.0, 0.0, 0.0,
    0.0, 0, 0.0, 0.0, 0.0);
}
}  // namespace

void ROS2Gimbal::send_mpc(
  bool control, bool fire, double big_yaw, double small_yaw, double pitch, double yaw_vel,
  double pitch_vel, double yaw_acc, double pitch_acc, uint8_t armor_id, double vx, double vy,
  double distance)
{
  if (!cmd_publisher_) return;
  try {
    // [[cyclonedds]] typed message publish replaces old Fast CDR serialization
    auto message = build_gimbal_cmd(
      control, fire, big_yaw, small_yaw, pitch, yaw_vel, pitch_vel, yaw_acc, pitch_acc,
      armor_id, vx, vy, distance);
    cmd_publisher_->publish(message);
  } catch (const std::exception & e) {
    tools::logger()->warn("[ROS2Gimbal] Failed to publish MPC gimbal cmd: {}", e.what());
  }
}

ROS2Gimbal::ROS2Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  imu_query_offset_ = read_imu_query_offset(yaml);
  bridge_config_ = load_bridge_config(config_path);

  tools::logger()->info(
    "[ROS2Gimbal] timing offset={:.2f}ms",
    static_cast<double>(imu_query_offset_.count()) / 1000.0);

  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
    self_initialized_ = true;
  }

  node_ = std::make_shared<rclcpp::Node>(bridge_config_.node_name);
  configure_topics();

  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spin_thread_ = std::thread([this]() { executor_->spin(); });

  tools::logger()->info(
    "[ROS2Gimbal] Waiting for gimbal status on '{}'...",
    bridge_config_.status_topic);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (sample_count_.load() >= 2 && prime_queue_if_ready()) {
      tools::logger()->info("[ROS2Gimbal] First IMU pair received.");
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  tools::logger()->warn("[ROS2Gimbal] No gimbal status received within 2s, continuing anyway.");
}

ROS2Gimbal::~ROS2Gimbal()
{
  if (executor_) executor_->cancel();
  if (spin_thread_.joinable()) spin_thread_.join();
  if (executor_ && node_) executor_->remove_node(node_);
  if (self_initialized_ && rclcpp::ok()) rclcpp::shutdown();
}

ROS2Gimbal::BridgeConfig ROS2Gimbal::load_bridge_config(const std::string & config_path)
{
  BridgeConfig config;
  auto yaml = tools::load(config_path);
  auto bridge = yaml["ros2_gimbal"];
  if (!bridge) return config;

  if (bridge["node_name"]) config.node_name = bridge["node_name"].as<std::string>();
  if (bridge["status_topic"]) config.status_topic = bridge["status_topic"].as<std::string>();
  if (bridge["cmd_topic"]) config.cmd_topic = bridge["cmd_topic"].as<std::string>();
  // [[deprecated(cyclonedds)]] typed messages no longer need type name strings
  // if (bridge["status_msg_type"]) config.status_msg_type = bridge["status_msg_type"].as<std::string>();
  // if (bridge["cmd_msg_type"]) config.cmd_msg_type = bridge["cmd_msg_type"].as<std::string>();
  return config;
}

void ROS2Gimbal::configure_topics()
{
  // [[deprecated(cyclonedds)]] old generic publisher/subscription with type candidates
  /*
  const auto candidates =
    build_type_candidates(bridge_config_.status_msg_type, bridge_config_.cmd_msg_type);
  if (candidates.empty()) {
    throw std::runtime_error(
      "[ROS2Gimbal] No message type candidates available. Configure ros2_gimbal.status_msg_type "
      "and ros2_gimbal.cmd_msg_type in YAML.");
  }

  std::string errors;
  for (const auto & [status_type, cmd_type] : candidates) {
    try {
      cmd_publisher_ =
        node_->create_generic_publisher(bridge_config_.cmd_topic, cmd_type, rclcpp::SensorDataQoS());
      status_subscription_ = node_->create_generic_subscription(
        bridge_config_.status_topic, status_type, rclcpp::SensorDataQoS(),
        [this](const std::shared_ptr<rclcpp::SerializedMessage> message) {
          this->status_callback(message);
        });

      bridge_config_.status_msg_type = status_type;
      bridge_config_.cmd_msg_type = cmd_type;
      tools::logger()->info(
        "[ROS2Gimbal] Using status type '{}' and cmd type '{}'.", status_type, cmd_type);
      return;
    } catch (const std::exception & e) {
      errors += "  - " + status_type + " / " + cmd_type + ": " + e.what() + "\n";
      cmd_publisher_.reset();
      status_subscription_.reset();
    }
  }

  throw std::runtime_error(
    "[ROS2Gimbal] Failed to create generic publisher/subscription for configured topics.\n" +
    errors +
    "Please set ros2_gimbal.status_msg_type and ros2_gimbal.cmd_msg_type to the actual interface "
    "package, for example rm_interfaces/msg/Gimbal.");
  */

  // [[cyclonedds]] typed publisher/subscription (no manual Fast CDR needed)
  cmd_publisher_ =
    node_->create_publisher<rm_interfaces::msg::GimbalCmd>(
      bridge_config_.cmd_topic, rclcpp::SensorDataQoS());
  status_subscription_ =
    node_->create_subscription<rm_interfaces::msg::Gimbal>(
      bridge_config_.status_topic, rclcpp::SensorDataQoS(),
      [this](const rm_interfaces::msg::Gimbal::SharedPtr msg) {
        this->status_callback(msg);
      });
  tools::logger()->info(
    "[ROS2Gimbal] Typed pub/sub on '{}' and '{}'.", bridge_config_.cmd_topic, bridge_config_.status_topic);
}

void ROS2Gimbal::status_callback(const rm_interfaces::msg::Gimbal::SharedPtr msg)
{
  try {
    // [[deprecated(cyclonedds)]] old manual Fast CDR deserialization
    /*
    double pitch_deg = 0.0;
    double roll_deg = 0.0;
    double yaw_deg = 0.0;
    uint8_t mode_raw = 0;
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    double yaw_vel_deg = 0.0;
    double pitch_vel_deg = 0.0;
    double bullet_speed = 0.0;
    double big_yaw_deg = 0.0;

    deserialize_gimbal_status(
      *message, pitch_deg, roll_deg, yaw_deg, mode_raw, q, yaw_vel_deg, pitch_vel_deg,
      bullet_speed, big_yaw_deg);
    */

    // [[cyclonedds]] read fields directly from typed message
    const double pitch_deg = msg->pitch;
    const double roll_deg = msg->roll;
    const double yaw_deg = msg->yaw;
    const double big_yaw_deg = msg->big_yaw;
    const uint8_t mode_raw = msg->mode;
    const double yaw_vel_deg = msg->small_yaw_speed;
    const double pitch_vel_deg = msg->pitch_speed;
    const double bullet_speed = msg->bullet_speed;

    Eigen::Quaterniond q(msg->w, msg->x, msg->y, msg->z);

    if (!is_valid_quaternion(q)) {
      q = quaternion_from_deg_euler(yaw_deg, pitch_deg, roll_deg);
    } else {
      q.normalize();
    }

    const auto now = std::chrono::steady_clock::now();
    queue_.push({q, now, deg2rad(big_yaw_deg)});
    sample_count_.fetch_add(1);

    {
      std::lock_guard<std::mutex> lk(mtx_);
      latest_q_ = q;
      has_latest_q_ = true;
      yaw_ = deg2rad(yaw_deg);
      pitch_ = deg2rad(pitch_deg);
      yaw_vel_ = deg2rad(yaw_vel_deg);
      pitch_vel_ = deg2rad(pitch_vel_deg);
      bullet_speed_ = bullet_speed;
      big_yaw_ = deg2rad(big_yaw_deg);

      switch (mode_raw) {
        case 0:
          mode_ = Mode::idle;
          break;
        case 1:
          mode_ = Mode::auto_aim;
          break;
        case 2:
          mode_ = Mode::small_buff;
          break;
        case 3:
          mode_ = Mode::big_buff;
          break;
        case 4:
          mode_ = Mode::outpost;
          break;
        default:
          mode_ = Mode::idle;
          break;
      }
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[ROS2Gimbal] Failed to parse status message: {}", e.what());
  }
}

bool ROS2Gimbal::prime_queue_if_ready()
{
  if (queue_primed_ || sample_count_.load() < 2) return queue_primed_;
  queue_.pop(data_ahead_);
  queue_.pop(data_behind_);
  data_prev_ = data_ahead_;
  queue_primed_ = true;
  return true;
}

Eigen::Quaterniond ROS2Gimbal::latest_q() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  if (!has_latest_q_.load()) return Eigen::Quaterniond::Identity();
  return latest_q_;
}

double ROS2Gimbal::latest_big_yaw() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return big_yaw_;
}

Eigen::Quaterniond ROS2Gimbal::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  if (!prime_queue_if_ready()) return latest_q();

  auto interpolate = [&](const IMUData & a, const IMUData & b) {
    auto q_a = a.q.normalized();
    auto q_b = b.q.normalized();
    const double dt = tools::delta_time(b.timestamp, a.timestamp);
    if (dt <= 1e-6) return q_b;
    const double k = std::clamp(tools::delta_time(timestamp, a.timestamp) / dt, 0.0, 1.0);
    return q_a.slerp(k, q_b).normalized();
  };

  if (timestamp <= data_ahead_.timestamp) {
    if (has_prev_ && timestamp >= data_prev_.timestamp) {
      return interpolate(data_prev_, data_ahead_);
    }
    return data_ahead_.q.normalized();
  }

  while (data_behind_.timestamp < timestamp) {
    has_prev_ = true;
    data_prev_ = data_ahead_;
    data_ahead_ = data_behind_;
    queue_.pop(data_behind_);
  }

  return interpolate(data_ahead_, data_behind_);
}

Eigen::Quaterniond ROS2Gimbal::imu_at_image(std::chrono::steady_clock::time_point image_timestamp)
{
  return imu_at(image_timestamp + imu_query_offset_);
}

double ROS2Gimbal::big_yaw_at(std::chrono::steady_clock::time_point timestamp)
{
  if (!prime_queue_if_ready()) return latest_big_yaw();

  auto pop_next = [this]() -> std::optional<IMUData> {
      IMUData next;
      if (!queue_.pop_for(next, std::chrono::milliseconds(0))) {
        return std::nullopt;
      }
      return next;
    };

  return detail::lookup_big_yaw_rad(
    timestamp, has_prev_, data_prev_, data_ahead_, data_behind_, pop_next);
}

double ROS2Gimbal::big_yaw_at_image(std::chrono::steady_clock::time_point image_timestamp)
{
  return big_yaw_at(image_timestamp + imu_query_offset_);
}

double ROS2Gimbal::offset_ms() const
{
  return static_cast<double>(imu_query_offset_.count()) / 1000.0;
}

void ROS2Gimbal::send(const io::Command & command)
{
  if (command.has_target_yaw) {
    send(command, command.big_yaw, command.small_yaw);
    return;
  }

  static std::atomic<bool> warned{false};
  send(command, command.yaw, command.yaw);
}

void ROS2Gimbal::send(const io::Command & command, double big_yaw, double small_yaw)
{
  if (!cmd_publisher_) return;

  try {
    // [[cyclonedds]] typed message publish replaces old Fast CDR serialization
    auto message = build_gimbal_cmd_from_command(command, big_yaw, small_yaw);
    cmd_publisher_->publish(message);
  } catch (const std::exception & e) {
    tools::logger()->warn("[ROS2Gimbal] Failed to publish gimbal cmd: {}", e.what());
  }
}

double ROS2Gimbal::bullet_speed() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return bullet_speed_;
}

Mode ROS2Gimbal::mode() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return mode_;
}

ROS2GimbalState ROS2Gimbal::state() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return {yaw_, yaw_vel_, pitch_, pitch_vel_, bullet_speed_, big_yaw_};
}

}  // namespace io
