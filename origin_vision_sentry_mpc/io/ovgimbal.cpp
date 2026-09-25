#include "ovgimbal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
namespace
{
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

std::chrono::microseconds read_imu_query_offset(const YAML::Node & yaml)
{
  double offset_s = -0.001;
  if (yaml["timing"].IsDefined() && yaml["timing"]["offset"].IsDefined()) {
    offset_s = yaml["timing"]["offset"].as<double>();
  }
  return std::chrono::microseconds(static_cast<int64_t>(std::llround(offset_s * 1e6)));
}
}  // namespace

OVGimbal::OVGimbal(const std::string & config_path)
{
  try {
    auto yaml = tools::load(config_path);
    imu_query_offset_ = read_imu_query_offset(yaml);
  } catch (const std::exception & e) {
    tools::logger()->warn(
      "[OVGimbal] Failed to load timing config from {}: {}", config_path, e.what());
  }

  tools::logger()->info(
    "[OVGimbal] timing offset={:.2f}ms",
    static_cast<double>(imu_query_offset_.count()) / 1000.0);

  try {
    serial_.setPort("/dev/ttyACM0");
    serial_.setBaudrate(115200);
    auto timeout = serial::Timeout::simpleTimeout(1000);
    serial_.setTimeout(timeout);
    serial_.open();
    tools::logger()->info("[OVGimbal] Serial port opened successfully: /dev/ttyACM0 @ 115200");
  } catch (const std::exception & e) {
    tools::logger()->error("[OVGimbal] Failed to open serial: {}", e.what());
    throw;
  }

  thread_ = std::thread(&OVGimbal::read_thread, this);

  tools::logger()->info("[OVGimbal] Waiting for IMU...");
  if (queue_.pop_for(data_ahead_, std::chrono::milliseconds(800)) &&
      queue_.pop_for(data_behind_, std::chrono::milliseconds(800)))
  {
    data_prev_ = data_ahead_;
    has_prev_ = true;
    tools::logger()->info("[OVGimbal] First IMU pair received.");
  } else {
    const auto now = std::chrono::steady_clock::now();
    data_prev_ = {Eigen::Quaterniond::Identity(), now};
    data_ahead_ = data_prev_;
    data_behind_ = data_prev_;
    tools::logger()->warn("[OVGimbal] IMU startup timeout, using latest-available fallback.");
  }
}

OVGimbal::~OVGimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  try {
    serial_.close();
  } catch (...) {
  }
}

Eigen::Quaterniond OVGimbal::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  if (!has_imu()) {
    return Eigen::Quaterniond::Identity();
  }

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
    if (!queue_.pop_for(data_behind_, std::chrono::milliseconds(2))) {
      std::lock_guard<std::mutex> lk(mtx_);
      if (has_latest_data_ && latest_data_.timestamp > data_ahead_.timestamp) {
        return interpolate(data_ahead_, latest_data_);
      }
      return data_ahead_.q.normalized();
    }
  }

  return interpolate(data_ahead_, data_behind_);
}

Eigen::Quaterniond OVGimbal::imu_at_image(std::chrono::steady_clock::time_point image_timestamp)
{
  return imu_at(image_timestamp + imu_query_offset_);
}

double OVGimbal::offset_ms() const
{
  return static_cast<double>(imu_query_offset_.count()) / 1000.0;
}

bool OVGimbal::has_imu() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return has_latest_data_;
}

void OVGimbal::send(const io::Command & command)
{
  const uint8_t header = 0xAA;
  const uint8_t cmd_id = 0x81;

  float pitch_deg = static_cast<float>(rad2deg(-command.pitch));
  float yaw_deg = static_cast<float>(rad2deg(command.yaw));
  uint8_t ad = command.shoot ? 1 : 0;

  uint8_t payload[sizeof(float) * 2 + 1];
  std::memcpy(payload + 0, &pitch_deg, sizeof(float));
  std::memcpy(payload + 4, &yaw_deg, sizeof(float));
  payload[8] = ad;

  const uint8_t length = static_cast<uint8_t>(3 + sizeof(payload));
  uint8_t frame[3 + sizeof(payload)];
  frame[0] = header;
  frame[1] = length;
  frame[2] = cmd_id;
  std::memcpy(frame + 3, payload, sizeof(payload));

  try {
    serial_.write(frame, sizeof(frame));
  } catch (const std::exception & e) {
    tools::logger()->warn("[OVGimbal] Failed to write serial: {}", e.what());
  }
}

double OVGimbal::bullet_speed() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return bullet_speed_;
}

Mode OVGimbal::mode() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return mode_;
}

OVGimbalState OVGimbal::state() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return {yaw_, yaw_vel_, pitch_, pitch_vel_, bullet_speed_};
}

bool OVGimbal::read_exact(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (...) {
    return false;
  }
}

void OVGimbal::read_thread()
{
  tools::logger()->info("[OVGimbal] read_thread started.");
  int error_count = 0;
  const uint8_t header = 0xAA;

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[OVGimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    uint8_t head = 0;
    if (!read_exact(&head, 1)) {
      error_count++;
      continue;
    }
    if (head != header) continue;

    uint8_t len_byte = 0;
    if (!read_exact(&len_byte, 1)) {
      error_count++;
      continue;
    }
    if (len_byte < 4 || len_byte > 64) {
      error_count++;
      continue;
    }

    std::vector<uint8_t> rest(len_byte - 2);
    if (!read_exact(rest.data(), rest.size()) || rest.empty()) {
      error_count++;
      continue;
    }

    const uint8_t cmd = rest[0];
    const auto now = std::chrono::steady_clock::now();

    if (cmd == 0x14) {
      if (rest.size() < 1 + 10 * sizeof(float) + 1 + 1) {
        error_count++;
        continue;
      }

      float pitch_rad = 0.0f;
      float roll_rad = 0.0f;
      float yaw_rad = 0.0f;
      std::memcpy(&pitch_rad, rest.data() + 1, sizeof(float));
      std::memcpy(&roll_rad, rest.data() + 5, sizeof(float));
      std::memcpy(&yaw_rad, rest.data() + 9, sizeof(float));

      const uint8_t mode_byte = rest[13];

      float q_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      std::memcpy(&q_data[0], rest.data() + 14, sizeof(float));
      std::memcpy(&q_data[1], rest.data() + 18, sizeof(float));
      std::memcpy(&q_data[2], rest.data() + 22, sizeof(float));
      std::memcpy(&q_data[3], rest.data() + 26, sizeof(float));

      float yaw_vel = 0.0f;
      float pitch_vel = 0.0f;
      std::memcpy(&yaw_vel, rest.data() + 30, sizeof(float));
      std::memcpy(&pitch_vel, rest.data() + 34, sizeof(float));

      float bullet_speed = 0.0f;
      std::memcpy(&bullet_speed, rest.data() + 38, sizeof(float));

      Eigen::Quaterniond q(q_data[0], q_data[1], q_data[2], q_data[3]);
      queue_.push({q, now});

      {
        std::lock_guard<std::mutex> lk(mtx_);
        latest_data_ = {q, now};
        has_latest_data_ = true;
        switch (mode_byte) {
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
            mode_ = Mode::auto_aim;
            break;
        }
        yaw_ = static_cast<double>(yaw_rad);
        pitch_ = static_cast<double>(pitch_rad);
        yaw_vel_ = static_cast<double>(yaw_vel);
        pitch_vel_ = static_cast<double>(pitch_vel);
        bullet_speed_ = static_cast<double>(bullet_speed);
      }

      error_count = 0;
    } else if (cmd == 0x15) {
      error_count = 0;
    } else {
      static std::unordered_map<uint8_t, int> unknown_cmd_counts;
      unknown_cmd_counts[cmd]++;
      if (unknown_cmd_counts[cmd] <= 3) {
        tools::logger()->warn(
          "[OVGimbal] Unknown command: 0x{:02x} (count: {})", cmd, unknown_cmd_counts[cmd]);
      }
      error_count++;
    }
  }

  tools::logger()->info("[OVGimbal] read_thread stopped.");
}

void OVGimbal::reconnect()
{
  for (int i = 0; i < 10 && !quit_; ++i) {
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      serial_.open();
      queue_.clear();
      {
        std::lock_guard<std::mutex> lk(mtx_);
        has_latest_data_ = false;
      }
      tools::logger()->info("[OVGimbal] Reconnected serial.");
      return;
    } catch (const std::exception & e) {
      tools::logger()->warn("[OVGimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
}

}  // namespace io
