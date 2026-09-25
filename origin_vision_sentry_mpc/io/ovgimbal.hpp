#ifndef IO__OVGIMBAL_HPP
#define IO__OVGIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct OVGimbalState
{
  double yaw;
  double yaw_vel;
  double pitch;
  double pitch_vel;
  double bullet_speed;
};

class OVGimbal
{
public:
  explicit OVGimbal(const std::string & config_path);
  ~OVGimbal();

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
  Eigen::Quaterniond imu_at_image(std::chrono::steady_clock::time_point image_timestamp);
  double offset_ms() const;
  bool has_imu() const;

  void send(const io::Command & command);

  double bullet_speed() const;
  Mode mode() const;
  OVGimbalState state() const;

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  serial::Serial serial_;
  std::thread thread_;
  std::atomic<bool> quit_{false};
  mutable std::mutex mtx_;

  tools::ThreadSafeQueue<IMUData> queue_{1000};
  Mode mode_{Mode::idle};
  double yaw_ = 0.0;
  double pitch_ = 0.0;
  double yaw_vel_ = 0.0;
  double pitch_vel_ = 0.0;
  double bullet_speed_ = 23.0;
  IMUData data_prev_;
  IMUData data_ahead_;
  IMUData data_behind_;
  bool has_prev_{false};
  bool has_latest_data_{false};
  IMUData latest_data_;
  std::chrono::microseconds imu_query_offset_{-1000};

  void read_thread();
  bool read_exact(uint8_t * buffer, size_t size);
  void reconnect();
};

}  // namespace io

#endif  // IO__OVGIMBAL_HPP
