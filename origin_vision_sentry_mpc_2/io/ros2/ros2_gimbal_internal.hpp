#ifndef IO__ROS2_GIMBAL_INTERNAL_HPP
#define IO__ROS2_GIMBAL_INTERNAL_HPP

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>

#include "tools/math_tools.hpp"

namespace io::detail
{
inline bool status_timestamp_is_fresh(
  bool timestamp_valid, std::size_t sample_count,
  std::chrono::steady_clock::time_point received_at,
  std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration max_age)
{
  if (!timestamp_valid || sample_count < 2 || now < received_at) return false;
  return now - received_at <= max_age;
}

inline Eigen::Quaterniond align_imu_world_yaw(
  const Eigen::Quaterniond & q, double world_yaw_offset_rad)
{
  Eigen::Quaterniond aligned =
    Eigen::AngleAxisd(world_yaw_offset_rad, Eigen::Vector3d::UnitZ()) * q.normalized();
  aligned.normalize();
  return aligned;
}

template <typename Sample>
double interpolate_big_yaw_rad(
  std::chrono::steady_clock::time_point timestamp, const Sample & a, const Sample & b)
{
  const double dt = tools::delta_time(b.timestamp, a.timestamp);
  if (dt <= 1e-6) return b.big_yaw_rad;

  const double k = std::clamp(tools::delta_time(timestamp, a.timestamp) / dt, 0.0, 1.0);
  return a.big_yaw_rad + (b.big_yaw_rad - a.big_yaw_rad) * k;
}

template <typename Sample>
double nearest_big_yaw_rad(
  std::chrono::steady_clock::time_point timestamp, const Sample & lhs, const Sample & rhs)
{
  const double lhs_dt = std::abs(tools::delta_time(timestamp, lhs.timestamp));
  const double rhs_dt = std::abs(tools::delta_time(timestamp, rhs.timestamp));
  return lhs_dt <= rhs_dt ? lhs.big_yaw_rad : rhs.big_yaw_rad;
}

template <typename Sample, typename PopNext>
double lookup_big_yaw_rad(
  std::chrono::steady_clock::time_point timestamp, bool & has_prev, Sample & prev, Sample & ahead,
  Sample & behind, PopNext pop_next)
{
  if (timestamp <= ahead.timestamp) {
    if (has_prev && timestamp >= prev.timestamp) {
      return interpolate_big_yaw_rad(timestamp, prev, ahead);
    }
    return ahead.big_yaw_rad;
  }

  while (behind.timestamp < timestamp) {
    auto next = pop_next();
    if (!next.has_value()) {
      return nearest_big_yaw_rad(timestamp, ahead, behind);
    }
    has_prev = true;
    prev = ahead;
    ahead = behind;
    behind = next.value();
  }

  return interpolate_big_yaw_rad(timestamp, ahead, behind);
}

}  // namespace io::detail

#endif  // IO__ROS2_GIMBAL_INTERNAL_HPP
