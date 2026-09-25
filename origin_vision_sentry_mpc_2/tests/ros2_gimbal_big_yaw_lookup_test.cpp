#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "io/ros2/ros2_gimbal_internal.hpp"

namespace
{
using namespace std::chrono_literals;

struct Sample
{
  std::chrono::steady_clock::time_point timestamp{};
  double big_yaw_rad = 0.0;
};

bool expect_near(double actual, double expected, double eps, const std::string & message)
{
  if (std::abs(actual - expected) <= eps) return true;
  std::cerr << message << ", actual=" << actual << ", expected=" << expected << std::endl;
  return false;
}

}  // namespace

int main()
{
  const auto t0 = std::chrono::steady_clock::now();

  if (io::detail::status_timestamp_is_fresh(false, 2, t0, t0, 200ms)) {
    std::cerr << "invalid status timestamp was accepted" << std::endl;
    return 1;
  }
  if (io::detail::status_timestamp_is_fresh(true, 1, t0, t0, 200ms)) {
    std::cerr << "a single status sample was accepted" << std::endl;
    return 1;
  }
  if (!io::detail::status_timestamp_is_fresh(true, 2, t0, t0 + 150ms, 200ms)) {
    std::cerr << "fresh status timestamp was rejected" << std::endl;
    return 1;
  }
  if (io::detail::status_timestamp_is_fresh(true, 2, t0, t0 + 250ms, 200ms)) {
    std::cerr << "stale status timestamp was accepted" << std::endl;
    return 1;
  }
  if (io::detail::status_timestamp_is_fresh(true, 2, t0 + 1ms, t0, 200ms)) {
    std::cerr << "future status timestamp was accepted" << std::endl;
    return 1;
  }

  {
    constexpr double pi = 3.14159265358979323846;
    const Eigen::Quaterniond imu_q(
      0.9742416143417358, 0.013855814933776855, -0.03302204608917236,
      0.2219984531402588);
    const Eigen::Quaterniond aligned = io::detail::align_imu_world_yaw(imu_q, -pi);
    const double aligned_yaw = std::atan2(
      2.0 * (aligned.w() * aligned.z() + aligned.x() * aligned.y()),
      1.0 - 2.0 * (aligned.y() * aligned.y() + aligned.z() * aligned.z()));
    const double aligned_yaw_deg = aligned_yaw * 180.0 / pi;
    if (!expect_near(
          aligned_yaw_deg, -154.35476684570312, 0.1,
          "IMU world yaw offset should align quaternion yaw with electrical yaw")) {
      return 1;
    }
  }

  {
    bool has_prev = true;
    Sample prev{t0, 0.0};
    Sample ahead{t0 + 10ms, 1.0};
    Sample behind{t0 + 20ms, 2.0};
    std::vector<Sample> pending;
    const auto query = t0 + 15ms;
    const double actual = io::detail::lookup_big_yaw_rad(
      query, has_prev, prev, ahead, behind,
      [&pending]() -> std::optional<Sample> {
        if (pending.empty()) return std::nullopt;
        auto next = pending.front();
        pending.erase(pending.begin());
        return next;
      });
    if (!expect_near(actual, 1.5, 1e-9, "big_yaw should be linearly interpolated in-range")) {
      return 1;
    }
  }

  {
    bool has_prev = false;
    Sample prev{};
    Sample ahead{t0 + 10ms, 1.0};
    Sample behind{t0 + 20ms, 2.0};
    std::vector<Sample> pending;
    const auto query = t0 + 5ms;
    const double actual = io::detail::lookup_big_yaw_rad(
      query, has_prev, prev, ahead, behind,
      [&pending]() -> std::optional<Sample> {
        if (pending.empty()) return std::nullopt;
        auto next = pending.front();
        pending.erase(pending.begin());
        return next;
      });
    if (!expect_near(actual, 1.0, 1e-9, "query before earliest sample should use nearest sample")) {
      return 1;
    }
  }

  {
    bool has_prev = true;
    Sample prev{t0 + 10ms, 1.0};
    Sample ahead{t0 + 20ms, 2.0};
    Sample behind{t0 + 30ms, 3.0};
    std::vector<Sample> pending;
    const auto query = t0 + 45ms;
    const double actual = io::detail::lookup_big_yaw_rad(
      query, has_prev, prev, ahead, behind,
      [&pending]() -> std::optional<Sample> {
        if (pending.empty()) return std::nullopt;
        auto next = pending.front();
        pending.erase(pending.begin());
        return next;
      });
    if (!expect_near(actual, 3.0, 1e-9, "query after latest sample should use nearest sample")) {
      return 1;
    }
  }

  return 0;
}
