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
