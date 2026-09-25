#include <cmath>
#include <iostream>
#include <string>

#include "tasks/auto_aim/sentry_yaw_control.hpp"

namespace
{
bool expect_near(double actual, double expected, double tolerance, const std::string & message)
{
  if (std::abs(actual - expected) <= tolerance) return true;
  std::cerr << message << ", actual=" << actual << ", expected=" << expected << std::endl;
  return false;
}
}  // namespace

int main()
{
  constexpr double pi = 3.14159265358979323846;
  constexpr double deg = pi / 180.0;

  io::Command command{true, false, 23.0 * deg, -4.0 * deg};
  auto_aim::sentry_yaw_control::apply(command, -179.0 * deg, 179.0 * deg, false);

  if (!expect_near(command.small_yaw, 23.0 * deg, 1e-12, "small yaw must come from MPC yaw")) {
    return 1;
  }
  if (!expect_near(command.big_yaw, 181.0 * deg, 1e-12, "big yaw must track center continuously")) {
    return 1;
  }
  if (!command.has_target_yaw) {
    std::cerr << "dual-yaw command flag was not set" << std::endl;
    return 1;
  }

  io::Command hold_command{true, false, -35.0 * deg, 2.0 * deg};
  auto_aim::sentry_yaw_control::apply(hold_command, 80.0 * deg, 12.0 * deg, true);
  if (!expect_near(hold_command.big_yaw, 12.0 * deg, 1e-12, "hold mode must keep big yaw")) {
    return 1;
  }
  if (!expect_near(
        hold_command.small_yaw, -35.0 * deg, 1e-12,
        "holding big yaw must not disable MPC small yaw")) {
    return 1;
  }

  return 0;
}
