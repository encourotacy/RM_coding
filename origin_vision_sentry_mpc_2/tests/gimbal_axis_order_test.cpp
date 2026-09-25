#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tools/math_tools.hpp"

namespace
{
bool expect_near(double actual, double expected, double eps, const std::string & message)
{
  if (std::abs(actual - expected) <= eps) return true;
  std::cerr << message << ", actual=" << actual << ", expected=" << expected << std::endl;
  return false;
}

bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

bool expect_vector_near(
  const Eigen::Vector3d & actual, const Eigen::Vector3d & expected, double eps,
  const std::string & message)
{
  if ((actual - expected).norm() <= eps) return true;
  std::cerr << message << ", actual=[" << actual.transpose() << "], expected=["
            << expected.transpose() << "]" << std::endl;
  return false;
}
}  // namespace

int main()
{
  constexpr double eps = 1e-9;
  constexpr double pi = 3.14159265358979323846;
  const double yaw = 0.4;
  const double elevation = 0.2;

  {
    const auto command =
      tools::gimbal_command_from_yaw_elevation(yaw, elevation, tools::GimbalAxisOrder::yaw_pitch);
    if (!expect_near(command[0], yaw, eps, "yaw_pitch yaw should equal horizontal yaw")) return 1;
    if (!expect_near(command[1], -elevation, eps, "yaw_pitch pitch should be negative elevation")) {
      return 1;
    }
  }

  const std::vector<std::pair<double, double>> samples{
    {0.0, 0.0},
    {0.6, 0.25},
    {-0.7, -0.2},
    {2.2, 0.15},
    {-2.4, -0.1},
  };

  for (const auto & [sample_yaw, sample_elevation] : samples) {
    const double cos_elevation = std::cos(sample_elevation);
    const Eigen::Vector3d expected{
      cos_elevation * std::cos(sample_yaw), cos_elevation * std::sin(sample_yaw),
      std::sin(sample_elevation)};
    const auto command = tools::gimbal_command_from_yaw_elevation(
      sample_yaw, sample_elevation, tools::GimbalAxisOrder::pitch_yaw);
    const auto actual = tools::gimbal_direction_from_command(
      command[0], command[1], tools::GimbalAxisOrder::pitch_yaw);
    if (!expect_vector_near(actual, expected, 1e-9, "pitch_yaw round trip mismatch")) {
      return 1;
    }
  }

  {
    const auto command = tools::gimbal_command_from_yaw_elevation(
      pi / 4.0, pi / 6.0, tools::GimbalAxisOrder::pitch_yaw);
    if (!expect(
          std::abs(command[0] - pi / 4.0) > 1e-3 &&
            std::abs(command[1] + pi / 6.0) > 1e-3,
          "pitch_yaw command should be coupled away from simple yaw/elevation angles")) {
      return 1;
    }
  }

  if (
    tools::parse_gimbal_axis_order("pitch-then-yaw") != tools::GimbalAxisOrder::pitch_yaw ||
    tools::parse_gimbal_axis_order("yaw_pitch") != tools::GimbalAxisOrder::yaw_pitch) {
    std::cerr << "axis order parser mismatch" << std::endl;
    return 1;
  }

  return 0;
}
