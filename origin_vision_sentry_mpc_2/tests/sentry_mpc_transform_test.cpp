#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <string>

#include "tasks/auto_aim/sentry_mpc_transform.hpp"

namespace
{
bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

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
  constexpr auto axis_order = tools::GimbalAxisOrder::pitch_yaw;

  const double center_yaw = 120.0 * deg;
  const double armor_yaw = 123.0 * deg;
  const double armor_elevation = 10.0 * deg;
  const double cos_elevation = std::cos(armor_elevation);
  const Eigen::Vector3d expected_world_direction{
    cos_elevation * std::cos(armor_yaw), cos_elevation * std::sin(armor_yaw),
    std::sin(armor_elevation)};

  const auto sentry_command =
    auto_aim::sentry_mpc_transform::world_direction_to_world_small_yaw_command(
      expected_world_direction, center_yaw, axis_order);
  if (!expect_near(
        sentry_command.small_yaw, armor_yaw, 1e-9,
        "small yaw must remain a world-referenced armor yaw")) {
    return 1;
  }

  const Eigen::Vector2d local_joints =
    auto_aim::sentry_mpc_transform::world_direction_to_big_yaw_local(
      expected_world_direction, center_yaw, axis_order);
  if (!expect_near(
        sentry_command.pitch, local_joints[1], 1e-9,
        "pitch must remain the physical outer joint after big-yaw compensation")) {
    return 1;
  }

  const Eigen::Vector3d reconstructed_world_direction =
    Eigen::AngleAxisd(center_yaw, Eigen::Vector3d::UnitZ()) *
    tools::gimbal_direction_from_command(local_joints[0], sentry_command.pitch, axis_order);

  if (!expect(
        (reconstructed_world_direction - expected_world_direction).norm() < 1e-9,
        "big yaw plus the physical pitch/yaw joints must reconstruct the world direction")) {
    return 1;
  }

  {
    const auto bottom =
      auto_aim::sentry_mpc_transform::world_yaw_elevation_to_world_small_yaw_command(
        armor_yaw, -10.0 * deg, center_yaw, axis_order);
    const auto middle =
      auto_aim::sentry_mpc_transform::world_yaw_elevation_to_world_small_yaw_command(
        armor_yaw, 0.0, center_yaw, axis_order);
    const auto top =
      auto_aim::sentry_mpc_transform::world_yaw_elevation_to_world_small_yaw_command(
        armor_yaw, 10.0 * deg, center_yaw, axis_order);

    if (!expect(
          bottom.pitch > middle.pitch && middle.pitch > top.pitch,
          "raw image bottom-to-top motion must monotonically decrease internal physical pitch")) {
      return 1;
    }
    if (!expect(
          bottom.pitch > 0.0 && std::abs(middle.pitch) < 1e-12 && top.pitch < 0.0,
          "pitch_yaw direction must be down=positive internal pitch and up=negative")) {
      return 1;
    }
    if (!expect(
          -bottom.pitch < 0.0 && -top.pitch > 0.0,
          "EC pitch sign must command down for a bottom target and up for a top target")) {
      return 1;
    }
  }

  {
    const double buff_yaw = -40.0 * deg;
    const double buff_elevation = 12.0 * deg;
    const Eigen::Vector2d configured_axis_command =
      tools::gimbal_command_from_yaw_elevation(buff_yaw, buff_elevation, axis_order);
    const Eigen::Vector3d recovered_world_direction =
      tools::gimbal_direction_from_command(
        configured_axis_command[0], configured_axis_command[1], axis_order);
    const auto buff_command =
      auto_aim::sentry_mpc_transform::world_direction_to_world_small_yaw_command(
        recovered_world_direction, buff_yaw, axis_order);
    if (!expect_near(
          buff_command.small_yaw, buff_yaw, 1e-9,
          "buff conversion must preserve world-referenced small yaw")) {
      return 1;
    }
  }

  for (double world_yaw_deg = 80.0; world_yaw_deg <= 100.0; world_yaw_deg += 1.0) {
    const double world_yaw = world_yaw_deg * deg;
    const double moving_center_yaw = world_yaw - 3.0 * deg;
    const auto command =
      auto_aim::sentry_mpc_transform::world_yaw_elevation_to_world_small_yaw_command(
        world_yaw, 10.0 * deg, moving_center_yaw, axis_order);
    if (!expect_near(
          command.small_yaw, world_yaw, 1e-9,
          "world small yaw must stay continuous across the mechanical 90-degree direction")) {
      return 1;
    }
    if (!expect(
          std::abs(command.pitch) < 20.0 * deg,
          "big-yaw compensation must keep the physical pitch away from the global singularity")) {
      return 1;
    }
  }

  return 0;
}
