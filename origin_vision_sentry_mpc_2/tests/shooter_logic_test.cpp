#include <cmath>
#include <iostream>
#include <string>

#include "tasks/auto_aim/shooter_logic.hpp"

namespace
{
bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}
}  // namespace

int main()
{
  constexpr double pi = 3.14159265358979323846;
  constexpr double deg = pi / 180.0;
  constexpr double tolerance = 2.0 * deg;

  io::Command command{true, false, 40.0 * deg, -6.0 * deg};
  command.small_yaw = 12.0 * deg;
  command.big_yaw = 35.0 * deg;
  command.has_target_yaw = true;

  if (!expect(
        std::abs(auto_aim::shooter_logic::commanded_yaw(command) - command.small_yaw) < 1e-12,
        "dual-yaw command must gate fire with the transmitted small_yaw")) {
    return 1;
  }

  if (!expect(
        auto_aim::shooter_logic::aim_is_aligned(
          command, 12.5 * deg, -5.5 * deg, tolerance, tolerance),
        "aligned small yaw and pitch should pass")) {
    return 1;
  }

  if (!expect(
        !auto_aim::shooter_logic::aim_is_aligned(
          command, 12.5 * deg, -2.0 * deg, tolerance, tolerance),
        "pitch error must block fire")) {
    return 1;
  }

  io::Command previous = command;
  previous.small_yaw = 179.0 * deg;
  command.small_yaw = -179.0 * deg;
  if (!expect(
        auto_aim::shooter_logic::command_is_stable(
          command, previous, tolerance, tolerance),
        "yaw stability must handle the +/-180 degree wrap")) {
    return 1;
  }

  command.pitch = previous.pitch + 5.0 * deg;
  if (!expect(
        !auto_aim::shooter_logic::command_is_stable(
          command, previous, tolerance, tolerance),
        "large pitch target changes must block fire")) {
    return 1;
  }

  return 0;
}
