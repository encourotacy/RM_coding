#ifndef AUTO_AIM__SHOOTER_LOGIC_HPP
#define AUTO_AIM__SHOOTER_LOGIC_HPP

#include <cmath>

#include "io/command.hpp"
#include "tasks/auto_aim/sentry_yaw_control.hpp"

namespace auto_aim::shooter_logic
{
inline double commanded_yaw(const io::Command & command)
{
  return command.has_target_yaw ? command.small_yaw : command.yaw;
}

inline bool command_is_stable(
  const io::Command & current, const io::Command & previous, double yaw_tolerance,
  double pitch_tolerance)
{
  if (!previous.control) return false;
  const double yaw_delta = sentry_yaw_control::wrap_to_pi(
    commanded_yaw(current) - commanded_yaw(previous));
  return std::abs(yaw_delta) < yaw_tolerance * 2.0 &&
         std::abs(current.pitch - previous.pitch) < pitch_tolerance * 2.0;
}

inline bool aim_is_aligned(
  const io::Command & command, double current_yaw, double current_pitch, double yaw_tolerance,
  double pitch_tolerance)
{
  const double yaw_error =
    sentry_yaw_control::wrap_to_pi(current_yaw - commanded_yaw(command));
  const double pitch_error = current_pitch - command.pitch;
  return std::abs(yaw_error) < yaw_tolerance && std::abs(pitch_error) < pitch_tolerance;
}
}  // namespace auto_aim::shooter_logic

#endif  // AUTO_AIM__SHOOTER_LOGIC_HPP
