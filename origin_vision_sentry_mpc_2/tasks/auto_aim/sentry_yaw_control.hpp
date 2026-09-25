#ifndef AUTO_AIM__SENTRY_YAW_CONTROL_HPP
#define AUTO_AIM__SENTRY_YAW_CONTROL_HPP

#include "io/command.hpp"

namespace auto_aim::sentry_yaw_control
{
inline double wrap_to_pi(double angle)
{
  constexpr double pi = 3.14159265358979323846;
  while (angle > pi) angle -= 2.0 * pi;
  while (angle <= -pi) angle += 2.0 * pi;
  return angle;
}

inline double nearest_continuous_yaw(double wrapped_yaw, double reference_yaw)
{
  return reference_yaw + wrap_to_pi(wrapped_yaw - reference_yaw);
}

inline void apply(
  io::Command & command, double center_world_yaw, double current_big_yaw, bool hold_big_yaw)
{
  if (!command.control) return;

  // command.yaw 是电控所需的世界系小 yaw 目标，不是相对 big yaw 的局部残差。
  command.small_yaw = command.yaw;

  // 大 yaw 只跟踪旋转中心，并保持连续角。
  command.big_yaw = hold_big_yaw
                      ? current_big_yaw
                      : nearest_continuous_yaw(center_world_yaw, current_big_yaw);
  command.has_target_yaw = true;
}
}  // namespace auto_aim::sentry_yaw_control

#endif  // AUTO_AIM__SENTRY_YAW_CONTROL_HPP
