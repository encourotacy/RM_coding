#ifndef AUTO_AIM__SENTRY_MPC_SAFETY_HPP
#define AUTO_AIM__SENTRY_MPC_SAFETY_HPP

#include <algorithm>
#include <cmath>

#include "io/command.hpp"
#include "tasks/auto_aim/planner/planner.hpp"

namespace auto_aim
{

// MPC入口只保留统一的物理pitch软件限位。目标类别、规划与实测角差、
// 帧间跳变等经验门限不在这里重复判断；轨迹连续性由Tracker、MPC和首次接管负责。
struct SentryMpcSafetyLimits
{
  double min_pitch = -60.0 / 57.3;
  double max_pitch = 30.0 / 57.3;
};

class SentryMpcSafetyGate
{
public:
  explicit SentryMpcSafetyGate(SentryMpcSafetyLimits limits) : limits_(limits)
  {
    const SentryMpcSafetyLimits defaults;
    if (!std::isfinite(limits_.min_pitch)) limits_.min_pitch = defaults.min_pitch;
    if (!std::isfinite(limits_.max_pitch)) limits_.max_pitch = defaults.max_pitch;
    if (limits_.min_pitch > limits_.max_pitch) {
      std::swap(limits_.min_pitch, limits_.max_pitch);
    }
  }

  bool plan_is_safe(const Plan & plan, double measured_pitch) const
  {
    return plan.control && plan_values_are_finite(plan) && pitch_is_safe(measured_pitch) &&
           pitch_is_safe(plan.target_pitch) && pitch_is_safe(plan.pitch);
  }

  bool setpoint_is_safe(
    const io::Command & command, double yaw_velocity, double pitch_velocity,
    double yaw_acceleration, double pitch_acceleration) const
  {
    if (!command.control) return true;
    return std::isfinite(command.yaw) && std::isfinite(command.small_yaw) &&
           std::isfinite(command.big_yaw) && pitch_is_safe(command.pitch) &&
           std::isfinite(yaw_velocity) && std::isfinite(pitch_velocity) &&
           std::isfinite(yaw_acceleration) && std::isfinite(pitch_acceleration);
  }

private:
  static constexpr double ANGLE_EPSILON = 1e-6;

  bool pitch_is_safe(double pitch) const
  {
    return std::isfinite(pitch) && pitch >= limits_.min_pitch - ANGLE_EPSILON &&
           pitch <= limits_.max_pitch + ANGLE_EPSILON;
  }

  static bool plan_values_are_finite(const Plan & plan)
  {
    return std::isfinite(plan.target_yaw) && std::isfinite(plan.target_pitch) &&
           std::isfinite(plan.yaw) && std::isfinite(plan.yaw_vel) &&
           std::isfinite(plan.yaw_acc) && std::isfinite(plan.pitch) &&
           std::isfinite(plan.pitch_vel) && std::isfinite(plan.pitch_acc);
  }

  SentryMpcSafetyLimits limits_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SENTRY_MPC_SAFETY_HPP
