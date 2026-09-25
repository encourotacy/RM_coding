#include "shooter.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
double outpost_moving_phase(const auto_aim::Target & target, const auto_aim::Aimer & aimer)
{
  const auto x = target.ekf_x();
  const double center_yaw = std::atan2(x[2], x[0]);
  const double spin_sign = x[7] >= 0.0 ? 1.0 : -1.0;
  const double aim_phase =
    tools::limit_rad(aimer.debug_aim_point.xyza[3] - center_yaw);
  return aim_phase * spin_sign;
}
}  // namespace

Shooter::Shooter(const std::string & config_path)
: last_command_{false, false, 0, 0}, high_spin_force_fire_active_{false}
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
  high_spin_force_fire_enabled_ = yaml["high_spin_force_fire_enabled"].as<bool>(false);
  high_spin_force_fire_enter_speed_ = yaml["high_spin_force_fire_enter_speed"].as<double>(12.0);
  high_spin_force_fire_exit_speed_ = yaml["high_spin_force_fire_exit_speed"].as<double>(9.0);
  outpost_fire_require_locked_ = yaml["outpost_fire_require_locked"].as<bool>(true);
  outpost_phase_fire_enabled_ = yaml["outpost_phase_fire_enabled"].as<bool>(true);
  outpost_static_fire_enabled_ = yaml["outpost_static_fire_enabled"].as<bool>(true);
  outpost_static_fire_require_locked_ =
    yaml["outpost_static_fire_require_locked"].as<bool>(false);
  outpost_static_speed_threshold_ =
    std::max(0.0, yaml["outpost_static_speed_threshold"].as<double>(0.15));
  outpost_fire_coming_angle_ =
    yaml["outpost_fire_coming_angle"].as<double>(3.0) / 57.3;
  outpost_fire_leaving_angle_ =
    yaml["outpost_fire_leaving_angle"].as<double>(45.0) / 57.3;
  if (high_spin_force_fire_exit_speed_ > high_spin_force_fire_enter_speed_) {
    high_spin_force_fire_exit_speed_ = std::max(0.0, high_spin_force_fire_enter_speed_ * 0.75);
  }
}

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos,
  bool tracker_tracking)
{
  if (!command.control || targets.empty() || !auto_fire_) {
    high_spin_force_fire_active_ = false;
    return false;
  }

  const auto & target = targets.front();
  const auto angular_speed = std::abs(target.ekf_x()[7]);
  const bool aim_locked = tracker_tracking && aimer.debug_aim_point.valid;

  if (target.name == ArmorName::outpost) {
    const bool locked = target.outpost_layer_locked();
    const bool static_mode =
      outpost_static_fire_enabled_ &&
      (target.outpost_static_direct_active() || angular_speed <= outpost_static_speed_threshold_);
    const bool require_locked =
      static_mode ? outpost_static_fire_require_locked_ : outpost_fire_require_locked_;
    if ((require_locked && !locked) || !aim_locked) {
      high_spin_force_fire_active_ = false;
      last_command_ = command;
      return false;
    }

    if (static_mode) {
      high_spin_force_fire_active_ = false;
    } else {
      const double moving_phase = outpost_moving_phase(target, aimer);
      if (
        moving_phase < -outpost_fire_coming_angle_ ||
        moving_phase > outpost_fire_leaving_angle_) {
        high_spin_force_fire_active_ = false;
        last_command_ = command;
        return false;
      }

      if (outpost_phase_fire_enabled_) {
        high_spin_force_fire_active_ = false;
        last_command_ = command;
        return true;
      }
    }
  }

  if (!high_spin_force_fire_enabled_ || !aim_locked) {
    high_spin_force_fire_active_ = false;
  } else if (high_spin_force_fire_active_) {
    if (angular_speed < high_spin_force_fire_exit_speed_) high_spin_force_fire_active_ = false;
  } else if (angular_speed > high_spin_force_fire_enter_speed_) {
    high_spin_force_fire_active_ = true;
  }

  if (high_spin_force_fire_active_) {
    last_command_ = command;
    return true;
  }

  auto target_x = target.ekf_x()[0];
  auto target_y = target.ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;
  // tools::logger()->debug("d(command.yaw) is {:.4f}", std::abs(last_command_.yaw - command.yaw));
  if (
    std::abs(last_command_.yaw - command.yaw) < tolerance * 2 &&  //此时认为command突变不应该射击
    std::abs(gimbal_pos[0] - last_command_.yaw) < tolerance &&    //应该减去上一次command的yaw值
    aimer.debug_aim_point.valid) {
    last_command_ = command;
    return true;
  }

  last_command_ = command;
  return false;
}

}  // namespace auto_aim
