#include "shooter.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tasks/auto_aim/shooter_logic.hpp"

namespace auto_aim
{
namespace
{
double outpost_aim_phase(const auto_aim::Target & target, const auto_aim::Aimer & aimer)
{
  const auto x = target.ekf_x();
  const double center_yaw = std::atan2(x[2], x[0]);
  return tools::limit_rad(aimer.debug_aim_point.xyza[3] - center_yaw);
}
}  // namespace

Shooter::Shooter(const std::string & config_path)
: last_command_{false, false, 0, 0}, high_spin_force_fire_active_{false},
  high_spin_center_aim_active_{false}, has_last_command_{false}
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
  high_spin_force_fire_enabled_ = yaml["high_spin_force_fire_enabled"].as<bool>(false);
  high_spin_force_fire_enter_speed_ = yaml["high_spin_force_fire_enter_speed"].as<double>(12.0);
  high_spin_force_fire_exit_speed_ = yaml["high_spin_force_fire_exit_speed"].as<double>(9.0);
  high_spin_center_aim_enabled_ = yaml["high_spin_center_aim_enabled"].as<bool>(true);
  high_spin_center_aim_enter_speed_ =
    yaml["high_spin_center_aim_enter_speed"].as<double>(10.0);
  high_spin_center_aim_exit_speed_ =
    yaml["high_spin_center_aim_exit_speed"].as<double>(8.0);
  outpost_fire_require_locked_ = yaml["outpost_fire_require_locked"].as<bool>(true);
  const double legacy_outpost_angle = yaml["outpost_fire_max_angle"].as<double>(18.0);
  // Negative enter angles intentionally delay fire until the armor has passed the center.
  outpost_fire_enter_angle_ =
    yaml["outpost_fire_enter_angle"].as<double>(legacy_outpost_angle) / 57.3;
  outpost_fire_exit_angle_ =
    std::max(0.0, yaml["outpost_fire_exit_angle"].as<double>(legacy_outpost_angle)) / 57.3;
  if (high_spin_force_fire_exit_speed_ > high_spin_force_fire_enter_speed_) {
    high_spin_force_fire_exit_speed_ = std::max(0.0, high_spin_force_fire_enter_speed_ * 0.75);
  }
  if (high_spin_center_aim_exit_speed_ > high_spin_center_aim_enter_speed_) {
    high_spin_center_aim_exit_speed_ =
      std::max(0.0, high_spin_center_aim_enter_speed_ * 0.8);
  }
}

void Shooter::update_high_spin_modes(const std::list<auto_aim::Target> & targets)
{
  if (targets.empty() || !auto_fire_) {
    high_spin_force_fire_active_ = false;
    high_spin_center_aim_active_ = false;
    return;
  }

  const auto & target = targets.front();
  const double angular_speed = std::abs(target.ekf_x()[7]);
  if (target.name == ArmorName::outpost || !std::isfinite(angular_speed)) {
    high_spin_force_fire_active_ = false;
    high_spin_center_aim_active_ = false;
    return;
  }

  if (!high_spin_force_fire_enabled_) {
    high_spin_force_fire_active_ = false;
  } else if (high_spin_force_fire_active_) {
    if (angular_speed < high_spin_force_fire_exit_speed_) high_spin_force_fire_active_ = false;
  } else if (angular_speed >= high_spin_force_fire_enter_speed_) {
    high_spin_force_fire_active_ = true;
  }

  if (!high_spin_center_aim_enabled_) {
    high_spin_center_aim_active_ = false;
  } else if (high_spin_center_aim_active_) {
    if (angular_speed < high_spin_center_aim_exit_speed_) high_spin_center_aim_active_ = false;
  } else if (angular_speed >= high_spin_center_aim_enter_speed_) {
    high_spin_center_aim_active_ = true;
  }

  // Center spraying is the stronger mode and must always imply continuous fire.
  if (high_spin_center_aim_active_) high_spin_force_fire_active_ = true;
}

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos,
  bool tracker_tracking)
{
  update_high_spin_modes(targets);
  if (!command.control || targets.empty() || !auto_fire_) {
    has_last_command_ = false;
    return false;
  }

  const auto & target = targets.front();
  const bool aim_locked = tracker_tracking && aimer.debug_aim_point.valid;

  if (target.name == ArmorName::outpost) {
    const bool locked = target.outpost_layer_locked();
    if ((outpost_fire_require_locked_ && !locked) || !aim_locked) {
      high_spin_force_fire_active_ = false;
      last_command_ = command;
      has_last_command_ = true;
      return false;
    }

    const double phase = outpost_aim_phase(target, aimer);
    const double yaw_rate = target.ekf_x()[7];
    const bool phase_in_fire_window =
      yaw_rate > 1e-3 ? (phase >= -outpost_fire_enter_angle_ && phase <= outpost_fire_exit_angle_)
      : yaw_rate < -1e-3 ? (phase <= outpost_fire_enter_angle_ && phase >= -outpost_fire_exit_angle_)
                         : std::abs(phase) <= std::min(outpost_fire_enter_angle_, outpost_fire_exit_angle_);
    if (!phase_in_fire_window) {
      high_spin_force_fire_active_ = false;
      last_command_ = command;
      has_last_command_ = true;
      return false;
    }
  }

  auto target_x = target.ekf_x()[0];
  auto target_y = target.ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;
  const bool aligned = shooter_logic::aim_is_aligned(
    command, gimbal_pos[0], gimbal_pos[1], tolerance, tolerance);
  const bool stable =
    has_last_command_ &&
    shooter_logic::command_is_stable(command, last_command_, tolerance, tolerance);
  const bool should_fire = aim_locked && aligned && (high_spin_force_fire_active_ || stable);
  last_command_ = command;
  has_last_command_ = true;
  return should_fire;
}

}  // namespace auto_aim
