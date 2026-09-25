#include "sentry_command.hpp"

#include <algorithm>
#include <cmath>

#include "tools/math_tools.hpp"

namespace auto_aim
{
bool is_unlocked_outpost_target(const Target & target)
{
  return target.name == ArmorName::outpost && !target.outpost_layer_locked();
}

namespace
{
double target_center_big_yaw_rad(const Target & target, double current_big_yaw_rad)
{
  const auto & ekf_x = target.ekf_x();
  const double wrapped_center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  return nearest_continuous_yaw_rad(wrapped_center_yaw, current_big_yaw_rad);
}

double horizon_distance(const Target & target)
{
  const auto & x = target.ekf_x();
  return std::sqrt(x[0] * x[0] + x[2] * x[2]);
}

}  // namespace

void apply_armor_target_mask(std::list<Armor> & armors, const ArmorTargetMask & mask)
{
  if (!mask.enabled) return;
  if (mask.ignored_ids.empty()) return;

  armors.remove_if([&](const Armor & armor) {
    const auto id = armor_name_to_nav_id(armor.name);
    return id != 0 && std::find(mask.ignored_ids.begin(), mask.ignored_ids.end(), id) !=
                      mask.ignored_ids.end();
  });
}

double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad)
{
  return reference_yaw_rad + tools::limit_rad(wrapped_yaw_rad - reference_yaw_rad);
}

void apply_sentry_tracking_yaws(
  io::Command & command, const Target & target, double current_big_yaw_rad)
{
  if (!command.control) return;
  command.small_yaw = command.yaw;
  command.big_yaw =
    is_unlocked_outpost_target(target) ? current_big_yaw_rad
                                      : target_center_big_yaw_rad(target, current_big_yaw_rad);
  command.has_target_yaw = true;
}

void fill_nav_target_info(io::Command & command, const std::list<Target> & targets)
{
  command.armor_id = 0;
  command.vx = 0.0;
  command.vy = 0.0;
  command.horizon_distance = 0.0;

  if (!command.control || targets.empty()) return;

  const auto & target = targets.front();
  const auto x = target.ekf_x();
  command.armor_id = armor_name_to_nav_id(target.name);
  command.vx = x[1];
  command.vy = x[3];
  command.horizon_distance = horizon_distance(target);
}

}  // namespace auto_aim
