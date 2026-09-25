#ifndef AUTO_AIM__SENTRY_COMMAND_HPP
#define AUTO_AIM__SENTRY_COMMAND_HPP

#include <cstdint>
#include <list>
#include <vector>

#include "armor.hpp"
#include "io/command.hpp"
#include "target.hpp"

namespace auto_aim
{

inline uint8_t armor_name_to_nav_id(ArmorName name)
{
  switch (name) {
    case ArmorName::one:
      return 1;
    case ArmorName::two:
      return 2;
    case ArmorName::three:
      return 3;
    case ArmorName::four:
      return 4;
    case ArmorName::five:
      return 5;
    case ArmorName::sentry:
      return 6;
    case ArmorName::outpost:
      return 7;
    case ArmorName::base:
      return 8;
    default:
      return 0;
  }
}

struct ArmorTargetMask
{
  bool enabled = false;
  std::vector<uint8_t> ignored_ids;
};

void apply_armor_target_mask(std::list<Armor> & armors, const ArmorTargetMask & mask);

double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad);

bool is_unlocked_outpost_target(const Target & target);

void apply_sentry_tracking_yaws(
  io::Command & command, const Target & target, double current_big_yaw_rad);

void fill_nav_target_info(io::Command & command, const std::list<Target> & targets);

}  // namespace auto_aim

#endif  // AUTO_AIM__SENTRY_COMMAND_HPP
