#include "auto_aim_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "tools/math_tools.hpp"

namespace ovsentry
{

std::string slot_name(omniperception::OmniCameraSlot slot)
{
  switch (slot) {
    case omniperception::OmniCameraSlot::left:
      return "LEFT";
    case omniperception::OmniCameraSlot::right:
      return "RIGHT";
    case omniperception::OmniCameraSlot::back:
      return "BACK";
    case omniperception::OmniCameraSlot::extra:
      return "EXTRA";
    default:
      return "UNKNOWN";
  }
}

bool better_armor(const auto_aim::Armor & lhs, const auto_aim::Armor & rhs)
{
  if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
  return lhs.confidence > rhs.confidence;
}

std::optional<auto_aim::Armor> pick_top_armor(const std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return std::nullopt;
  auto best_it = armors.begin();
  for (auto it = std::next(armors.begin()); it != armors.end(); ++it) {
    if (better_armor(*it, *best_it)) best_it = it;
  }
  return *best_it;
}

uint8_t armor_name_to_nav_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;
    case auto_aim::ArmorName::two:
      return 2;
    case auto_aim::ArmorName::three:
      return 3;
    case auto_aim::ArmorName::four:
      return 4;
    case auto_aim::ArmorName::five:
      return 5;
    case auto_aim::ArmorName::sentry:
      return 6;
    case auto_aim::ArmorName::outpost:
      return 7;
    case auto_aim::ArmorName::base:
      return 8;
    default:
      return 0;
  }
}

void apply_armor_target_mask(std::list<auto_aim::Armor> & armors, const ArmorTargetMask & mask)
{
  if (!mask.enabled) return;
  if (mask.ignored_ids.empty()) return;

  armors.remove_if([&](const auto_aim::Armor & armor) {
    const auto id = armor_name_to_nav_id(armor.name);
    return id != 0 && std::find(mask.ignored_ids.begin(), mask.ignored_ids.end(), id) !=
                        mask.ignored_ids.end();
  });
}

std::pair<double, double> calc_delta_angle_deg(
  const auto_aim::Armor & armor, const OmniCamConfig & cam)
{
  const double delta_yaw =
    cam.spec.center_yaw_deg + (0.5 - armor.center_norm.x) * cam.spec.fov_h_deg;
  const double delta_pitch = (armor.center_norm.y - 0.5) * cam.spec.fov_v_deg;
  return {delta_yaw, delta_pitch};
}

double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}

double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad)
{
  return reference_yaw_rad + tools::limit_rad(wrapped_yaw_rad - reference_yaw_rad);
}

double target_center_big_yaw_rad(const auto_aim::Target & target, double current_big_yaw_rad)
{
  const auto & ekf_x = target.ekf_x();
  const double wrapped_center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  return nearest_continuous_yaw_rad(wrapped_center_yaw, current_big_yaw_rad);
}

bool is_unlocked_outpost_target(const auto_aim::Target & target)
{
  return target.name == auto_aim::ArmorName::outpost && !target.outpost_layer_locked();
}

void apply_sentry_tracking_yaws(
  io::Command & command, const auto_aim::Target & target, double current_big_yaw_rad)
{
  if (!command.control) return;
  command.small_yaw = command.yaw;
  command.big_yaw = is_unlocked_outpost_target(target)
                      ? current_big_yaw_rad
                      : target_center_big_yaw_rad(target, current_big_yaw_rad);
  command.has_target_yaw = true;
}

void apply_abs_yaw_target(io::Command & command, double abs_yaw_rad)
{
  command.control = true;
  command.yaw = tools::limit_rad(abs_yaw_rad);
  command.big_yaw = abs_yaw_rad;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
}

std::optional<omniperception::OmniCandidate> build_omni_candidate(
  const OmniInferenceResult & result, std::chrono::steady_clock::time_point timestamp,
  double base_big_yaw_rad)
{
  if (!result.top_armor.has_value()) return std::nullopt;

  const auto & armor = result.top_armor.value();
  omniperception::OmniCandidate candidate;
  candidate.slot = result.cam.spec.slot;
  candidate.armor_name = armor.name;
  candidate.priority = armor.priority;
  candidate.confidence = armor.confidence;
  candidate.timestamp = timestamp;
  candidate.base_big_yaw_rad = base_big_yaw_rad;
  candidate.abs_yaw_rad = base_big_yaw_rad + result.delta_yaw_deg / 57.3;
  apply_abs_yaw_target(candidate.command, candidate.abs_yaw_rad);
  candidate.command.armor_id = armor_name_to_nav_id(armor.name);
  candidate.command.pitch = 0.26;
  return candidate;
}

omniperception::AcceptedOmniTarget make_accepted_omni_target(
  const omniperception::OmniCandidate & candidate)
{
  omniperception::AcceptedOmniTarget accepted_target;
  accepted_target.slot = candidate.slot;
  accepted_target.armor_name = candidate.armor_name;
  accepted_target.priority = candidate.priority;
  accepted_target.confidence = candidate.confidence;
  accepted_target.timestamp = candidate.timestamp;
  accepted_target.base_big_yaw_rad = candidate.base_big_yaw_rad;
  accepted_target.abs_yaw_rad = candidate.abs_yaw_rad;
  accepted_target.command = candidate.command;
  return accepted_target;
}

bool same_candidate_frame(
  const OmniCandidateFrame & frame, const omniperception::OmniCandidate & candidate)
{
  if (!frame.candidate.has_value()) return false;
  return frame.candidate->slot == candidate.slot &&
         frame.candidate->armor_name == candidate.armor_name &&
         frame.candidate->timestamp == candidate.timestamp;
}

bool same_omni_target_continuation(
  const omniperception::AcceptedOmniTarget & lhs, const omniperception::AcceptedOmniTarget & rhs,
  double retarget_min_delta_deg)
{
  if (lhs.slot != rhs.slot) return false;
  if (lhs.armor_name != rhs.armor_name) return false;
  return angular_distance_deg(lhs.abs_yaw_rad, rhs.abs_yaw_rad) < retarget_min_delta_deg;
}

double horizon_distance(const auto_aim::Target & target)
{
  const auto & x = target.ekf_x();
  return std::sqrt(x[0] * x[0] + x[2] * x[2]);
}

void fill_nav_target_info(io::Command & command, const std::list<auto_aim::Target> & targets)
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

}  // namespace ovsentry
