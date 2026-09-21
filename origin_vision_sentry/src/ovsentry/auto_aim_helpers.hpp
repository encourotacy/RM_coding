#ifndef OVSENTRY__AUTO_AIM_HELPERS_HPP
#define OVSENTRY__AUTO_AIM_HELPERS_HPP

#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <utility>

#include "io/command.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"
#include "types.hpp"

namespace ovsentry
{

std::string slot_name(omniperception::OmniCameraSlot slot);
bool better_armor(const auto_aim::Armor & lhs, const auto_aim::Armor & rhs);
std::optional<auto_aim::Armor> pick_top_armor(const std::list<auto_aim::Armor> & armors);
uint8_t armor_name_to_nav_id(auto_aim::ArmorName name);
void apply_armor_target_mask(std::list<auto_aim::Armor> & armors, const ArmorTargetMask & mask);
std::pair<double, double> calc_delta_angle_deg(
  const auto_aim::Armor & armor, const OmniCamConfig & cam);
double angular_distance_deg(double lhs_rad, double rhs_rad);
double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad);
double target_center_big_yaw_rad(const auto_aim::Target & target, double current_big_yaw_rad);
bool is_unlocked_outpost_target(const auto_aim::Target & target);
void apply_sentry_tracking_yaws(
  io::Command & command, const auto_aim::Target & target, double current_big_yaw_rad);
void apply_abs_yaw_target(io::Command & command, double abs_yaw_rad);
std::optional<omniperception::OmniCandidate> build_omni_candidate(
  const OmniInferenceResult & result, std::chrono::steady_clock::time_point timestamp,
  double base_big_yaw_rad);
omniperception::AcceptedOmniTarget make_accepted_omni_target(
  const omniperception::OmniCandidate & candidate);
bool same_candidate_frame(
  const OmniCandidateFrame & frame, const omniperception::OmniCandidate & candidate);
bool same_omni_target_continuation(
  const omniperception::AcceptedOmniTarget & lhs, const omniperception::AcceptedOmniTarget & rhs,
  double retarget_min_delta_deg);
double horizon_distance(const auto_aim::Target & target);
void fill_nav_target_info(io::Command & command, const std::list<auto_aim::Target> & targets);

}  // namespace ovsentry

#endif  // OVSENTRY__AUTO_AIM_HELPERS_HPP
