#include "ovsentry_omni_logic.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "tasks/auto_aim/sentry_command.hpp"
#include "tasks/auto_aim/sentry_mpc_transform.hpp"
#include "tools/math_tools.hpp"

namespace omniperception
{
double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}

namespace
{
bool better_armor(const auto_aim::Armor & lhs, const auto_aim::Armor & rhs)
{
  if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
  return lhs.confidence > rhs.confidence;
}

bool same_target_continuation(
  const OmniCandidate & candidate, const AcceptedOmniTarget & accepted_target,
  double retarget_min_delta_deg)
{
  if (candidate.slot != accepted_target.slot) return false;
  if (candidate.armor_name != accepted_target.armor_name) return false;
  return angular_distance_deg(candidate.abs_yaw_rad, accepted_target.abs_yaw_rad) <
         retarget_min_delta_deg;
}

double reference_delta_deg(
  const OmniCandidate & candidate, const std::optional<AcceptedOmniTarget> & reference_target,
  double current_abs_yaw_rad)
{
  if (reference_target.has_value()) {
    return angular_distance_deg(candidate.abs_yaw_rad, reference_target->abs_yaw_rad);
  }
  return angular_distance_deg(candidate.abs_yaw_rad, current_abs_yaw_rad);
}

}  // namespace

std::optional<auto_aim::Armor> pick_top_armor(const std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return std::nullopt;
  auto best_it = armors.begin();
  for (auto it = std::next(armors.begin()); it != armors.end(); ++it) {
    if (better_armor(*it, *best_it)) best_it = it;
  }
  return *best_it;
}

std::pair<double, double> calc_delta_angle_deg(
  const auto_aim::Armor & armor, const CameraSpec & cam)
{
  const double delta_yaw = cam.center_yaw_deg + (0.5 - armor.center_norm.x) * cam.fov_h_deg;
  const double delta_pitch = (armor.center_norm.y - 0.5) * cam.fov_v_deg;
  return {delta_yaw, delta_pitch};
}

OmniCandidate build_omni_candidate(
  const auto_aim::Armor & armor, OmniCameraSlot slot, double delta_yaw_deg,
  std::chrono::steady_clock::time_point timestamp, double base_big_yaw_rad,
  tools::GimbalAxisOrder gimbal_axis_order)
{
  OmniCandidate candidate;
  candidate.slot = slot;
  candidate.armor_name = armor.name;
  candidate.priority = armor.priority;
  candidate.confidence = armor.confidence;
  candidate.timestamp = timestamp;
  candidate.base_big_yaw_rad = base_big_yaw_rad;
  candidate.abs_yaw_rad = base_big_yaw_rad + delta_yaw_deg / 57.3;
  const auto sentry_command =
    auto_aim::sentry_mpc_transform::world_yaw_elevation_to_world_small_yaw_command(
      candidate.abs_yaw_rad, -0.26, candidate.abs_yaw_rad, gimbal_axis_order);
  candidate.command.control = true;
  candidate.command.yaw = sentry_command.small_yaw;
  candidate.command.pitch = sentry_command.pitch;
  candidate.command.big_yaw = candidate.abs_yaw_rad;
  candidate.command.small_yaw = candidate.command.yaw;
  candidate.command.has_target_yaw = true;
  candidate.command.armor_id = auto_aim::armor_name_to_nav_id(armor.name);
  return candidate;
}

double fill_omni_candidate(
  OmniCandidateFrame & frame, const CameraSpec & spec, tools::GimbalAxisOrder gimbal_axis_order)
{
  frame.result.top_armor = pick_top_armor(frame.result.armors);
  if (!frame.result.top_armor.has_value()) {
    frame.candidate.reset();
    frame.result.delta_yaw_deg = 0.0;
    frame.result.delta_pitch_deg = 0.0;
    return 0.0;
  }

  const auto [delta_yaw_deg, delta_pitch_deg] =
    calc_delta_angle_deg(frame.result.top_armor.value(), spec);
  frame.result.delta_yaw_deg = delta_yaw_deg;
  frame.result.delta_pitch_deg = delta_pitch_deg;
  frame.candidate = build_omni_candidate(
    frame.result.top_armor.value(), spec.slot, delta_yaw_deg, frame.timestamp, frame.base_big_yaw_rad,
    gimbal_axis_order);
  return delta_pitch_deg;
}

AcceptedOmniTarget make_accepted_omni_target(const OmniCandidate & candidate)
{
  AcceptedOmniTarget accepted_target;
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

bool same_omni_target_continuation(
  const AcceptedOmniTarget & lhs, const AcceptedOmniTarget & rhs, double retarget_min_delta_deg)
{
  if (lhs.slot != rhs.slot) return false;
  if (lhs.armor_name != rhs.armor_name) return false;
  return angular_distance_deg(lhs.abs_yaw_rad, rhs.abs_yaw_rad) < retarget_min_delta_deg;
}

std::optional<AcceptedOmniTarget> select_omni_retarget_reference_target(
  const std::optional<AcceptedOmniTarget> & session_target,
  const std::optional<AcceptedOmniTarget> & cooldown_anchor_target, bool cooldown_active)
{
  if (session_target.has_value()) return session_target;
  if (cooldown_active && cooldown_anchor_target.has_value()) return cooldown_anchor_target;
  return std::nullopt;
}

std::optional<OmniCandidate> select_omni_candidate(
  const std::vector<OmniCandidate> & candidates,
  const std::optional<AcceptedOmniTarget> & reference_target, double current_abs_yaw_rad,
  double retarget_min_delta_deg)
{
  if (candidates.empty()) return std::nullopt;

  std::vector<OmniCandidate> sorted = candidates;
  std::sort(
    sorted.begin(), sorted.end(),
    [&](const OmniCandidate & lhs, const OmniCandidate & rhs) {
      if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;

      if (reference_target.has_value()) {
        const bool lhs_same =
          same_target_continuation(lhs, reference_target.value(), retarget_min_delta_deg);
        const bool rhs_same =
          same_target_continuation(rhs, reference_target.value(), retarget_min_delta_deg);
        if (lhs_same != rhs_same) return lhs_same;
      }

      const double lhs_delta = reference_delta_deg(lhs, reference_target, current_abs_yaw_rad);
      const double rhs_delta = reference_delta_deg(rhs, reference_target, current_abs_yaw_rad);
      if (std::abs(lhs_delta - rhs_delta) > 1e-6) return lhs_delta < rhs_delta;

      if (std::abs(lhs.confidence - rhs.confidence) > 1e-6) {
        return lhs.confidence > rhs.confidence;
      }

      if (lhs.timestamp != rhs.timestamp) return lhs.timestamp > rhs.timestamp;
      if (lhs.slot != rhs.slot) return static_cast<int>(lhs.slot) < static_cast<int>(rhs.slot);
      return static_cast<int>(lhs.armor_name) < static_cast<int>(rhs.armor_name);
    });

  return sorted.front();
}

OmniRetargetDecision evaluate_omni_retarget(
  const OmniCandidate & candidate, const std::optional<AcceptedOmniTarget> & reference_target,
  double current_abs_yaw_rad, bool cooldown_active, double retarget_min_delta_deg)
{
  OmniRetargetDecision decision;
  decision.accept = true;

  decision.candidate_delta_deg =
    reference_delta_deg(candidate, reference_target, current_abs_yaw_rad);

  if (reference_target.has_value()) {
    decision.same_target_continuation =
      same_target_continuation(candidate, reference_target.value(), retarget_min_delta_deg);
  }

  const bool large_retarget =
    !decision.same_target_continuation && decision.candidate_delta_deg >= retarget_min_delta_deg;
  if (large_retarget && cooldown_active && reference_target.has_value()) {
    decision.accept = false;
    decision.blocked = true;
    decision.block_reason = "cooldown_large_retarget";
  }

  return decision;
}

bool should_start_omni_retarget_cooldown(
  const OmniRetargetDecision & decision, double retarget_min_delta_deg)
{
  return decision.accept && !decision.same_target_continuation &&
         decision.candidate_delta_deg >= retarget_min_delta_deg;
}

}  // namespace omniperception
