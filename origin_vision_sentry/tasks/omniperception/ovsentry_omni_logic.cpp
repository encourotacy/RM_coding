#include "ovsentry_omni_logic.hpp"

#include <algorithm>
#include <cmath>

#include "tools/math_tools.hpp"

namespace omniperception
{
namespace
{
double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
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

      const double lhs_delta =
        reference_delta_deg(lhs, reference_target, current_abs_yaw_rad);
      const double rhs_delta =
        reference_delta_deg(rhs, reference_target, current_abs_yaw_rad);
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
    !decision.same_target_continuation &&
    decision.candidate_delta_deg >= retarget_min_delta_deg;
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
