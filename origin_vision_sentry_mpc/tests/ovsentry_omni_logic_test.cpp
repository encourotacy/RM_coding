#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "tasks/omniperception/ovsentry_omni_logic.hpp"

namespace
{
using namespace std::chrono_literals;

bool expect(bool condition, const std::string & message)
{
  if (condition) return true;
  std::cerr << message << std::endl;
  return false;
}

omniperception::OmniCandidate make_candidate(
  omniperception::OmniCameraSlot slot, auto_aim::ArmorName armor_name,
  auto_aim::ArmorPriority priority, double confidence, double abs_yaw_deg,
  std::chrono::steady_clock::time_point timestamp)
{
  omniperception::OmniCandidate candidate;
  candidate.slot = slot;
  candidate.armor_name = armor_name;
  candidate.priority = priority;
  candidate.confidence = confidence;
  candidate.timestamp = timestamp;
  candidate.base_big_yaw_rad = 0.0;
  candidate.abs_yaw_rad = abs_yaw_deg / 57.3;
  candidate.command = {true, false, candidate.abs_yaw_rad, 0.001};
  candidate.command.big_yaw = candidate.abs_yaw_rad;
  candidate.command.small_yaw = candidate.abs_yaw_rad;
  candidate.command.has_target_yaw = true;
  return candidate;
}

omniperception::AcceptedOmniTarget make_accepted_target(const omniperception::OmniCandidate & candidate)
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

}  // namespace

int main()
{
  const auto now = std::chrono::steady_clock::now();
  const double retarget_min_delta_deg = 20.0;
  const double current_abs_yaw_deg = 0.0;

  {
    const auto candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.7, 15.0, now);
    const auto decision = omniperception::evaluate_omni_retarget(
      candidate, std::nullopt, current_abs_yaw_deg / 57.3, false, retarget_min_delta_deg);
    if (!expect(decision.accept, "candidate without accepted target should be accepted")) return 1;
  }

  {
    const auto candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.7, 35.0, now);
    const auto decision = omniperception::evaluate_omni_retarget(
      candidate, std::nullopt, current_abs_yaw_deg / 57.3, false, retarget_min_delta_deg);
    if (!expect(
          omniperception::should_start_omni_retarget_cooldown(decision, retarget_min_delta_deg),
          "first large retarget should start cooldown")) {
      return 1;
    }
  }

  {
    const auto accepted_candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.8, 30.0, now);
    const auto continuation_candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.6, 42.0, now + 5ms);
    const auto decision = omniperception::evaluate_omni_retarget(
      continuation_candidate, make_accepted_target(accepted_candidate),
      current_abs_yaw_deg / 57.3, true, retarget_min_delta_deg);
    if (!expect(decision.accept, "same-target continuation should ignore cooldown")) return 1;
    if (!expect(
          decision.same_target_continuation,
          "same slot/name with small yaw delta should be treated as continuation")) {
      return 1;
    }
  }

  {
    const auto accepted_candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.8, 10.0, now);
    const auto blocked_candidate = make_candidate(
      omniperception::OmniCameraSlot::right, auto_aim::ArmorName::three,
      auto_aim::ArmorPriority::first, 0.9, 50.0, now + 5ms);
    const auto decision = omniperception::evaluate_omni_retarget(
      blocked_candidate, make_accepted_target(accepted_candidate),
      current_abs_yaw_deg / 57.3, true, retarget_min_delta_deg);
    if (!expect(!decision.accept, "large cross-target retarget should be blocked during cooldown")) {
      return 1;
    }
    if (!expect(decision.blocked, "blocked retarget should report blocked=true")) return 1;
  }

  {
    const auto cooldown_anchor_candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.8, 30.0, now);
    const auto cooldown_anchor = make_accepted_target(cooldown_anchor_candidate);
    const auto reference_target = omniperception::select_omni_retarget_reference_target(
      std::nullopt, cooldown_anchor, true);
    if (!expect(reference_target.has_value(), "active cooldown should reuse cooldown anchor target")) {
      return 1;
    }

    const auto blocked_candidate = make_candidate(
      omniperception::OmniCameraSlot::right, auto_aim::ArmorName::three,
      auto_aim::ArmorPriority::first, 0.9, 70.0, now + 5ms);
    const auto blocked_decision = omniperception::evaluate_omni_retarget(
      blocked_candidate, reference_target, current_abs_yaw_deg / 57.3, true, retarget_min_delta_deg);
    if (!expect(
          !blocked_decision.accept,
          "cross-lost cooldown should still block large cross-target retarget")) {
      return 1;
    }

    const auto continuation_candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.7, 38.0, now + 10ms);
    const auto continuation_decision = omniperception::evaluate_omni_retarget(
      continuation_candidate, reference_target, current_abs_yaw_deg / 57.3, true,
      retarget_min_delta_deg);
    if (!expect(
          continuation_decision.accept && continuation_decision.same_target_continuation,
          "cross-lost cooldown should still allow same-target continuation")) {
      return 1;
    }
  }

  {
    const auto cooldown_anchor_candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::first, 0.8, 30.0, now);
    const auto reference_target = omniperception::select_omni_retarget_reference_target(
      std::nullopt, make_accepted_target(cooldown_anchor_candidate), false);
    if (!expect(!reference_target.has_value(), "expired cooldown should not keep reference target")) {
      return 1;
    }

    const auto candidate = make_candidate(
      omniperception::OmniCameraSlot::right, auto_aim::ArmorName::three,
      auto_aim::ArmorPriority::first, 0.9, 55.0, now + 5ms);
    const auto decision = omniperception::evaluate_omni_retarget(
      candidate, reference_target, current_abs_yaw_deg / 57.3, false, retarget_min_delta_deg);
    if (!expect(decision.accept, "candidate should be accepted after cooldown expires")) return 1;
    if (!expect(
          omniperception::should_start_omni_retarget_cooldown(decision, retarget_min_delta_deg),
          "accepted large retarget after cooldown expiry should start a new cooldown")) {
      return 1;
    }
  }

  {
    const auto accepted_candidate = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::second, 0.4, 15.0, now);
    const auto same_target_low_conf = make_candidate(
      omniperception::OmniCameraSlot::left, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::second, 0.5, 18.0, now + 2ms);
    const auto different_slot_high_conf = make_candidate(
      omniperception::OmniCameraSlot::right, auto_aim::ArmorName::one,
      auto_aim::ArmorPriority::second, 0.95, 17.0, now + 3ms);

    const auto selected = omniperception::select_omni_candidate(
      {same_target_low_conf, different_slot_high_conf}, make_accepted_target(accepted_candidate),
      0.0, retarget_min_delta_deg);
    if (!expect(selected.has_value(), "selection should return a candidate")) return 1;
    if (!expect(
          selected->slot == omniperception::OmniCameraSlot::left &&
            selected->armor_name == auto_aim::ArmorName::one,
          "same-target continuation should beat higher-confidence cross-slot candidate")) {
      return 1;
    }
  }

  return 0;
}
