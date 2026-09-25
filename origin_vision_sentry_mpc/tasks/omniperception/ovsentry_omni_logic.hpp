#ifndef OMNIPERCEPTION__OVSENTRY_OMNI_LOGIC_HPP
#define OMNIPERCEPTION__OVSENTRY_OMNI_LOGIC_HPP

#include <chrono>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "detection.hpp"
#include "io/command.hpp"

namespace omniperception
{
struct OmniCandidate
{
  OmniCameraSlot slot = OmniCameraSlot::unknown;
  auto_aim::ArmorName armor_name = auto_aim::ArmorName::not_armor;
  auto_aim::ArmorPriority priority = auto_aim::ArmorPriority::fifth;
  double confidence = 0.0;
  std::chrono::steady_clock::time_point timestamp{};
  double base_big_yaw_rad = 0.0;
  double abs_yaw_rad = 0.0;
  io::Command command{false, false, 0.0, 0.0};
};

struct AcceptedOmniTarget
{
  OmniCameraSlot slot = OmniCameraSlot::unknown;
  auto_aim::ArmorName armor_name = auto_aim::ArmorName::not_armor;
  auto_aim::ArmorPriority priority = auto_aim::ArmorPriority::fifth;
  double confidence = 0.0;
  std::chrono::steady_clock::time_point timestamp{};
  double base_big_yaw_rad = 0.0;
  double abs_yaw_rad = 0.0;
  io::Command command{false, false, 0.0, 0.0};
};

struct OmniRetargetDecision
{
  bool accept = false;
  bool blocked = false;
  bool same_target_continuation = false;
  double candidate_delta_deg = 0.0;
  std::string block_reason = "none";
};

struct OmniCamConfig
{
  CameraSpec spec;
  std::string dev_name;
};

struct OmniInferenceResult
{
  OmniCamConfig cam;
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Armor> top_armor;
  double delta_yaw_deg = 0.0;
};

struct OmniCandidateFrame
{
  OmniInferenceResult result;
  std::chrono::steady_clock::time_point timestamp{};
  double base_big_yaw_rad = 0.0;
  bool has_base_big_yaw = false;
  std::optional<OmniCandidate> candidate;
};

double angular_distance_deg(double lhs_rad, double rhs_rad);

std::optional<auto_aim::Armor> pick_top_armor(const std::list<auto_aim::Armor> & armors);

std::pair<double, double> calc_delta_angle_deg(
  const auto_aim::Armor & armor, const CameraSpec & cam);

std::optional<OmniCandidate> build_omni_candidate(
  const auto_aim::Armor & armor, OmniCameraSlot slot, double delta_yaw_deg,
  std::chrono::steady_clock::time_point timestamp, double base_big_yaw_rad);

double fill_omni_candidate(OmniCandidateFrame & frame, const CameraSpec & spec);

AcceptedOmniTarget make_accepted_omni_target(const OmniCandidate & candidate);

bool same_omni_target_continuation(
  const AcceptedOmniTarget & lhs, const AcceptedOmniTarget & rhs, double retarget_min_delta_deg);

std::optional<AcceptedOmniTarget> select_omni_retarget_reference_target(
  const std::optional<AcceptedOmniTarget> & session_target,
  const std::optional<AcceptedOmniTarget> & cooldown_anchor_target, bool cooldown_active);

std::optional<OmniCandidate> select_omni_candidate(
  const std::vector<OmniCandidate> & candidates,
  const std::optional<AcceptedOmniTarget> & reference_target, double current_abs_yaw_rad,
  double retarget_min_delta_deg);

OmniRetargetDecision evaluate_omni_retarget(
  const OmniCandidate & candidate, const std::optional<AcceptedOmniTarget> & reference_target,
  double current_abs_yaw_rad, bool cooldown_active, double retarget_min_delta_deg);

bool should_start_omni_retarget_cooldown(
  const OmniRetargetDecision & decision, double retarget_min_delta_deg);

}  // namespace omniperception

#endif
