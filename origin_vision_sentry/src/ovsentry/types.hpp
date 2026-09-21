#ifndef OVSENTRY__TYPES_HPP
#define OVSENTRY__TYPES_HPP

#include <chrono>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/omniperception/detection.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"

namespace ovsentry
{

struct OmniCamConfig
{
  omniperception::CameraSpec spec;
  std::string dev_name;
  cv::Scalar color;
};

struct OmniInferenceResult
{
  OmniCamConfig cam;
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Armor> top_armor;
  double delta_yaw_deg = 0.0;
  double delta_pitch_deg = 0.0;
  double infer_ms = 0.0;
};

struct OmniCandidateFrame
{
  OmniInferenceResult result;
  std::chrono::steady_clock::time_point timestamp{};
  double base_big_yaw_rad = 0.0;
  bool has_base_big_yaw = false;
  std::optional<omniperception::OmniCandidate> candidate;
};

struct ArmorTargetMask
{
  bool enabled = false;
  std::vector<uint8_t> ignored_ids;
};

}  // namespace ovsentry

#endif  // OVSENTRY__TYPES_HPP
